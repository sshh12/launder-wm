"""Check 6 — the tight fence, TECH_PLAN.md §7.1 row 6, §7.7. L4 only.

"Close paraphrase only" fails as an LLM instruction because "close" has no
anchor. Five bounded lexical metrics replace it, none of which asks a model for
a judgement call, and **all of them run on the same word tokenization the
scorer uses** — so the fence and the scoreboard can never disagree.

| Metric | Definition | Default | Catches |
|---|---|---|---|
| `content_word_retention` | shared non-stopword lemmas / original's | >= 0.55 | total vocabulary replacement |
| `word_distance_ratio` | `damerau(orig, sub) / len(orig)` | <= 0.45 | rewrites masquerading as edits |
| `length_ratio` | `len(sub) / len(orig)` | [0.80, 1.25] | compression to a summary; padding |
| `sentence_count_delta` | absolute difference in sentence count | <= 1 | discourse restructuring |
| `sentence_alignment` | greedy 1:1 match by content-word Jaccard | >= 0.40 | "merged three sentences into one" |

The last one earns its place: merging three sentences into one is invisible to
every corpus-level metric above it.

These same numbers render live in the UI as a fence meter under the textarea,
on the same debounce as the needle. A constraint the player can watch is a game
mechanic; a constraint discovered only on rejection is a bug report. The client
recomputes them from this definition, so this module is a port target too.

**Lemmatization is a small COMMITTED TABLE, never a runtime NLP dependency.**
A silently updated lemmatizer would change verdicts underneath the judge cache.
The table is two files plus one rule set:

* `data/config/lemma.v1.tsv` — the irregular forms.
* `data/config/stopwords.v1.txt` — the function words.
* the suffix rules in `lemma()` — plural / past / gerund / -ly / -ness. This is
  the `stemmer` `scoring.toml [lemma]` names, and it is `suffix_rules_v1`, NOT
  Porter2: the config said "porter2" for a while and no Porter2 has ever been
  in this file. Deeper stemming conflates words a reader would not call the
  same word, and this metric answers "did you keep the original's vocabulary",
  not "build me a search index".

All three are versioned by `scoring_version`. The dicts below are the FALLBACK
used when `data/` is not on disk (the bare-venv import CI enforces in §2.1);
when it is, the files win, and `assert_tables_match_files()` is what stops the
two copies from drifting.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any, ClassVar, Final

from launder_core.gates.feedback import CopyError, data_dir
from launder_core.gates.registry import (
    META_COPY_KEY,
    GateConfigError,
    GateContext,
    Phase,
    register,
)
from launder_core.schemas import CheckResult
from launder_core.scoring.damerau import damerau_levenshtein
from launder_core.scoring.normalize import normalize, words

__all__ = [
    "IRREGULAR_LEMMAS",
    "LEMMA_TABLE_PATH",
    "STOPWORDS",
    "STOPWORDS_PATH",
    "CloseParaphrase",
    "assert_tables_match_files",
    "content_lemmas",
    "irregular_lemmas",
    "lemma",
    "sentence_alignment",
    "sentences",
    "stopwords",
]

#: Paths relative to `data/`, mirrored in `scoring.toml [lemma]`.
LEMMA_TABLE_PATH: Final[str] = "config/lemma.v1.tsv"
STOPWORDS_PATH: Final[str] = "config/stopwords.v1.txt"

# ---------------------------------------------------------------------------
# The committed tables
# ---------------------------------------------------------------------------
#: Function words. Excluded from `content_word_retention` because keeping "the"
#: is not evidence that the player kept the original's vocabulary, and because
#: a fence that counted them would be trivially satisfiable by any English
#: sentence at all.
_STOPWORD_SOURCE: Final[str] = """
    a about above after again against all am an and any are as at be because been before being
    below between both but by can cannot could did do does doing down during each few for from
    further had has have having he her here hers herself him himself his how i if in into is it
    its itself just me more most my myself no nor not now of off on once only or other our ours
    ourselves out over own same she should so some such than that the their theirs them
    themselves then there these they this those through to too under until up very was we were
    what when where which while who whom why will with would you your yours yourself yourselves
"""

STOPWORDS: Final[frozenset[str]] = frozenset(_STOPWORD_SOURCE.split())

#: Irregular forms, the part no suffix rule can reach. Small on purpose: the
#: metric is a ratio over a ~200-word passage, so a handful of misses moves it
#: by half a percent, while a 40 MB lemmatizer would move the judge cache.
IRREGULAR_LEMMAS: Final[dict[str, str]] = {
    "am": "be",
    "are": "be",
    "is": "be",
    "was": "be",
    "were": "be",
    "been": "be",
    "being": "be",
    "has": "have",
    "had": "have",
    "having": "have",
    "does": "do",
    "did": "do",
    "done": "do",
    "doing": "do",
    "goes": "go",
    "went": "go",
    "gone": "go",
    "going": "go",
    "made": "make",
    "making": "make",
    "said": "say",
    "saying": "say",
    "says": "say",
    "took": "take",
    "taken": "take",
    "taking": "take",
    "came": "come",
    "coming": "come",
    "found": "find",
    "finding": "find",
    "gave": "give",
    "given": "give",
    "giving": "give",
    "knew": "know",
    "known": "know",
    "knowing": "know",
    "saw": "see",
    "seen": "see",
    "seeing": "see",
    "thought": "think",
    "thinking": "think",
    "told": "tell",
    "telling": "tell",
    "became": "become",
    "becoming": "become",
    "began": "begin",
    "begun": "begin",
    "beginning": "begin",
    "brought": "bring",
    "bringing": "bring",
    "built": "build",
    "building": "build",
    "bought": "buy",
    "buying": "buy",
    "chose": "choose",
    "chosen": "choose",
    "choosing": "choose",
    "children": "child",
    "men": "man",
    "women": "woman",
    "people": "person",
    "feet": "foot",
    "teeth": "tooth",
    "mice": "mouse",
    "geese": "goose",
    "data": "datum",
    "criteria": "criterion",
    "analyses": "analysis",
    "hypotheses": "hypothesis",
    "left": "leave",
    "leaving": "leave",
    "lost": "lose",
    "losing": "lose",
    "meant": "mean",
    "meaning": "mean",
    "met": "meet",
    "meeting": "meet",
    "paid": "pay",
    "paying": "pay",
    "put": "put",
    "ran": "run",
    "running": "run",
    "sent": "send",
    "sending": "send",
    "showed": "show",
    "shown": "show",
    "showing": "show",
    "wrote": "write",
    "written": "write",
    "writing": "write",
    "held": "hold",
    "holding": "hold",
    "kept": "keep",
    "keeping": "keep",
    "led": "lead",
    "leading": "lead",
    "spent": "spend",
    "spending": "spend",
    "stood": "stand",
    "standing": "stand",
    "understood": "understand",
    "understanding": "understand",
    "better": "good",
    "best": "good",
    "worse": "bad",
    "worst": "bad",
}


@lru_cache(maxsize=1)
def _load_tables() -> tuple[frozenset[str], dict[str, str]] | None:
    """`(stopwords, irregulars)` from `data/config/`, or `None` if unavailable.

    Lazy and cached, NOT loaded at import: `launder_core` must import in a venv
    holding only pydantic, blake3 and numpy with no `data/` tree anywhere (§2.1,
    asserted in CI), and an import-time filesystem read would break that.

    Returning `None` rather than raising is the fallback path, and it is only
    reachable when `data/` is genuinely absent — in which case the module-level
    dicts, which `assert_tables_match_files()` pins to the files, are used.

    THE EXCEPTION LIST IS THE WHOLE POINT. This caught `FileNotFoundError`,
    which is what `launder_core.watermark.config.data_dir` raises — but the
    `data_dir` imported here is `gates.feedback`'s, and that one raises
    `CopyError` (a plain `RuntimeError`) both when it cannot find `data/` and
    when `LAUNDER_DATA_DIR` points somewhere without a `config/` in it. So the
    fallback path was unreachable: in exactly the bare venv §2.1 promises to
    support, `lemma()` and `stopwords()` raised `CopyError` instead of using the
    in-code tables they exist for.
    """
    try:
        root = data_dir()
    except (CopyError, FileNotFoundError):
        return None
    lemma_path = root / LEMMA_TABLE_PATH
    stop_path = root / STOPWORDS_PATH
    if not (lemma_path.is_file() and stop_path.is_file()):
        return None

    irregulars: dict[str, str] = {}
    for lineno, raw in enumerate(lemma_path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("	")
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise GateConfigError(
                f"{lemma_path}:{lineno}: expected exactly two TAB-separated columns "
                f"(form, lemma), got {raw!r}. This table decides L4 verdicts; a "
                "half-parsed row is a fence that silently stops fencing."
            )
        irregulars[parts[0].lower()] = parts[1].lower()

    stops = {
        line.strip().lower()
        for line in stop_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    if not stops or not irregulars:
        raise GateConfigError(
            f"{lemma_path} / {stop_path}: one of the lemma tables is empty. "
            "content_word_retention would then count function words and pass anything."
        )
    return frozenset(stops), irregulars


def stopwords() -> frozenset[str]:
    """The function-word set actually in force."""
    tables = _load_tables()
    return STOPWORDS if tables is None else tables[0]


def irregular_lemmas() -> Mapping[str, str]:
    """The irregular-forms table actually in force."""
    tables = _load_tables()
    return IRREGULAR_LEMMAS if tables is None else tables[1]


def assert_tables_match_files() -> None:
    """The in-code fallback IS the committed file. Raises when they drift.

    Two copies of a table that decides verdicts is exactly the drift `forge
    lint-copy` catches for copy; this is the same rule for the lemma table.
    """
    tables = _load_tables()
    if tables is None:
        return
    stops, irregulars = tables
    if stops != STOPWORDS:
        only_file = sorted(stops - STOPWORDS)[:5]
        only_code = sorted(STOPWORDS - stops)[:5]
        raise GateConfigError(
            f"data/{STOPWORDS_PATH} and the STOPWORDS fallback in close_paraphrase.py "
            f"disagree (file-only: {only_file}, code-only: {only_code}). They decide L4's "
            "content_word_retention, so a split value means two different fences."
        )
    if irregulars != IRREGULAR_LEMMAS:
        keys = sorted(set(irregulars) ^ set(IRREGULAR_LEMMAS))[:5]
        raise GateConfigError(
            f"data/{LEMMA_TABLE_PATH} and the IRREGULAR_LEMMAS fallback in "
            f"close_paraphrase.py disagree (differing forms: {keys})."
        )


#: Stripped from a word before lemmatizing. Note this is the LEMMA path only —
#: the SCORER keeps punctuation attached, because "study." -> "study," is one
#: thing a player did and must cost 1 (§8.1). Here it would only add noise to a
#: vocabulary-overlap ratio.
_PUNCT_STRIP: Final[str] = "\"'`.,;:!?()[]{}<>-*_/" + chr(92) + "|@#$%^&+=~" + "\u2013\u2014\u2026"


def lemma(word: str) -> str:
    """Lowercase, strip attached punctuation, then irregulars, then suffixes.

    Suffix stripping is deliberately shallow — plural, past, gerund and the
    -ly/-ness adverbial pair. Deeper stemming (Porter's later steps) starts
    conflating words a reader would not call the same word, and this metric's
    job is to answer "did you keep the original's vocabulary", not to build a
    search index.
    """
    w = word.strip(_PUNCT_STRIP).lower()
    if not w:
        return ""
    irregular = irregular_lemmas().get(w)
    if irregular is not None:
        return irregular
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith(("sses", "shes", "ches", "xes", "zes")):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    if len(w) > 4 and w.endswith("ing"):
        stem = w[:-3]
        if len(stem) > 2 and stem[-1] == stem[-2] and stem[-1] not in "aeiou":
            stem = stem[:-1]
        return stem + "e" if len(stem) == 2 else stem
    if len(w) > 3 and w.endswith("ed"):
        stem = w[:-2]
        if len(stem) > 2 and stem[-1] == stem[-2] and stem[-1] not in "aeiou":
            stem = stem[:-1]
        return stem
    if len(w) > 4 and w.endswith("ly"):
        return w[:-2]
    return w


def content_lemmas(word_seq: Sequence[str]) -> frozenset[str]:
    """The non-stopword lemma SET. A set, not a bag: the metric asks whether the
    original's vocabulary survived, not how often each word was used."""
    out: set[str] = set()
    stops = stopwords()
    for w in word_seq:
        stem = lemma(w)
        if stem and stem not in stops and w.strip(_PUNCT_STRIP).lower() not in stops:
            out.add(stem)
    return frozenset(out)


_TERMINATORS: Final[tuple[str, ...]] = (".", "!", "?")
#: Closing marks that belong to the sentence they end, not to the next one.
_CLOSERS: Final[str] = "\"')]}\u00bb\u201d\u2019"
_WS_RUN: Final[re.Pattern[str]] = re.compile(r"\s+")


def sentences(text: str) -> tuple[str, ...]:
    """Split at whitespace that follows terminal punctuation.

    Written as a scan rather than one regex because the closing quote in
    ``"Then what?" she asked.`` has to stay with the sentence it closes, and
    Python's `re` cannot express a variable-width lookbehind. Deliberately
    simple otherwise — every heuristic beyond this ("Dr. Smith") trades a rare
    improvement for one more rule the TS port has to reproduce exactly, and both
    metrics built on it (a COUNT DELTA and a match quality) survive a
    consistently wrong splitter.
    """
    out: list[str] = []
    start = 0
    for gap in _WS_RUN.finditer(text):
        head = text[start : gap.start()]
        if head.rstrip(_CLOSERS).endswith(_TERMINATORS):
            stripped = head.strip()
            if stripped:
                out.append(stripped)
            start = gap.end()
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return tuple(out)


def sentence_alignment(original: str, submission: str) -> float:
    """Greedy 1:1 sentence match by content-word Jaccard; the score is the
    WEAKEST original sentence's match.

    §7.7 requires that *every* original sentence needs a partner, so the metric
    is a minimum rather than a mean: merging three sentences into one leaves two
    originals unpartnered at 0.0, and no amount of good matching elsewhere hides
    that. An original with no sentences at all scores 1.0 — vacuously aligned.
    """
    a_sents = [content_lemmas(words(s)) for s in sentences(original)]
    b_sents = [content_lemmas(words(s)) for s in sentences(submission)]
    if not a_sents:
        return 1.0
    available = list(range(len(b_sents)))
    worst = 1.0
    for a_set in a_sents:
        best_score = 0.0
        best_idx = -1
        for idx in available:
            b_set = b_sents[idx]
            union = a_set | b_set
            jaccard = 1.0 if not union else len(a_set & b_set) / len(union)
            if jaccard > best_score:
                best_score, best_idx = jaccard, idx
        if best_idx >= 0:
            available.remove(best_idx)
        worst = min(worst, best_score)
    return worst


def _pct(x: float) -> int:
    """Metrics render as whole percentages. The player reads "41% of key words
    kept, needs 55%", not "0.4142857"."""
    return round(x * 100)


@register
class CloseParaphrase:
    name: ClassVar[str] = "close_paraphrase"
    phase: ClassVar[int] = Phase.FENCE
    fail_open: ClassVar[bool] = False
    config_params: ClassVar[frozenset[str]] = frozenset(
        {
            "min_content_word_retention",
            "max_word_distance_ratio",
            "length_ratio",
            "max_sentence_count_delta",
            "min_sentence_alignment",
        }
    )
    required_params: ClassVar[frozenset[str]] = config_params
    template_params: ClassVar[frozenset[str]] = frozenset(
        {
            "content_word_retention_pct",
            "min_content_word_retention_pct",
            "word_distance_ratio_pct",
            "max_word_distance_ratio_pct",
            "length_ratio_pct",
            "length_ratio_min_pct",
            "length_ratio_max_pct",
            "sentence_count_delta",
            "max_sentence_count_delta",
            "sentence_alignment_pct",
            "min_sentence_alignment_pct",
        }
    )
    copy_keys: ClassVar[frozenset[str]] = frozenset(
        {"reject", "reject_distance", "reject_length", "reject_sentences", "reject_alignment"}
    )

    def __call__(self, ctx: GateContext, params: Mapping[str, Any]) -> CheckResult:
        bounds = params["length_ratio"]
        if not isinstance(bounds, Sequence) or len(bounds) != 2:
            raise GateConfigError(
                "close_paraphrase: length_ratio must be a two-element [min, max] array, got "
                f"{bounds!r}"
            )
        lo, hi = float(bounds[0]), float(bounds[1])
        min_retention = float(params["min_content_word_retention"])
        max_ratio = float(params["max_word_distance_ratio"])
        max_delta = int(params["max_sentence_count_delta"])
        min_alignment = float(params["min_sentence_alignment"])

        cfg = ctx.deps.normalize_config
        original = normalize(ctx.passage.text, cfg)
        a_words = words(original)
        b_words = ctx.words

        a_lemmas = content_lemmas(a_words)
        b_lemmas = content_lemmas(b_words)
        retention = 1.0 if not a_lemmas else len(a_lemmas & b_lemmas) / len(a_lemmas)
        distance_ratio = (
            0.0 if not a_words else damerau_levenshtein(a_words, b_words) / len(a_words)
        )
        length_ratio = 1.0 if not a_words else len(b_words) / len(a_words)
        a_sentences, b_sentences = sentences(original), sentences(ctx.normalized)
        delta = abs(len(b_sentences) - len(a_sentences))
        alignment = sentence_alignment(original, ctx.normalized)

        rendered: dict[str, Any] = {
            "content_word_retention_pct": _pct(retention),
            "min_content_word_retention_pct": _pct(min_retention),
            "word_distance_ratio_pct": _pct(distance_ratio),
            "max_word_distance_ratio_pct": _pct(max_ratio),
            "length_ratio_pct": _pct(length_ratio),
            "length_ratio_min_pct": _pct(lo),
            "length_ratio_max_pct": _pct(hi),
            "sentence_count_delta": delta,
            "max_sentence_count_delta": max_delta,
            "sentence_alignment_pct": _pct(alignment),
            "min_sentence_alignment_pct": _pct(min_alignment),
        }
        meta: dict[str, Any] = {
            "content_word_retention": retention,
            "word_distance_ratio": distance_ratio,
            "length_ratio": length_ratio,
            "sentence_count_delta": delta,
            "sentence_alignment": alignment,
        }

        # Evaluation order is a UX decision: the first thing the player is told
        # should be the most actionable one. Vocabulary first, then shape.
        #
        # THE COPY KEYS. All five now exist in copy.toml, and `copy_keys` above
        # declares them so `forge lint-copy` requires them: rule 1 used to check
        # `label`/`blurb`/`reject` only — the keys that EXIST rather than the
        # keys a check can ASK FOR — so `reject_distance` and `reject_alignment`
        # fell back to `reject`, and a player who failed on distance was shown
        # the RETENTION message quoting retention numbers that were fine, with
        # the lint reporting "OK".
        failures: list[tuple[str, str]] = []
        if retention < min_retention:
            failures.append((f"{self.name}_retention", "reject"))
        if distance_ratio > max_ratio:
            failures.append((f"{self.name}_distance", "reject_distance"))
        if not (lo <= length_ratio <= hi):
            failures.append((f"{self.name}_length", "reject_length"))
        if delta > max_delta:
            failures.append((f"{self.name}_sentences", "reject_sentences"))
        if alignment < min_alignment:
            failures.append((f"{self.name}_alignment", "reject_alignment"))

        if failures:
            code, copy_key = failures[0]
            return CheckResult(
                status="fail",
                check=self.name,
                code=code,
                params=rendered,
                meta={**meta, META_COPY_KEY: copy_key},
            )
        return CheckResult(status="pass", check=self.name, params=rendered, meta=meta)
