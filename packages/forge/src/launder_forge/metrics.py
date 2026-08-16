"""The §6.5 difficulty metrics — cheap projections of data already computed.

Every one of these is a *selection criterion*, not decoration. In particular
`upstream_leverage` is the metric that enforces CONCEPT.md's guardrail
structurally (§4.6 mechanism 3): a passage where attacking the hot word beats
cutting upstream teaches the wrong lesson and is rejected, so the shipped
passage set is *selected* for the honest skill rather than merely described by
it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from launder_forge.corebridge import words as split_words
from launder_forge.numerics import ScoredPassage, U8Array, score_ids

__all__ = ["Difficulty", "WordMap", "compute_difficulty", "gini", "hot_runs", "map_words_to_tokens"]


def gini(values: np.ndarray) -> float:
    """Gini coefficient of per-token heat. High -> a few tokens carry the mark
    -> a surgical kill exists -> good L2/L4 material."""
    x = np.sort(np.asarray(values, dtype=np.float64))
    n = x.shape[0]
    if n == 0:
        return 0.0
    total = x.sum()
    if total <= 0:
        return 0.0
    index = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * (index * x).sum()) / (n * total) - (n + 1.0) / n)


def hot_runs(
    heat: np.ndarray, *, quantile: float = 0.75, min_len: int = 3
) -> list[tuple[int, int]]:
    """Maximal runs of >= `min_len` consecutive tokens with heat above p75.

    Returned as half-open token-index ranges over the *current-token* indexing
    the mirror colours, i.e. row `i` is current token `i + ngram_len - 1`; the
    caller supplies heat already in that frame.
    """
    if heat.size == 0:
        return []
    threshold = float(np.quantile(heat, quantile))
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, v in enumerate(heat):
        if v > threshold:
            if start is None:
                start = i
        elif start is not None:
            if i - start >= min_len:
                runs.append((start, i))
            start = None
    if start is not None and heat.size - start >= min_len:
        runs.append((start, int(heat.size)))
    return runs


@dataclass(frozen=True, slots=True)
class WordMap:
    """Alignment between the word sequence and the token sequence.

    Built by re-encoding growing prefixes, which is exact and costs one encode
    per word (~0.65 ms each) — cheap at authoring time and immune to the
    leading-space merge traps that make character-offset arithmetic wrong on a
    byte-fallback BPE (§4.4a).
    """

    words: tuple[str, ...]
    token_spans: tuple[tuple[int, int], ...]

    def tokens_for(self, word_index: int) -> tuple[int, int]:
        return self.token_spans[word_index]


def map_words_to_tokens(text: str, tokenizer: Any) -> WordMap:
    ws = split_words(text)
    spans: list[tuple[int, int]] = []
    prev = 0
    for i in range(len(ws)):
        prefix = " ".join(ws[: i + 1])
        n = len(tokenizer.encode(prefix))
        spans.append((prev, max(prev, n)))
        prev = n
    return WordMap(words=tuple(ws), token_spans=tuple(spans))


@dataclass(slots=True)
class Difficulty:
    """§6.5's table, as computed values plus the provenance of how."""

    margin_z: float
    n_scored: int
    hot_runs: int
    concentration: float
    upstream_leverage: float
    median_eff_choices: float
    masked_fraction: float
    best_single_edit_drop_z: float | None = None
    par_upper: int | None = None
    exhaustive_clear_free_upto_k: int = 0
    locked_share: float | None = None
    retokenize_ok: bool = True
    judge_baseline_pass: bool = True
    z_total: float = 0.0
    n_words: int = 0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "margin_z": self.margin_z,
            "n_scored": self.n_scored,
            "hot_runs": self.hot_runs,
            "concentration": self.concentration,
            "upstream_leverage": self.upstream_leverage,
            "median_eff_choices": self.median_eff_choices,
            "masked_fraction": self.masked_fraction,
            "best_single_edit_drop_z": self.best_single_edit_drop_z,
            "par_upper": self.par_upper,
            "exhaustive_clear_free_upto_k": self.exhaustive_clear_free_upto_k,
            "locked_share": self.locked_share,
            "retokenize_ok": self.retokenize_ok,
            "judge_baseline_pass": self.judge_baseline_pass,
            "z_total": self.z_total,
            "n_words": self.n_words,
        }


def _heat_in_token_frame(scored: ScoredPassage, ngram_len: int, n_tokens: int) -> np.ndarray:
    """Row `i` of the g-matrix is current token `i + ngram_len - 1` (§4.4)."""
    heat = np.zeros(n_tokens, dtype=np.float64)
    offset = ngram_len - 1
    take = min(scored.heat.shape[0], n_tokens - offset)
    if take > 0:
        heat[offset : offset + take] = scored.heat[:take]
    return heat


def _delete_word(ws: list[str], index: int) -> str:
    return " ".join(ws[:index] + ws[index + 1 :])


def compute_difficulty(
    text: str,
    token_ids: list[int],
    *,
    tokenizer: Any,
    keys: tuple[int, ...],
    ngram_len: int,
    table: U8Array,
    context_history_size: int,
    z_star: float,
    inflation: float = 1.0,
    eff_choices: list[float] | None = None,
    locked_phrases: tuple[str, ...] = (),
    judge_baseline_pass: bool = True,
) -> Difficulty:
    """Every §6.5 metric that does not need the solver or the judge."""
    scored = score_ids(
        token_ids,
        keys=keys,
        ngram_len=ngram_len,
        table=table,
        context_history_size=context_history_size,
        inflation=inflation,
    )
    encoded = tokenizer.encode(text)
    retok_ok = list(encoded) == list(token_ids)

    heat_tokens = _heat_in_token_frame(scored, ngram_len, len(token_ids))
    runs = hot_runs(heat_tokens)
    wmap = map_words_to_tokens(text, tokenizer)
    ws = list(wmap.words)

    leverage = _upstream_leverage(
        ws,
        runs,
        wmap,
        base_z=scored.z,
        tokenizer=tokenizer,
        keys=keys,
        ngram_len=ngram_len,
        table=table,
        context_history_size=context_history_size,
        inflation=inflation,
    )

    locked_share = None
    if locked_phrases:
        locked_share = _locked_share(text, locked_phrases, wmap, heat_tokens)

    return Difficulty(
        margin_z=scored.z - z_star,
        n_scored=scored.n_scored,
        hot_runs=len(runs),
        concentration=gini(scored.heat),
        upstream_leverage=leverage,
        median_eff_choices=float(np.median(eff_choices)) if eff_choices else 0.0,
        masked_fraction=scored.masked_fraction,
        locked_share=locked_share,
        retokenize_ok=retok_ok,
        judge_baseline_pass=judge_baseline_pass,
        z_total=scored.z,
        n_words=len(ws),
        notes=[] if retok_ok else ["encode(text) != token_ids — this passage desyncs the browser"],
    )


def _z_of_text(
    text: str,
    *,
    tokenizer: Any,
    keys: tuple[int, ...],
    ngram_len: int,
    table: U8Array,
    context_history_size: int,
    inflation: float,
) -> float:
    ids = tokenizer.encode(text)
    if len(ids) < ngram_len:
        return 0.0
    return score_ids(
        ids,
        keys=keys,
        ngram_len=ngram_len,
        table=table,
        context_history_size=context_history_size,
        inflation=inflation,
    ).z


def _upstream_leverage(
    ws: list[str],
    runs: list[tuple[int, int]],
    wmap: WordMap,
    *,
    base_z: float,
    tokenizer: Any,
    keys: tuple[int, ...],
    ngram_len: int,
    table: U8Array,
    context_history_size: int,
    inflation: float,
) -> float:
    """THE core skill metric (§6.5).

    Mean over hot runs of `dz(delete the word immediately upstream)` divided by
    `dz(delete the run's first word)`. Above 1.3 means cutting upstream
    genuinely beats cutting the hot word — the passage actually teaches the
    ripple. Below 1.0 the passage teaches the wrong lesson and triage rejects
    it by name.
    """
    if not runs or not ws:
        return 0.0
    ratios: list[float] = []
    for run_start, _ in runs:
        first_word = _word_containing_token(wmap, run_start)
        if first_word is None or first_word == 0:
            continue
        z_hot = _z_of_text(
            _delete_word(ws, first_word),
            tokenizer=tokenizer,
            keys=keys,
            ngram_len=ngram_len,
            table=table,
            context_history_size=context_history_size,
            inflation=inflation,
        )
        z_up = _z_of_text(
            _delete_word(ws, first_word - 1),
            tokenizer=tokenizer,
            keys=keys,
            ngram_len=ngram_len,
            table=table,
            context_history_size=context_history_size,
            inflation=inflation,
        )
        drop_hot = base_z - z_hot
        drop_up = base_z - z_up
        if drop_hot <= 1e-9:
            continue
        ratios.append(drop_up / drop_hot)
    return float(np.mean(ratios)) if ratios else 0.0


def _word_containing_token(wmap: WordMap, token_index: int) -> int | None:
    for i, (lo, hi) in enumerate(wmap.token_spans):
        if lo <= token_index < hi:
            return i
    return None


def _locked_share(
    text: str, phrases: tuple[str, ...], wmap: WordMap, heat_tokens: np.ndarray
) -> float:
    """Fraction of total evidence inside the locked phrase. L3 wants 0.15-0.45."""
    total = float(heat_tokens.sum())
    if total <= 0:
        return 0.0
    inside = 0.0
    ws = list(wmap.words)
    for phrase in phrases:
        pw = split_words(phrase)
        if not pw:
            continue
        for i in range(len(ws) - len(pw) + 1):
            if ws[i : i + len(pw)] == pw:
                lo = wmap.token_spans[i][0]
                hi = wmap.token_spans[i + len(pw) - 1][1]
                inside += float(heat_tokens[lo:hi].sum())
                break
    return inside / total
