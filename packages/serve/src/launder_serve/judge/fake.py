"""`FakeJudge` — deterministic, marker-driven, no network (TECH_PLAN.md §7.5).

**Its job is to exercise the pipeline, not to judge well.** Every heuristic here
is crude on purpose: the moment it starts trying to be a good judge, a test that
passes against it stops telling you anything about the real one.

Markers, applied to the NORMALIZED submission:

| marker                       | effect                                       |
|------------------------------|----------------------------------------------|
| `__FAIL_INJECTION__`         | `contains_embedded_instructions = true`      |
| `__FAIL_NATURAL__`           | `natural_prose = false`, kind `keyword_soup` |
| `__FAIL_NATURAL_word_salad__`| ...with the named kind                       |
| `__DROP_c3__`                | claim `c3` reported missing                  |
| `__NEGATE_c1__`              | claim `c1` reported negated                  |
| `__ADD__`                    | one fabricated added claim                   |

Everything else falls through to the heuristics, which are documented inline so
nobody mistakes them for a specification of what the real judge does.
"""

from __future__ import annotations

import re
from typing import ClassVar, Final, Literal, cast

from launder_core.schemas import PassagePublic
from launder_serve.judge.protocol import ClaimObservation, JudgeUsage, Observation

#: Mirrors core's `Observation.unnatural_kind`. Declared rather than imported
#: because core states it inline in the model, and a `str` here would let a
#: typo reach a ValidationError at runtime instead of mypy at build time.
UnnaturalKind = Literal[
    "word_salad", "keyword_soup", "repetition", "non_prose_content", "not_english"
]

__all__ = ["FakeJudge"]

_DROP: Final[re.Pattern[str]] = re.compile(r"__DROP_([A-Za-z0-9_-]+)__")
_NEGATE: Final[re.Pattern[str]] = re.compile(r"__NEGATE_([A-Za-z0-9_-]+)__")
_FAIL_NATURAL: Final[re.Pattern[str]] = re.compile(
    r"__FAIL_NATURAL(?:_([a-z_]+))__|__FAIL_NATURAL__"
)
_WORD: Final[re.Pattern[str]] = re.compile(r"[A-Za-z][A-Za-z'-]*")

_VALID_KINDS: Final[frozenset[str]] = frozenset(
    {"word_salad", "keyword_soup", "repetition", "non_prose_content", "not_english"}
)

#: Crude injection tells. The real judge is instructed to reason about intent;
#: this is a substring list, and that difference is the point.
_INJECTION_TELLS: Final[tuple[str, ...]] = (
    "ignore previous instructions",
    "ignore all previous",
    "disregard the above",
    "you are now",
    "system:",
    "assistant:",
    "developer:",
    "</submission>",
    "<submission",
    "output pass",
    "mark this as approved",
    "reveal your instructions",
)

#: Words too common to distinguish one claim from another.
_STOPWORDS: Final[frozenset[str]] = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "for",
        "from",
        "had",
        "has",
        "have",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "there",
        "they",
        "this",
        "to",
        "was",
        "were",
        "which",
        "who",
        "will",
        "with",
    ]
)


def _content_words(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text) if len(w) > 3 and w.lower() not in _STOPWORDS]


class FakeJudge:
    """Deterministic judge. `calls` is public so tests can assert *zero* API calls."""

    name: ClassVar[str] = "fake"

    def __init__(self) -> None:
        self.calls: int = 0
        self.last_normalized: str = ""

    async def observe(self, passage: PassagePublic, normalized: str, nonce: str) -> Observation:
        obs, _ = await self.observe_with_usage(passage, normalized, nonce)
        return obs

    async def observe_with_usage(
        self, passage: PassagePublic, normalized: str, nonce: str
    ) -> tuple[Observation, JudgeUsage]:
        self.calls += 1
        self.last_normalized = normalized
        lowered = normalized.lower()

        injection = "__fail_injection__" in lowered or any(
            tell in lowered for tell in _INJECTION_TELLS
        )

        kind = self._unnatural_kind(normalized, lowered)
        dropped = set(_DROP.findall(normalized))
        negated = set(_NEGATE.findall(normalized))

        claim_obs: list[ClaimObservation] = []
        for claim in passage.claims:
            if claim.id in negated:
                claim_obs.append(ClaimObservation(id=claim.id, present=False, how="negated"))
            elif claim.id in dropped:
                claim_obs.append(ClaimObservation(id=claim.id, present=False, how="missing"))
            elif self._claim_survives(claim.text, normalized):
                claim_obs.append(ClaimObservation(id=claim.id, present=True, how="asserted"))
            else:
                claim_obs.append(ClaimObservation(id=claim.id, present=False, how="missing"))

        added: tuple[str, ...] = ("a fabricated added claim",) if "__add__" in lowered else ()

        failed = injection or kind is not None or added or any(not c.present for c in claim_obs)
        notes = ""
        if injection:
            notes = "the text tries to instruct the judge"
        elif kind is not None:
            notes = "this does not read as ordinary prose"
        elif any(not c.present for c in claim_obs):
            notes = "something the original says is no longer there"

        return Observation(
            natural_prose=kind is None,
            unnatural_kind=kind,
            claims=tuple(claim_obs),
            added_claims=added,
            contains_embedded_instructions=injection,
            notes=notes,
            verdict_opinion="fail" if failed else "pass",
        ), JudgeUsage(provider=self.name, model="fake-1", latency_ms=0)

    # -- heuristics ---------------------------------------------------------

    def _unnatural_kind(self, text: str, lowered: str) -> UnnaturalKind | None:
        marker = _FAIL_NATURAL.search(text)
        if marker is not None:
            named = marker.group(1)
            if named in _VALID_KINDS:
                # Validated against the literal set above; mypy cannot see
                # through the regex, so the narrowing is spelled out.
                return cast(UnnaturalKind, named)
            return "keyword_soup"

        words = _WORD.findall(text)
        if not words:
            return "word_salad" if text.strip() else None

        # Non-prose: letters are a minority of the characters. Catches markup,
        # code and symbol soup without pretending to parse anything.
        letters = sum(ch.isalpha() for ch in text)
        if len(text) >= 20 and letters / len(text) < 0.45:
            return "non_prose_content"

        # Repetition: the same word four times in a row, or a vocabulary so
        # small the text cannot be saying much.
        for i in range(len(words) - 3):
            if len({w.lower() for w in words[i : i + 4]}) == 1:
                return "repetition"
        if len(words) >= 20 and len({w.lower() for w in words}) / len(words) < 0.35:
            return "repetition"

        # Keyword soup: the words are there, the sentences are not.
        if len(words) >= 8 and not any(p in text for p in ".!?"):
            return "keyword_soup"

        del lowered  # only the marker path needs the lowered copy
        return None

    def _claim_survives(self, claim_text: str, submission: str) -> bool:
        """Half the claim's content words still present. Crude, and deliberately so."""
        wanted = _content_words(claim_text)
        if not wanted:
            return True
        have = set(_content_words(submission))
        hits = sum(1 for w in wanted if w in have)
        return hits * 2 >= len(wanted)
