"""Check 4 — TECH_PLAN.md §7.1 row 4.

Substring match after whitespace normalization. **Never ask an LLM "does this
phrase appear verbatim" — it will paraphrase-match.** That is the whole reason
this is a deterministic check sitting five positions before the judge.

The phrase carries watermark signal the player cannot touch, which is the point
of L3: launder AROUND fixed signal.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Final

from launder_core.gates.registry import (
    GateConfigError,
    GateContext,
    GateDataError,
    Phase,
    register,
)
from launder_core.schemas import CheckResult
from launder_core.scoring.normalize import normalize

__all__ = ["LockedPhrase", "resolve_ref"]

#: `phrase_ref` is a dotted path into the passage rather than a literal, so a
#: level config never has to know a passage's content. Only these paths exist;
#: an unknown one is a boot error, not a runtime surprise.
_REFS: Final[dict[str, str]] = {
    "passage.rules.locked_phrases": "locked_phrases",
    "passage.rules.unit_test_id": "unit_test_id",
}


def resolve_ref(ref: str, ctx: GateContext) -> Any:
    attribute = _REFS.get(ref)
    if attribute is None:
        raise GateConfigError(
            f"unknown reference {ref!r}; the resolvable paths are {sorted(_REFS)}. "
            "A ref that points nowhere would silently make its check vacuous."
        )
    return getattr(ctx.passage.rules, attribute)


@register
class LockedPhrase:
    name: ClassVar[str] = "locked_phrase"
    phase: ClassVar[int] = Phase.SHAPE
    fail_open: ClassVar[bool] = False
    config_params: ClassVar[frozenset[str]] = frozenset({"phrase_ref", "match"})
    required_params: ClassVar[frozenset[str]] = frozenset({"phrase_ref", "match"})
    #: The phrase is AUTHORED DATA, not player text, so rendering it back is
    #: safe — which is what lets the rejection be specific.
    template_params: ClassVar[frozenset[str]] = frozenset({"phrase"})

    _MATCH_MODES: ClassVar[frozenset[str]] = frozenset({"whitespace_insensitive", "exact"})

    def __call__(self, ctx: GateContext, params: Mapping[str, Any]) -> CheckResult:
        mode = str(params["match"])
        if mode not in self._MATCH_MODES:
            raise GateConfigError(
                f"locked_phrase: unknown match mode {mode!r}; expected one of "
                f"{sorted(self._MATCH_MODES)}"
            )
        phrases = resolve_ref(str(params["phrase_ref"]), ctx)
        if not isinstance(phrases, Sequence) or isinstance(phrases, str):
            raise GateConfigError(
                f"locked_phrase: {params['phrase_ref']} resolved to {type(phrases).__name__}, "
                "expected a sequence of phrases"
            )
        if not phrases:
            raise GateDataError(
                f"passage {ctx.passage.id} runs level {ctx.level.id}, which includes "
                "locked_phrase, but the passage declares no rules.locked_phrases. The check "
                "would be vacuous and the level would be a lie. Fix the passage, not this "
                "check — `forge verify` should have caught it before it shipped."
            )

        cfg = ctx.deps.normalize_config
        haystack = ctx.normalized if mode == "whitespace_insensitive" else ctx.raw
        for phrase in phrases:
            needle = normalize(str(phrase), cfg) if mode == "whitespace_insensitive" else phrase
            if needle not in haystack:
                return CheckResult(
                    status="fail",
                    check=self.name,
                    code=self.name,
                    params={"phrase": str(phrase)},
                )
        return CheckResult(status="pass", check=self.name, params={"phrase": str(phrases[0])})
