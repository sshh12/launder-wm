"""Check 2 — TECH_PLAN.md §7.1 row 2.

Tokenize and count. It exists because deleting your way to a short passage is
not laundering: a short text carries too little evidence for the detector either
way, so without a floor the puzzle solves itself.

"Floor" means the length minimum and NOTHING ELSE in this product. §10.7's
one-word-one-meaning rule: an earlier draft used "floor" for the detector
threshold as well, on the same screen. The threshold is "the line".
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from launder_core.gates.registry import GateContext, Phase, register
from launder_core.schemas import CheckResult

__all__ = ["WordFloor"]


@register
class WordFloor:
    name: ClassVar[str] = "word_floor"
    phase: ClassVar[int] = Phase.LENGTH
    fail_open: ClassVar[bool] = False
    config_params: ClassVar[frozenset[str]] = frozenset({"min_words"})
    required_params: ClassVar[frozenset[str]] = frozenset({"min_words"})
    template_params: ClassVar[frozenset[str]] = frozenset({"min_words", "n_words"})

    def __call__(self, ctx: GateContext, params: Mapping[str, Any]) -> CheckResult:
        min_words = int(params["min_words"])
        n_words = len(ctx.words)
        rendered = {"n_words": n_words, "min_words": min_words}
        if n_words < min_words:
            return CheckResult(status="fail", check=self.name, code=self.name, params=rendered)
        return CheckResult(status="pass", check=self.name, params=rendered)
