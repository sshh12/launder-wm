"""Check 3 — TECH_PLAN.md §7.1 row 3, §8.3.

**It calls the same `score()` the scoreboard uses.** That is the entire design:
the budget and the scoreboard are one number computed once, so they cannot
disagree, and a player who can see "7 changed" under the textarea knows exactly
what the gate will say. If this check ever grows its own distance function, the
first bug report will be "it says 12 but it rejected me at 12".
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from launder_core.gates.registry import GateContext, Phase, register
from launder_core.schemas import CheckResult
from launder_core.scoring.damerau import score

__all__ = ["EditBudget"]


@register
class EditBudget:
    name: ClassVar[str] = "edit_budget"
    phase: ClassVar[int] = Phase.SHAPE
    fail_open: ClassVar[bool] = False
    config_params: ClassVar[frozenset[str]] = frozenset({"max_word_distance"})
    required_params: ClassVar[frozenset[str]] = frozenset({"max_word_distance"})
    template_params: ClassVar[frozenset[str]] = frozenset({"distance", "max_word_distance"})

    def __call__(self, ctx: GateContext, params: Mapping[str, Any]) -> CheckResult:
        budget = int(params["max_word_distance"])
        result = score(ctx.passage.text, ctx.normalized, ctx.deps.normalize_config)
        rendered = {"distance": result.distance, "max_word_distance": budget}
        # `ops` deliberately does NOT go into meta: CheckResult.trace ships to
        # the client in SubmitResponse, and the diff is already there once, in
        # ScoreSummary. Twice is a payload bug waiting to happen.
        if result.distance > budget:
            return CheckResult(status="fail", check=self.name, code=self.name, params=rendered)
        return CheckResult(status="pass", check=self.name, params=rendered)
