"""Check — "you may only edit the opening".

Every edit must land inside a window of the ORIGINAL word sequence. It exists
to teach the one thing about this watermark that is genuinely counter-intuitive
and that no other check forces you to confront: **an edit changes the g-values
of the tokens that follow it, not the ones before it.**

Because the context window runs left to right, a word you change re-keys the
n-grams downstream of it and nothing upstream. So the same edit is worth much
more at the start of a passage than at the end, and a player who has been
editing wherever the mirror looks hottest has probably been leaving most of the
available drop on the table. Confining them to the opening makes that lever the
only lever they have, and the reading falls further than they expect.

It reuses `score()` — the same Damerau-Levenshtein backtrace the budget and the
scoreboard use — rather than diffing the words itself. One distance function in
the product, so "6 changed" means the same thing to this check, to the budget,
and to the number under the textarea.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from launder_core.gates.registry import GateConfigError, GateContext, Phase, register
from launder_core.schemas import CheckResult
from launder_core.scoring.damerau import score

__all__ = ["EditRegion"]


@register
class EditRegion:
    name: ClassVar[str] = "edit_region"
    #: SHAPE, beside `edit_budget`: both are deterministic word-diff checks that
    #: cost microseconds, and both must run long before the paid judge.
    phase: ClassVar[int] = Phase.SHAPE
    fail_open: ClassVar[bool] = False
    config_params: ClassVar[frozenset[str]] = frozenset({"editable_prefix_words"})
    required_params: ClassVar[frozenset[str]] = frozenset({"editable_prefix_words"})
    template_params: ClassVar[frozenset[str]] = frozenset(
        {"editable_prefix_words", "first_bad_word_index", "n_outside"}
    )

    def __call__(self, ctx: GateContext, params: Mapping[str, Any]) -> CheckResult:
        window = int(params["editable_prefix_words"])
        if window < 1:
            raise GateConfigError(
                "edit_region: editable_prefix_words must be >= 1. A window of zero is a "
                "level nobody can submit to, which is a config bug rather than a hard level."
            )
        result = score(ctx.passage.text, ctx.normalized, ctx.deps.normalize_config)

        # `i` indexes the ORIGINAL word sequence, which is the sequence the
        # window is defined over — the player is told "the first N words of the
        # passage", and the passage is what they can see. Using `j` would move
        # the goalposts as soon as they inserted a word.
        outside = [op.i for op in result.ops if op.i >= window]
        rendered: dict[str, Any] = {
            "editable_prefix_words": window,
            "n_outside": len(outside),
            # 1-based: the rejection is read by a person counting words, not by
            # a programmer indexing an array.
            "first_bad_word_index": (min(outside) + 1) if outside else 0,
        }
        if outside:
            return CheckResult(status="fail", check=self.name, code=self.name, params=rendered)
        return CheckResult(status="pass", check=self.name, params=rendered)
