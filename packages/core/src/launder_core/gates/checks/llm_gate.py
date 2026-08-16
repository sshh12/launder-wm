"""Check 8 — TECH_PLAN.md §7.1 row 8, §7.3, §7.4, §7.5. The only paid step.

**The LLM does not emit a verdict.** It emits observations against a per-passage
claim list shipped as repo data, and the code below computes the verdict from
those observations plus the level config. That collapses three problems into one
mechanism: prompt injection has no "output pass" to target, the tight fence
becomes claim-completeness rather than vibes, and all six levels share one
prompt, one cache namespace and one eval set.

Only two questions in this whole product have no closed form and therefore
justify a model: *is this readable human prose* (perplexity proxies fail exactly
on the interesting case — plain, flat prose is low-perplexity AND legitimate)
and *does this still assert claim c2* (embedding similarity cannot separate
"reworded" from "reworded and dropped a claim", and that separation is the whole
game).

Two policies here are not negotiable:

* `contains_embedded_instructions` is **fail-closed** and never provisional.
  Reporting it is the correct and complete response to injected text; acting on
  it is not.
* Everything else fails **open** on a provider error (§7.5). After the provider
  has exhausted its one retry, the submission clears PROVISIONALLY: the chime
  plays and the UI says "gate unavailable", but the campaign does not advance to
  the next level, nothing is written to the player's progress, and the run is
  excluded from the per-level best-distance ranking. The deterministic checks
  already caught every mechanical exploit, so a false clear costs one unranked
  run while a false rejection costs a player.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, ClassVar, Final

from launder_core.gates.registry import (
    META_COPY_KEY,
    META_COPY_TABLE,
    GateConfigError,
    GateContext,
    GateDependencyError,
    JudgeUnavailable,
    Observation,
    Phase,
    register,
)
from launder_core.schemas import CheckResult, Claim

__all__ = ["LlmGate", "derive_verdict"]

log = logging.getLogger(__name__)

#: Copy for these lives in `[check.llm_gate.reject_by_code]`.
_BY_CODE: Final[str] = "reject_by_code"
#: ...and for an unnatural-prose rejection, in
#: `[check.llm_gate.reject_by_unnatural_kind]`, keyed by the observed kind.
_BY_KIND: Final[str] = "reject_by_unnatural_kind"

_REQUIRE_CLAIMS: Final[frozenset[str]] = frozenset({"all", "required_only"})


def _fail(
    code: str, table: str, key: str, params: dict[str, Any], meta: dict[str, Any]
) -> CheckResult:
    return CheckResult(
        status="fail",
        check=LlmGate.name,
        code=code,
        params=params,
        meta={**meta, META_COPY_TABLE: table, META_COPY_KEY: key},
    )


def derive_verdict(
    obs: Observation, params: Mapping[str, Any], claims: tuple[Claim, ...]
) -> CheckResult:
    """§7.4, verbatim in structure. **Code computes the verdict, not the model.**

    `verdict_opinion` is read here only to be recorded: a persistent gap between
    it and the derived verdict means the prompt and the level config disagree,
    which is a prompt bug worth mining (§13, M5), never an input to the answer.
    """
    require = str(params["require_claims"])
    if require not in _REQUIRE_CLAIMS:
        raise GateConfigError(
            f"llm_gate: require_claims must be one of {sorted(_REQUIRE_CLAIMS)}, got {require!r}"
        )
    meta: dict[str, Any] = {
        "verdict_opinion": obs.verdict_opinion,
        "unnatural_kind": obs.unnatural_kind,
        "n_added_claims": len(obs.added_claims),
    }
    notes = obs.notes

    if obs.contains_embedded_instructions:
        # ALWAYS fail-closed. Never provisional, never soft, and the message
        # never quotes the offending text back — that is both an injection
        # surface and bad UX.
        return _fail("injection_attempt", _BY_CODE, "injection_attempt", {}, meta)

    if not obs.natural_prose:
        kind = obs.unnatural_kind or "word_salad"
        return _fail("not_natural_language", _BY_KIND, kind, {}, {**meta, "notes": notes})

    by_id = {c.id: c for c in obs.claims}
    required = [c for c in claims if require == "all" or c.required]
    missing = [c for c in required if not (by_id[c.id].present if c.id in by_id else False)]
    inverted = [
        c for c in required if c.id in by_id and by_id[c.id].how in ("negated", "reattributed")
    ]

    if inverted:
        return _fail(
            "meaning_inverted",
            _BY_CODE,
            "meaning_inverted",
            {"claim_label": inverted[0].label},
            {**meta, "notes": notes},
        )
    if len(missing) > int(params["max_missing_claims"]):
        return _fail(
            "meaning_drift",
            _BY_CODE,
            "meaning_drift",
            {"claim_label": missing[0].label},
            {**meta, "notes": notes, "n_missing": len(missing)},
        )
    if len(obs.added_claims) > int(params["max_added_claims"]):
        return _fail(
            "meaning_added",
            _BY_CODE,
            "meaning_added",
            {"added": obs.added_claims[0]},
            {**meta, "notes": notes},
        )
    return CheckResult(status="pass", check=LlmGate.name, params={}, meta=meta)


@register
class LlmGate:
    name: ClassVar[str] = "llm_gate"
    phase: ClassVar[int] = Phase.JUDGE
    #: The ONLY fail-open check in the registry (§7.5).
    fail_open: ClassVar[bool] = True
    config_params: ClassVar[frozenset[str]] = frozenset(
        {"require_claims", "max_missing_claims", "max_added_claims"}
    )
    required_params: ClassVar[frozenset[str]] = config_params
    #: `claim_label` is AUTHORED OFFLINE (`Claim.label`), which is why feedback
    #: can name a dropped claim without ever quoting the player's own text.
    #: `added` is the model's own <=8-word paraphrase, filtered before render.
    template_params: ClassVar[frozenset[str]] = frozenset({"claim_label", "added"})
    required_deps: ClassVar[frozenset[str]] = frozenset({"judge"})

    async def __call__(self, ctx: GateContext, params: Mapping[str, Any]) -> CheckResult:
        judge = ctx.deps.judge
        if judge is None:
            raise GateDependencyError(
                "llm_gate ran with no JudgeProvider in Deps. Set JUDGE_PROVIDER and wire one; "
                "the `fake` provider exists precisely so that a test or a local run can "
                "exercise this path without a network or an API key."
            )
        nonce = ctx.deps.nonce_factory()
        try:
            observation = await judge.observe(ctx.passage, ctx.normalized, nonce)
        except JudgeUnavailable as exc:
            log.warning("judge unavailable for passage %s: %s", ctx.passage.id, exc)
            return CheckResult(
                status="error",
                check=self.name,
                code="judge_unavailable",
                meta={"error": type(exc).__name__},
            )
        except Exception as exc:
            # A provider that raises anything else is buggy: §7.5 makes the
            # retry policy the PROVIDER's job, so by the time it reaches here
            # it should already be a JudgeUnavailable. Log it as the bug it is,
            # then treat it the same way — a bug in our provider must not cost
            # the player their submission.
            log.exception("judge provider %s raised an unexpected error", type(judge).__name__)
            return CheckResult(
                status="error",
                check=self.name,
                code="judge_error",
                meta={"error": type(exc).__name__},
            )

        return derive_verdict(observation, params, ctx.claims())
