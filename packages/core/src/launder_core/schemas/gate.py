"""Gate contracts: check results, level configuration, the outcome envelope.

The gate is an ordered list of checks and ORDER IS THE SEMANTICS
(TECH_PLAN.md §7.1). Every deterministic check runs before `llm_gate`, so a
submission that busts the budget, the floor, the locked phrase, the unit test
or unicode sanitation costs zero API calls — and `detector_threshold` at
position 7 means only submissions that already beat the watermark ever reach
the LLM.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "CheckResult",
    "CheckSpec",
    "CheckStatus",
    "GateFailure",
    "GateResult",
    "LevelConfig",
    "LevelsFile",
    "ParSource",
]

CheckStatus = Literal["pass", "fail", "error"]
ParSource = Literal["authored", "first_n_plays"]


class CheckResult(BaseModel):
    """One check's verdict (TECH_PLAN.md §7.6).

    `params` fills the feedback template from `data/config/copy.toml`; `meta`
    carries instrumentation (cost, cache_hit, metrics) and is never rendered.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: CheckStatus
    check: str = Field(min_length=1)
    code: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    meta: dict[str, Any] = Field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "pass"


class GateFailure(BaseModel):
    """The rendered failure, as it appears in a `/api/submit` response.

    `message` is rendered from a template keyed on `(code, claim.label)` where
    `label` is authored offline — feedback never quotes the player's own text
    back at them (injection surface AND bad UX).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    check: str = Field(min_length=1)
    code: str = Field(min_length=1)
    message: str = ""
    params: dict[str, Any] = Field(default_factory=dict)
    #: Model-generated, derived from player-controlled input. Rendered only
    #: when non-empty AND it passes the §7.4 output filter.
    notes: str = ""


class GateResult(BaseModel):
    """The gate pipeline's outcome. `trace` ALWAYS returns, pass or fail — it
    powers the gate pips and makes the gate legible rather than oracular."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cleared: bool
    provisional: bool = False
    failure: CheckResult | None = None
    trace: tuple[CheckResult, ...] = ()

    @model_validator(mode="after")
    def _failure_matches_cleared(self) -> GateResult:
        if self.cleared and self.failure is not None:
            raise ValueError("a cleared gate cannot carry a failure")
        if not self.cleared and self.failure is None:
            raise ValueError("a rejected gate must name the check that rejected it")
        return self


class CheckSpec(BaseModel):
    """One entry in a level's ordered check list."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    check: str = Field(min_length=1, description="key into launder_core.gates.registry.REGISTRY")
    params: dict[str, Any] = Field(default_factory=dict)


class LevelConfig(BaseModel):
    """A level, loaded from `data/config/levels.toml` (TECH_PLAN.md §7.6).

    `load_levels()` in `launder_core.levels` raises at BOOT on an unknown check
    name — never at play. Defined here rather than in `levels.py` so that
    `serve`, `forge` and the tests all agree on one type.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^L[1-9][0-9]*$")
    name: str = Field(min_length=1)
    teaches: str = ""
    checks: tuple[CheckSpec, ...] = Field(min_length=1)
    par_source: ParSource = "authored"

    @field_validator("checks")
    @classmethod
    def _no_duplicate_checks(cls, v: tuple[CheckSpec, ...]) -> tuple[CheckSpec, ...]:
        names = [c.check for c in v]
        if len(set(names)) != len(names):
            raise ValueError(f"level lists the same check twice: {names}")
        return v

    def param_for(self, check: str, key: str, default: Any = None) -> Any:
        for spec in self.checks:
            if spec.check == check:
                return spec.params.get(key, default)
        return default


class LevelsFile(BaseModel):
    """The whole of `levels.toml`, including the `[defaults.*]` blocks that are
    merged under each level's per-check params."""

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    schema_id: Literal["launder.levels/1"] = Field(default="launder.levels/1", alias="schema")
    judge_version: str = Field(min_length=1)
    defaults: dict[str, dict[str, Any]] = Field(default_factory=dict)
    levels: tuple[LevelConfig, ...] = Field(min_length=1)

    @field_validator("levels")
    @classmethod
    def _unique_ids(cls, v: tuple[LevelConfig, ...]) -> tuple[LevelConfig, ...]:
        ids = [x.id for x in v]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate level ids: {ids}")
        return v

    def by_id(self) -> dict[str, LevelConfig]:
        return {x.id: x for x in self.levels}

    def resolved_params(self, level: LevelConfig, spec: CheckSpec) -> dict[str, Any]:
        """`[defaults.<check>]` merged under the level's own params.

        The level always wins. This is why L5 can say
        `unicode_sanitation, params = { homoglyph_policy = "reject_always" }`
        and still inherit `reject_categories` from the defaults block.
        """
        merged = dict(self.defaults.get(spec.check, {}))
        merged.update(spec.params)
        return merged
