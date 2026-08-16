"""The ordered check pipeline — TECH_PLAN.md §7.1, §7.6.

**ORDER IS THE SEMANTICS.** Every deterministic check runs before `llm_gate`,
so a submission that busts the budget, the floor, the locked phrase, the unit
test or unicode sanitation costs zero API calls. `detector_threshold` sitting
immediately before the judge is the single largest cost saver: only submissions
that already beat the watermark ever reach the LLM.

Adding a check is adding a file. Write `packages/core/src/launder_core/gates/
checks/<name>.py`, decorate the class with `@register`, add the module to
`gates/checks/__init__.py`, write its `[check.<name>]` block in
`data/config/copy.toml`, and list it in `data/config/levels.toml`. Nothing else
in this package changes. Both of those data steps are enforced —
`launder_core.gates.feedback.lint_copy` fails the build on a check with no
copy, and `launder_core.levels.load_levels` fails at BOOT on a check that is
not in `REGISTRY`.

This module also owns the four INJECTED contracts (`Detector`, `JudgeProvider`,
`UnitTestRunner`, and the `Observation` payload). They live here rather than in
`schemas/` because they are protocols and behaviour, not wire shapes, and here
rather than in the individual check modules because `Deps` has to name them and
`serve` has to implement them without importing a check.
"""

from __future__ import annotations

import logging
import secrets
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar, Final, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from launder_core.schemas import (
    CheckResult,
    Claim,
    DetectorReading,
    GateResult,
    LevelConfig,
    PassagePublic,
)
from launder_core.scoring.normalize import DEFAULT_NORMALIZE, NormalizeConfig

__all__ = [
    "CTX_NORMALIZED",
    "CTX_WORDS",
    "META_COPY_KEY",
    "META_COPY_TABLE",
    "REGISTRY",
    "ClaimObservation",
    "Deps",
    "Detector",
    "GateCheck",
    "GateConfigError",
    "GateContext",
    "GateDataError",
    "GateDependencyError",
    "JudgeProvider",
    "JudgeUnavailable",
    "Observation",
    "Phase",
    "UnitTestOutcome",
    "UnitTestRunner",
    "copy_keys_of",
    "register",
    "required_deps_of",
    "run_gate",
]

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reserved keys
# ---------------------------------------------------------------------------
#: `unicode_sanitation` is the first check and it FILLS `normalized` and
#: `words` for everything downstream (§7.1 table, row 1). `GateContext` is
#: frozen, so the check publishes them through `meta` under these keys and
#: `run_gate` rebuilds the context. A check that wants to rewrite the text for
#: later checks does the same thing; nothing else may.
CTX_NORMALIZED: Final[str] = "ctx.normalized"
CTX_WORDS: Final[str] = "ctx.words"

#: How a check tells `feedback.py` WHICH template in its `[check.<name>]` block
#: to render, without `feedback.py` holding any player-facing string and
#: without this module holding a table of copy keys. `META_COPY_TABLE` names a
#: sub-table (`llm_gate` uses `reject_by_code` and `reject_by_unnatural_kind`),
#: `META_COPY_KEY` names the entry inside it. Both are structural identifiers,
#: not copy.
META_COPY_TABLE: Final[str] = "copy.table"
META_COPY_KEY: Final[str] = "copy.key"


class Phase:
    """Pipeline position, coarser than the §7.1 table's 1-8 ranking.

    §7.1 orders `edit_budget` (3) before `locked_phrase` (4). §7.6's shipped L3
    lists them the other way round. Both are in TECH_PLAN.md, so a strict total
    order would reject the plan's own config. What the ordering actually has to
    guarantee is a cost ladder — µs checks before ms checks before the paid one
    — and those two checks are both µs-to-ms deterministic, so they share a
    phase and either order is legal. Everything else keeps the plan's ranking,
    and `load_levels` rejects any level whose checks are not phase-ascending.
    """

    SANITATION = 0  # §7.1 row 1 — fills `normalized` and `words`
    LENGTH = 10  # row 2
    SHAPE = 20  # rows 3 and 4 — edit_budget, locked_phrase
    EXECUTION = 30  # row 5 — unit_test
    FENCE = 40  # row 6 — close_paraphrase
    DETECTOR = 50  # row 7 — the actual win condition
    JUDGE = 60  # row 8 — the only paid step


# ---------------------------------------------------------------------------
# Errors. Three kinds, deliberately distinct.
# ---------------------------------------------------------------------------
class GateConfigError(RuntimeError):
    """A level, a param or a registry entry is wrong. Raised at BOOT."""


class GateDataError(RuntimeError):
    """A passage does not carry data a check on its level requires — an L3
    passage with no locked phrase, an L5 passage with no unit-test id.

    Deliberately NOT a player-visible rejection. The level is unplayable
    because of our data, and telling the player they failed a check they could
    not have passed is worse than a 500. `forge verify` is where this gets
    caught before a passage ships.
    """


class GateDependencyError(RuntimeError):
    """A check needs an injected dependency the caller did not provide.

    Raised rather than silently passing: a `detector_threshold` that returns
    `pass` because nobody wired a detector would clear every submission on
    every level, which is the exact silent-failure shape §4.5 exists to
    prevent.
    """


class JudgeUnavailable(RuntimeError):
    """The judge could not be reached after retry AND failover (§7.5).

    Providers raise this; `llm_gate` turns it into `status="error"`, which the
    pipeline turns into a PROVISIONAL clear. Any other exception from a
    provider is a bug, and is logged as one before being treated the same way —
    a provider bug must not cost the player their submission.
    """


# ---------------------------------------------------------------------------
# The injected contracts
# ---------------------------------------------------------------------------
class ClaimObservation(BaseModel):
    """One row of the judge's per-claim report (§7.4 schema)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    present: bool
    how: Literal["asserted", "missing", "negated", "altered", "hedged", "reattributed"]


class Observation(BaseModel):
    """What the judge emits. **Observations, not verdicts** (§7.3).

    Code computes the verdict from these plus the level config, which is why
    prompt injection has no "output pass" to target: there is no field here a
    player could ask the model to set that would clear them, and
    `contains_embedded_instructions` fails closed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    natural_prose: bool
    unnatural_kind: (
        Literal["word_salad", "keyword_soup", "repetition", "non_prose_content", "not_english"]
        | None
    ) = None
    claims: tuple[ClaimObservation, ...] = ()
    added_claims: tuple[str, ...] = ()
    contains_embedded_instructions: bool = False
    notes: str = ""
    #: Telemetry ONLY (§7.4). A persistent gap between this and the derived
    #: verdict means the prompt and the level config disagree — a prompt bug
    #: worth mining, never an input to the decision.
    verdict_opinion: Literal["pass", "fail"] = "pass"


@runtime_checkable
class JudgeProvider(Protocol):
    """§7.5. Implemented by `serve` as `openai`, `anthropic`, `fake` and
    `cassette`. Core depends on the protocol and nothing else — no SDK, no
    network, no API key ever reaches this package."""

    name: ClassVar[str]

    async def observe(self, passage: PassagePublic, normalized: str, nonce: str) -> Observation: ...


@runtime_checkable
class Detector(Protocol):
    """The server-side re-check that IS the win condition (§7.1 row 7).

    Synchronous: it is a few milliseconds of CPU over a NumPy array, and making
    it async would buy nothing but a context switch.
    """

    def read(
        self, text: str, passage: PassagePublic, calibration: str | None = None
    ) -> DetectorReading: ...


class UnitTestOutcome(BaseModel):
    """What a sandboxed suite run reports back."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    passed: bool
    failed_test: str = ""
    timed_out: bool = False
    #: Never rendered to the player unfiltered: it can contain their own code.
    detail: str = Field(default="", repr=False)


@runtime_checkable
class UnitTestRunner(Protocol):
    """Runs a committed suite against player-authored code (§7.6, L5).

    Core defines the protocol and REFUSES to ship an implementation. Executing
    player code needs a real sandbox, and the two knobs the level config asks
    for — `timeout_ms` and `memory_mb` — have no portable implementation:
    `RLIMIT_AS` is POSIX-only and does not exist on the Windows box this repo is
    developed on. A plausible-looking `subprocess` runner in core would be an
    remote-code-execution hole that reviewers would assume was sandboxed.
    """

    def run(
        self, suite_id: str, source: str, timeout_ms: int, memory_mb: int
    ) -> UnitTestOutcome: ...


def _default_nonce() -> str:
    """`secrets.token_hex(6)` (§7.3). EXCLUDED from the judge cache key — the
    key hashes the normalized submission, not the rendered prompt."""
    return secrets.token_hex(6)


@dataclass(frozen=True)
class Deps:
    """Everything a check may reach outside its own arguments.

    All optional, all `None` by default: the deterministic checks need nothing,
    and a test that exercises them should not have to fake a judge. A check
    whose dependency is missing raises `GateDependencyError` rather than
    passing.
    """

    detector: Detector | None = None
    judge: JudgeProvider | None = None
    unit_tests: UnitTestRunner | None = None
    normalize_config: NormalizeConfig = DEFAULT_NORMALIZE
    nonce_factory: Callable[[], str] = _default_nonce
    #: Claims live in the server sidecar, not the public bundle (§11.3). When
    #: the caller has loaded them separately they go here and win over
    #: `passage.claims`.
    claims: tuple[Claim, ...] | None = None


@dataclass(frozen=True)
class GateContext:
    """One submission, mid-pipeline (§7.6).

    `normalized` and `words` start empty and are filled by `unicode_sanitation`,
    which is why that check is mandatory and first. `words` is the SAME
    tokenization the scorer uses — that is what makes the fence, the budget and
    the scoreboard incapable of disagreeing.
    """

    passage: PassagePublic
    level: LevelConfig
    raw: str
    normalized: str = ""
    words: tuple[str, ...] = ()
    deps: Deps = field(default_factory=Deps)

    def claims(self) -> tuple[Claim, ...]:
        return self.deps.claims if self.deps.claims is not None else self.passage.claims


CheckReturn = CheckResult | Awaitable[CheckResult]


class GateCheck(Protocol):
    """One check. Stateless, instantiated once by `@register`.

    The four ClassVars are not decoration: `phase` is what `load_levels`
    validates order against, `config_params` and `required_params` are what it
    validates a level's params against, and `template_params` is what
    `lint_copy` validates `copy.toml`'s placeholders against. A check that
    declares them wrongly fails the build, not a player's submission.
    """

    #: REGISTRY key, `[check.<name>]` key in copy.toml, and the name in
    #: levels.toml. One string, three files, no mapping table.
    name: ClassVar[str]
    phase: ClassVar[int]
    #: True only for checks whose failure to RUN may be waved through as a
    #: provisional clear (§7.5). The deterministic checks already caught every
    #: mechanical exploit, so only the judge qualifies.
    fail_open: ClassVar[bool]
    #: Keys this check accepts from `levels.toml` (after defaults are merged).
    config_params: ClassVar[frozenset[str]]
    #: Keys it cannot run without.
    required_params: ClassVar[frozenset[str]]
    #: Keys it guarantees to put in `CheckResult.params`, i.e. everything a
    #: copy template for this check is allowed to interpolate.
    template_params: ClassVar[frozenset[str]]
    #: Field names on `Deps` this check cannot run without, e.g.
    #: `{"unit_tests"}`. Declared so the caller can find out at BOOT that a
    #: configured level names a check whose dependency it does not wire —
    #: `load_levels` validated check NAMES and not the DEPENDENCIES those checks
    #: declare, so L5 booted clean and then 500'd on the first submit with
    #: `GateDependencyError: unit_test ran with no UnitTestRunner in Deps`.
    #
    # TWO MORE ClassVars are OPTIONAL and read through `required_deps_of` /
    # `copy_keys_of` rather than declared here, so a check that needs neither
    # stays a four-line class:
    #
    #   required_deps  field names on `Deps` the check cannot run without, e.g.
    #                  `{"unit_tests"}`. `load_levels` validated check NAMES and
    #                  not the DEPENDENCIES those checks declare, so L5 booted
    #                  clean and then 500'd on its first submit with
    #                  `GateDependencyError`. Defaults to empty.
    #   copy_keys      every `[check.<name>]` copy key the check can select via
    #                  `meta[META_COPY_KEY]`. Lint rule 1 checked
    #                  `label`/`blurb`/`reject` — the keys that EXIST, not the
    #                  keys a check can ASK FOR — so `close_paraphrase` emitted
    #                  `reject_distance` and `reject_alignment` with neither in
    #                  copy.toml, a distance failure rendered the RETENTION
    #                  message, and the lint reported "OK". Defaults to
    #                  `{"reject"}`.

    def __call__(self, ctx: GateContext, params: Mapping[str, Any]) -> CheckReturn: ...


def copy_keys_of(check: GateCheck) -> frozenset[str]:
    """Copy keys this check can select, always including `reject`."""
    return frozenset(getattr(check, "copy_keys", frozenset())) | {"reject"}


def required_deps_of(check: GateCheck) -> frozenset[str]:
    """`check.required_deps`, defaulting to empty for checks that declare none."""
    return frozenset(getattr(check, "required_deps", frozenset()))


#: Populated by `@register` at import time. `gates/checks/__init__.py` imports
#: every check module, and `launder_core.gates` imports that, so
#: `from launder_core.gates import REGISTRY` is always complete.
REGISTRY: dict[str, GateCheck] = {}


def register[CheckClass: type[GateCheck]](cls: CheckClass) -> CheckClass:
    """Class decorator. `REGISTRY[cls.name] = cls()`."""
    name = cls.name
    if not name:
        raise GateConfigError(f"{cls.__qualname__} must declare a non-empty `name`")
    existing = REGISTRY.get(name)
    if existing is not None and type(existing).__qualname__ != cls.__qualname__:
        raise GateConfigError(
            f"two checks claim the name {name!r}: {type(existing).__qualname__} and "
            f"{cls.__qualname__}. The name keys the REGISTRY, the copy.toml block and the "
            "levels.toml entry; it has to be unique."
        )
    REGISTRY[name] = cls()
    return cls


async def _call(check: GateCheck, ctx: GateContext, params: Mapping[str, Any]) -> CheckResult:
    """Deterministic checks are plain functions; only `llm_gate` is a
    coroutine. Keeping the sync ones sync means the whole deterministic ladder
    is callable from a CLI, a test or the solver without an event loop."""
    outcome = check(ctx, params)
    if isinstance(outcome, CheckResult):
        return outcome
    return await outcome


def _absorb(ctx: GateContext, result: CheckResult) -> GateContext:
    """Apply a check's published context updates (see `CTX_NORMALIZED`).

    Runs even when the check FAILED, so a rejection still carries the
    normalized text for logging and for the `notes` filter.
    """
    updates: dict[str, Any] = {}
    if CTX_NORMALIZED in result.meta:
        updates["normalized"] = str(result.meta[CTX_NORMALIZED])
    if CTX_WORDS in result.meta:
        updates["words"] = tuple(result.meta[CTX_WORDS])
    return replace(ctx, **updates) if updates else ctx


async def run_gate(ctx: GateContext) -> GateResult:
    """The pipeline of §7.1, with the §7.5 fail-open policy for errors.

    Departure from the pseudocode, and why: §7.1 writes the error branch as
    `return handle_error(...)`. Returning on a fail-open error would SKIP every
    remaining check, so this loop instead marks the run provisional and keeps
    going. In every shipped level `llm_gate` is last, so the two are identical
    today; they stop being identical the moment somebody adds a check after it,
    and the version that keeps going is the one that stays correct.

    An error from a check that is NOT fail-open ends the run as a rejection.
    Those checks are pure functions of the submission — an error in one is a
    bug in us, and clearing a submission because our own code raised is how a
    watermark game silently becomes a random number generator.
    """
    trace: list[CheckResult] = []
    provisional = False

    for step in ctx.level.checks:
        check = REGISTRY.get(step.check)
        if check is None:
            raise GateConfigError(
                f"level {ctx.level.id} runs unregistered check {step.check!r}. "
                "load_levels() should have caught this at boot; something bypassed it."
            )
        result = await _call(check, ctx, step.params)
        ctx = _absorb(ctx, result)
        trace.append(result)

        if result.status == "fail":
            return GateResult(cleared=False, provisional=False, failure=result, trace=tuple(trace))
        if result.status == "error":
            if not check.fail_open:
                return GateResult(
                    cleared=False, provisional=False, failure=result, trace=tuple(trace)
                )
            log.warning(
                "gate check %s errored on passage %s; clearing PROVISIONALLY (code=%s)",
                step.check,
                ctx.passage.id,
                result.code,
            )
            provisional = True

    return GateResult(cleared=True, provisional=provisional, failure=None, trace=tuple(trace))
