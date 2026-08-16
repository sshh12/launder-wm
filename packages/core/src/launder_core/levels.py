"""Level configuration — TECH_PLAN.md §7.6.

Six levels, data-driven, ordered check lists. **ORDER IS THE SEMANTICS.**

Everything in this module exists to move failures from PLAY to BOOT. A level
that names a check nobody wrote, a check missing a param it cannot run without,
a param nobody reads, or a pipeline that puts the paid judge before the free
detector are all startup errors here — because every one of them, discovered at
play, is either a 500 in front of a player mid-puzzle or (worse) a check that
silently does nothing while the checklist claims it is enforcing something.

`LevelConfig` and `CheckSpec` themselves live in `launder_core.schemas.gate`, so
`serve`, `forge` and the tests share one definition. This module is the LOADER
and the VALIDATOR; it re-exports the two types for convenience and defines no
second copy of them.
"""

from __future__ import annotations

import tomllib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from launder_core.gates.feedback import data_dir
from launder_core.gates.registry import (
    REGISTRY,
    GateCheck,
    GateConfigError,
    required_deps_of,
)
from launder_core.schemas import CheckSpec, LevelConfig, LevelsFile

__all__ = [
    "CheckSpec",
    "LevelConfig",
    "LevelsFile",
    "assert_dependencies_available",
    "levels_path",
    "load_levels",
    "load_levels_file",
    "unsatisfiable_levels",
    "validate_level",
]


def levels_path() -> Path:
    return data_dir() / "config" / "levels.toml"


def load_levels_file(path: Path | None = None) -> LevelsFile:
    """Parse and schema-validate, without touching the registry."""
    resolved = path if path is not None else levels_path()
    with resolved.open("rb") as fh:
        raw = tomllib.load(fh)
    return LevelsFile.model_validate(raw)


def validate_level(level: LevelConfig, registry: Mapping[str, GateCheck] | None = None) -> None:
    """Every rule that must hold before the server accepts a single submission.

    Raises `GateConfigError` naming the level, the check and the fix.
    """
    registry = registry if registry is not None else REGISTRY
    names = [spec.check for spec in level.checks]

    # 1. Every check exists. The error names the file to write, because "adding
    #    a check is adding a file" only works if the failure says which file.
    for name in names:
        if name not in registry:
            raise GateConfigError(
                f"level {level.id} lists check {name!r}, which is not registered. Write "
                f"packages/core/src/launder_core/gates/checks/{name}.py, decorate the class "
                "with @register, and add it to gates/checks/__init__.py. Known checks: "
                f"{sorted(registry)}."
            )

    # 2. unicode_sanitation is first, and present. It FILLS ctx.normalized and
    #    ctx.words; a level without it hands every later check empty text, and a
    #    level that runs it second scores the raw string.
    sanitation = "unicode_sanitation"
    if sanitation not in names:
        raise GateConfigError(
            f"level {level.id} does not run {sanitation}. It is what fills ctx.normalized and "
            "ctx.words for every check after it, and it is the only thing standing between "
            "the detector and a zero-width-character exploit (TECH_PLAN.md §7.2)."
        )
    if names[0] != sanitation:
        raise GateConfigError(
            f"level {level.id} runs {names[0]!r} before {sanitation}. Every downstream check "
            "reads ctx.normalized, which does not exist yet at that point."
        )

    # 3. Phase-ascending. See registry.Phase for why this is a partial order.
    phases = [registry[n].phase for n in names]
    for i in range(1, len(phases)):
        if phases[i] < phases[i - 1]:
            raise GateConfigError(
                f"level {level.id} runs {names[i]!r} (phase {phases[i]}) after {names[i - 1]!r} "
                f"(phase {phases[i - 1]}). Order is the semantics: a cheap deterministic check "
                "must never sit behind an expensive one, or a mechanical failure starts "
                "costing an API call (TECH_PLAN.md §7.1)."
            )

    # 4. The paid step is last, and the detector immediately precedes it. This
    #    is the single largest cost saver in the product: only submissions that
    #    already beat the watermark ever reach the LLM.
    paid = [n for n in names if registry[n].phase >= registry["llm_gate"].phase]
    if paid:
        if names[-1] != paid[0]:
            raise GateConfigError(
                f"level {level.id} runs {paid[0]!r} at position {names.index(paid[0]) + 1} of "
                f"{len(names)}. The paid check must be last so that every mechanical rejection "
                "costs zero API calls."
            )
        if len(names) < 2 or names[-2] != "detector_threshold":
            raise GateConfigError(
                f"level {level.id} does not run detector_threshold immediately before "
                f"{names[-1]!r}. That ordering is what keeps submissions which have not yet "
                "beaten the watermark away from the judge (TECH_PLAN.md §7.1)."
            )

    # 5. Params: nothing missing, nothing unread.
    for spec in level.checks:
        check = registry[spec.check]
        supplied = set(spec.params)
        missing = check.required_params - supplied
        if missing:
            raise GateConfigError(
                f"level {level.id}, check {spec.check}: missing required param(s) "
                f"{sorted(missing)}. Set them on the level or in [defaults.{spec.check}]."
            )
        unknown = supplied - check.config_params
        if unknown:
            raise GateConfigError(
                f"level {level.id}, check {spec.check}: param(s) {sorted(unknown)} are not read "
                f"by that check (it reads {sorted(check.config_params)}). A param nobody reads "
                "is a rule nobody enforces — most often a rename that only landed in one of "
                "the two files."
            )


def unsatisfiable_levels(
    levels: Mapping[str, LevelConfig],
    available: Iterable[str],
    registry: Mapping[str, GateCheck] | None = None,
) -> dict[str, frozenset[str]]:
    """`{level_id: {missing Deps field, ...}}` for levels this process cannot run.

    A level whose checks declare a dependency nobody wired is UNSERVICEABLE: the
    check raises rather than passing (passing would clear submissions nothing
    checked), so every submission on it is a 500. Reporting that at boot is what
    lets the caller refuse the level cleanly instead of discovering it in front
    of a player.
    """
    registry = registry if registry is not None else REGISTRY
    have = set(available)
    out: dict[str, frozenset[str]] = {}
    for level in levels.values():
        missing: set[str] = set()
        for spec in level.checks:
            check = registry.get(spec.check)
            if check is not None:
                missing |= required_deps_of(check) - have
        if missing:
            out[level.id] = frozenset(missing)
    return out


def assert_dependencies_available(
    levels: Mapping[str, LevelConfig],
    available: Iterable[str],
    registry: Mapping[str, GateCheck] | None = None,
) -> None:
    """Refuse to serve a level whose checks need a dependency nobody wired.

    **`load_levels` validates check NAMES; it does not validate the DEPENDENCIES
    those checks declare.** L5 therefore booted perfectly clean and then raised
    `GateDependencyError: unit_test ran with no UnitTestRunner in Deps` as an
    unhandled 500 on the first submit — a server that passed every startup
    assertion breaking at play time, on a level nobody had played, which is the
    precise shape §7.6's "raises at BOOT, never at play" rule exists to prevent.

    Refusing rather than passing the check by default is still right (`unit_test`
    is the ONLY meaning check on L5, so a default pass would clear any code that
    beats the detector, working or not). This just moves the refusal to boot.

    Call it from the app factory with the field names of the `Deps` the server
    will actually build.
    """
    registry = registry if registry is not None else REGISTRY
    have = set(available)
    for level in levels.values():
        for spec in level.checks:
            check = registry.get(spec.check)
            if check is None:
                continue
            missing = sorted(required_deps_of(check) - have)
            if missing:
                raise GateConfigError(
                    f"level {level.id} runs check {spec.check!r}, which requires "
                    f"Deps.{missing[0]} — and this process does not provide it. That "
                    "check raises GateDependencyError rather than passing by default "
                    "(passing would clear submissions nothing checked), so the level "
                    "would be an unhandled 500 on its first submit. Either wire "
                    f"Deps.{missing[0]}, or remove {spec.check!r} from {level.id} in "
                    "data/config/levels.toml."
                )


def load_levels(
    path: Path | None = None, registry: Mapping[str, GateCheck] | None = None
) -> dict[str, LevelConfig]:
    """Load `levels.toml`, merge `[defaults.*]`, validate, return by id.

    RAISES AT BOOT on an unknown check name, a missing param, an unread param or
    a check listed out of pipeline order. Call it during startup, not lazily on
    the first request: the whole point is that a bad config never reaches a
    player.

    The returned `LevelConfig`s carry FULLY RESOLVED params — the `[defaults.*]`
    blocks are already merged under each level's own values — so `run_gate` and
    the checks never have to know that defaults exist.
    """
    parsed = load_levels_file(path)
    registry = registry if registry is not None else REGISTRY

    known_defaults = set(parsed.defaults) - set(registry)
    if known_defaults:
        raise GateConfigError(
            f"levels.toml declares [defaults.{sorted(known_defaults)[0]}] for a check that is "
            "not registered. Those defaults would silently apply to nothing."
        )

    resolved: dict[str, LevelConfig] = {}
    for level in parsed.levels:
        merged: list[CheckSpec] = []
        for spec in level.checks:
            params: dict[str, Any] = parsed.resolved_params(level, spec)
            merged.append(CheckSpec(check=spec.check, params=params))
        full = level.model_copy(update={"checks": tuple(merged)})
        validate_level(full, registry)
        resolved[full.id] = full
    return resolved
