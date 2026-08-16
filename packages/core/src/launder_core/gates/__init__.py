"""The gate — an ordered pipeline of checks, TECH_PLAN.md §7.

Importing this package registers every check, so `REGISTRY` is complete from
here on. The four injected contracts (`Detector`, `JudgeProvider`,
`UnitTestRunner`, `Observation`) are re-exported so `serve` can implement them
without importing an individual check module.
"""

from __future__ import annotations

from launder_core.gates import checks as checks
from launder_core.gates.feedback import (
    CopyBook,
    CopyError,
    assert_copy_complete,
    filter_notes,
    lint_copy,
    load_copy,
    render_feedback,
)
from launder_core.gates.registry import (
    REGISTRY,
    ClaimObservation,
    Deps,
    Detector,
    GateCheck,
    GateConfigError,
    GateContext,
    GateDataError,
    GateDependencyError,
    JudgeProvider,
    JudgeUnavailable,
    Observation,
    Phase,
    UnitTestOutcome,
    UnitTestRunner,
    register,
    run_gate,
)

__all__ = [
    "REGISTRY",
    "ClaimObservation",
    "CopyBook",
    "CopyError",
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
    "assert_copy_complete",
    "checks",
    "filter_notes",
    "lint_copy",
    "load_copy",
    "register",
    "render_feedback",
    "run_gate",
]
