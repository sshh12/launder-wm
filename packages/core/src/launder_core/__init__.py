"""launder-core — the watermark, the detector, the scorer, the gate, the schemas.

Pure Python. No torch, no FastAPI, no network. HARD INVARIANT (TECH_PLAN.md
§2.1): importing this package must succeed in a venv containing only
`pydantic`, `blake3` and `numpy`. CI enforces it with a bare-venv import test.
That invariant is what keeps torch out of the Railway image.

Why the lazy `__getattr__`
--------------------------
The §3 module tree is built across milestones M1-M4 by different hands. Eager
`from .watermark.hash import accumulate_hash` at the top of this file would make
`import launder_core` — and therefore every schema import, and therefore every
test in the repo — fail until the last of those modules lands.

So the public surface is declared here as a name -> "module:attribute" table and
resolved on first access. Consequences worth knowing:

* `from launder_core import PassagePublic` works today (schemas are eager).
* `from launder_core import accumulate_hash` works the moment
  `launder_core/watermark/hash.py` defines it, with no edit to this file.
* Asking for a name whose module does not exist yet raises an `AttributeError`
  that names the file that must be written, instead of an opaque ImportError.
* `dir(launder_core)` lists the full intended surface, implemented or not, so
  the contract is discoverable.

Do NOT convert this into eager imports until every module below exists, and do
NOT create empty placeholder modules to make eager imports work — a module that
imports cleanly and returns nothing is exactly the silent-failure shape this
project spends §4.3 and §4.5 defending against.
"""

from __future__ import annotations

import importlib
from typing import Any, Final

# Eager: this package owns them, they depend on nothing heavier than pydantic.
from launder_core import schemas as schemas
from launder_core.schemas import (
    CANONICAL_KEYS,
    FORBIDDEN_PUBLIC_FIELDS,
    LCG_MULTIPLIER,
    CheckResult,
    CheckSpec,
    Claim,
    DetectRequest,
    DetectResponse,
    EditOp,
    GateFailure,
    GateOutcome,
    GateResult,
    HealthResponse,
    LevelConfig,
    LevelsFile,
    PassageAuthor,
    PassagePublic,
    PassageServer,
    ProgressResponse,
    ScoreResult,
    ScoringConfig,
    SubmitRequest,
    SubmitResponse,
    SynthIDConfig,
    TokenHeat,
    canonical_json,
    wm_config_id_of,
)

__version__ = "0.1.0"

#: Public name -> "relative.module:attribute". The right-hand modules are owned
#: by the M1-M4 implementation passes; this table is the contract they satisfy.
_LAZY: Final[dict[str, str]] = {
    # --- watermark (M1) ---
    "wrap_int64": "watermark.hash:wrap_int64",
    "accumulate_hash": "watermark.hash:accumulate_hash",
    "load_sampling_table": "watermark.table:load_sampling_table",
    "sampling_table_digest": "watermark.table:sampling_table_digest",
    "compute_g_values": "watermark.gvalues:compute_g_values",
    "compute_context_hashes": "watermark.gvalues:compute_context_hashes",
    "compute_context_repetition_mask": "watermark.gvalues:compute_context_repetition_mask",
    "compute_eos_mask": "watermark.gvalues:compute_eos_mask",
    "ripple_span": "watermark.gvalues:ripple_span",
    # --- detector (M1) ---
    "weighted_mean_score": "detect.weighted_mean:weighted_mean_score",
    "contributions": "detect.weighted_mean:contributions",
    "depth_weights": "detect.weighted_mean:depth_weights",
    "bayesian_score": "detect.bayesian:bayesian_score",
    "Calibration": "detect.calibration:Calibration",
    "load_thresholds": "detect.calibration:load_thresholds",
    "z_from_score": "detect.calibration:z_from_score",
    # --- scoring (M4) ---
    "NormalizeConfig": "scoring.normalize:NormalizeConfig",
    "normalize": "scoring.normalize:normalize",
    "words": "scoring.normalize:words",
    "damerau_levenshtein": "scoring.damerau:damerau_levenshtein",
    "score": "scoring.damerau:score",
    # --- gates (M4) ---
    "REGISTRY": "gates.registry:REGISTRY",
    "register": "gates.registry:register",
    "GateCheck": "gates.registry:GateCheck",
    "GateContext": "gates.registry:GateContext",
    "run_gate": "gates.registry:run_gate",
    "render_feedback": "gates.feedback:render_feedback",
    # --- levels (M4) ---
    "load_levels": "levels:load_levels",
}

# NOTE for whoever lands M1-M4: once every module in `_LAZY` exists, a
# `if TYPE_CHECKING:` block re-importing these names would give static checkers
# real signatures instead of `Any`. It is deliberately absent today, because
# mypy resolves those imports eagerly and would fail on modules that have not
# been written yet — which would make the type checker a blocker on work order
# rather than on correctness.


def __getattr__(name: str) -> Any:
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module 'launder_core' has no attribute {name!r}")
    modpath, _, attr = target.partition(":")
    try:
        module = importlib.import_module(f"launder_core.{modpath}")
    except ModuleNotFoundError as exc:
        if exc.name is not None and not exc.name.startswith("launder_core"):
            raise  # a real missing third-party dependency; do not disguise it
        raise AttributeError(
            f"launder_core.{name} is declared in the public API but "
            f"packages/core/src/launder_core/{modpath.replace('.', '/')}.py does not exist yet. "
            "See TECH_PLAN.md §3 for what belongs in it."
        ) from exc
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise AttributeError(
            f"launder_core.{modpath} exists but does not define {attr!r}; "
            f"launder_core.{name} is part of the declared public API (TECH_PLAN.md §3)."
        ) from exc


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(_LAZY))


__all__ = [
    # Sorted, not grouped: ruff's RUF022 owns the ordering. The grouping that
    # matters is in `_LAZY` above — everything not in that table is eager.
    "CANONICAL_KEYS",
    "FORBIDDEN_PUBLIC_FIELDS",
    "LCG_MULTIPLIER",
    "REGISTRY",
    "Calibration",
    "CheckResult",
    "CheckSpec",
    "Claim",
    "DetectRequest",
    "DetectResponse",
    "EditOp",
    "GateCheck",
    "GateContext",
    "GateFailure",
    "GateOutcome",
    "GateResult",
    "HealthResponse",
    "LevelConfig",
    "LevelsFile",
    "NormalizeConfig",
    "PassageAuthor",
    "PassagePublic",
    "PassageServer",
    "ProgressResponse",
    "ScoreResult",
    "ScoringConfig",
    "SubmitRequest",
    "SubmitResponse",
    "SynthIDConfig",
    "TokenHeat",
    "__version__",
    "accumulate_hash",
    "bayesian_score",
    "canonical_json",
    "compute_context_hashes",
    "compute_context_repetition_mask",
    "compute_eos_mask",
    "compute_g_values",
    "contributions",
    "damerau_levenshtein",
    "depth_weights",
    "load_levels",
    "load_sampling_table",
    "load_thresholds",
    "normalize",
    "register",
    "render_feedback",
    "ripple_span",
    "run_gate",
    "sampling_table_digest",
    "schemas",
    "score",
    "weighted_mean_score",
    "wm_config_id_of",
    "words",
    "wrap_int64",
    "z_from_score",
]
