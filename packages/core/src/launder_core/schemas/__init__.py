"""Pydantic v2 models for every wire and disk contract.

This package is the shared contract. `forge`, `serve`, the tests and the TS
port's golden vectors all agree here or they do not agree at all. It imports
nothing outside pydantic and the standard library — no numpy, no blake3 — so
that importing a schema can never drag in a heavy dependency.

Import from the package root:

    from launder_core.schemas import PassagePublic, DetectResponse, SynthIDConfig
"""

from __future__ import annotations

from launder_core.schemas.api import (
    CLIENT_SCORE_FIELDS,
    MAX_TEXT_BYTES,
    DetectorReading,
    DetectRequest,
    DetectResponse,
    HealthResponse,
    ProgressResponse,
    ScoreSummary,
    SubmitClient,
    SubmitRequest,
    SubmitResponse,
    TokenHeat,
)
from launder_core.schemas.gate import (
    CheckResult,
    CheckSpec,
    CheckStatus,
    GateFailure,
    GateResult,
    LevelConfig,
    LevelsFile,
    ParSource,
)
from launder_core.schemas.passage import (
    FORBIDDEN_PUBLIC_FIELDS,
    Claim,
    DetectorExpectation,
    IntroConfig,
    PassageAuthor,
    PassageDifficulty,
    PassageOptionality,
    PassageProvenance,
    PassagePublic,
    PassageRules,
    PassageServer,
    ReferenceSolution,
    SolverReport,
)
from launder_core.schemas.scoring import EditOp, EditOpKind, ScoreResult, ScoringConfig
from launder_core.schemas.watermark import (
    CANONICAL_KEYS,
    LCG_MULTIPLIER,
    AssetBundleId,
    Blake3Digest,
    Sha256Digest,
    SynthIDConfig,
    WmConfigId,
    canonical_json,
    wm_config_id_of,
)

#: `GateOutcome` is the name TECH_PLAN.md §7.1 uses in the pipeline pseudocode;
#: `GateResult` is the name the task brief uses. Same object, one definition.
GateOutcome = GateResult

__all__ = [
    "CANONICAL_KEYS",
    "CLIENT_SCORE_FIELDS",
    "FORBIDDEN_PUBLIC_FIELDS",
    "LCG_MULTIPLIER",
    "MAX_TEXT_BYTES",
    "AssetBundleId",
    "Blake3Digest",
    "CheckResult",
    "CheckSpec",
    "CheckStatus",
    "Claim",
    "DetectRequest",
    "DetectResponse",
    "DetectorExpectation",
    "DetectorReading",
    "EditOp",
    "EditOpKind",
    "GateFailure",
    "GateOutcome",
    "GateResult",
    "HealthResponse",
    "IntroConfig",
    "LevelConfig",
    "LevelsFile",
    "ParSource",
    "PassageAuthor",
    "PassageDifficulty",
    "PassageOptionality",
    "PassageProvenance",
    "PassagePublic",
    "PassageRules",
    "PassageServer",
    "ProgressResponse",
    "ReferenceSolution",
    "ScoreResult",
    "ScoreSummary",
    "ScoringConfig",
    "Sha256Digest",
    "SolverReport",
    "SubmitClient",
    "SubmitRequest",
    "SubmitResponse",
    "SynthIDConfig",
    "TokenHeat",
    "WmConfigId",
    "canonical_json",
    "wm_config_id_of",
]
