"""Wire contracts for the three dynamic endpoints plus /healthz.

Two properties of these models are load-bearing and must not be "improved":

1. **`DetectResponse` is exactly what the local detector emits** (char offsets,
   per-token heat, masked flags). One renderer serves both the SERVER and LOCAL
   paths, which is why M2 is playable before the TS detector exists and why the
   handover is a no-op in the view layer (TECH_PLAN.md §5.4, §9.2).

2. **`SubmitRequest` carries no scores.** Distance, detector score, z and every
   gate verdict are computed server-side. A client that lies about its needle is
   lying only to itself — which is what makes publishing the watermark keys
   costless to integrity (TECH_PLAN.md §1, §12 "not modular").
"""

from __future__ import annotations

from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from launder_core.schemas.gate import CheckResult, GateFailure
from launder_core.schemas.scoring import EditOp
from launder_core.schemas.watermark import AssetBundleId, Sha256Digest, WmConfigId

__all__ = [
    "CLIENT_SCORE_FIELDS",
    "MAX_TEXT_BYTES",
    "DetectRequest",
    "DetectResponse",
    "DetectorReading",
    "HealthResponse",
    "ProgressResponse",
    "ScoreSummary",
    "SubmitClient",
    "SubmitRequest",
    "SubmitResponse",
    "TokenHeat",
]

#: Body size cap for /api/detect and /api/submit (TECH_PLAN.md §9.2, §9.3).
#: 413 above this. Enforced by the ASGI layer as well; this bound is the
#: schema-level backstop so a fuzzer cannot allocate a 40 MB string.
MAX_TEXT_BYTES: Final[int] = 64 * 1024

#: Keys a client must never send on /api/submit. Present so the rejection says
#: WHY rather than "extra inputs are not permitted".
CLIENT_SCORE_FIELDS: Final[frozenset[str]] = frozenset(
    {"score", "z", "distance", "detector", "detector_score", "n_scored", "cleared", "ops"}
)


class TokenHeat(BaseModel):
    """One token's contribution, addressed by CHARACTER offsets into the text.

    Char offsets are why the mirror needs no client tokenizer in SERVER mode.
    `heat = sum_L w[L]*g[i][L] / m` in [0,1], 0.5 neutral. `masked` positions
    contribute nothing to numerator or denominator and render visibly grey —
    they are structural free wins and the player should see that.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    s: int = Field(ge=0, description="start char offset, inclusive")
    e: int = Field(ge=0, description="end char offset, exclusive")
    heat: float = Field(ge=0.0, le=1.0)
    masked: bool = False

    @model_validator(mode="after")
    def _ordered(self) -> TokenHeat:
        if self.e < self.s:
            raise ValueError(f"token span end {self.e} precedes start {self.s}")
        return self


class DetectRequest(BaseModel):
    """POST /api/detect. Not rate limited — a player hammering the detector
    costs one CPU-millisecond and throttling it would make the game feel
    broken."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    passage_id: str = Field(min_length=1, max_length=64)
    text: str = Field(max_length=MAX_TEXT_BYTES)
    seq: int = Field(ge=0, description="monotonic; the client drops out-of-order responses")


class DetectResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    seq: int = Field(ge=0)
    #: sha256 of the exact text this reading describes. The client applies a
    #: reading only if `seq >= lastSeq` AND `text_hash` matches the current
    #: textarea contents (TECH_PLAN.md §10.2) — the one rule that kills needle
    #: flicker from out-of-order responses, stale worker results and the
    #: SERVER->LOCAL handover simultaneously.
    text_hash: Sha256Digest
    score: float = Field(ge=0.0, le=1.0)
    z: float
    z_star: float
    n_scored: int = Field(ge=0)
    n_tokens: int = Field(ge=0)
    tokens: tuple[TokenHeat, ...] = ()
    preview_distance: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _scored_within_tokens(self) -> DetectResponse:
        if self.n_scored > self.n_tokens:
            raise ValueError(
                f"n_scored ({self.n_scored}) exceeds n_tokens ({self.n_tokens}); "
                "scored rows are a masked subset of the ngram windows"
            )
        return self


class SubmitClient(BaseModel):
    """Client-side telemetry. Advisory, never trusted, never scored."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    detector: Literal["local", "server"] = "server"
    elapsed_ms: int = Field(default=0, ge=0)
    asset_bundle_id: AssetBundleId | None = None


class SubmitRequest(BaseModel):
    """POST /api/submit — NOTE: no scores, no z, no distance.

    The client asserts nothing. See the module docstring.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    passage_id: str = Field(min_length=1, max_length=64)
    level_id: str = Field(pattern=r"^L[1-9][0-9]*$")
    text: str = Field(min_length=1, max_length=MAX_TEXT_BYTES)
    client: SubmitClient = SubmitClient()
    #: localStorage UUID. NOT identity. NOT trusted. It is the key campaign
    #: progress is recorded under, so a player who clears a level without one
    #: keeps their progress in localStorage alone.
    session_id: str | None = Field(default=None, max_length=64)

    @model_validator(mode="before")
    @classmethod
    def _reject_client_supplied_scores(cls, data: Any) -> Any:
        if isinstance(data, dict):
            offending = sorted(CLIENT_SCORE_FIELDS.intersection(data))
            if offending:
                raise ValueError(
                    f"submit payload carries client-computed field(s) {offending}. The submit "
                    "contract has no scores: distance, detector score, z and every verdict are "
                    "computed server-side. Adding a client-supplied number would undo the "
                    "anti-cheat property in one line (TECH_PLAN.md §8.3, §12)."
                )
        return data


class ScoreSummary(BaseModel):
    """Server-computed distance plus the shareable diff."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    distance: int = Field(ge=0)
    ops: tuple[EditOp, ...] = ()


class DetectorReading(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    score: float = Field(ge=0.0, le=1.0)
    z: float
    z_star: float
    n_scored: int = Field(ge=0)
    masked_fraction: float = Field(default=0.0, ge=0.0, le=1.0)


class SubmitResponse(BaseModel):
    """A rejection is a game outcome, not an HTTP error — this is a 200 either
    way. Real HTTP errors are reserved for real errors (TECH_PLAN.md §9.3)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cleared: bool
    #: True when the judge was unreachable after retry and failover. The chime
    #: plays; the board and the recorded progress exclude it; the UI says "gate
    #: unavailable". `injection_attempt` is NEVER provisional (§7.5).
    provisional: bool = False
    score: ScoreSummary
    detector: DetectorReading
    failure: GateFailure | None = None
    trace: tuple[CheckResult, ...] = ()

    par: int | None = Field(default=None, ge=1)
    #: Rank among the clears OF THIS LEVEL, or None when this submit did not
    #: clear. Not a daily ranking: the campaign has no day to rank within.
    rank: int | None = Field(default=None, ge=1)
    share: str | None = None

    @model_validator(mode="after")
    def _failure_iff_rejected(self) -> SubmitResponse:
        if self.cleared and self.failure is not None:
            raise ValueError("a cleared submission cannot carry a failure")
        if not self.cleared and self.failure is None:
            raise ValueError("a rejected submission must say which check rejected it")
        return self


class ProgressResponse(BaseModel):
    """GET /api/progress. `Cache-Control: no-store`.

    The server's half of the campaign progress the client also keeps in
    localStorage. An unknown session id is not an error — it is a first visit,
    a cleared cache or a second device — and it answers "nothing cleared, level
    1 unlocked" rather than 404ing a player who has done nothing wrong.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    cleared: tuple[int, ...] = ()
    #: `min(max(cleared, default 0) + 1, level_count)`. Derived, never stored:
    #: a stored value would drift the moment the campaign grew a level.
    unlocked: int = Field(ge=1)
    level_count: int = Field(ge=1)


class HealthResponse(BaseModel):
    """GET /healthz. Executes SELECT 1 — a 200 here promotes the deploy, so it
    must be honest. Exposing `wm_config_id` gives a one-curl answer to "is the
    needle lying?"."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ok: bool
    sha: str = Field(min_length=1)
    wm_config_id: WmConfigId
    asset_bundle_id: AssetBundleId
    judge_version: str = Field(min_length=1)
    scoring_version: str = Field(min_length=1)
