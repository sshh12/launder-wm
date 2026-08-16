"""`POST /api/submit` — the gate (TECH_PLAN.md §9.3).

**The request carries no scores.** No `z`, no `distance`, no `cleared`. The
client asserts nothing, the server recomputes everything, and that is what makes
publishing the watermark keys costless to integrity: a client that lies about
its needle is lying only to itself. `SubmitRequest` rejects any score-shaped
field by name, with a message saying why, so the next person to add one gets an
explanation rather than "extra inputs are not permitted".

**A rejection is a game outcome, not an HTTP error.** Clearing and failing are
both 200s carrying a `trace`, which returns pass or fail and stops at the first
failing check. That is what powers the gate pips, and it is also the proof that
a deterministic failure cost nothing: if `llm_gate` is not in the trace, it was
not called.

The ordered pipeline itself is core's (`launder_core.run_gate`) — ORDER IS THE
SEMANTICS, and a second copy of that loop here would be a second thing to keep
in step with `levels.toml`.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Request, Response

from launder_core.gates.feedback import render_feedback
from launder_core.gates.registry import Deps, GateContext, run_gate
from launder_core.schemas import (
    DetectorReading,
    GateFailure,
    GateResult,
    ScoreSummary,
    SubmitRequest,
    SubmitResponse,
)
from launder_serve.api.deps import AppState, state_of
from launder_serve.errors import RateLimited
from launder_serve.judge.gate import RATE_LIMITED, REQUEST_CLIENT_KEY
from launder_serve.repo.protocol import SubmissionRecord

__all__ = ["router"]

_log = logging.getLogger("launder.submit")
router = APIRouter(prefix="/api", tags=["submit"])


@router.post("/submit", response_model=SubmitResponse, summary="Run the gate")
async def submit(request: Request, body: SubmitRequest, response: Response) -> SubmitResponse:
    state = state_of(request)
    response.headers["Cache-Control"] = "no-store"

    bundle = state.passage_or_404(body.passage_id)
    ruleset = state.ruleset_or_404(body.level_id)
    # The request does not carry `level_n` — the client asserts nothing about
    # where it is in the campaign, exactly as it asserts nothing about its own
    # score. The position is the passage's, and the server owns the mapping.
    level_n = state.level_n_or_404(body.passage_id)

    # Everything below is server-computed. This is the entire anti-cheat story
    # and it is three lines long.
    normalized = state.scorer.normalize(body.text)
    score = state.scorer.score(bundle.public.text, body.text)

    # The per-IP bucket keys on `X-Real-IP` and is consulted on judge cache
    # MISSES ONLY. A ContextVar rather than a `Deps` field: client identity is
    # an HTTP concern and core should not grow a place to put one.
    REQUEST_CLIENT_KEY.set(state.rate_key(request.headers))
    RATE_LIMITED.set(None)

    gate: GateResult = await run_gate(
        GateContext(
            passage=bundle.public,
            level=ruleset,
            raw=body.text,
            deps=Deps(
                detector=state.detector,
                judge=state.judge,
                normalize_config=state.scorer.config,
                claims=bundle.claims,
            ),
        )
    )

    # A rate-limited judge miss is the ONE gate outcome that is a real HTTP
    # error rather than a game outcome: the player has not lost, we declined to
    # spend on them right now, and `Retry-After` says when to come back.
    retry_after = RATE_LIMITED.get()
    if retry_after is not None:
        raise RateLimited(retry_after)

    detector = _reading_from(gate) or _fallback_reading(state)
    failure = _failure(state, gate)

    record = SubmissionRecord(
        level_n=level_n,
        passage_id=bundle.id,
        level_id=body.level_id,
        text_hash=hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        text=body.text,
        cleared=gate.cleared,
        provisional=gate.provisional,
        distance=score.distance,
        ops=score.ops,
        detector_score=detector.score,
        detector_z=detector.z,
        n_scored=detector.n_scored,
        masked_fraction=detector.masked_fraction,
        failure_code=failure.code if failure is not None else None,
        scoring_version=state.content.scoring_version,
        wm_config_id=state.content.wm_config_id,
        session_id=body.session_id,
        elapsed_ms=body.client.elapsed_ms or None,
        created_at=datetime.now(UTC),
    )
    await state.submissions.record(record)

    rank: int | None = None
    share: str | None = None
    # A PROVISIONAL clear does not advance the campaign. The judge never
    # answered, so the meaning arm of the gate never ran, and unlocking the next
    # level on one would let a judge outage walk somebody through all fifteen.
    # A player with no `session_id` (localStorage refused) still clears the
    # level; their progress simply lives in the cookie and nowhere else.
    if gate.cleared and not gate.provisional:
        rank = await state.submissions.rank_of(level_n, score.distance)
        share = _share(state, level_n, score.distance)
        if body.session_id:
            await state.progress.record(body.session_id, level_n, score.distance)

    return SubmitResponse(
        cleared=gate.cleared,
        provisional=gate.provisional,
        score=ScoreSummary(distance=score.distance, ops=score.ops),
        detector=detector,
        failure=failure,
        trace=gate.trace,
        par=bundle.par,
        rank=rank,
        share=share,
    )


def _reading_from(gate: GateResult) -> DetectorReading | None:
    """The reading `detector_threshold` already computed — never a second read.

    Reading twice would let the number the player is judged on differ from the
    number they are shown, which is the one disagreement this game cannot
    survive.
    """
    for result in gate.trace:
        meta = result.meta
        if result.check == "detector_threshold" and "score" in meta and "z" in meta:
            return DetectorReading(
                score=float(meta["score"]),
                z=float(meta["z"]),
                z_star=float(meta.get("z_star", 0.0)),
                n_scored=int(meta.get("n_scored", 0)),
                masked_fraction=float(meta.get("masked_fraction", 0.0)),
            )
    return None


def _fallback_reading(state: AppState) -> DetectorReading:
    """A level with no `detector_threshold` (L5's suite runs instead) still owes
    the client a reading-shaped object. Zeros, honestly labelled, not a guess."""
    del state
    return DetectorReading(score=0.5, z=0.0, z_star=0.0, n_scored=0, masked_fraction=0.0)


def _failure(state: AppState, gate: GateResult) -> GateFailure | None:
    if gate.failure is None:
        return None
    # Core renders it: every string comes from copy.toml, `claim_label` is
    # authored offline, and `notes` passes the §7.4 output filter before it is
    # allowed anywhere near a screen.
    return render_feedback(gate.failure, state.content.copy)


def _share(state: AppState, level_n: int, distance: int) -> str:
    """ "Launder WM — level 3 cleared in 4 🧼", from copy.toml.

    The level number is the campaign position and nothing else: it does not
    depend on when the player played, so a share string posted today still
    names the same level next year.
    """
    readout: Any = state.content.copy.raw.get("readout", {})
    template = str(readout.get("share_template", "")) if isinstance(readout, dict) else ""
    if not template:
        return ""
    return template.format(level_n=level_n, distance=distance)
