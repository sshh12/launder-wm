"""`GET /healthz` (TECH_PLAN.md §9.5).

**A 200 here promotes the deploy, so it must be honest.** It executes
`SELECT 1`; a database that is configured but unreachable returns 503 and the
old release keeps serving.

Exposing `wm_config_id` gives a one-curl answer to "is the needle lying?" — the
id the server computed from `data/config/watermark.toml` is the same id the
browser asserts against every passage it loads, so a mismatch is visible from
outside without reading a line of code.

Railway's probe arrives from hostname `healthcheck.railway.app`; if
`TrustedHostMiddleware` is ever added, allowlist it or every deploy fails.
"""

from __future__ import annotations

import logging

import sqlalchemy as sa
from fastapi import APIRouter, Request, Response

from launder_core.schemas import HealthResponse
from launder_serve.api.deps import AppState, state_of
from launder_serve.errors import DatabaseUnavailable

__all__ = ["router"]

_log = logging.getLogger("launder.health")
router = APIRouter(tags=["health"])


async def _select_1(state: AppState) -> None:
    engine = state.engine
    if engine is None:
        # In-memory repositories: there is no database to be honest about, and
        # claiming one is up would be exactly the dishonesty this endpoint
        # exists to prevent.
        return
    try:
        async with engine.connect() as conn:
            await conn.execute(sa.text("SELECT 1"))
    except Exception as exc:
        _log.error("healthz SELECT 1 failed: %s", type(exc).__name__)
        raise DatabaseUnavailable(type(exc).__name__) from exc


@router.get("/healthz", response_model=HealthResponse, summary="Deploy gate")
async def healthz(request: Request, response: Response) -> HealthResponse:
    state = state_of(request)
    await _select_1(state)
    response.headers["Cache-Control"] = "no-store"
    return HealthResponse(
        ok=True,
        sha=state.settings.sha,
        wm_config_id=state.content.wm_config_id,
        asset_bundle_id=state.content.asset_bundle_id,
        judge_version=state.content.judge_version,
        scoring_version=state.content.scoring_version,
    )
