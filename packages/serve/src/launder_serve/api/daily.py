"""`GET /api/daily` and `GET /api/leaderboard/{day}` (TECH_PLAN.md §9.1, §9.4).

`schedule.toml` is the source of truth for which passage runs on which day —
editing that file and restarting is the whole of "change the daily" (§12 row
14). The database contributes exactly one field, `observed_par`, because that
one is discovered by play rather than authored.

`GET /api/daily` is `public, max-age=60`; the leaderboard is `max-age=30`. The
index HTML inlines the passage JSON, so a first-time player makes **zero** API
calls before playing — this endpoint exists for the second visit and for
clients that want the day's metadata without re-parsing the page.

Diffs are public by design (§8.3): the brag, the anti-cheat mechanism and the
teaching tool are the same object. No names, no session ids, no identity.
"""

from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Path, Query, Request, Response

from launder_core.schemas import DailyResponse, LeaderboardEntry, LeaderboardResponse
from launder_serve.api.deps import AppState, state_of, today_utc
from launder_serve.content import DailySlotSpec
from launder_serve.errors import NotFound

__all__ = ["router"]

router = APIRouter(prefix="/api", tags=["daily"])


def _slot_or_404(state: AppState, day: date) -> DailySlotSpec:
    slot = state.content.schedule.for_date(day)
    if slot is None or slot.passage_id not in state.content.passages:
        raise NotFound("unknown_passage", day=day.isoformat())
    return slot


@router.get("/daily", response_model=DailyResponse, summary="Today's puzzle")
async def daily(response: Response, request: Request) -> DailyResponse:
    state = state_of(request)
    day = today_utc()
    slot = _slot_or_404(state, day)
    bundle = state.content.passage(slot.passage_id)
    assert bundle is not None  # _slot_or_404 checked membership

    stored = await state.dailies.for_date(day)
    response.headers["Cache-Control"] = "public, max-age=60"
    return DailyResponse(
        day=day,
        passage_id=slot.passage_id,
        level_id=slot.level_id,
        par=bundle.par,
        observed_par=stored.observed_par if stored is not None else None,
        puzzle_number=state.content.schedule.puzzle_number(day),
        passage_url=f"/data/passages/{slot.passage_id}.public.json",
        asset_bundle_id=state.content.asset_bundle_id,
    )


@router.get(
    "/leaderboard/{day}",
    response_model=LeaderboardResponse,
    summary="Public diffs for one day and level",
)
async def leaderboard(
    request: Request,
    response: Response,
    day: date = Path(description="UTC date, YYYY-MM-DD"),  # noqa: B008 - FastAPI's declaration idiom
    level_id: str = Query(default="", pattern=r"^(L[1-9][0-9]*)?$"),
    limit: int = Query(default=20, ge=1, le=100),
) -> LeaderboardResponse:
    state = state_of(request)
    slot = _slot_or_404(state, day)
    resolved_level = level_id or slot.level_id
    if resolved_level not in state.content.levels:
        raise NotFound("unknown_passage", level_id=resolved_level)

    bundle = state.content.passage(slot.passage_id)
    rows = await state.submissions.best_for_day(day, resolved_level, limit)

    response.headers["Cache-Control"] = "public, max-age=30"
    return LeaderboardResponse(
        day=day,
        level_id=resolved_level,
        par=bundle.par if bundle is not None else None,
        # `machine_par` is the solver's upper bound, which lives in the
        # author-side bundle and is deliberately excluded from the image
        # (§11.3). It stays None here until `forge pack` promotes it.
        machine_par=None,
        rows=tuple(
            LeaderboardEntry(
                rank=i + 1,
                distance=row.distance,
                ops=row.ops,
                elapsed_ms=row.elapsed_ms,
                at=row.at,
            )
            for i, row in enumerate(rows)
        ),
    )
