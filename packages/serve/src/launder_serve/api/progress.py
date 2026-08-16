"""`GET /api/progress` — how far this session got (TECH_PLAN.md §9.1).

Progress lives in TWO places on purpose. `localStorage` is the fast copy: it is
what `GET /` reconciles against before the first paint, and it costs no network
at all. This endpoint is the durable copy, keyed on the same anonymous
localStorage session id, and it is what survives a cleared cache or a second
device.

**A session id is not identity.** It is a uuid the browser minted for itself, it
is never joined to anything, and it is the only key here — there is no name, no
email, no cookie beyond `launder_level`, and nothing to log in to. An unknown id
is therefore an ordinary answer ("nothing cleared, level 1 unlocked") rather
than a 404: a first visit, a cleared cache and a typo are indistinguishable and
all three deserve a playable game.

`unlocked` is DERIVED (`min(max(cleared) + 1, level_count)`) rather than stored.
A stored value would be wrong the moment the campaign grew a sixteenth level,
and it would be a second thing that could disagree with the `cleared` list.

`Cache-Control: no-store`: it is per-session and it changes on every clear.
"""

from __future__ import annotations

from fastapi import APIRouter, Query, Request, Response

from launder_core.schemas import ProgressResponse
from launder_serve.api.deps import state_of

__all__ = ["router"]

router = APIRouter(prefix="/api", tags=["progress"])


@router.get("/progress", response_model=ProgressResponse, summary="Campaign progress")
async def progress(
    request: Request,
    response: Response,
    session_id: str = Query(default="", max_length=64),
) -> ProgressResponse:
    state = state_of(request)
    response.headers["Cache-Control"] = "no-store"

    # Floored at 1: a server with no passages at all (a clone with no dev
    # fixture) still owes a well-formed answer rather than a validation 500 on
    # `level_count >= 1`.
    level_count = max(state.content.level_count, 1)
    cleared = await state.progress.cleared_levels(session_id) if session_id else []
    # Levels the campaign no longer has are dropped rather than counted: a
    # shortened campaign must not leave `unlocked` pointing past its own end.
    cleared = [n for n in cleared if 1 <= n <= level_count]
    unlocked = min(max(cleared, default=0) + 1, level_count)
    return ProgressResponse(cleared=tuple(cleared), unlocked=unlocked, level_count=level_count)
