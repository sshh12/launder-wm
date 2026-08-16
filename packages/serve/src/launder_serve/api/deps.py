"""The app state every endpoint reads, and the accessors that reach it.

One object, constructed once in the app factory, attached to `app.state`. No
module-level globals: two test apps in one process must not share a rate-limit
bucket or a judge cache, and a global would make `--workers 1` load-bearing for
a reason unrelated to the one §9.6 gives.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any, cast

from fastapi import Request
from starlette.datastructures import Headers

from launder_core.schemas import LevelConfig
from launder_serve.content import Content, PassageBundle
from launder_serve.engine import CoreScorer, ServerDetector
from launder_serve.errors import NotFound
from launder_serve.judge.protocol import JudgeProvider
from launder_serve.limits import TokenBucketLimiter, client_key
from launder_serve.repo.protocol import DailyRepo, SubmissionRepo
from launder_serve.settings import Settings

__all__ = ["AppState", "state_of", "today_utc"]


def today_utc() -> date:
    """Daily rollover is UTC midnight, stated in the UI (§9.8).

    Any local-time scheme means two players see different puzzles and the
    shared leaderboard becomes incoherent.
    """
    return datetime.now(UTC).date()


@dataclass
class AppState:
    settings: Settings
    content: Content
    prompt_hash: str
    scorer: CoreScorer
    detector: ServerDetector
    judge: JudgeProvider
    submissions: SubmissionRepo
    dailies: DailyRepo
    limiter: TokenBucketLimiter
    engine: Any | None = None
    http: Any | None = None
    boot_warnings: tuple[str, ...] = ()

    # -- lookups that must 404 rather than 500 -------------------------------

    def passage_or_404(self, passage_id: str) -> PassageBundle:
        bundle = self.content.passage(passage_id)
        if bundle is None:
            raise NotFound("unknown_passage", passage_id=passage_id)
        return bundle

    def level_or_404(self, level_id: str) -> LevelConfig:
        level = self.content.level(level_id)
        if level is None:
            raise NotFound("unknown_passage", level_id=level_id)
        return level

    def rate_key(self, headers: Headers) -> str:
        """§7.5 / §14.2 item 6: `X-Real-IP`, shared bucket when absent."""
        limits = self.content.judge.limits
        return client_key(headers, header=limits.rate_limit_header)


def state_of(request: Request) -> AppState:
    state = getattr(request.app.state, "launder", None)
    if state is None:  # pragma: no cover - only reachable if the factory changed
        raise RuntimeError("app.state.launder is unset; build the app with create_app().")
    return cast(AppState, state)
