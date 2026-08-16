"""The repository interfaces (TECH_PLAN.md §9.6).

**Narrow, hand-written, no ORM in the interface.** SQLAlchemy lives strictly in
`repo/sqlalchemy.py`; if a `Table`, a `Session`, a `select()` or a `Row` ever
appears in this file, the abstraction has stopped being one. The test for that
is mechanical and it is in CI: `packages/serve/tests/test_repo_contract.py`
runs ONE suite against BOTH implementations, and `memory.py` cannot import
SQLAlchemy at all.

The row types below are plain frozen dataclasses rather than pydantic models on
purpose: they are internal boundary types, they never touch the wire, and
`EditOp`/`ScoreResult` (which DO touch the wire) already live in
`launder_core.schemas`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Protocol

from launder_core.schemas import EditOp

__all__ = [
    "CacheStats",
    "CachedVerdict",
    "DailyRepo",
    "DailySlot",
    "JudgeCacheRepo",
    "LeaderRow",
    "SpendRepo",
    "SubmissionRecord",
    "SubmissionRepo",
]


@dataclass(frozen=True)
class SubmissionRecord:
    """One row of `submission` (§9.7).

    `text` is UNTRUSTED. Never feed it to an LLM that reads logs; the judge sees
    it only through the nonce sandwich, and nothing else should see it at all.
    `distance` is server-computed and is the only authority.
    """

    day: date
    passage_id: str
    level_id: str
    text_hash: str
    text: str
    cleared: bool
    distance: int
    detector_score: float
    detector_z: float
    n_scored: int
    masked_fraction: float
    scoring_version: str
    wm_config_id: str
    created_at: datetime
    provisional: bool = False
    ops: tuple[EditOp, ...] = ()
    failure_code: str | None = None
    session_id: str | None = None
    elapsed_ms: int | None = None


@dataclass(frozen=True)
class LeaderRow:
    """A public leaderboard row. No names, no session ids, no identity (§9.4)."""

    distance: int
    at: datetime
    ops: tuple[EditOp, ...] = ()
    elapsed_ms: int | None = None


@dataclass(frozen=True)
class CachedVerdict:
    """A judge result, content-addressed and version-scoped.

    `observation` is stored rather than only the derived verdict, so a level
    whose `max_missing_claims` changes re-derives from the same observation
    instead of paying for the same API call again. `status=error` is NEVER
    cached — a provisional clear must not become permanent (§7.5).
    """

    judge_version: str
    passage_id: str
    level_id: str
    cleared: bool
    observation: dict[str, Any]
    provider: str
    model: str
    created_at: datetime
    failure_code: str | None = None
    feedback: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class CacheStats:
    entries: int = 0
    cleared: int = 0
    rejected: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    by_version: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class DailySlot:
    day: date
    passage_id: str
    level_id: str
    created_at: datetime
    authored_par: int | None = None
    #: Best clear in the first N plays; self-balancing (§9.1).
    observed_par: int | None = None


class SubmissionRepo(Protocol):
    async def record(self, s: SubmissionRecord) -> None: ...

    async def best_for_day(self, day: date, level_id: str, limit: int) -> list[LeaderRow]: ...

    async def rank_of(self, day: date, level_id: str, distance: int) -> int: ...

    async def count_for_day(self, day: date) -> int: ...


class JudgeCacheRepo(Protocol):
    async def get(self, key: str) -> CachedVerdict | None: ...

    async def put(self, key: str, v: CachedVerdict) -> None: ...

    async def purge_version(self, judge_version: str) -> int: ...

    async def stats(self, since: datetime) -> CacheStats: ...


class SpendRepo(Protocol):
    async def reserve(self, day: date, est_usd: float, cap_usd: float) -> bool:
        """ATOMIC reserve-then-call. `False` means refused -> degrade to provisional.

        Never raise on refusal and never 500: a spend cap is an expected
        operating condition, and the player's submission is still worth a
        chime and a "gate unavailable" label (§7.5).
        """
        ...


class DailyRepo(Protocol):
    async def for_date(self, d: date) -> DailySlot | None: ...

    async def set_observed_par(self, day: date, level_id: str, par: int) -> None: ...
