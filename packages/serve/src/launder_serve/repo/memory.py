"""In-memory repositories. **This module must never import SQLAlchemy.**

That constraint is the point of the exercise: if `repo/protocol.py` ever leaks
an ORM concept, this file stops compiling, and the dual-engine contract suite
stops proving anything. It is also what makes the whole API testable with no
database at all — every serve test that is not specifically about SQL runs
against these.

Semantics match `repo/sqlalchemy.py` exactly, including the two that are easy
to get subtly different and are therefore pinned by the shared contract suite:

* `record()` is an upsert on `(day, level_id, text_hash)` — a player
  resubmitting identical text must not create a second leaderboard row.
* `reserve()` is atomic and refuses the whole reservation when it would cross
  the cap; it never partially spends.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime

from launder_serve.repo.protocol import (
    CachedVerdict,
    CacheStats,
    DailySlot,
    LeaderRow,
    SubmissionRecord,
)

__all__ = ["MemoryDailyRepo", "MemoryJudgeCacheRepo", "MemorySpendRepo", "MemorySubmissionRepo"]


class MemorySubmissionRepo:
    def __init__(self) -> None:
        self._rows: dict[tuple[date, str, str], SubmissionRecord] = {}
        self._order: list[tuple[date, str, str]] = []
        self._lock = asyncio.Lock()

    async def record(self, s: SubmissionRecord) -> None:
        key = (s.day, s.level_id, s.text_hash)
        async with self._lock:
            if key not in self._rows:
                self._order.append(key)
            self._rows[key] = s

    async def best_for_day(self, day: date, level_id: str, limit: int) -> list[LeaderRow]:
        rows = [
            r
            for r in self._rows.values()
            if r.day == day and r.level_id == level_id and r.cleared and not r.provisional
        ]
        # (distance asc, created_at asc) — the earliest submission wins a tie,
        # which is the same ordering the SQL implementation asks the engine for.
        rows.sort(key=lambda r: (r.distance, r.created_at))
        return [
            LeaderRow(distance=r.distance, at=r.created_at, ops=r.ops, elapsed_ms=r.elapsed_ms)
            for r in rows[: max(limit, 0)]
        ]

    async def rank_of(self, day: date, level_id: str, distance: int) -> int:
        better = sum(
            1
            for r in self._rows.values()
            if r.day == day
            and r.level_id == level_id
            and r.cleared
            and not r.provisional
            and r.distance < distance
        )
        return better + 1

    async def count_for_day(self, day: date) -> int:
        return sum(1 for r in self._rows.values() if r.day == day)


class MemoryJudgeCacheRepo:
    def __init__(self) -> None:
        self._rows: dict[str, CachedVerdict] = {}

    async def get(self, key: str) -> CachedVerdict | None:
        return self._rows.get(key)

    async def put(self, key: str, v: CachedVerdict) -> None:
        self._rows[key] = v

    async def purge_version(self, judge_version: str) -> int:
        doomed = [k for k, v in self._rows.items() if v.judge_version == judge_version]
        for k in doomed:
            del self._rows[k]
        return len(doomed)

    async def stats(self, since: datetime) -> CacheStats:
        rows = [v for v in self._rows.values() if v.created_at >= since]
        by_version: dict[str, int] = {}
        for v in rows:
            by_version[v.judge_version] = by_version.get(v.judge_version, 0) + 1
        return CacheStats(
            entries=len(rows),
            cleared=sum(1 for v in rows if v.cleared),
            rejected=sum(1 for v in rows if not v.cleared),
            input_tokens=sum(v.input_tokens for v in rows),
            output_tokens=sum(v.output_tokens for v in rows),
            by_version=by_version,
        )


class MemorySpendRepo:
    def __init__(self) -> None:
        self._spent: dict[date, float] = {}
        self._lock = asyncio.Lock()

    async def reserve(self, day: date, est_usd: float, cap_usd: float) -> bool:
        async with self._lock:
            current = self._spent.get(day, 0.0)
            if current + est_usd > cap_usd:
                return False
            self._spent[day] = current + est_usd
            return True

    async def spent(self, day: date) -> float:
        return self._spent.get(day, 0.0)


class MemoryDailyRepo:
    def __init__(self) -> None:
        self._slots: dict[date, DailySlot] = {}

    async def for_date(self, d: date) -> DailySlot | None:
        return self._slots.get(d)

    async def set_observed_par(self, day: date, level_id: str, par: int) -> None:
        slot = self._slots.get(day)
        if slot is None or slot.level_id != level_id:
            return
        self._slots[day] = DailySlot(
            day=slot.day,
            passage_id=slot.passage_id,
            level_id=slot.level_id,
            created_at=slot.created_at,
            authored_par=slot.authored_par,
            observed_par=par,
        )

    async def upsert(self, slot: DailySlot) -> None:
        """Not part of the §9.6 interface: seeding, used by boot and by tests."""
        self._slots[slot.day] = slot
