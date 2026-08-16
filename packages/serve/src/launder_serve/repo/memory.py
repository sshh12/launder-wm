"""In-memory repositories. **This module must never import SQLAlchemy.**

That constraint is the point of the exercise: if `repo/protocol.py` ever leaks
an ORM concept, this file stops compiling, and the dual-engine contract suite
stops proving anything. It is also what makes the whole API testable with no
database at all — every serve test that is not specifically about SQL runs
against these.

Semantics match `repo/sqlalchemy.py` exactly, including the three that are easy
to get subtly different and are therefore pinned by the shared contract suite:

* `record()` is an upsert on `(level_n, text_hash)` — a player resubmitting
  identical text must not create a second leaderboard row.
* `reserve()` is atomic and refuses the whole reservation when it would cross
  the cap; it never partially spends.
* `ProgressRepo.record()` keeps the LOWER distance on conflict — replaying a
  cleared level with a sloppier solve must not worsen the record.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime

from launder_serve.repo.protocol import (
    CachedVerdict,
    CacheStats,
    LeaderRow,
    SubmissionRecord,
)

__all__ = [
    "MemoryJudgeCacheRepo",
    "MemoryProgressRepo",
    "MemorySpendRepo",
    "MemorySubmissionRepo",
]


class MemorySubmissionRepo:
    def __init__(self) -> None:
        self._rows: dict[tuple[int, str], SubmissionRecord] = {}
        self._lock = asyncio.Lock()

    async def record(self, s: SubmissionRecord) -> None:
        key = (s.level_n, s.text_hash)
        async with self._lock:
            self._rows[key] = s

    async def best_for_level(self, level_n: int, limit: int) -> list[LeaderRow]:
        rows = [
            r
            for r in self._rows.values()
            if r.level_n == level_n and r.cleared and not r.provisional
        ]
        # (distance asc, created_at asc) — the earliest submission wins a tie,
        # which is the same ordering the SQL implementation asks the engine for.
        rows.sort(key=lambda r: (r.distance, r.created_at))
        return [
            LeaderRow(distance=r.distance, at=r.created_at, ops=r.ops, elapsed_ms=r.elapsed_ms)
            for r in rows[: max(limit, 0)]
        ]

    async def rank_of(self, level_n: int, distance: int) -> int:
        better = sum(
            1
            for r in self._rows.values()
            if r.level_n == level_n and r.cleared and not r.provisional and r.distance < distance
        )
        return better + 1

    async def count_for_level(self, level_n: int) -> int:
        return sum(1 for r in self._rows.values() if r.level_n == level_n)


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


class MemoryProgressRepo:
    def __init__(self) -> None:
        self._best: dict[tuple[str, int], int] = {}
        self._lock = asyncio.Lock()

    async def record(self, session_id: str, level_n: int, distance: int) -> None:
        key = (session_id, level_n)
        async with self._lock:
            current = self._best.get(key)
            if current is None or distance < current:
                self._best[key] = distance

    async def cleared_levels(self, session_id: str) -> list[int]:
        return sorted(n for (sid, n) in self._best if sid == session_id)
