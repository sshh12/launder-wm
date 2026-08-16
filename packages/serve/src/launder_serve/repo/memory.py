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
import dataclasses
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


#: The columns `SqlSubmissionRepo.record`'s `ON CONFLICT DO UPDATE` actually
#: writes on a duplicate `(level_n, text_hash)`. Everything NOT named here —
#: `created_at`, `passage_id`, `level_id`, `text`, `session_id`,
#: `scoring_version`, `wm_config_id` — belongs to the FIRST insert and the SQL
#: statement leaves it alone.
#:
#: This list exists because `record()` here replaced the whole row instead, and
#: `created_at` is the leaderboard's tie-break: `best_for_level` orders by
#: (distance asc, created_at asc), so a player resubmitting identical text moved
#: to the BACK of a tie under the memory repo and stayed where they were under
#: the SQL one. One suite runs against both, and it could not see the difference
#: because every record it wrote carried the same timestamp.
_UPSERT_FIELDS: tuple[str, ...] = (
    "cleared",
    "provisional",
    "distance",
    "ops",
    "detector_score",
    "detector_z",
    "n_scored",
    "masked_fraction",
    "failure_code",
    "elapsed_ms",
)


class MemorySubmissionRepo:
    def __init__(self) -> None:
        self._rows: dict[tuple[int, str], SubmissionRecord] = {}
        self._lock = asyncio.Lock()

    async def record(self, s: SubmissionRecord) -> None:
        key = (s.level_n, s.text_hash)
        async with self._lock:
            existing = self._rows.get(key)
            if existing is None:
                self._rows[key] = s
                return
            self._rows[key] = dataclasses.replace(
                existing, **{name: getattr(s, name) for name in _UPSERT_FIELDS}
            )

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
        existing = self._rows.get(key)
        if existing is None:
            self._rows[key] = v
            return
        # Same rule as `MemorySubmissionRepo.record`: `SqlJudgeCacheRepo.put`'s
        # `ON CONFLICT DO UPDATE` writes only these five columns, so the
        # provider, the model and the token counts belong to the call that first
        # produced this key. Replacing the whole row here made `stats()` report
        # the LAST write's token counts on one implementation and the FIRST
        # write's on the other, for the same sequence of calls.
        self._rows[key] = dataclasses.replace(
            existing,
            cleared=v.cleared,
            failure_code=v.failure_code,
            feedback=v.feedback,
            observation=v.observation,
            created_at=v.created_at,
        )

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
