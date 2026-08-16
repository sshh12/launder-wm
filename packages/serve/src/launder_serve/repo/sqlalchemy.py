"""SQLAlchemy Core implementations. Postgres *and* SQLite through the same code.

`postgresql+psycopg://` and `sqlite+aiosqlite://` differ in exactly one place in
this file — `_insert()` — and that single branch is the §9.6 rule "one dialect
branch, in one function" made literal. Everything else is dialect-neutral, and
the dual-engine contract suite is what proves it stayed that way.

**psycopg3, not asyncpg.** asyncpg is not libpq and rejects Railway's
`DATABASE_URL` query params (`sslmode=`, `channel_binding=`), forcing URL
surgery *and* a second driver for Alembic. Throughput is irrelevant at
daily-puzzle volume; accepting `${{Postgres.DATABASE_URL}}` verbatim is worth
more.

One hazard worth naming because it bites silently: **SQLite does not store
timezones.** A `datetime` written as UTC-aware comes back naive, and a naive
`datetime` compared against an aware one raises. `_as_utc()` re-attaches UTC on
the way out, so both engines hand the caller the same aware value.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, date, datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from launder_core.schemas import EditOp
from launder_serve.repo.models import METADATA, daily_slot, judge_cache, spend_ledger, submission
from launder_serve.repo.protocol import (
    CachedVerdict,
    CacheStats,
    DailySlot,
    LeaderRow,
    SubmissionRecord,
)

__all__ = [
    "SqlDailyRepo",
    "SqlJudgeCacheRepo",
    "SqlSpendRepo",
    "SqlSubmissionRepo",
    "create_all",
    "make_engine",
]


def make_engine(url: str) -> AsyncEngine:
    """TECH_PLAN.md §9.6, reproduced.

    Do NOT set `pool_size=20` "for the HN spike": the bottleneck is the judge
    call and the rate limiter, and a fat pool converts a traffic spike into
    `FATAL: too many connections`.
    """
    if url.startswith("postgres://"):
        url = "postgresql://" + url[11:]
    if url.startswith("postgresql://"):
        return create_async_engine(
            "postgresql+psycopg://" + url[13:],
            pool_size=5,
            max_overflow=5,  # 1 replica x 1 worker x 10 conns; PG default max is 100
            pool_timeout=10,  # fail fast under load rather than pile up
            pool_recycle=1800,  # Railway proxies drop idle conns
            pool_pre_ping=True,  # non-negotiable on managed PG
            connect_args={"prepare_threshold": None},  # safe if pgbouncer ever appears
        )
    return create_async_engine(url, connect_args={"check_same_thread": False})


async def create_all(engine: AsyncEngine) -> None:
    """Schema without Alembic. Tests and local dev only — production runs
    `alembic upgrade head` as `preDeployCommand` (§9.8)."""
    async with engine.begin() as conn:
        if conn.dialect.name == "sqlite":
            # Concurrent writers on the dev file DB (§9.6 hazard table).
            await conn.exec_driver_sql("PRAGMA journal_mode=WAL")
        await conn.run_sync(METADATA.create_all)


def _insert(table: sa.Table, dialect: str) -> Any:
    """THE dialect branch. One function, one line, nothing else in this module."""
    return sqlite_insert(table) if dialect == "sqlite" else pg_insert(table)


def _as_utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _ops_json(ops: Sequence[EditOp]) -> list[dict[str, Any]]:
    #: `EditOp`'s serializer emits the wire key `from`; no `by_alias` needed.
    return [op.model_dump() for op in ops]


def _ops_from_json(raw: Any) -> tuple[EditOp, ...]:
    if not raw:
        return ()
    return tuple(EditOp.model_validate(item) for item in raw)


class SqlSubmissionRepo:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def record(self, s: SubmissionRecord) -> None:
        stmt = _insert(submission, self._engine.dialect.name).values(
            day=s.day,
            passage_id=s.passage_id,
            level_id=s.level_id,
            text_hash=s.text_hash,
            text=s.text,
            cleared=s.cleared,
            provisional=s.provisional,
            distance=s.distance,
            ops=_ops_json(s.ops),
            detector_score=s.detector_score,
            detector_z=s.detector_z,
            n_scored=s.n_scored,
            masked_fraction=s.masked_fraction,
            failure_code=s.failure_code,
            scoring_version=s.scoring_version,
            wm_config_id=s.wm_config_id,
            session_id=s.session_id,
            elapsed_ms=s.elapsed_ms,
            created_at=s.created_at,
        )
        # The dedup index is (day, level_id, text_hash): a player nudging a
        # broken submission and resending identical text updates the row rather
        # than minting a second leaderboard entry.
        stmt = stmt.on_conflict_do_update(
            index_elements=[submission.c.day, submission.c.level_id, submission.c.text_hash],
            set_={
                "cleared": stmt.excluded.cleared,
                "provisional": stmt.excluded.provisional,
                "distance": stmt.excluded.distance,
                "ops": stmt.excluded.ops,
                "detector_score": stmt.excluded.detector_score,
                "detector_z": stmt.excluded.detector_z,
                "n_scored": stmt.excluded.n_scored,
                "masked_fraction": stmt.excluded.masked_fraction,
                "failure_code": stmt.excluded.failure_code,
                "elapsed_ms": stmt.excluded.elapsed_ms,
            },
        )
        async with self._engine.begin() as conn:
            await conn.execute(stmt)

    async def best_for_day(self, day: date, level_id: str, limit: int) -> list[LeaderRow]:
        stmt = (
            sa.select(
                submission.c.distance,
                submission.c.ops,
                submission.c.elapsed_ms,
                submission.c.created_at,
            )
            .where(
                submission.c.day == day,
                submission.c.level_id == level_id,
                submission.c.cleared.is_(True),
                submission.c.provisional.is_(False),
            )
            .order_by(submission.c.distance.asc(), submission.c.created_at.asc())
            .limit(max(limit, 0))
        )
        async with self._engine.connect() as conn:
            rows = (await conn.execute(stmt)).all()
        return [
            LeaderRow(
                distance=row.distance,
                at=_as_utc(row.created_at),
                ops=_ops_from_json(row.ops),
                elapsed_ms=row.elapsed_ms,
            )
            for row in rows
        ]

    async def rank_of(self, day: date, level_id: str, distance: int) -> int:
        stmt = sa.select(sa.func.count()).where(
            submission.c.day == day,
            submission.c.level_id == level_id,
            submission.c.cleared.is_(True),
            submission.c.provisional.is_(False),
            submission.c.distance < distance,
        )
        async with self._engine.connect() as conn:
            better = (await conn.execute(stmt)).scalar_one()
        return int(better) + 1

    async def count_for_day(self, day: date) -> int:
        stmt = sa.select(sa.func.count()).where(submission.c.day == day)
        async with self._engine.connect() as conn:
            return int((await conn.execute(stmt)).scalar_one())


class SqlJudgeCacheRepo:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def get(self, key: str) -> CachedVerdict | None:
        stmt = sa.select(judge_cache).where(judge_cache.c.key == key)
        async with self._engine.connect() as conn:
            row = (await conn.execute(stmt)).mappings().first()
        if row is None:
            return None
        return CachedVerdict(
            judge_version=row["judge_version"],
            passage_id=row["passage_id"],
            level_id=row["level_id"],
            cleared=bool(row["cleared"]),
            observation=dict(row["observation"]),
            provider=row["provider"],
            model=row["model"],
            created_at=_as_utc(row["created_at"]),
            failure_code=row["failure_code"],
            feedback=row["feedback"],
            input_tokens=int(row["input_tokens"]),
            output_tokens=int(row["output_tokens"]),
        )

    async def put(self, key: str, v: CachedVerdict) -> None:
        stmt = _insert(judge_cache, self._engine.dialect.name).values(
            key=key,
            judge_version=v.judge_version,
            passage_id=v.passage_id,
            level_id=v.level_id,
            cleared=v.cleared,
            failure_code=v.failure_code,
            feedback=v.feedback,
            observation=v.observation,
            provider=v.provider,
            model=v.model,
            input_tokens=v.input_tokens,
            output_tokens=v.output_tokens,
            created_at=v.created_at,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[judge_cache.c.key],
            set_={
                "cleared": stmt.excluded.cleared,
                "failure_code": stmt.excluded.failure_code,
                "feedback": stmt.excluded.feedback,
                "observation": stmt.excluded.observation,
                "created_at": stmt.excluded.created_at,
            },
        )
        async with self._engine.begin() as conn:
            await conn.execute(stmt)

    async def purge_version(self, judge_version: str) -> int:
        stmt = sa.delete(judge_cache).where(judge_cache.c.judge_version == judge_version)
        async with self._engine.begin() as conn:
            result = await conn.execute(stmt)
        return int(result.rowcount or 0)

    async def stats(self, since: datetime) -> CacheStats:
        stmt = sa.select(
            judge_cache.c.judge_version,
            judge_cache.c.cleared,
            judge_cache.c.input_tokens,
            judge_cache.c.output_tokens,
        ).where(judge_cache.c.created_at >= since)
        async with self._engine.connect() as conn:
            rows = (await conn.execute(stmt)).all()
        by_version: dict[str, int] = {}
        for row in rows:
            by_version[row.judge_version] = by_version.get(row.judge_version, 0) + 1
        return CacheStats(
            entries=len(rows),
            cleared=sum(1 for r in rows if r.cleared),
            rejected=sum(1 for r in rows if not r.cleared),
            input_tokens=sum(int(r.input_tokens) for r in rows),
            output_tokens=sum(int(r.output_tokens) for r in rows),
            by_version=by_version,
        )


class SqlSpendRepo:
    """Atomic reserve-then-call (§9.7)."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def reserve(self, day: date, est_usd: float, cap_usd: float) -> bool:
        # A single call larger than the whole cap must be refused before the
        # statement runs: the ON CONFLICT guard only fires on the *second* and
        # later reservations of a day, so the first insert would otherwise
        # sail past a cap it already exceeds.
        if est_usd > cap_usd:
            return False
        stmt = _insert(spend_ledger, self._engine.dialect.name).values(day=day, spent_usd=est_usd)
        new_total = spend_ledger.c.spent_usd + est_usd
        stmt = stmt.on_conflict_do_update(
            index_elements=[spend_ledger.c.day],
            set_={"spent_usd": new_total},
            where=new_total <= cap_usd,
        ).returning(spend_ledger.c.spent_usd)
        async with self._engine.begin() as conn:
            row = (await conn.execute(stmt)).first()
        # No row returned == the DO UPDATE's WHERE refused it == over cap.
        return row is not None

    async def spent(self, day: date) -> float:
        stmt = sa.select(spend_ledger.c.spent_usd).where(spend_ledger.c.day == day)
        async with self._engine.connect() as conn:
            value = (await conn.execute(stmt)).scalar_one_or_none()
        return float(value or 0.0)


class SqlDailyRepo:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def for_date(self, d: date) -> DailySlot | None:
        stmt = sa.select(daily_slot).where(daily_slot.c.day == d)
        async with self._engine.connect() as conn:
            row = (await conn.execute(stmt)).mappings().first()
        if row is None:
            return None
        return DailySlot(
            day=row["day"],
            passage_id=row["passage_id"],
            level_id=row["level_id"],
            created_at=_as_utc(row["created_at"]),
            authored_par=row["authored_par"],
            observed_par=row["observed_par"],
        )

    async def set_observed_par(self, day: date, level_id: str, par: int) -> None:
        stmt = (
            sa.update(daily_slot)
            .where(daily_slot.c.day == day, daily_slot.c.level_id == level_id)
            .values(observed_par=par)
        )
        async with self._engine.begin() as conn:
            await conn.execute(stmt)

    async def upsert(self, slot: DailySlot) -> None:
        """Not part of the §9.6 interface: seeding, used by boot and by tests."""
        stmt = _insert(daily_slot, self._engine.dialect.name).values(
            day=slot.day,
            passage_id=slot.passage_id,
            level_id=slot.level_id,
            authored_par=slot.authored_par,
            observed_par=slot.observed_par,
            created_at=slot.created_at,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[daily_slot.c.day],
            set_={
                "passage_id": stmt.excluded.passage_id,
                "level_id": stmt.excluded.level_id,
                "authored_par": stmt.excluded.authored_par,
            },
        )
        async with self._engine.begin() as conn:
            await conn.execute(stmt)
