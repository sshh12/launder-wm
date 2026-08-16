"""ONE suite, BOTH implementations (TECH_PLAN.md §9.6).

This is the whole point of `repo/protocol.py`. Every test below is
parametrized over the in-memory repositories and the SQLAlchemy ones running
against `sqlite+aiosqlite` in a tmp file, and neither implementation is allowed
to be special-cased. If a test needs to know which one it is talking to, the
abstraction has leaked and the fix is in the implementation, not here.

There is a THIRD leg: Postgres, when `TEST_POSTGRES_URL` is set. §11.6 says this
suite is "parametrized over **both** `sqlite+aiosqlite://` and the Postgres
service", and CI spins up a `postgres:17` service and exports the URL — but
nothing in the repository read it. The only consumer was the `alembic upgrade
head` step, so the service proved that migrations apply and NOTHING else, while
`pool_pre_ping`, `prepare_threshold`, the JSONB columns, the upsert shape,
`.is_(True)` and the timezone round-trip in `repo/sqlalchemy.py` were all
unexercised behind a green tick.

The Postgres leg SKIPS when the variable is unset (a laptop with no server) and
RUNS whenever it is set. `test_ci_runs_the_postgres_leg` below is what stops the
skip from being how the gate passes in CI.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from launder_core.schemas import EditOp
from launder_serve.repo.memory import (
    MemoryJudgeCacheRepo,
    MemoryProgressRepo,
    MemorySpendRepo,
    MemorySubmissionRepo,
)
from launder_serve.repo.protocol import CachedVerdict, SubmissionRecord

#: The BILLING day of the spend ledger — the one date left in the schema.
DAY = date(2026, 9, 1)
NOW = datetime(2026, 9, 1, 6, 11, 2, tzinfo=UTC)
LEVEL = 7


class Repos:
    def __init__(
        self,
        submissions: Any,
        cache: Any,
        spend: Any,
        progress: Any,
        best_distance: Callable[[str, int], Awaitable[int | None]],
        kind: str,
    ) -> None:
        self.submissions = submissions
        self.cache = cache
        self.spend = spend
        self.progress = progress
        #: `ProgressRepo` deliberately has no reader for the stored distance —
        #: nothing in the product needs one. The FIXTURE supplies it per
        #: implementation so the tests below can pin the lower-distance-wins
        #: upsert without ever asking which engine they are talking to.
        self.best_distance = best_distance
        self.kind = kind


#: CI exports this alongside the `postgres:17` service. Empty = skip the leg.
TEST_POSTGRES_URL = "TEST_POSTGRES_URL"
#: Set by CI so a skipped Postgres leg is a FAILURE there rather than a pass.
REQUIRE_POSTGRES = "REQUIRE_POSTGRES"


def postgres_url() -> str:
    """The URL for the Postgres leg, verbatim.

    NOT rewritten to a driver here: `make_engine` already normalizes
    `postgres://` and `postgresql://` to `postgresql+psycopg://` (psycopg3, not
    asyncpg — §9.6: asyncpg is not libpq and rejects Railway's `sslmode=`), and
    a second normalization in the tests would be a second source of truth for
    the one thing this leg is meant to exercise.
    """
    return os.environ.get(TEST_POSTGRES_URL, "").strip()


def _memory_best_distance(repo: MemoryProgressRepo) -> Callable[[str, int], Awaitable[int | None]]:
    async def read(session_id: str, level_n: int) -> int | None:
        return repo._best.get((session_id, level_n))

    return read


def _sql_best_distance(engine: Any) -> Callable[[str, int], Awaitable[int | None]]:
    import sqlalchemy as sa

    from launder_serve.repo.models import progress

    async def read(session_id: str, level_n: int) -> int | None:
        stmt = sa.select(progress.c.distance).where(
            progress.c.session_id == session_id, progress.c.level_n == level_n
        )
        async with engine.connect() as conn:
            value = (await conn.execute(stmt)).scalar_one_or_none()
        return None if value is None else int(value)

    return read


@pytest.fixture(params=["memory", "sqlite", "postgres"])
async def repo(request: pytest.FixtureRequest, tmp_path: Path) -> AsyncIterator[Repos]:
    if request.param == "memory":
        progress = MemoryProgressRepo()
        yield Repos(
            MemorySubmissionRepo(),
            MemoryJudgeCacheRepo(),
            MemorySpendRepo(),
            progress,
            _memory_best_distance(progress),
            "memory",
        )
        return

    from launder_serve.repo.sqlalchemy import (
        SqlJudgeCacheRepo,
        SqlProgressRepo,
        SqlSpendRepo,
        SqlSubmissionRepo,
        create_all,
        make_engine,
    )

    if request.param == "postgres":
        url = postgres_url()
        if not url:
            pytest.skip(
                f"${TEST_POSTGRES_URL} is unset, so the Postgres leg cannot run. CI sets it "
                "from the postgres:17 service; set it locally to exercise JSONB, the upsert "
                "shape and the timezone round-trip against the real engine."
            )
        engine = make_engine(url)
        try:
            await _reset(engine)
            await create_all(engine)
            yield Repos(
                SqlSubmissionRepo(engine),
                SqlJudgeCacheRepo(engine),
                SqlSpendRepo(engine),
                SqlProgressRepo(engine),
                _sql_best_distance(engine),
                "postgres",
            )
        finally:
            await engine.dispose()
        return

    db = tmp_path / "contract.sqlite3"
    engine = make_engine(f"sqlite+aiosqlite:///{db.as_posix()}")
    await create_all(engine)
    try:
        yield Repos(
            SqlSubmissionRepo(engine),
            SqlJudgeCacheRepo(engine),
            SqlSpendRepo(engine),
            SqlProgressRepo(engine),
            _sql_best_distance(engine),
            "sqlite",
        )
    finally:
        await engine.dispose()


async def _reset(engine: Any) -> None:
    """Drop and recreate the schema, so each Postgres test starts empty.

    The sqlite leg gets a fresh tmp file per test for free; a shared server does
    not, and a suite whose assertions depend on leftover rows is a suite that
    passes in the wrong order.
    """
    from launder_serve.repo.models import METADATA

    async with engine.begin() as conn:
        await conn.run_sync(METADATA.drop_all)


def _record(**overrides: Any) -> SubmissionRecord:
    base: dict[str, Any] = {
        "level_n": LEVEL,
        "passage_id": "p07",
        "level_id": "L2",
        "text_hash": "a" * 64,
        "text": "a laundered passage",
        "cleared": True,
        "distance": 4,
        "detector_score": 0.5108,
        "detector_z": 1.74,
        "n_scored": 171,
        "masked_fraction": 0.12,
        "scoring_version": "sc1",
        "wm_config_id": "wm1:" + "0" * 64,
        "created_at": NOW,
        "ops": (EditOp(op="sub", i=2, j=2, **{"from": "holds"}, to="keeps"),),
        "elapsed_ms": 184320,
    }
    base.update(overrides)
    return SubmissionRecord(**base)


# ---------------------------------------------------------------------------
# submissions
# ---------------------------------------------------------------------------


async def test_record_then_read_back_the_diff(repo: Repos) -> None:
    await repo.submissions.record(_record())
    rows = await repo.submissions.best_for_level(LEVEL, 10)
    assert len(rows) == 1
    assert rows[0].distance == 4
    assert rows[0].elapsed_ms == 184320
    # The diff IS the leaderboard row: the brag, the anti-cheat mechanism and
    # the teaching tool are the same object (§8.3).
    assert rows[0].ops[0].op == "sub"
    assert rows[0].ops[0].from_ == "holds"
    assert rows[0].ops[0].to == "keeps"
    # Timestamps come back UTC-aware from both engines, SQLite included.
    assert rows[0].at.tzinfo is not None
    assert rows[0].at.astimezone(UTC) == NOW


async def test_the_board_is_ordered_by_distance_then_time(repo: Repos) -> None:
    await repo.submissions.record(_record(text_hash="c" * 64, distance=7))
    await repo.submissions.record(
        _record(text_hash="b" * 64, distance=3, created_at=NOW + timedelta(minutes=5))
    )
    await repo.submissions.record(_record(text_hash="d" * 64, distance=3))
    rows = await repo.submissions.best_for_level(LEVEL, 10)
    assert [r.distance for r in rows] == [3, 3, 7]
    # Equal distance: the earlier submission ranks first.
    assert rows[0].at < rows[1].at


async def test_only_cleared_non_provisional_rows_reach_the_board(repo: Repos) -> None:
    await repo.submissions.record(_record(text_hash="e" * 64, cleared=False, distance=1))
    await repo.submissions.record(
        _record(text_hash="f" * 64, cleared=True, provisional=True, distance=2)
    )
    await repo.submissions.record(_record(text_hash="g" * 64, distance=9))
    rows = await repo.submissions.best_for_level(LEVEL, 10)
    # A provisional clear is real for the player — the chime plays — but the
    # judge never answered, so it does not advance the campaign and it is
    # excluded from the per-level best-distance ranking this method computes
    # (§7.5).
    assert [r.distance for r in rows] == [9]


async def test_limit_is_respected(repo: Repos) -> None:
    for i in range(5):
        await repo.submissions.record(_record(text_hash=str(i) * 64, distance=i + 1))
    assert len(await repo.submissions.best_for_level(LEVEL, 3)) == 3


async def test_rank_of_counts_strictly_better_clears(repo: Repos) -> None:
    for i, d in enumerate((2, 3, 3, 8)):
        await repo.submissions.record(_record(text_hash=str(i) * 64, distance=d))
    assert await repo.submissions.rank_of(LEVEL, 2) == 1
    assert await repo.submissions.rank_of(LEVEL, 3) == 2
    assert await repo.submissions.rank_of(LEVEL, 9) == 5


async def test_record_is_an_upsert_on_the_dedup_index(repo: Repos) -> None:
    """A player nudging broken text and resending it must not mint a second row."""
    await repo.submissions.record(_record(distance=9, cleared=False))
    await repo.submissions.record(_record(distance=4, cleared=True))
    assert await repo.submissions.count_for_level(LEVEL) == 1
    rows = await repo.submissions.best_for_level(LEVEL, 10)
    assert [r.distance for r in rows] == [4]


async def test_levels_do_not_bleed_into_each_other(repo: Repos) -> None:
    """The board is per LEVEL. A clear of level 8 is not a clear of level 7."""
    await repo.submissions.record(_record())
    await repo.submissions.record(_record(level_n=LEVEL + 1, distance=1))
    assert [r.distance for r in await repo.submissions.best_for_level(LEVEL, 10)] == [4]
    assert await repo.submissions.count_for_level(LEVEL) == 1
    assert await repo.submissions.count_for_level(LEVEL + 1) == 1


async def test_the_dedup_index_is_the_level_and_the_text_only(repo: Repos) -> None:
    """Identical text on the SAME level is one row whatever ruleset it names.

    The ruleset is a property of how the level is played, not a second identity
    for the submission: keying dedup on it would let a client re-post the same
    text under a different `level_id` and mint a second board entry for it.
    """
    await repo.submissions.record(_record(distance=9))
    await repo.submissions.record(_record(level_id="L1", distance=4))
    assert await repo.submissions.count_for_level(LEVEL) == 1
    assert [r.distance for r in await repo.submissions.best_for_level(LEVEL, 10)] == [4]


# ---------------------------------------------------------------------------
# judge cache
# ---------------------------------------------------------------------------


def _verdict(**overrides: Any) -> CachedVerdict:
    base: dict[str, Any] = {
        "judge_version": "g3",
        "passage_id": "p07",
        "level_id": "L2",
        "cleared": True,
        "observation": {
            "natural_prose": True,
            "unnatural_kind": None,
            "claims": [{"id": "c1", "present": True, "how": "asserted"}],
            "added_claims": [],
            "contains_embedded_instructions": False,
            "notes": "",
            "verdict_opinion": "pass",
        },
        "provider": "openai",
        "model": "test-model",
        "created_at": NOW,
        "input_tokens": 1380,
        "output_tokens": 90,
    }
    base.update(overrides)
    return CachedVerdict(**base)


async def test_judge_cache_round_trips_the_observation(repo: Repos) -> None:
    await repo.cache.put("k1", _verdict())
    got = await repo.cache.get("k1")
    assert got is not None
    # The OBSERVATION is what is stored, so a level whose params change
    # re-derives instead of paying for the same call again.
    assert got.observation["claims"][0]["id"] == "c1"
    assert got.cleared is True
    assert got.created_at.astimezone(UTC) == NOW


async def test_judge_cache_miss_is_none_not_an_error(repo: Repos) -> None:
    assert await repo.cache.get("nope") is None


async def test_purge_version_orphans_only_its_own_version(repo: Repos) -> None:
    await repo.cache.put("k1", _verdict(judge_version="g3"))
    await repo.cache.put("k2", _verdict(judge_version="g4"))
    assert await repo.cache.purge_version("g3") == 1
    assert await repo.cache.get("k1") is None
    assert await repo.cache.get("k2") is not None


async def test_cache_stats_since(repo: Repos) -> None:
    await repo.cache.put("k1", _verdict())
    await repo.cache.put("k2", _verdict(cleared=False, failure_code="meaning_drift"))
    stats = await repo.cache.stats(NOW - timedelta(days=1))
    assert stats.entries == 2
    assert stats.cleared == 1
    assert stats.rejected == 1
    assert stats.input_tokens == 2760
    assert await repo.cache.stats(NOW + timedelta(days=1)) == type(stats)()


# ---------------------------------------------------------------------------
# spend ledger
# ---------------------------------------------------------------------------


async def test_reserve_accumulates_and_refuses_at_the_cap(repo: Repos) -> None:
    assert await repo.spend.reserve(DAY, 0.4, 1.0) is True
    assert await repo.spend.reserve(DAY, 0.4, 1.0) is True
    # 0.4 + 0.4 + 0.4 > 1.00: refused, and NOT partially spent.
    assert await repo.spend.reserve(DAY, 0.4, 1.0) is False
    assert await repo.spend.reserve(DAY, 0.2, 1.0) is True


async def test_reserve_refuses_a_single_call_larger_than_the_cap(repo: Repos) -> None:
    """The ON CONFLICT guard only fires on the second reservation of a day."""
    assert await repo.spend.reserve(DAY, 5.0, 2.0) is False


async def test_spend_is_per_day(repo: Repos) -> None:
    assert await repo.spend.reserve(DAY, 0.9, 1.0) is True
    assert await repo.spend.reserve(DAY + timedelta(days=1), 0.9, 1.0) is True


# ---------------------------------------------------------------------------
# campaign progress
# ---------------------------------------------------------------------------


async def test_cleared_levels_come_back_ascending(repo: Repos) -> None:
    for level_n in (3, 1, 2):
        await repo.progress.record("s_one", level_n, 4)
    # Ascending whatever order they were cleared in: the client unions this into
    # localStorage and derives `unlocked` from the last element.
    assert await repo.progress.cleared_levels("s_one") == [1, 2, 3]


async def test_an_unknown_session_has_cleared_nothing(repo: Repos) -> None:
    assert await repo.progress.cleared_levels("s_never_seen") == []


async def test_sessions_do_not_bleed(repo: Repos) -> None:
    await repo.progress.record("s_one", 1, 4)
    await repo.progress.record("s_two", 5, 4)
    assert await repo.progress.cleared_levels("s_one") == [1]
    assert await repo.progress.cleared_levels("s_two") == [5]


async def test_recording_a_level_twice_keeps_the_lower_distance(repo: Repos) -> None:
    """A player coming back to improve a level must not be able to make their
    own record worse by playing it badly."""
    await repo.progress.record("s_one", 2, 6)
    await repo.progress.record("s_one", 2, 4)
    assert await repo.progress.cleared_levels("s_one") == [2]
    assert await repo.best_distance("s_one", 2) == 4

    await repo.progress.record("s_one", 2, 9)
    assert await repo.best_distance("s_one", 2) == 4
    assert await repo.progress.cleared_levels("s_one") == [2]


# ---------------------------------------------------------------------------
# the gate on the gate
# ---------------------------------------------------------------------------


def test_ci_runs_the_postgres_leg() -> None:
    """A gate that can pass by not running is not a gate.

    CI stands up a `postgres:17` service and exports `TEST_POSTGRES_URL`; §11.6
    claims this suite runs against it. For a year it did not — the fixture was
    parametrized over `["memory", "sqlite"]` and nothing in the repository read
    the variable. This asserts that where CI says the leg runs, it runs.
    """
    if os.environ.get(REQUIRE_POSTGRES, "").strip().lower() not in {"1", "true", "yes"}:
        pytest.skip(f"${REQUIRE_POSTGRES} is not set; this assertion is for CI")
    assert postgres_url(), (
        f"${REQUIRE_POSTGRES} is set but ${TEST_POSTGRES_URL} is empty, so the Postgres "
        "leg of the repository contract silently skipped and pool_pre_ping, JSONB, the "
        "upsert shape and the timezone round-trip went unexercised."
    )
