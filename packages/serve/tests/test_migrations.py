"""`alembic upgrade head` on SQLite (TECH_PLAN.md §9.8).

Migrations run as Railway's `preDeployCommand`, and a non-zero exit BLOCKS THE
DEPLOYMENT. That is the behaviour we want, which makes "does the migration
actually apply" a thing CI has to answer before the deploy does.

SQLite is the cheap leg. It is also the one that catches `render_as_batch`
regressions and the `BigInteger` primary-key variant, both of which pass
silently on Postgres.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config

from launder_serve.repo.models import METADATA
from launder_serve.settings import repo_root


@pytest.fixture
def alembic_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Config, Path]:
    root = repo_root()
    ini = root / "alembic.ini"
    if not ini.is_file():
        pytest.skip("alembic.ini is not present in this checkout")
    db = tmp_path / "migrated.sqlite3"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db.as_posix()}")
    cfg = Config(str(ini))
    # `script_location` is relative to the working directory in the ini; pytest
    # does not run from the repo root.
    cfg.set_main_option("script_location", str(root / "alembic"))
    return cfg, db


def test_upgrade_head_creates_the_whole_schema(alembic_config: tuple[Config, Path]) -> None:
    cfg, db = alembic_config
    command.upgrade(cfg, "head")

    with sqlite3.connect(db) as conn:
        tables = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        indexes = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
        }

    assert {"daily_slot", "submission", "judge_cache", "spend_ledger"} <= tables
    assert "alembic_version" in tables
    assert {"submission_board_idx", "submission_dedup_idx", "judge_cache_version_idx"} <= indexes


def test_the_migration_matches_the_models(alembic_config: tuple[Config, Path]) -> None:
    """`repo/models.py` and the migration must not drift.

    Alembic's autogenerate would catch this too, but only for somebody who
    remembers to run it. Comparing the applied schema against the metadata is
    the version that runs on every commit.
    """
    cfg, db = alembic_config
    command.upgrade(cfg, "head")

    with sqlite3.connect(db) as conn:
        for table in METADATA.sorted_tables:
            applied = {row[1] for row in conn.execute(f"PRAGMA table_info('{table.name}')")}
            declared = {column.name for column in table.columns}
            assert applied == declared, f"{table.name}: migration and models disagree"


def test_downgrade_is_reversible_on_the_first_revision(
    alembic_config: tuple[Config, Path],
) -> None:
    """Tested here, never run in production: a Railway rollback restores the
    old IMAGE, not the old schema, so the production answer is forward-fix."""
    cfg, db = alembic_config
    command.upgrade(cfg, "head")
    command.downgrade(cfg, "base")

    with sqlite3.connect(db) as conn:
        tables = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert "submission" not in tables
