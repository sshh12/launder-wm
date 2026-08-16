"""Alembic environment (TECH_PLAN.md §9.8).

Runs as Railway's `preDeployCommand`, between build and deploy, from the built
image. A non-zero exit BLOCKS THE DEPLOYMENT and is not retried — which is the
behaviour we want, and the reason nothing in here is allowed to swallow an
error.

Two things this file does that are not boilerplate:

* **`render_as_batch` when the dialect is sqlite.** SQLite has no
  `ALTER COLUMN`; batch mode rebuilds the table instead. Without it, the first
  migration that alters a column passes in CI (Postgres) and fails on a
  developer's local file DB.
* **It reads `DATABASE_URL` from the environment and converts async drivers to
  their sync spelling.** Alembic runs synchronously; the app runs
  `postgresql+psycopg://` and `sqlite+aiosqlite://`. psycopg3 is both, so the
  Postgres URL needs no surgery at all — which is precisely why §9.6 chose it
  over asyncpg.
"""

from __future__ import annotations

import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# `prepend_sys_path` in alembic.ini covers the normal invocation; this covers
# `alembic.command.upgrade()` called programmatically from a test, where the
# ini's paths are resolved against a working directory we do not control.
_ROOT = Path(__file__).resolve().parents[1]
for _pkg in ("packages/core/src", "packages/serve/src"):
    _path = str(_ROOT / _pkg)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from launder_serve.repo.models import METADATA  # noqa: E402

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = METADATA


def _sync_url() -> str:
    """`DATABASE_URL`, in the spelling a synchronous engine understands."""
    url = os.environ.get("DATABASE_URL", "").strip()
    if not url:
        url = config.get_main_option("sqlalchemy.url") or ""
    if not url:
        raise RuntimeError(
            "DATABASE_URL is unset and alembic.ini's sqlalchemy.url is empty (on "
            "purpose: a URL pasted there would be a credential in git). Set "
            "DATABASE_URL — Railway supplies ${{Postgres.DATABASE_URL}} verbatim."
        )
    if url.startswith("postgres://"):
        url = "postgresql://" + url[11:]
    if url.startswith("postgresql://"):
        # One driver for the async app AND sync Alembic; this is the payoff for
        # choosing psycopg3 over asyncpg (§9.6).
        url = "postgresql+psycopg://" + url[13:]
    return url.replace("+aiosqlite", "").replace("+asyncpg", "+psycopg")


def run_migrations_offline() -> None:
    context.configure(
        url=_sync_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        compare_type=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = _sync_url()
    connectable = engine_from_config(section, prefix="sqlalchemy.", poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # SQLite has no ALTER COLUMN.
            render_as_batch=connection.dialect.name == "sqlite",
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
