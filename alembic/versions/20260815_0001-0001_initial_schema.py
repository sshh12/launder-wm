"""initial schema: submission, progress, judge_cache, spend_ledger

Revision ID: 0001_initial
Revises:
Create Date: 2026-08-15

TECH_PLAN.md §9.7, reproduced. Two variants are load-bearing rather than
cosmetic and both come from §9.6's hazard table:

* `JSON().with_variant(JSONB, "postgresql")` — and **never query inside JSON in
  SQL**. If you need to filter on something in `ops` or `observation`, promote
  it to a real column in a later migration.
* `BigInteger().with_variant(Integer, "sqlite")` for the surrogate key: SQLite
  aliases the rowid only for a declared `INTEGER PRIMARY KEY`, so a literal
  `BIGINT` primary key silently loses autoincrement there.

`created_at` is `TIMESTAMPTZ` everywhere and is written from Python with
`datetime.now(UTC)`. `server_default=func.now()` is deliberately absent: it is
the server clock and it differs across engines.

The only `day` left in this schema is `spend_ledger.day`, and it is a BILLING
day — the calendar day the judge's USD cap resets on. The game itself is a
linear campaign: a submission is keyed on `level_n`, its position in that
campaign, and nothing here rolls over at midnight.

EXPAND/CONTRACT ONLY from here on. `downgrade()` drops everything because this
is the first revision and there is nothing to preserve; later revisions must
not assume the same freedom, and production never runs downgrade at all.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_BIGPK = sa.BigInteger().with_variant(sa.Integer(), "sqlite")
_JSON = sa.JSON().with_variant(JSONB, "postgresql")


def upgrade() -> None:
    op.create_table(
        "submission",
        sa.Column("id", _BIGPK, primary_key=True, autoincrement=True, nullable=False),
        # The campaign position, 1..level_count. `level_id` beside it is the
        # RULESET (`L1..L6`); the two are independent.
        sa.Column("level_n", sa.Integer(), nullable=False),
        sa.Column("passage_id", sa.Text(), nullable=False),
        sa.Column("level_id", sa.Text(), nullable=False),
        # sha256(normalized)
        sa.Column("text_hash", sa.String(length=64), nullable=False),
        # UNTRUSTED. Never feed to an LLM that reads logs.
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("cleared", sa.Boolean(), nullable=False),
        sa.Column("provisional", sa.Boolean(), nullable=False, server_default=sa.false()),
        # Server-computed. The only authority.
        sa.Column("distance", sa.Integer(), nullable=False),
        # The shareable diff.
        sa.Column("ops", _JSON, nullable=False),
        sa.Column("detector_score", sa.Float(), nullable=False),
        sa.Column("detector_z", sa.Float(), nullable=False),
        sa.Column("n_scored", sa.Integer(), nullable=False),
        # Instruments the repetition exploit (§4.4c).
        sa.Column("masked_fraction", sa.Float(), nullable=False),
        sa.Column("failure_code", sa.Text(), nullable=True),
        sa.Column("scoring_version", sa.Text(), nullable=False),
        sa.Column("wm_config_id", sa.Text(), nullable=False),
        # localStorage uuid. NOT identity. NOT trusted. 64 because the client's
        # `crypto.randomUUID()` is 36 characters.
        sa.Column("session_id", sa.String(length=64), nullable=True),
        sa.Column("elapsed_ms", sa.BigInteger(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("submission_board_idx", "submission", ["level_n", "cleared", "distance"])
    # UNIQUE: a player nudging a broken submission and resending identical text
    # updates the row rather than minting a second leaderboard entry.
    op.create_index(
        "submission_dedup_idx",
        "submission",
        ["level_n", "text_hash"],
        unique=True,
    )

    op.create_table(
        "progress",
        # The composite primary key IS the upsert target: one row per
        # (session, level), holding that session's BEST clear of that level.
        sa.Column("session_id", sa.String(length=64), primary_key=True, nullable=False),
        sa.Column("level_n", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("distance", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "judge_cache",
        # sha256(judge_version ⋮ prompt_hash ⋮ scoring_version ⋮ passage_id ⋮
        #        level_id ⋮ sha256(normalized))
        sa.Column("key", sa.String(length=64), primary_key=True, nullable=False),
        sa.Column("judge_version", sa.Text(), nullable=False),
        sa.Column("passage_id", sa.Text(), nullable=False),
        sa.Column("level_id", sa.Text(), nullable=False),
        sa.Column("cleared", sa.Boolean(), nullable=False),
        sa.Column("failure_code", sa.Text(), nullable=True),
        sa.Column("feedback", sa.Text(), nullable=True),
        sa.Column("observation", _JSON, nullable=False),
        sa.Column("provider", sa.Text(), nullable=False),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=False),
        sa.Column("output_tokens", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("judge_cache_version_idx", "judge_cache", ["judge_version", "created_at"])

    op.create_table(
        "spend_ledger",
        # A BILLING day, not a puzzle day: the calendar day the judge's USD cap
        # resets on.
        sa.Column("day", sa.Date(), primary_key=True, nullable=False),
        sa.Column("spent_usd", sa.Numeric(precision=10, scale=6), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("spend_ledger")
    op.drop_index("judge_cache_version_idx", table_name="judge_cache")
    op.drop_table("judge_cache")
    op.drop_table("progress")
    op.drop_index("submission_dedup_idx", table_name="submission")
    op.drop_index("submission_board_idx", table_name="submission")
    op.drop_table("submission")
