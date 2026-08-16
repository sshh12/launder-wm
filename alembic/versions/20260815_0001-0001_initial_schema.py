"""initial schema: daily_slot, submission, judge_cache, spend_ledger

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
the server clock, it differs across engines, and the product's daily rollover is
UTC midnight regardless of what the database believes the time is.

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
        "daily_slot",
        sa.Column("day", sa.Date(), primary_key=True, nullable=False),
        sa.Column("passage_id", sa.Text(), nullable=False),
        sa.Column("level_id", sa.Text(), nullable=False),
        sa.Column("authored_par", sa.Integer(), nullable=True),
        # Best clear in the first N plays; self-balancing (§9.1).
        sa.Column("observed_par", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "submission",
        sa.Column("id", _BIGPK, primary_key=True, autoincrement=True, nullable=False),
        sa.Column("day", sa.Date(), nullable=False),
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
        # localStorage uuid. NOT identity. NOT trusted.
        sa.Column("session_id", sa.String(length=32), nullable=True),
        sa.Column("elapsed_ms", sa.BigInteger(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "submission_board_idx", "submission", ["day", "level_id", "cleared", "distance"]
    )
    op.create_index(
        "submission_dedup_idx",
        "submission",
        ["day", "level_id", "text_hash"],
        unique=True,
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
        sa.Column("day", sa.Date(), primary_key=True, nullable=False),
        sa.Column("spent_usd", sa.Numeric(precision=10, scale=6), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("spend_ledger")
    op.drop_index("judge_cache_version_idx", table_name="judge_cache")
    op.drop_table("judge_cache")
    op.drop_index("submission_dedup_idx", table_name="submission")
    op.drop_index("submission_board_idx", table_name="submission")
    op.drop_table("submission")
    op.drop_table("daily_slot")
