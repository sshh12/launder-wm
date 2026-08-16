"""SQLAlchemy **Core** table definitions (TECH_PLAN.md §9.7).

Core, not ORM, and it lives behind `repo/protocol.py`. Every convention below
is one of the §9.6 SQLite-vs-Postgres hazards, and the dual-engine contract
suite is what keeps them honest:

| hazard          | convention here                                        |
|-----------------|--------------------------------------------------------|
| JSON columns    | `JSON().with_variant(JSONB, "postgresql")`; never queried into |
| Upsert          | `on_conflict_do_*` — same shape in both dialects, one branch   |
| Timestamps      | `DateTime(timezone=True)`, UTC, `datetime.now(UTC)` from Python |
| Booleans        | `.is_(True)` at the call sites, never bare truthiness  |
| NULL ordering   | `.nulls_last()` explicitly                             |
| Integer width   | `BigInteger` for anything that counts events           |
| `LIKE`          | never used — we hash submissions, we don't search them |

`func.now()` is deliberately absent: it is the *server* clock, it differs
across engines, and the daily rollover is a UTC-midnight product rule rather
than whatever the database thinks the time is.
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

__all__ = [
    "METADATA",
    "daily_slot",
    "judge_cache",
    "spend_ledger",
    "submission",
]

METADATA = sa.MetaData()

#: `BIGSERIAL` on Postgres; SQLite needs a plain `INTEGER PRIMARY KEY` for
#: rowid aliasing, so the variant is not cosmetic.
_BIGPK = sa.BigInteger().with_variant(sa.Integer(), "sqlite")
_JSON = sa.JSON().with_variant(JSONB, "postgresql")


daily_slot = sa.Table(
    "daily_slot",
    METADATA,
    sa.Column("day", sa.Date(), primary_key=True),
    sa.Column("passage_id", sa.Text(), nullable=False),
    sa.Column("level_id", sa.Text(), nullable=False),
    sa.Column("authored_par", sa.Integer(), nullable=True),
    # Best clear in the first N plays; self-balancing (§9.1).
    sa.Column("observed_par", sa.Integer(), nullable=True),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
)


submission = sa.Table(
    "submission",
    METADATA,
    sa.Column("id", _BIGPK, primary_key=True, autoincrement=True),
    sa.Column("day", sa.Date(), nullable=False),
    sa.Column("passage_id", sa.Text(), nullable=False),
    sa.Column("level_id", sa.Text(), nullable=False),
    # sha256(normalized)
    sa.Column("text_hash", sa.String(64), nullable=False),
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
    sa.Column("session_id", sa.String(32), nullable=True),
    sa.Column("elapsed_ms", sa.BigInteger(), nullable=True),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    sa.Index("submission_board_idx", "day", "level_id", "cleared", "distance"),
    sa.Index("submission_dedup_idx", "day", "level_id", "text_hash", unique=True),
)


judge_cache = sa.Table(
    "judge_cache",
    METADATA,
    # sha256(judge_version ⋮ prompt_hash ⋮ scoring_version ⋮ passage_id ⋮
    #        level_id ⋮ sha256(normalized))
    sa.Column("key", sa.String(64), primary_key=True),
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
    sa.Index("judge_cache_version_idx", "judge_version", "created_at"),
)


spend_ledger = sa.Table(
    "spend_ledger",
    METADATA,
    sa.Column("day", sa.Date(), primary_key=True),
    sa.Column("spent_usd", sa.Numeric(10, 6), nullable=False),
)
