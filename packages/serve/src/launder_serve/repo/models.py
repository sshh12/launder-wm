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

`func.now()` is deliberately absent: it is the *server* clock and it differs
across engines, so every timestamp here is written from Python as
`datetime.now(UTC)`.
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

__all__ = [
    "METADATA",
    "judge_cache",
    "progress",
    "spend_ledger",
    "submission",
]

METADATA = sa.MetaData()

#: `BIGSERIAL` on Postgres; SQLite needs a plain `INTEGER PRIMARY KEY` for
#: rowid aliasing, so the variant is not cosmetic.
_BIGPK = sa.BigInteger().with_variant(sa.Integer(), "sqlite")
_JSON = sa.JSON().with_variant(JSONB, "postgresql")


progress = sa.Table(
    "progress",
    METADATA,
    # localStorage uuid. NOT identity. NOT trusted. `crypto.randomUUID()` is 36
    # characters, so the column is sized for the hyphenated form and not for the
    # 32-char hex one.
    sa.Column("session_id", sa.String(64), primary_key=True),
    sa.Column("level_n", sa.Integer(), primary_key=True),
    # The BEST clear, not the latest: the upsert keeps the lower value.
    sa.Column("distance", sa.Integer(), nullable=False),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
)


submission = sa.Table(
    "submission",
    METADATA,
    sa.Column("id", _BIGPK, primary_key=True, autoincrement=True),
    # The campaign position, 1..level_count. There is no date on a submission:
    # a level is a place in an ordered campaign, not a day.
    sa.Column("level_n", sa.Integer(), nullable=False),
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
    # localStorage uuid. NOT identity. NOT trusted. 64, matching `progress`: the
    # client's `crypto.randomUUID()` is 36 characters and a 32-char column
    # rejected every submit that carried one, on Postgres only.
    sa.Column("session_id", sa.String(64), nullable=True),
    sa.Column("elapsed_ms", sa.BigInteger(), nullable=True),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    sa.Index("submission_board_idx", "level_n", "cleared", "distance"),
    sa.Index("submission_dedup_idx", "level_n", "text_hash", unique=True),
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
    # A BILLING day — the calendar day the judge's USD cap resets on. It is the
    # only `day` left in the schema and it has nothing to do with the campaign.
    sa.Column("day", sa.Date(), primary_key=True),
    sa.Column("spent_usd", sa.Numeric(10, 6), nullable=False),
)
