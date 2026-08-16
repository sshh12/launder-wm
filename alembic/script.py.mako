"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}

EXPAND/CONTRACT ONLY (TECH_PLAN.md §9.8). `overlapSeconds` means old and new
code serve simultaneously: add nullable columns and new tables in one deploy,
backfill, then drop or set NOT NULL in a later one. Never DROP COLUMN in the
release that stops writing it — and never run `alembic downgrade` in
production, because a Railway rollback restores the old IMAGE, not the old
schema. Forward-fix.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

${imports if imports else ""}

revision: str = ${repr(up_revision)}
down_revision: str | None = ${repr(down_revision)}
branch_labels: str | Sequence[str] | None = ${repr(branch_labels)}
depends_on: str | Sequence[str] | None = ${repr(depends_on)}


def upgrade() -> None:
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    ${downgrades if downgrades else "pass"}
