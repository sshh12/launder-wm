"""Repositories: the narrow interfaces (§9.6) and their two implementations.

Import the Protocols and the row types from here. `repo.sqlalchemy` is imported
lazily by the app factory so that a test suite using `repo.memory` never pays
for the SQLAlchemy import — and so that an accidental SQLAlchemy dependency in
the memory path shows up as an ImportError rather than as nothing at all.
"""

from __future__ import annotations

from launder_serve.repo.memory import (
    MemoryDailyRepo,
    MemoryJudgeCacheRepo,
    MemorySpendRepo,
    MemorySubmissionRepo,
)
from launder_serve.repo.protocol import (
    CachedVerdict,
    CacheStats,
    DailyRepo,
    DailySlot,
    JudgeCacheRepo,
    LeaderRow,
    SpendRepo,
    SubmissionRecord,
    SubmissionRepo,
)

__all__ = [
    "CacheStats",
    "CachedVerdict",
    "DailyRepo",
    "DailySlot",
    "JudgeCacheRepo",
    "LeaderRow",
    "MemoryDailyRepo",
    "MemoryJudgeCacheRepo",
    "MemorySpendRepo",
    "MemorySubmissionRepo",
    "SpendRepo",
    "SubmissionRecord",
    "SubmissionRepo",
]
