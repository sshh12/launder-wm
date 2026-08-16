"""Scoring — normalization, word tokenization and the word-level distance.

One implementation, ported once, versioned (TECH_PLAN.md §8). Client for
preview, server for authority; `edit_budget` and `close_paraphrase` call the
same `score()`, so the budget and the scoreboard can never disagree.
"""

from __future__ import annotations

from launder_core.scoring.damerau import (
    TIE_BREAK,
    apply_ops,
    damerau_levenshtein,
    damerau_levenshtein_ops,
    score,
)
from launder_core.scoring.normalize import (
    DEFAULT_NORMALIZE,
    FOLD_MAP,
    WHITESPACE_CHARS,
    WHITESPACE_CODEPOINTS,
    WHITESPACE_PATTERN,
    NormalizeConfig,
    normalize,
    normalize_config_from,
    words,
)

__all__ = [
    "DEFAULT_NORMALIZE",
    "FOLD_MAP",
    "TIE_BREAK",
    "WHITESPACE_CHARS",
    "WHITESPACE_CODEPOINTS",
    "WHITESPACE_PATTERN",
    "NormalizeConfig",
    "apply_ops",
    "damerau_levenshtein",
    "damerau_levenshtein_ops",
    "normalize",
    "normalize_config_from",
    "score",
    "words",
]
