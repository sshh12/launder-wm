"""Access to `launder_core` symbols that land across M1-M4.

`launder_core.__init__` resolves its public surface lazily from a name ->
module table, so `from launder_core import damerau_levenshtein` raises an
`AttributeError` naming the file that must be written until that file exists.
That is the right behaviour, and this module does not paper over it — it only
decides, per symbol, whether forge can proceed without it:

* **Required.** `damerau_levenshtein` / `score` are THE authority on edit
  distance (§8.2). Forge never reimplements them; a missing one is a hard,
  named failure.
* **Degradable.** `words()` is needed by the authoring metrics long before the
  server needs it. Forge falls back to a whitespace split, warns once, and
  every artifact it writes records `word_split: "whitespace-fallback"` so a
  number computed under the fallback is never mistaken for a real one.

Nothing here defines watermark or scoring *semantics*. If you find yourself
wanting to add one, it belongs in `launder_core`.
"""

from __future__ import annotations

import warnings
from typing import Any, Final

__all__ = [
    "WORD_SPLIT_FALLBACK",
    "core_symbol",
    "damerau_distance",
    "have_core_scoring",
    "have_core_watermark",
    "word_split_provenance",
    "words",
]

WORD_SPLIT_FALLBACK: Final[str] = "whitespace-fallback"
_WARNED = False


def core_symbol(name: str) -> Any | None:
    """Return `launder_core.<name>` or None if that module has not landed."""
    import launder_core

    try:
        return getattr(launder_core, name)
    except AttributeError:
        return None


def have_core_watermark() -> bool:
    return core_symbol("compute_g_values") is not None


def have_core_scoring() -> bool:
    return core_symbol("damerau_levenshtein") is not None


def words(text: str) -> list[str]:
    """Word sequence for the authoring metrics.

    Prefers `launder_core.words` (NFC + whitespace + quote folding, §8.1). The
    fallback is a bare whitespace split, which is close enough to count hot
    runs and word budgets during authoring and NOT close enough to publish a
    `par` against — `word_split_provenance()` is recorded next to every number
    derived from it.
    """
    global _WARNED
    fn = core_symbol("words")
    if fn is not None:
        return list(fn(text))
    if not _WARNED:
        _WARNED = True
        warnings.warn(
            "launder_core.scoring.normalize has not landed yet; forge is splitting words on "
            "whitespace. Metrics computed this way are recorded as "
            f"word_split={WORD_SPLIT_FALLBACK!r} and must be recomputed before publishing.",
            RuntimeWarning,
            stacklevel=2,
        )
    return text.split()


def word_split_provenance() -> str:
    return "launder_core.words" if core_symbol("words") is not None else WORD_SPLIT_FALLBACK


def damerau_distance(a: list[str], b: list[str]) -> int:
    """Unrestricted Damerau-Levenshtein over words. Core is the only authority."""
    fn = core_symbol("damerau_levenshtein")
    if fn is None:
        raise RuntimeError(
            "launder_core.scoring.damerau.damerau_levenshtein does not exist yet, and forge "
            "will not ship a second implementation of it: `edit_budget`, `close_paraphrase` "
            "and the scoreboard all call the same function precisely so the budget and the "
            "score can never disagree (TECH_PLAN.md §8.2). Write "
            "packages/core/src/launder_core/scoring/damerau.py."
        )
    return int(fn(a, b))
