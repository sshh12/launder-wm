r"""Normalization and word tokenization — TECH_PLAN.md §8.1.

This module is the ORIGIN of every word index in the product: the scoreboard's
distance, `edit_budget`, `close_paraphrase`'s five metrics, the diff exhibit and
the text handed to the judge all start here. One implementation, ported once,
versioned (`scoring_version` in `data/config/scoring.toml`, hashed into the
judge cache key — changing normalization changes scores, so it must invalidate
caches).

Three properties are load-bearing and must survive any edit:

1. **Idempotence.** ``normalize(normalize(x)) == normalize(x)``, asserted by a
   Hypothesis property test. The cache key hashes the normalized submission, so
   a non-idempotent normalizer would produce two keys for one submission.

2. **NFC, not NFKC.** NFKC folds the fi ligature to "fi" and circled-1 to "1",
   which changes tokenization in ways the player did not author — and the judge
   eval set contains NFKC-bait as a MUST-FAIL case. Folding it here would let
   the exploit through. `unicode_sanitation` REJECTS confusables; normalization
   does not silently repair them.

3. **Nothing invisible is stripped.** Zero-width characters, soft hyphens, bidi
   controls, variation selectors and private-use codepoints pass through
   untouched, because stripping them would let the exploit succeed at the
   detector while the judge sees clean text (§7.2). U+FEFF is deliberately NOT
   in the whitespace class below for exactly that reason — JavaScript's \s
   *does* match it, which is the single most likely place for the TS port to
   silently diverge.

Order of operations, and why it is idempotent
---------------------------------------------
fold -> collapse whitespace -> NFC.

NFC runs last because whitespace collapse can move a combining mark next to a
different base character, and we want the composed form of whatever survives
collapse. Running it last is safe because canonical composition can never
produce a fold source (U+2013, U+2014, U+2018, U+2019, U+201C, U+201D, U+2026
and U+00A0 are not canonical composition targets — U+00A0's decomposition is
``<noBreak>``, a *compatibility* mapping NFC does not apply) and can never
produce whitespace. So a second pass changes nothing.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import Final

__all__ = [
    "DEFAULT_NORMALIZE",
    "FOLD_MAP",
    "WHITESPACE_CHARS",
    "WHITESPACE_CODEPOINTS",
    "WHITESPACE_PATTERN",
    "NormalizeConfig",
    "normalize",
    "normalize_config_from",
    "words",
]

# ---------------------------------------------------------------------------
# The whitespace class, written out rather than inherited from \s
# ---------------------------------------------------------------------------
# Python's \s and JavaScript's \s are NOT the same set: Python matches
# U+001C-U+001F and U+0085 but not U+FEFF; JavaScript matches U+FEFF but not
# U+001C-U+001F. Either default would be a client/server parity bug, so the set
# is spelled out here and in the TS port. It is exactly the Unicode White_Space
# property MINUS U+FEFF, which `unicode_sanitation` rejects as a zero-width
# character — collapsing it here would silently repair the exploit.
#
# U+001C-U+001F (FILE/GROUP/RECORD/UNIT SEPARATOR) are excluded for the same
# reason and the decision is paired: they are C0 controls, not Unicode
# whitespace, so `normalize()` leaves them alone and `unicode_sanitation`
# rejects them under the `control_char` category. Python's \s matching them is
# an `re` quirk, not a spec, and it is not expressible in a JS character class
# without hand-writing it — a shorthand that means two different things in the
# two runtimes is the exact shape of the bug this repo exists to prevent.
WHITESPACE_CODEPOINTS: Final[tuple[int, ...]] = (
    0x09,  # CHARACTER TABULATION
    0x0A,  # LINE FEED
    0x0B,  # LINE TABULATION
    0x0C,  # FORM FEED
    0x0D,  # CARRIAGE RETURN
    0x20,  # SPACE
    0x85,  # NEXT LINE
    0xA0,  # NO-BREAK SPACE
    0x1680,  # OGHAM SPACE MARK
    *range(0x2000, 0x200B),  # EN QUAD .. HAIR SPACE
    0x2028,  # LINE SEPARATOR
    0x2029,  # PARAGRAPH SEPARATOR
    0x202F,  # NARROW NO-BREAK SPACE
    0x205F,  # MEDIUM MATHEMATICAL SPACE
    0x3000,  # IDEOGRAPHIC SPACE
)

WHITESPACE_CHARS: Final[str] = "".join(chr(cp) for cp in WHITESPACE_CODEPOINTS)

#: The exact regex the TS port must use. Built from the codepoint list rather
#: than typed out, so the two can never drift apart.
WHITESPACE_PATTERN: Final[str] = (
    "[" + "".join(f"\\u{cp:04x}" for cp in WHITESPACE_CODEPOINTS) + "]+"
)

_WS_RE: Final[re.Pattern[str]] = re.compile(WHITESPACE_PATTERN)

# ---------------------------------------------------------------------------
# The fold table, written out for the same reason
# ---------------------------------------------------------------------------
#: Codepoint -> replacement, grouped by the switch that enables it. These are
#: keyboard and autocorrect artifacts, not authored edits: an iOS keyboard
#: turning ' into U+2019 must not cost the player a word of distance.
FOLD_MAP: Final[dict[str, dict[str, str]]] = {
    "fold_smart_quotes": {
        "\u201c": '"',  # LEFT DOUBLE QUOTATION MARK
        "\u201d": '"',  # RIGHT DOUBLE QUOTATION MARK
        "\u2018": "'",  # LEFT SINGLE QUOTATION MARK
        "\u2019": "'",  # RIGHT SINGLE QUOTATION MARK
    },
    "fold_dashes": {
        "\u2013": "-",  # EN DASH
        "\u2014": "-",  # EM DASH
    },
    "fold_ellipsis": {
        "\u2026": "...",  # HORIZONTAL ELLIPSIS
    },
    "nbsp_to_space": {
        "\u00a0": " ",  # NO-BREAK SPACE
    },
}


@dataclass(frozen=True)
class NormalizeConfig:
    """The normalization switches (TECH_PLAN.md §8.1).

    `ScoringConfig` in `launder_core.schemas` is the on-DISK contract for
    `data/config/scoring.toml`; this is the in-memory one the algorithm reads.
    Build it with `normalize_config_from()` rather than declaring the defaults
    twice.

    `lowercase` and `strip_punctuation` exist so that turning them on is a
    visible diff sitting next to the reason it is wrong — and, because "NEVER"
    in the plan means never, constructing a config with either set raises.
    """

    nfc: bool = True
    collapse_whitespace: bool = True
    fold_smart_quotes: bool = True
    fold_dashes: bool = True
    fold_ellipsis: bool = True
    nbsp_to_space: bool = True
    lowercase: bool = False
    strip_punctuation: bool = False

    def __post_init__(self) -> None:
        if self.lowercase:
            raise ValueError(
                "NormalizeConfig.lowercase must stay False: casing is evidence for the judge "
                "(TECH_PLAN.md §8.1). Folding case here would hide a whole class of "
                "not-natural-prose submissions from the only check that can see them."
            )
        if self.strip_punctuation:
            raise ValueError(
                "NormalizeConfig.strip_punctuation must stay False: punctuation is sentence "
                "structure, and 'study.' -> 'study,' must cost 1 because it is one thing a "
                "player did. Stripping it makes repunctuation free — a degenerate move the "
                "tokenizer DOES notice (TECH_PLAN.md §8.1)."
            )


DEFAULT_NORMALIZE: Final[NormalizeConfig] = NormalizeConfig()


def normalize_config_from(cfg: object) -> NormalizeConfig:
    """Build a `NormalizeConfig` from a `schemas.ScoringConfig` (or anything
    with the same attribute names). Typed loosely on purpose: importing the
    pydantic model here would make every scoring import pay for pydantic."""
    return NormalizeConfig(
        nfc=bool(getattr(cfg, "nfc", True)),
        collapse_whitespace=bool(getattr(cfg, "collapse_whitespace", True)),
        fold_smart_quotes=bool(getattr(cfg, "fold_smart_quotes", True)),
        fold_dashes=bool(getattr(cfg, "fold_dashes", True)),
        fold_ellipsis=bool(getattr(cfg, "fold_ellipsis", True)),
        nbsp_to_space=bool(getattr(cfg, "nbsp_to_space", True)),
    )


@lru_cache(maxsize=8)
def _fold_table(
    fold_smart_quotes: bool, fold_dashes: bool, fold_ellipsis: bool, nbsp_to_space: bool
) -> dict[int, str] | None:
    enabled = {
        "fold_smart_quotes": fold_smart_quotes,
        "fold_dashes": fold_dashes,
        "fold_ellipsis": fold_ellipsis,
        "nbsp_to_space": nbsp_to_space,
    }
    table: dict[int, str] = {}
    for switch, mapping in FOLD_MAP.items():
        if not enabled[switch]:
            continue
        for source, replacement in mapping.items():
            table[ord(source)] = replacement
    return table or None


def normalize(s: str, cfg: NormalizeConfig = DEFAULT_NORMALIZE) -> str:
    """Fold, collapse, compose. MUST be idempotent — see the module docstring."""
    table = _fold_table(
        cfg.fold_smart_quotes, cfg.fold_dashes, cfg.fold_ellipsis, cfg.nbsp_to_space
    )
    if table is not None:
        s = s.translate(table)
    if cfg.collapse_whitespace:
        # Every run becomes exactly one U+0020, so stripping literal spaces at
        # the ends is exactly right. `str.strip()` with no argument would strip
        # by `str.isspace()`, a DIFFERENT set (it includes U+001C-U+001F), and
        # would make this function disagree with its own whitespace class.
        s = _WS_RE.sub(" ", s).strip(" ")
    if cfg.nfc:
        s = unicodedata.normalize("NFC", s)
    return s


def words(s: str, cfg: NormalizeConfig = DEFAULT_NORMALIZE) -> tuple[str, ...]:
    """`normalize` then split on the whitespace class. Punctuation STAYS
    ATTACHED to its word (TECH_PLAN.md §8.1: "normalize -> split on `/\\s+/`").

    **It normalizes first, and that is load-bearing rather than convenient.**
    `web/src/scoring/normalize.ts` exports a function of the same name that
    always normalized; this one used to require pre-normalized input and split
    raw text differently, so `words("“smart”")` was one thing in the
    browser and another on the server — a word count, and therefore a word
    floor and an edit budget, that disagreed across the wire. The parity gate
    caught it. `normalize` is idempotent, so every existing caller that already
    normalized is unaffected.
    """
    return tuple(w for w in _WS_RE.split(normalize(s, cfg)) if w)
