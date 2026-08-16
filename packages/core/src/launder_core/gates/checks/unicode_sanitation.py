"""Check 1 — TECH_PLAN.md §7.1 row 1, §7.2.

**Deviation from CONCEPT.md #1: unicode is deterministic, not judged.**
CONCEPT.md wants the LLM gate to close zero-width and soft-hyphen tokenizer
exploits. It should not:

1. It is a codepoint classification with a closed form. Asking a model is both
   unreliable and wasteful.
2. The LLM sees POST-SANITATION text, so it cannot see what it is being asked
   to judge.
3. There is a related 1-point clear the judge would happily approve: a single
   typo (`the` -> `teh`) is distance 1, retokenizes into byte-fallback tokens,
   wipes a whole context window, and reads as perfectly human. This check
   closes the family; a judge cannot.

**It rejects rather than silently strips.** Stripping would let the exploit
succeed at the detector while the judge sees clean text. The UI's "clean it for
me" button applies the same normalization visibly, as an ordinary edit that
costs ordinary distance.

This is also the check that FILLS `ctx.normalized` and `ctx.words` for
everything downstream, which is why `load_levels` requires it first on every
level.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping
from typing import Any, ClassVar, Final

from launder_core.gates.registry import (
    CTX_NORMALIZED,
    CTX_WORDS,
    GateConfigError,
    GateContext,
    Phase,
    register,
)
from launder_core.schemas import CheckResult
from launder_core.scoring.normalize import normalize, words

__all__ = ["CONFUSABLES", "REJECT_CATEGORIES", "UnicodeSanitation", "classify"]

# ---------------------------------------------------------------------------
# The categories, exactly as named in data/config/levels.toml
# ---------------------------------------------------------------------------
_ZERO_WIDTH: Final[frozenset[int]] = frozenset(
    {
        0x200B,  # ZERO WIDTH SPACE
        0x200C,  # ZERO WIDTH NON-JOINER
        0x200D,  # ZERO WIDTH JOINER
        0x2060,  # WORD JOINER
        0x2061,  # FUNCTION APPLICATION
        0x2062,  # INVISIBLE TIMES
        0x2063,  # INVISIBLE SEPARATOR
        0x2064,  # INVISIBLE PLUS
        0xFEFF,  # ZERO WIDTH NO-BREAK SPACE / BOM
        0x180E,  # MONGOLIAN VOWEL SEPARATOR
    }
)
_SOFT_HYPHEN: Final[frozenset[int]] = frozenset({0x00AD})
#: C0 and C1 controls that are NOT whitespace, plus DEL.
#:
#: This category exists because of a cross-language decision recorded in
#: `scoring/normalize.py`: Python's `re` `\s` matches U+001C-U+001F (FILE,
#: GROUP, RECORD and UNIT SEPARATOR) and JavaScript's does not, so `normalize()`
#: uses Unicode `White_Space` — a property both languages spell identically —
#: and does NOT collapse them. That is the right call for normalization (§8.1:
#: normalization must not silently repair an exploit) but it leaves an invisible
#: character in the text, which is exactly what this check is for. Rejecting it
#: here is the other half of that decision, not an afterthought.
#:
#: `\t`, `\n`, `\v`, `\f`, `\r` and U+0085 are excluded: they ARE whitespace,
#: `normalize()` collapses them, and a player who pressed Enter has not
#: exploited anything.
_CONTROL_CHARS: Final[frozenset[int]] = frozenset(
    {cp for cp in range(0x00, 0x20) if cp not in (0x09, 0x0A, 0x0B, 0x0C, 0x0D)}
    | {0x7F}
    | {cp for cp in range(0x80, 0xA0) if cp != 0x85}
)
_BIDI_CONTROL: Final[frozenset[int]] = frozenset(
    {
        0x061C,  # ARABIC LETTER MARK
        0x200E,  # LEFT-TO-RIGHT MARK
        0x200F,  # RIGHT-TO-LEFT MARK
        *range(0x202A, 0x202F),  # LRE, RLE, PDF, LRO, RLO
        *range(0x2066, 0x206A),  # LRI, RLI, FSI, PDI
    }
)

#: Name -> membership test. The keys are the strings `reject_categories` uses.
REJECT_CATEGORIES: Final[tuple[str, ...]] = (
    "zero_width",
    "soft_hyphen",
    "bidi_control",
    "control_char",
    "variation_selector",
    "private_use",
    "homoglyph",
    "combining_marks",
)


def _is_variation_selector(cp: int) -> bool:
    return 0xFE00 <= cp <= 0xFE0F or 0xE0100 <= cp <= 0xE01EF


def _is_private_use(cp: int) -> bool:
    return 0xE000 <= cp <= 0xF8FF or 0xF0000 <= cp <= 0xFFFFD or 0x100000 <= cp <= 0x10FFFD


# ---------------------------------------------------------------------------
# Homoglyphs
# ---------------------------------------------------------------------------
# Two mechanisms, because two different things are called "homoglyph":
#
# 1. Characters with a COMPATIBILITY decomposition to ASCII alphanumerics —
#    fullwidth letters, mathematical alphanumeric symbols, the fi ligature,
#    circled digits, roman numerals. Detected by asking Unicode: if NFKC(ch) is
#    a different, purely ASCII alphanumeric string, ch is a lookalike. This is
#    exactly the NFKC-bait that §8.1 keeps out of `normalize()` — normalization
#    must not repair it, so it is rejected here instead.
#
# 2. Cross-script lookalikes with NO decomposition — Cyrillic a, Greek omicron.
#    Unicode cannot help; they need the table below. It is deliberately small
#    and confined to characters that are visually identical to ASCII letters in
#    a normal reading font. It is NOT a general confusables database, and it
#    does not need to be: the mechanism it defends against is "swap one letter
#    for an identical-looking one to wipe a context window", which only works
#    with characters a player can actually produce and a reader cannot see.
#
# WRITTEN AS ESCAPES, DELIBERATELY. A table of characters that look identical
# to ASCII cannot be reviewed by looking at it — that is the definition of the
# thing it catalogues. Escapes make the diff readable and make it impossible to
# introduce a wrong entry by pasting.
CONFUSABLES: Final[dict[str, str]] = {
    # --- Cyrillic, lower case ---
    "\u0430": "a",  # CYRILLIC SMALL LETTER A
    "\u0435": "e",  # CYRILLIC SMALL LETTER IE
    "\u043e": "o",  # CYRILLIC SMALL LETTER O
    "\u0440": "p",  # CYRILLIC SMALL LETTER ER
    "\u0441": "c",  # CYRILLIC SMALL LETTER ES
    "\u0445": "x",  # CYRILLIC SMALL LETTER HA
    "\u0443": "y",  # CYRILLIC SMALL LETTER U
    "\u0456": "i",  # CYRILLIC SMALL LETTER BYELORUSSIAN-UKRAINIAN I
    "\u0458": "j",  # CYRILLIC SMALL LETTER JE
    "\u0455": "s",  # CYRILLIC SMALL LETTER DZE
    "\u0501": "d",  # CYRILLIC SMALL LETTER KOMI DE
    "\u04bb": "h",  # CYRILLIC SMALL LETTER SHHA
    "\u04cf": "l",  # CYRILLIC SMALL LETTER PALOCHKA
    # --- Cyrillic, upper case ---
    "\u0410": "A",  # CYRILLIC CAPITAL LETTER A
    "\u0412": "B",  # CYRILLIC CAPITAL LETTER VE
    "\u0415": "E",  # CYRILLIC CAPITAL LETTER IE
    "\u041a": "K",  # CYRILLIC CAPITAL LETTER KA
    "\u041c": "M",  # CYRILLIC CAPITAL LETTER EM
    "\u041d": "H",  # CYRILLIC CAPITAL LETTER EN
    "\u041e": "O",  # CYRILLIC CAPITAL LETTER O
    "\u0420": "P",  # CYRILLIC CAPITAL LETTER ER
    "\u0421": "C",  # CYRILLIC CAPITAL LETTER ES
    "\u0422": "T",  # CYRILLIC CAPITAL LETTER TE
    "\u0425": "X",  # CYRILLIC CAPITAL LETTER HA
    "\u0405": "S",  # CYRILLIC CAPITAL LETTER DZE
    "\u0406": "I",  # CYRILLIC CAPITAL LETTER BYELORUSSIAN-UKRAINIAN I
    "\u0408": "J",  # CYRILLIC CAPITAL LETTER JE
    "\u04ae": "Y",  # CYRILLIC CAPITAL LETTER STRAIGHT U
    # --- Greek, lower case ---
    "\u03b1": "a",  # GREEK SMALL LETTER ALPHA
    "\u03b5": "e",  # GREEK SMALL LETTER EPSILON
    "\u03b9": "i",  # GREEK SMALL LETTER IOTA
    "\u03ba": "k",  # GREEK SMALL LETTER KAPPA
    "\u03bd": "v",  # GREEK SMALL LETTER NU
    "\u03bf": "o",  # GREEK SMALL LETTER OMICRON
    "\u03c1": "p",  # GREEK SMALL LETTER RHO
    "\u03c4": "t",  # GREEK SMALL LETTER TAU
    "\u03c5": "u",  # GREEK SMALL LETTER UPSILON
    "\u03c7": "x",  # GREEK SMALL LETTER CHI
    # --- Greek, upper case ---
    "\u0391": "A",  # GREEK CAPITAL LETTER ALPHA
    "\u0392": "B",  # GREEK CAPITAL LETTER BETA
    "\u0395": "E",  # GREEK CAPITAL LETTER EPSILON
    "\u0396": "Z",  # GREEK CAPITAL LETTER ZETA
    "\u0397": "H",  # GREEK CAPITAL LETTER ETA
    "\u0399": "I",  # GREEK CAPITAL LETTER IOTA
    "\u039a": "K",  # GREEK CAPITAL LETTER KAPPA
    "\u039c": "M",  # GREEK CAPITAL LETTER MU
    "\u039d": "N",  # GREEK CAPITAL LETTER NU
    "\u039f": "O",  # GREEK CAPITAL LETTER OMICRON
    "\u03a1": "P",  # GREEK CAPITAL LETTER RHO
    "\u03a4": "T",  # GREEK CAPITAL LETTER TAU
    "\u03a5": "Y",  # GREEK CAPITAL LETTER UPSILON
    "\u03a7": "X",  # GREEK CAPITAL LETTER CHI
    # --- Latin lookalikes that are not ASCII ---
    "\u0131": "i",  # LATIN SMALL LETTER DOTLESS I
    "\u0237": "j",  # LATIN SMALL LETTER DOTLESS J
    "\u01c0": "l",  # LATIN LETTER DENTAL CLICK
    # --- Armenian ---
    "\u0585": "o",  # ARMENIAN SMALL LETTER OH
    "\u0578": "n",  # ARMENIAN SMALL LETTER VO
    "\u057d": "u",  # ARMENIAN SMALL LETTER SEH
}


def homoglyph_target(ch: str) -> str | None:
    """The ASCII this character imitates, or None if it imitates nothing.

    Restricted to letters and digits on purpose. An en dash is not a homoglyph
    of a hyphen for our purposes — `normalize()` already folds it, and the
    player who pasted it did not gain a tokenization exploit.
    """
    if ch.isascii():
        return None
    if not (ch.isalpha() or ch.isdigit()):
        return None
    folded = unicodedata.normalize("NFKC", ch)
    if folded != ch and folded.isascii() and folded.isalnum():
        return folded
    return CONFUSABLES.get(ch)


_MARK_CATEGORIES: Final[frozenset[str]] = frozenset({"Mn", "Mc", "Me"})


def classify(
    text: str,
    reject_categories: frozenset[str],
    homoglyph_policy: str,
    max_combining_marks: int,
    allowed_chars: frozenset[str],
) -> dict[str, int]:
    """Count offending characters per category. Empty dict means clean.

    `allowed_chars` is the character set of the ORIGINAL passage, used by
    `homoglyph_policy = "reject_unless_in_original"`: a passage that legitimately
    contains a Greek letter must not become unplayable.
    """
    counts: dict[str, int] = {}

    def bump(category: str, by: int = 1) -> None:
        if category in reject_categories:
            counts[category] = counts.get(category, 0) + by

    run = 0
    for ch in text:
        cp = ord(ch)
        if cp in _ZERO_WIDTH:
            bump("zero_width")
        elif cp in _SOFT_HYPHEN:
            bump("soft_hyphen")
        elif cp in _BIDI_CONTROL:
            bump("bidi_control")
        elif cp in _CONTROL_CHARS:
            bump("control_char")
        elif _is_variation_selector(cp):
            bump("variation_selector")
        elif _is_private_use(cp):
            bump("private_use")

        if (
            homoglyph_policy != "allow"
            and homoglyph_target(ch) is not None
            # `reject_unless_in_original`: a passage that legitimately contains
            # a lookalike must not become unplayable.
            and (homoglyph_policy == "reject_always" or ch not in allowed_chars)
        ):
            bump("homoglyph")

        if unicodedata.category(ch) in _MARK_CATEGORIES:
            run += 1
            if run > max_combining_marks:
                bump("combining_marks")
        else:
            run = 0

    return counts


@register
class UnicodeSanitation:
    """Reject invisible and lookalike characters. Fill the context."""

    name: ClassVar[str] = "unicode_sanitation"
    phase: ClassVar[int] = Phase.SANITATION
    fail_open: ClassVar[bool] = False
    config_params: ClassVar[frozenset[str]] = frozenset(
        {"reject_categories", "homoglyph_policy", "max_combining_marks"}
    )
    required_params: ClassVar[frozenset[str]] = frozenset(
        {"reject_categories", "homoglyph_policy", "max_combining_marks"}
    )
    template_params: ClassVar[frozenset[str]] = frozenset({"count"})

    _POLICIES: ClassVar[frozenset[str]] = frozenset(
        {"reject_always", "reject_unless_in_original", "allow"}
    )

    def __call__(self, ctx: GateContext, params: Mapping[str, Any]) -> CheckResult:
        cfg = ctx.deps.normalize_config
        normalized = normalize(ctx.raw, cfg)
        context_update: dict[str, Any] = {
            CTX_NORMALIZED: normalized,
            CTX_WORDS: words(normalized),
        }

        policy = str(params["homoglyph_policy"])
        if policy not in self._POLICIES:
            raise GateConfigError(
                f"unicode_sanitation: unknown homoglyph_policy {policy!r}; "
                f"expected one of {sorted(self._POLICIES)}"
            )
        categories = frozenset(str(c) for c in params["reject_categories"])
        unknown = categories - set(REJECT_CATEGORIES)
        if unknown:
            raise GateConfigError(
                f"unicode_sanitation: unknown reject_categories {sorted(unknown)}; "
                f"known categories are {list(REJECT_CATEGORIES)}"
            )
        # `homoglyph` and `combining_marks` are governed by their own params, so
        # they are always live even though levels.toml does not list them among
        # the categories.
        categories = categories | {"homoglyph", "combining_marks"}

        counts = classify(
            normalized,
            reject_categories=categories,
            homoglyph_policy=policy,
            max_combining_marks=int(params["max_combining_marks"]),
            allowed_chars=frozenset(normalize(ctx.passage.text, cfg)),
        )
        total = sum(counts.values())
        if total:
            return CheckResult(
                status="fail",
                check=self.name,
                code=self.name,
                params={"count": total},
                meta={**context_update, "by_category": counts},
            )
        return CheckResult(status="pass", check=self.name, params={"count": 0}, meta=context_update)
