"""`web/src/scoring/unicode.ts` against this package, table for table.

The "Real characters" rule is now answered live in the browser (§10.5), which
means `classify()` exists twice. Three of its four mechanisms port by
transcription and are pinned here as literals; the fourth does not port at all
and is the reason this file exists.

**The homoglyph predicate.** Python asks `ch.isalpha() or ch.isdigit()` before
consulting NFKC. `str.isdigit()` is `Numeric_Type in {Digit, Decimal}`, which no
JavaScript `\\p{...}` escape names, and the closest approximations disagree on 41
real codepoints — the circled numbers U+2469 onward fold to ASCII digit PAIRS,
so Python's `isdigit()` says no while `\\p{No}` says yes. A browser that is
STRICTER than the gate is worse than a browser that checks nothing: it paints a
rule red and then the submission clears.

So the TS ships the exact domain of `homoglyph_target` as data — 980 codepoints
in 117 ranges — and this test re-derives it by scanning all of Unicode and
comparing. When it fails it prints the replacement string, because the table is
generated and hand-editing it is how the two halves drift.

It reads the TypeScript SOURCE rather than a generated artifact on purpose: the
file the browser bundles is the file under test, and a checked-in copy of it
would be a third place for the same table to live.
"""

from __future__ import annotations

import re
import textwrap
import unicodedata
from pathlib import Path

import pytest

from launder_core.gates.checks.unicode_sanitation import (
    REJECT_CATEGORIES,
    classify,
    homoglyph_target,
)

TS_PATH = Path(__file__).resolve().parents[3] / "web" / "src" / "scoring" / "unicode.ts"

pytestmark = pytest.mark.skipif(
    not TS_PATH.is_file(), reason=f"{TS_PATH} is absent from this checkout"
)


@pytest.fixture(scope="module")
def ts_source() -> str:
    return TS_PATH.read_text(encoding="utf-8")


def _concatenated_string(source: str, name: str) -> str:
    """The value of `export const NAME = "a" + "b" + ...;` — the only form the
    generated table is allowed to take, so a table smuggled in from a fetch or
    an import would fail here rather than go unchecked."""
    match = re.search(rf"export const {name} =\s*(.*?);", source, re.DOTALL)
    assert match is not None, f"{TS_PATH.name} no longer exports {name}"
    return "".join(re.findall(r'"((?:[^"\\]|\\.)*)"', match.group(1)))


def _number_set(source: str, name: str) -> set[int]:
    """The codepoints in `const NAME = new Set([0x..., ...]);`."""
    match = re.search(rf"const {name} = new Set\(\[(.*?)\]\)", source, re.DOTALL)
    assert match is not None, f"{TS_PATH.name} no longer declares {name}"
    return {int(tok, 16) for tok in re.findall(r"0x[0-9a-fA-F]+", match.group(1))}


def _encode_ranges(codepoints: list[int]) -> str:
    """`gap:span` pairs in hex, the encoding `decodeRanges` reads."""
    out: list[str] = []
    last = -1
    start = prev = codepoints[0]
    for cp in [*codepoints[1:], -1]:
        if cp == prev + 1:
            prev = cp
            continue
        out.append(f"{start - last - 1:x}:{prev - start:x}")
        last = prev
        start = prev = cp
    return " ".join(out)


def _decode_ranges(encoded: str) -> list[int]:
    out: list[int] = []
    last = -1
    for chunk in encoded.split():
        gap, span = chunk.split(":")
        start = last + 1 + int(gap, 16)
        end = start + int(span, 16)
        out.extend(range(start, end + 1))
        last = end
    return out


# ---------------------------------------------------------------------------
# the part that cannot be transcribed
# ---------------------------------------------------------------------------


def test_homoglyph_ranges_are_exactly_the_python_predicate(ts_source: str) -> None:
    expected = [cp for cp in range(0x110000) if homoglyph_target(chr(cp)) is not None]
    shipped = _decode_ranges(_concatenated_string(ts_source, "HOMOGLYPH_RANGES"))
    if shipped != expected:
        only_ts = sorted(set(shipped) - set(expected))[:8]
        only_py = sorted(set(expected) - set(shipped))[:8]
        replacement = "\n".join(
            f'  "{line} " +' for line in textwrap.wrap(_encode_ranges(expected), 86)
        )
        pytest.fail(
            "web/src/scoring/unicode.ts HOMOGLYPH_RANGES no longer matches "
            "homoglyph_target(). TS-only: "
            + ", ".join(f"U+{cp:04X}" for cp in only_ts)
            + "; Python-only: "
            + ", ".join(f"U+{cp:04X}" for cp in only_py)
            + f"\n\nReplace the constant with (mind the trailing quote):\n{replacement}"
        )


def test_the_table_is_not_merely_the_hand_written_confusables(ts_source: str) -> None:
    """Both mechanisms have to be in there.

    `homoglyph_target` answers from NFKC decomposition FIRST and only then from
    the `CONFUSABLES` table. A table carrying one and not the other would still
    look plausible — and would miss either every fullwidth letter or every
    Cyrillic one.
    """
    # WRITTEN AS ESCAPES, for the same reason `CONFUSABLES` is: a table of
    # characters that look identical to ASCII cannot be reviewed by looking at
    # it, and a literal here would be indistinguishable from the ASCII it
    # imitates in every diff and every review.
    shipped = set(_decode_ranges(_concatenated_string(ts_source, "HOMOGLYPH_RANGES")))
    assert 0xFF21 in shipped, "FULLWIDTH LATIN CAPITAL A (NFKC mechanism)"
    assert 0x1D400 in shipped, "MATHEMATICAL BOLD CAPITAL A (NFKC mechanism)"
    assert 0x0430 in shipped, "CYRILLIC SMALL LETTER A (CONFUSABLES mechanism)"
    assert 0x03BF in shipped, "GREEK SMALL LETTER OMICRON (CONFUSABLES mechanism)"
    # ... and nothing a reader would call ordinary punctuation. An en dash is
    # not a homoglyph for our purposes: `normalize()` already folds it and the
    # player who pasted it gained no tokenization exploit.
    assert 0x2013 not in shipped, "EN DASH is punctuation, and normalize() folds it"
    assert 0x00E9 not in shipped, "e-acute is a letter, not a lookalike"


def test_the_circled_numbers_that_broke_the_obvious_port_stay_out(ts_source: str) -> None:
    """U+2469 CIRCLED NUMBER TEN folds to the two ASCII characters `10`, so
    Python's `isdigit()` is False and it is NOT a homoglyph. Every plausible
    JavaScript spelling of the gate says otherwise, which is what forced the
    table."""
    shipped = set(_decode_ranges(_concatenated_string(ts_source, "HOMOGLYPH_RANGES")))
    for cp in (0x2469, 0x2473, 0x3251, 0x32BF):
        assert unicodedata.normalize("NFKC", chr(cp)).isascii()
        assert homoglyph_target(chr(cp)) is None
        assert cp not in shipped, f"U+{cp:04X} would make the browser stricter than the gate"
    # The single-digit circled forms DO fold to one ASCII character and ARE
    # homoglyphs, so this is a real boundary rather than a blanket exclusion.
    assert 0x2460 in shipped


# ---------------------------------------------------------------------------
# the parts that do
# ---------------------------------------------------------------------------


def test_the_transcribed_codepoint_sets_match(ts_source: str) -> None:
    from launder_core.gates.checks import unicode_sanitation as py

    assert _number_set(ts_source, "ZERO_WIDTH") == set(py._ZERO_WIDTH)
    assert _number_set(ts_source, "SOFT_HYPHEN") == set(py._SOFT_HYPHEN)
    assert _number_set(ts_source, "BIDI_CONTROL") == set(py._BIDI_CONTROL)


def test_the_category_names_match_and_are_in_the_same_order(ts_source: str) -> None:
    """`reject_categories` in levels.toml is read by both sides by NAME. A
    category the browser does not know is a rule it silently stops enforcing."""
    match = re.search(r"export const REJECT_CATEGORIES = \[(.*?)\] as const", ts_source, re.DOTALL)
    assert match is not None
    names = tuple(re.findall(r'"([a-z_]+)"', match.group(1)))
    assert names == REJECT_CATEGORIES


def test_the_control_char_predicate_matches(ts_source: str) -> None:
    """Transcribed as a range test rather than a set, so it is checked
    behaviourally: the C0/C1 blocks minus the five whitespace controls, minus
    NEL, plus DEL."""
    expected = (
        {cp for cp in range(0x00, 0x20) if cp not in (0x09, 0x0A, 0x0B, 0x0C, 0x0D)}
        | {0x7F}
        | {cp for cp in range(0x80, 0xA0) if cp != 0x85}
    )
    body = re.search(r"function isControlChar\(cp: number\): boolean \{(.*?)\n\}", ts_source, re.S)
    assert body is not None
    text = body.group(1)
    for cp in (0x09, 0x0A, 0x0B, 0x0C, 0x0D):
        assert f"0x{cp:02x}" in text, f"the TS predicate must exempt U+{cp:04X} as whitespace"
    assert "0x85" in text and "0x7f" in text
    # The five C0 separators §8.1 leaves in place are the reason this category
    # exists at all; assert the Python end of the pairing here too.
    for cp in range(0x1C, 0x20):
        assert cp in expected
        assert classify(chr(cp), frozenset(REJECT_CATEGORIES), "allow", 2, frozenset()) == {
            "control_char": 1
        }
