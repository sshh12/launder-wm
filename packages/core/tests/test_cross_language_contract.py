"""The three cross-language decisions the integration pass had to settle.

Each of these was a place where Python and TypeScript computed different things
from the same input, quietly, in a way no single-language test could see. Each
is now pinned from BOTH sides — here, in `web/tests/scoring.test.ts`, and in the
end-to-end gate at `packages/forge/tests/test_parity_ts.py`.

1. **The whitespace class** is Unicode `White_Space` minus U+FEFF. Neither
   language's `\\s` is usable: Python's matches U+001C-U+001F and U+0085,
   JavaScript's matches U+FEFF instead. The two exclusions are paired with
   `unicode_sanitation`, because §8.1 forbids normalization from silently
   repairing an exploit — so anything invisible that `normalize()` leaves in
   place has to be *rejected* downstream.
2. **`words()` normalizes first.** The TS export of the same name always did;
   this one required pre-normalized input and split raw text differently, so a
   word count computed in the browser and one computed on the server could
   disagree on any text carrying a smart quote.
3. **The calibration interpolates kappa in `log n`**, matching the shipped
   file's own `sigma_model.inflation.kind`.
"""

from __future__ import annotations

import pytest

from launder_core.detect.calibration import load_thresholds, sigma_closed_form
from launder_core.gates.checks.unicode_sanitation import REJECT_CATEGORIES, classify
from launder_core.scoring.normalize import WHITESPACE_CODEPOINTS, normalize, words

ALL_CATEGORIES = frozenset(REJECT_CATEGORIES)


def _classify(text: str) -> dict[str, int]:
    return classify(text, ALL_CATEGORIES, "allow", 2, frozenset())


# --------------------------------------------------------------------------
# 1. the whitespace class, and its other half
# --------------------------------------------------------------------------


def test_whitespace_class_is_unicode_white_space_minus_bom() -> None:
    assert 0xFEFF not in WHITESPACE_CODEPOINTS
    for cp in range(0x1C, 0x20):
        assert cp not in WHITESPACE_CODEPOINTS, f"U+{cp:04X} is a C0 control, not whitespace"
    # NEL is whitespace and JavaScript's \s misses it, which is why the class is
    # written out rather than inherited on either side.
    assert 0x85 in WHITESPACE_CODEPOINTS


@pytest.mark.parametrize("cp", [0x1C, 0x1D, 0x1E, 0x1F])
def test_c0_separators_survive_normalize_and_are_rejected(cp: int) -> None:
    """The pairing, asserted as one fact: not eaten here, not allowed through."""
    ch = chr(cp)
    text = f"and{ch}then"
    assert normalize(text) == text, "normalize must not silently delete a control character"
    assert words(text) == (text,), "it stays glued to its word, visible to the gate"
    assert _classify(text) == {"control_char": 1}


@pytest.mark.parametrize(
    "ch",
    ["\ufeff", "\u200b", "\u00ad"],
    ids=["bom", "zero_width_space", "soft_hyphen"],
)
def test_invisible_characters_survive_normalize_and_are_rejected(ch: str) -> None:
    text = f"a{ch}b"
    assert ch in normalize(text)
    assert _classify(text), f"{ch!r} survives normalize and must be rejected by the gate"


@pytest.mark.parametrize(
    "ch",
    ["\t", "\n", "\r", "\x0b", "\x0c", "\x85", "\u2028", "\u00a0"],
    ids=["tab", "lf", "cr", "vt", "ff", "nel", "line_sep", "nbsp"],
)
def test_real_whitespace_is_collapsed_and_not_rejected(ch: str) -> None:
    """A player who pressed Enter has not exploited anything."""
    assert normalize(f"a{ch}b") == "a b"
    assert _classify(f"a{ch}b") == {}


def test_control_char_is_a_declared_category() -> None:
    assert "control_char" in REJECT_CATEGORIES


def test_tab_and_newline_are_not_control_chars() -> None:
    assert _classify("line\nbreak\tand tab") == {}


# --------------------------------------------------------------------------
# 2. words() normalizes
# --------------------------------------------------------------------------


def test_words_normalizes_before_splitting() -> None:
    assert words("\u201csmart\u201d \u2018quotes\u2019") == ('"smart"', "'quotes'")
    assert words("a\u00a0b") == ("a", "b")
    assert words("  ragged   spacing  ") == ("ragged", "spacing")


def test_words_is_idempotent_under_normalize() -> None:
    """Every existing caller already normalized; normalize is idempotent."""
    raw = "  \u201cThe study\u2014it held\u2026\u201d  "
    assert words(raw) == words(normalize(raw))


def test_words_keeps_punctuation_attached() -> None:
    assert words("study. it ended") == ("study.", "it", "ended")


# --------------------------------------------------------------------------
# 3. the calibration axis
# --------------------------------------------------------------------------


def test_kappa_interpolates_in_log_n() -> None:
    cal = load_thresholds()
    if len(cal.buckets) < 2:
        pytest.skip("no empirical buckets: data/assets/thresholds.v1.json is absent")
    lo, hi = cal.buckets[0], cal.buckets[1]
    import math

    mid = math.exp(0.5 * (math.log(lo.n_scored) + math.log(hi.n_scored)))
    expected = 0.5 * (lo.kappa + hi.kappa)
    assert cal.kappa(mid) == pytest.approx(expected, rel=1e-12)
    # ... and NOT the linear-in-n midpoint, which is the convention this repo
    # rejected. They differ by enough to move z in the fourth decimal.
    linear_mid = 0.5 * (lo.n_scored + hi.n_scored)
    assert cal.kappa(linear_mid) != pytest.approx(expected, rel=1e-9)


def test_shipped_buckets_carry_sigma_so_the_browser_needs_no_phi_inv() -> None:
    """`web/src/detector/calibration.ts` refuses a bucket without `sigma`."""
    import json
    from pathlib import Path

    from launder_core.watermark.config import data_dir

    p = Path(data_dir()) / "assets" / "thresholds.v1.json"
    if not p.is_file():
        pytest.skip("data/assets/thresholds.v1.json is absent")
    raw = json.loads(p.read_text(encoding="utf-8"))
    for name, group in raw["calibrations"].items():
        for b in group["buckets"]:
            assert "sigma" in b, f"{name} bucket n_scored={b['n_scored']} has no sigma"
            derived = (b["score_at_fpr"] - 0.5) / raw["z_star"]
            assert b["sigma"] == pytest.approx(derived, rel=1e-9)
            assert b["sigma"] > 0


def test_sigma_keeps_the_closed_form_shape() -> None:
    """The formula is the shape; the percentile is only the level (§14.2 #7)."""
    assert sigma_closed_form(400) == pytest.approx(sigma_closed_form(100) / 2.0, rel=1e-12)
