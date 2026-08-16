"""`edit_region` and `detector_floor` — the two checks the campaign added.

Both exist because the campaign was measured and found to be scriptable: a
greedy "delete whichever word most lowers z" loop cleared every shipped level.
So the assertions that matter here are not "the check runs" but the two
properties that make the levels resist that loop — a region check that rejects
an edit by WHERE it landed rather than how large it was, and a floor that
rejects a reading for being too LOW.
"""

from __future__ import annotations

import pytest

from launder_core.gates import Deps, GateContext, GateDependencyError
from launder_core.gates.checks.detector_floor import DetectorFloor
from launder_core.gates.checks.edit_region import EditRegion
from launder_core.gates.registry import REGISTRY, GateConfigError, Phase
from launder_core.schemas import (
    DetectorExpectation,
    DetectorReading,
    PassagePublic,
    SynthIDConfig,
)

# Twelve words per sentence, so word indices are easy to reason about.
ORIGINAL = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima "
    "mike november oscar papa quebec romeo sierra tango uniform victor whiskey xray"
)


def passage() -> PassagePublic:
    return PassagePublic(
        id="p_test",
        level_id="L7",
        wm_config_id=SynthIDConfig().wm_config_id,
        asset_bundle_id="ab1:" + "0" * 64,
        scoring_version="sc1",
        text=ORIGINAL,
        n_words=len(ORIGINAL.split()),
        detector=DetectorExpectation(
            expected_n_scored=10,
            expected_score=0.5,
            expected_z=0.0,
            g_digest="blake3:" + "0" * 64,
        ),
        claims=(),
        par=1,
        par_source="authored_reference",
        judge_prompt_id="judge.observe.v3",
    )


class Detector:
    def __init__(self, z: float, z_star: float = 2.3263) -> None:
        self.z, self.z_star = z, z_star

    def read(self, text: str, p: PassagePublic, calibration: str | None = None) -> DetectorReading:
        return DetectorReading(score=0.5, z=self.z, z_star=self.z_star, n_scored=120)


def ctx(raw: str, *, z: float = 0.0, detector: bool = True) -> GateContext:
    """A context as `unicode_sanitation` would leave it.

    `normalized` and `words` start EMPTY and are filled by that check, which is
    why it is mandatory and first. A test that constructs the context by hand
    and forgets them is not testing the check — every word reads as deleted and
    any diff-based check fails on identical text.
    """
    from launder_core.levels import load_levels
    from launder_core.scoring.normalize import normalize

    normalized = normalize(raw)
    return GateContext(
        passage=passage(),
        level=load_levels()["L7"],
        raw=raw,
        normalized=normalized,
        words=tuple(normalized.split()),
        deps=Deps(detector=Detector(z) if detector else None, judge=None),
    )


# ---------------------------------------------------------------------------
# edit_region
# ---------------------------------------------------------------------------


def test_both_checks_are_registered_and_phase_ordered() -> None:
    assert REGISTRY["edit_region"].phase == Phase.SHAPE
    assert REGISTRY["detector_floor"].phase == Phase.DETECTOR
    # The floor must be able to sit beside the threshold, never after it: the
    # threshold has to stay adjacent to the judge.
    assert REGISTRY["detector_floor"].phase == REGISTRY["detector_threshold"].phase


def test_an_edit_inside_the_window_passes() -> None:
    words = ORIGINAL.split()
    words[2] = "CHANGED"  # index 2, well inside a window of 6
    out = EditRegion()(ctx(" ".join(words)), {"editable_prefix_words": 6})
    assert out.status == "pass"
    assert out.params["n_outside"] == 0


def test_an_edit_outside_the_window_fails_and_names_where() -> None:
    words = ORIGINAL.split()
    words[15] = "CHANGED"
    out = EditRegion()(ctx(" ".join(words)), {"editable_prefix_words": 6})
    assert out.status == "fail"
    assert out.code == "edit_region"
    assert out.params["n_outside"] == 1
    # 1-based, because the message is read by a person counting words.
    assert out.params["first_bad_word_index"] == 16


def test_the_check_is_about_position_not_size() -> None:
    """The whole point: a LARGE edit inside the window is fine, and a single
    word outside it is not. That is what distinguishes this from edit_budget."""
    words = ORIGINAL.split()
    for i in range(6):
        words[i] = f"w{i}"
    big_but_inside = EditRegion()(ctx(" ".join(words)), {"editable_prefix_words": 6})

    words = ORIGINAL.split()
    words[-1] = "CHANGED"
    tiny_but_outside = EditRegion()(ctx(" ".join(words)), {"editable_prefix_words": 6})

    assert big_but_inside.status == "pass"
    assert tiny_but_outside.status == "fail"


def test_an_unedited_submission_passes() -> None:
    assert EditRegion()(ctx(ORIGINAL), {"editable_prefix_words": 6}).status == "pass"


def test_a_zero_width_window_is_a_config_error_not_an_unwinnable_level() -> None:
    with pytest.raises(GateConfigError):
        EditRegion()(ctx(ORIGINAL), {"editable_prefix_words": 0})


# ---------------------------------------------------------------------------
# detector_floor
# ---------------------------------------------------------------------------


def test_a_reading_inside_the_window_passes() -> None:
    # z* = 2.3263, min_z = -1.2 -> floor at ~1.13
    out = DetectorFloor()(ctx(ORIGINAL, z=1.5), {"min_z": -1.2})
    assert out.status == "pass"


def test_over_scrubbing_fails() -> None:
    out = DetectorFloor()(ctx(ORIGINAL, z=-2.0), {"min_z": -1.2})
    assert out.status == "fail"
    assert out.code == "detector_floor"
    # It reports the reading that rejected it, so /api/submit can show the
    # player the number rather than a zero.
    assert out.meta["z"] == pytest.approx(-2.0)
    assert out.meta["z_floor"] == pytest.approx(2.3263 - 1.2)


def test_the_floor_is_relative_to_the_notch_like_max_z_is() -> None:
    """Both bounds share an origin, so re-calibrating z* moves them together."""
    tight = DetectorFloor()(ctx(ORIGINAL, z=1.0), {"min_z": -1.0})
    loose = DetectorFloor()(ctx(ORIGINAL, z=1.0), {"min_z": -2.0})
    assert tight.status == "fail"  # floor 1.33, reading 1.0
    assert loose.status == "pass"  # floor 0.33, reading 1.0


def test_the_floor_never_passes_by_default_when_the_detector_is_missing() -> None:
    with pytest.raises(GateDependencyError):
        DetectorFloor()(ctx(ORIGINAL, detector=False), {"min_z": -1.2})
