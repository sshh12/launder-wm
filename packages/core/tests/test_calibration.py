"""sigma_null(T), z, and the two-part calibration of TECH_PLAN.md §14.2 item 7.

The rule being encoded: **the closed form is the SHAPE, the empirical percentile
is the LEVEL.** These tests assert both halves independently, and assert that the
missing-file path is loud rather than silent — a deployment running on the
unverified level must be able to say so.
"""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Any

import pytest

from launder_core.detect.calibration import (
    DEFAULT_FPR,
    Z_STAR,
    Calibration,
    CalibrationBucket,
    closed_form_calibration,
    load_thresholds,
    parse_thresholds,
    phi,
    phi_inv,
    sigma_closed_form,
    sigma_weighted_closed_form,
    weighting_kappa,
    z_from_score,
)

# Exact standard-normal quantiles, to 12 significant figures.
KNOWN_QUANTILES = [
    (0.5, 0.0),
    (0.75, 0.674489750196),
    (0.9, 1.281551565545),
    (0.95, 1.644853626951),
    (0.975, 1.959963984540),
    (0.99, 2.326347874041),
    (0.999, 3.090232306168),
    (0.9999, 3.719016485455),
    (1e-6, -4.753424308823),
]


# ---------------------------------------------------------------------------
# Phi and Phi^-1 without scipy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("p", "expected"), KNOWN_QUANTILES)
def test_phi_inv_matches_known_quantiles(p: float, expected: float) -> None:
    assert phi_inv(p) == pytest.approx(expected, abs=1e-11)


def test_phi_and_phi_inv_are_inverse() -> None:
    for i in range(1, 1000):
        p = i / 1000.0
        assert phi(phi_inv(p)) == pytest.approx(p, rel=1e-12, abs=1e-15)


def test_phi_inv_matches_scipy_where_scipy_exists() -> None:
    """`launder-core` must not depend on scipy; `forge` may. Pin them together."""
    stats = pytest.importorskip("scipy.stats", reason="scipy lives in launder-forge")
    for p in (1e-5, 0.001, 0.01, 0.2, 0.5, 0.8, 0.99, 0.999):
        assert phi_inv(p) == pytest.approx(float(stats.norm.ppf(p)), abs=1e-10)
    # Far upper tail: `Phi(x) - p` cancels catastrophically once `1 - p` is at the
    # edge of double precision, so agreement is relative, not absolute. Nothing in
    # this project evaluates a quantile past 1 - 1e-6 (z* is Phi^-1(0.99)).
    for p in (1 - 1e-9, 1 - 1e-12):
        assert phi_inv(p) == pytest.approx(float(stats.norm.ppf(p)), rel=1e-8)


def test_phi_inv_rejects_impossible_probabilities() -> None:
    for p in (0.0, 1.0, -0.1, 1.5, math.nan):
        with pytest.raises(ValueError, match="requires 0 < p < 1"):
            phi_inv(p)


def test_z_star_is_the_1_percent_quantile() -> None:
    """z* = 2.3263 is Phi^-1(0.99): Google's published 1% FPR, and the reason
    'you are beating a detector calibrated at their headline rate' is sayable."""
    assert pytest.approx(phi_inv(1.0 - DEFAULT_FPR), abs=5e-5) == Z_STAR
    assert DEFAULT_FPR == 0.01


# ---------------------------------------------------------------------------
# the closed form — the SHAPE
# ---------------------------------------------------------------------------


def test_closed_form_reproduces_the_papers_numbers() -> None:
    """§14.2 item 7's own sanity check: the paper reports a 1%-FPR threshold of
    0.510, which `tau = 1/2 + Phi_inv(0.99)/(2*sqrt(m*T))` solves at T ~= 450 —
    matching their ~400-token texts. If this drifts, the formula is wrong."""
    # `sigma_closed_form` directly, NOT `closed_form_calibration()`: the paper's
    # number is the UNWEIGHTED closed form, and the fallback calibration now
    # carries `weighting_kappa` on top of it (that factor is algebra, see
    # test_weighting_kappa_is_the_algebraic_part_of_the_measured_kappa).
    tau = 0.5 + phi_inv(0.99) * sigma_closed_form(450, 30)
    assert tau == pytest.approx(0.510, abs=5e-4)

    # And solving the other way lands on the same place.
    z = phi_inv(0.99)
    t_solved = (z / (2.0 * (0.510 - 0.5))) ** 2 / 30
    assert t_solved == pytest.approx(450, rel=0.01)


def test_sigma_falls_as_one_over_sqrt_t() -> None:
    """The intro slider's lesson, as an instrument reading: z grows like sqrt(T)."""
    assert sigma_closed_form(100, 30) / sigma_closed_form(400, 30) == pytest.approx(2.0)
    cal = closed_form_calibration()
    z100 = cal.z(0.52, 100)
    z400 = cal.z(0.52, 400)
    assert z400 / z100 == pytest.approx(2.0, rel=1e-12)


def test_sigma_is_infinite_for_nothing_scored_and_z_is_zero() -> None:
    assert sigma_closed_form(0, 30) == math.inf
    assert closed_form_calibration().z(0.9, 0) == 0.0
    assert closed_form_calibration().sigma(0) == math.inf


def test_z_is_zero_at_the_neutral_score() -> None:
    cal = closed_form_calibration()
    assert cal.z(0.5, 200) == 0.0
    assert cal.z(0.4, 200) < 0.0


def test_cleared_is_strictly_below_the_notch() -> None:
    cal = closed_form_calibration()
    n = 160
    at_notch = 0.5 + cal.z_star * cal.sigma(n)
    assert cal.z(at_notch, n) == pytest.approx(cal.z_star, rel=1e-12)
    assert not cal.cleared(at_notch + 1e-9, n)
    assert cal.cleared(at_notch - 1e-9, n)


def test_z_from_score_accepts_an_explicit_calibration() -> None:
    cal = closed_form_calibration()
    assert z_from_score(0.52, 200, cal) == pytest.approx(cal.z(0.52, 200))


def test_closed_form_calibration_reports_its_source() -> None:
    cal = closed_form_calibration()
    assert cal.source == "closed_form"
    assert cal.empirical is False
    assert cal.depth == 30
    # NOT 1.0. `sigma_closed_form` is the sd of an UNWEIGHTED mean; the shipped
    # detector takes a weighted one, whose null sd is exactly
    # `sqrt(sum(w^2)/m)` times larger under independence. Shipping kappa = 1
    # here overstated every fallback z by 5-11%, i.e. ran a notch labelled
    # 1% FPR at a true 2-3%.
    assert cal.kappa(123) == pytest.approx(weighting_kappa(30), abs=0.0)
    assert cal.sigma(123) == pytest.approx(sigma_weighted_closed_form(123, 30), rel=1e-15)


# ---------------------------------------------------------------------------
# the empirical percentile — the LEVEL
# ---------------------------------------------------------------------------


def _thresholds_doc(buckets: list[dict[str, float | int]]) -> dict[str, object]:
    return {
        "schema": "launder.thresholds/1",
        "wm_config_id": "wm1:" + "0" * 64,
        "z_star": 2.3263,
        "fpr": 0.01,
        "depth": 30,
        "calibrations": {
            "default": {"buckets": buckets},
            "code": {"buckets": [{"n_scored": 60, "sigma": 0.02, "n_samples": 20000}]},
        },
    }


def test_empirical_sigma_is_derived_from_the_percentile() -> None:
    """`sigma = (score_at_fpr - 0.5) / Phi_inv(1 - fpr)` — the level, measured."""
    bucket = CalibrationBucket(n_scored=100, sigma=0.0, n_samples=0)
    assert bucket.n_scored == 100
    doc = _thresholds_doc([{"n_scored": 100, "score_at_fpr": 0.52, "n_samples": 20000}])
    cal = parse_thresholds(doc, name="default", source="empirical:test")
    (only,) = cal.buckets
    assert only.sigma == pytest.approx((0.52 - 0.5) / phi_inv(0.99), abs=1e-15)
    assert only.n_samples == 20000
    assert cal.empirical is True


def test_empirical_level_multiplies_the_closed_form_shape() -> None:
    """kappa = sigma_empirical / sigma_closed, and sigma(T) = kappa(T)*closed(T).

    kappa is expressed against the UNWEIGHTED closed form on both sides of the
    language boundary, so the comparison here is against `sigma_closed_form`
    rather than against the fallback calibration (which carries
    `weighting_kappa` and would make the ratio 3/1.1129, not 3).
    """
    closed = sigma_closed_form(100, 30)
    doc = _thresholds_doc([{"n_scored": 100, "sigma": 3.0 * closed, "n_samples": 20000}])
    cal = parse_thresholds(doc, name="default", source="empirical:test")
    assert cal.kappa(100) == pytest.approx(3.0)
    assert cal.sigma(100) == pytest.approx(3.0 * closed)
    assert cal.z(0.52, 100) == pytest.approx((0.52 - 0.5) / (3.0 * closed))
    # ... and the fallback differs from it by exactly the weighting factor.
    assert cal.z(0.52, 100) == pytest.approx(
        closed_form_calibration().z(0.52, 100) * weighting_kappa(30) / 3.0
    )


def test_kappa_interpolates_in_log_t_and_clamps_outside() -> None:
    lo_closed = sigma_closed_form(100, 30)
    hi_closed = sigma_closed_form(400, 30)
    doc = _thresholds_doc(
        [
            {"n_scored": 100, "sigma": 2.0 * lo_closed, "n_samples": 20000},
            {"n_scored": 400, "sigma": 4.0 * hi_closed, "n_samples": 20000},
        ]
    )
    cal = parse_thresholds(doc, name="default", source="empirical:test")

    assert cal.kappa(100) == pytest.approx(2.0)
    assert cal.kappa(400) == pytest.approx(4.0)
    assert cal.kappa(200) == pytest.approx(3.0)  # log-midpoint of 100 and 400
    # Clamped, not extrapolated: beyond the measured range only the shape is trusted.
    assert cal.kappa(10) == pytest.approx(2.0)
    assert cal.kappa(10_000) == pytest.approx(4.0)
    # The sqrt(T) shape still applies outside the range.
    assert cal.sigma(1600) == pytest.approx(4.0 * sigma_closed_form(1600, 30))


def test_buckets_are_sorted_on_load_and_duplicates_rejected() -> None:
    closed = sigma_closed_form(200, 30)
    doc = _thresholds_doc(
        [
            {"n_scored": 400, "sigma": closed, "n_samples": 1},
            {"n_scored": 100, "sigma": closed, "n_samples": 1},
        ]
    )
    cal = parse_thresholds(doc, name="default", source="empirical:test")
    assert [b.n_scored for b in cal.buckets] == [100, 400]

    with pytest.raises(ValueError, match="duplicate n_scored"):
        Calibration(
            buckets=(
                CalibrationBucket(n_scored=100, sigma=0.01),
                CalibrationBucket(n_scored=100, sigma=0.02),
            )
        )
    with pytest.raises(ValueError, match="must be sorted"):
        Calibration(
            buckets=(
                CalibrationBucket(n_scored=200, sigma=0.01),
                CalibrationBucket(n_scored=100, sigma=0.02),
            )
        )


def test_a_measured_bucket_is_reproduced_at_the_files_own_depth() -> None:
    """`kappa` is a RATIO against the closed form, and the closed form has an
    `m` in it — so the `m` on the bucket and the `m` on the curve have to be the
    same one.

    `CalibrationBucket.kappa` hardcoded 30 while `Calibration.sigma` divided by
    `sigma_closed_form(T, self.depth)`, so the two only cancelled while the
    thresholds file said `depth = 30`. Any other depth silently replaced a
    MEASURED sigma with `sigma * sqrt(30/depth)`, which is the one thing a
    measured bucket may never mean.
    """
    doc = _thresholds_doc([{"n_scored": 100, "sigma": 0.01, "n_samples": 20000}])
    doc["depth"] = 15
    cal = parse_thresholds(doc, name="default", source="empirical:test")
    assert cal.depth == 15
    assert cal.sigma(100) == pytest.approx(0.01, rel=1e-15)
    assert cal.z(0.52, 100) == pytest.approx(0.02 / 0.01, rel=1e-15)
    # ...and the shipped depth still behaves exactly as before.
    at30 = parse_thresholds(
        _thresholds_doc([{"n_scored": 100, "sigma": 0.01, "n_samples": 20000}]),
        name="default",
        source="empirical:test",
    )
    assert at30.sigma(100) == pytest.approx(0.01, rel=1e-15)


def test_a_calibration_with_no_buckets_carries_the_weighting_factor_not_one() -> None:
    """The field comment said "NOT 1.0 for the shipped detector" three lines
    above a default of 1.0. Both constructors in the module passed
    `weighting_kappa` explicitly, so the shipped paths were right and the
    default was loaded for the next caller: `Calibration(depth=30)`, the obvious
    spelling of "the fallback curve", reinstated the 5-11% overstatement."""
    assert Calibration().kappa(200) == pytest.approx(weighting_kappa(30), abs=0.0)
    assert Calibration(depth=12).kappa(200) == pytest.approx(weighting_kappa(12), abs=0.0)
    assert Calibration().sigma(200) == pytest.approx(sigma_weighted_closed_form(200, 30), rel=1e-15)
    # An explicit value still wins — that is what makes it a knob.
    assert Calibration(base_kappa=1.0).kappa(200) == 1.0


def test_a_bucket_without_a_measurement_is_rejected() -> None:
    doc = _thresholds_doc([{"n_scored": 100, "n_samples": 20000}])
    with pytest.raises(ValueError, match="neither 'sigma' nor 'score_at_fpr'"):
        parse_thresholds(doc, name="default", source="empirical:test")


def test_l5_gets_its_own_bucket_set() -> None:
    """§7.6: code has far fewer scored tokens and far lower optionality, so its
    sigma(T) curve is fit separately."""
    doc = _thresholds_doc([{"n_scored": 100, "sigma": 0.01, "n_samples": 20000}])
    code = parse_thresholds(doc, name="code", source="empirical:test")
    assert [b.n_scored for b in code.buckets] == [60]
    assert code.name == "code"
    with pytest.raises(KeyError, match="not found in thresholds file"):
        parse_thresholds(doc, name="prose", source="empirical:test")


def test_load_thresholds_reads_a_real_file(tmp_path: Path) -> None:
    closed = sigma_closed_form(120, 30)
    doc = _thresholds_doc([{"n_scored": 120, "sigma": 2.5 * closed, "n_samples": 20000}])
    path = tmp_path / "thresholds.v1.json"
    path.write_text(json.dumps(doc), encoding="utf-8")

    cal = load_thresholds(path)
    assert cal.source == f"empirical:{path}"
    assert cal.z_star == 2.3263
    assert cal.kappa(120) == pytest.approx(2.5)
    assert cal.wm_config_id == "wm1:" + "0" * 64
    assert load_thresholds(path) is cal, "result must be cached; it is read per request"


def test_flat_bucket_list_is_accepted(tmp_path: Path) -> None:
    path = tmp_path / "flat.json"
    path.write_text(
        json.dumps({"buckets": [{"n_scored": 80, "score_at_fpr": 0.515, "n_samples": 20000}]}),
        encoding="utf-8",
    )
    cal = load_thresholds(path)
    assert cal.buckets[0].score_at_fpr == 0.515
    assert cal.buckets[0].sigma > 0.0


def test_wrong_schema_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "wrong.json"
    path.write_text(json.dumps({"schema": "launder.thresholds/2"}), encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape("expected 'launder.thresholds/1'")):
        load_thresholds(path)


# ---------------------------------------------------------------------------
# the fallback must be loud
# ---------------------------------------------------------------------------


def test_missing_file_warns_and_falls_back(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    missing = tmp_path / "absent.json"
    with caplog.at_level(logging.WARNING, logger="launder_core.detect.calibration"):
        cal = load_thresholds(missing)
    assert cal.source == "closed_form"
    assert cal.buckets == ()
    text = caplog.text
    assert "CLOSED-FORM" in text
    assert "OVERSTATED" in text
    assert "forge calibrate" in text, "the warning must name the command that fixes it"


def test_missing_file_is_fatal_when_required(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="forge calibrate"):
        load_thresholds(tmp_path / "also-absent.json", required=True)


def test_shipped_calibration_state_is_whatever_the_repo_actually_has() -> None:
    """Documents the current state rather than asserting a file that M7 produces.

    Once `forge calibrate` lands `data/assets/thresholds.v1.json`, this flips to
    the empirical branch on its own — and `required=True` at boot is what a
    calibrated deployment should use.
    """
    cal = load_thresholds()
    assert cal.z_star > 0.0
    assert cal.depth == 30
    assert cal.source == "closed_form" or cal.source.startswith("empirical:")


# ---------------------------------------------------------------------------
# kappa's algebraic part (the correction that is NOT a correlation correction)
# ---------------------------------------------------------------------------


def test_weighting_kappa_is_sqrt_sum_w_squared_over_m() -> None:
    """`kappa` is mostly algebra, and this is the algebra.

    `sigma_closed_form` is the null sd of an UNWEIGHTED mean of m*T independent
    Bernoulli(1/2)s. The shipped detector takes a WEIGHTED mean, so its null sd
    is `sqrt(sum(w^2))/(2*m*sqrt(T))` — a constant factor `sqrt(sum(w^2)/m)`
    larger, with no correlation anywhere in the derivation.
    """
    from launder_core.detect.weighted_mean import depth_weights

    w = depth_weights(30)
    assert w.sum() == pytest.approx(30.0, rel=1e-15)
    expected = math.sqrt(float((w * w).sum()) / 30.0)
    assert weighting_kappa(30) == pytest.approx(expected, abs=0.0)
    assert weighting_kappa(30) == pytest.approx(1.1128924007211063, rel=1e-12)
    assert sigma_weighted_closed_form(180, 30) == pytest.approx(
        weighting_kappa(30) * sigma_closed_form(180, 30), rel=1e-15
    )


def test_weighting_kappa_is_the_algebraic_part_of_the_measured_kappa() -> None:
    """Every SHIPPED bucket sits within a few percent of the analytic factor.

    This is the load-bearing fact the module docstring used to get wrong: the
    measured kappa is not "the 30 tournament layers are strongly correlated", it
    is the weight vector. Anyone who edits `depth_weights` must regenerate
    `thresholds.v1.json` and every packed `expected_z`, for algebraic reasons,
    before correlation is even discussed.
    """
    cal = load_thresholds()
    if not cal.buckets:  # pragma: no cover - only if the asset is absent
        pytest.skip("data/assets/thresholds.v1.json is absent")
    analytic = weighting_kappa(cal.depth)
    for bucket in cal.buckets:
        ratio = bucket.kappa / analytic
        assert 0.90 < ratio < 1.10, (
            f"bucket T={bucket.n_scored} measures kappa {bucket.kappa:.6f}, which is "
            f"{ratio:.4f}x the analytic weighting factor {analytic:.6f}. Either the "
            "weight vector changed without a recalibration, or the null corpus grew "
            "real row-to-row correlation — both are reportable, neither is silent."
        )


def test_the_measured_kappa_is_reproduced_by_INDEPENDENT_rows() -> None:
    """Monte-Carlo over iid Bernoulli g-values, i.e. zero correlation by construction.

    If the ~1.11 came from correlated tournament layers this could not match: the
    rows here are independent by construction and still reproduce it.
    """
    import numpy as np

    from launder_core.detect.weighted_mean import weighted_mean_score

    rng = np.random.default_rng(20260815)
    n_scored, trials = 180, 4000
    scores = np.empty(trials, dtype=np.float64)
    g = rng.integers(0, 2, size=(trials, n_scored, 30), dtype=np.uint8)
    for i in range(trials):
        scores[i] = weighted_mean_score(g[i]).score
    measured = float(scores.std(ddof=1))
    predicted = sigma_weighted_closed_form(n_scored, 30)
    assert measured == pytest.approx(predicted, rel=0.05)
    # ... and is emphatically NOT the unweighted closed form.
    assert abs(measured / sigma_closed_form(n_scored, 30) - 1.0) > 0.05


# ---------------------------------------------------------------------------
# the documented knob is the real knob (TECH_PLAN.md §12 row 8)
# ---------------------------------------------------------------------------


def test_a_group_may_repeat_z_star_and_fpr() -> None:
    """The SHAPE THE SHIPPED FILE IS IN. `forge calibrate` writes both levels."""
    doc = _thresholds_doc([{"n_scored": 100, "sigma": 0.01, "n_samples": 20000}])
    groups: Any = doc["calibrations"]
    groups["default"]["z_star"] = doc["z_star"]
    groups["default"]["fpr"] = doc["fpr"]
    cal = parse_thresholds(doc, name="default", source="empirical:test")
    assert cal.z_star == doc["z_star"]


def test_a_group_may_not_shadow_the_top_level_z_star() -> None:
    """Editing the top-level `z_star` has to actually move the notch.

    §12 row 8 documents exactly that edit. A group key of the same name used to
    win silently, so the documented knob was inert and the real one was
    undocumented and nested.
    """
    doc = _thresholds_doc([{"n_scored": 100, "sigma": 0.01, "n_samples": 20000}])
    groups: Any = doc["calibrations"]
    groups["default"]["z_star"] = 1.5
    with pytest.raises(ValueError, match="contradicts the top-level z_star"):
        parse_thresholds(doc, name="default", source="empirical:test")


def test_editing_the_top_level_z_star_moves_the_notch() -> None:
    doc = _thresholds_doc([{"n_scored": 100, "sigma": 0.01, "n_samples": 20000}])
    doc["z_star"] = 2.0
    cal = parse_thresholds(doc, name="default", source="empirical:test")
    assert cal.z_star == 2.0
