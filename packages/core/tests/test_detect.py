"""The detector: weights, per-token decomposition, the end-to-end golden, the Protocol.

Golden case #6 (TECH_PLAN.md §4.5) is `test_end_to_end_score_on_ids50`, at
`tol = 1e-9` in float64 — the tolerance the TS port is held to as well.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from launder_core.detect import (
    BayesianDetector,
    Detector,
    DetectorScore,
    WeightedMeanDetector,
    bayesian_score,
    closed_form_calibration,
    contributions,
    default_params,
    depth_weights,
    detect_ids,
    fit_params,
    heat_values,
    weighted_mean_score,
)
from launder_core.watermark import compute_frame, compute_g_values

# The golden token sequences live with the golden hashes they were verified
# against. pytest's importlib mode resolves this; mypy has no package here to
# resolve it in, hence the ignore.
from .test_watermark import IDS50, REPEATED_IDS  # type: ignore[import-not-found]

TOL = 1e-9

# Golden case #6: the end-to-end weighted mean on the fixed 50-id sequence,
# with the CLOSED-FORM calibration pinned so the value does not move when
# `forge calibrate` lands data/assets/thresholds.v1.json.
GOLDEN_SCORE = 0.5119985461814548
# CHANGED DELIBERATELY, and this is the audit trail. It was 0.8914524158844032,
# which is `(score - 0.5) / (1/(2*sqrt(30*46)))` — the null sd of an UNWEIGHTED
# mean. The shipped detector takes a WEIGHTED mean, whose null sd under
# independence is `sqrt(sum(w^2)/m)` = 1.1128924007211063 times larger, so
# `closed_form_calibration()` now carries that factor and the fallback z is the
# old value DIVIDED by it:
#
#     0.8914524158844032 / 1.1128924007211063 == 0.8010230057342295
#
# The old value overstated every fallback z by ~11%, i.e. ran a notch labelled
# 1% FPR at a true 2-3%. `data/golden/vectors.json` is unaffected: its z values
# use the EMPIRICAL calibration, where kappa is measured against the unweighted
# closed form and the two factors cancel exactly.
GOLDEN_Z_CLOSED_FORM = 0.8010230057342295
GOLDEN_N_SCORED = 46
GOLDEN_HEAT_HEAD = [0.5009404388714734, 0.3646812957157784, 0.5882967607105538]

# The same, on a sequence with two rows removed by the repetition mask.
GOLDEN_MASKED_SCORE = 0.5215836526181353
GOLDEN_MASKED_N_SCORED = 9
GOLDEN_MASKED_ROWS = 11


# ---------------------------------------------------------------------------
# weights
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("m", [1, 2, 9, 30, 64])
def test_depth_weights_sum_to_m(m: int) -> None:
    """`sum(w) == m` is what keeps heat in [0,1] and 0.5 neutral."""
    w = depth_weights(m)
    assert w.shape == (m,)
    assert float(w.sum()) == pytest.approx(m, abs=1e-12)
    assert (w > 0).all()


def test_depth_weights_are_the_10_to_1_ramp() -> None:
    w = depth_weights(30)
    assert w[0] / w[-1] == pytest.approx(10.0)
    assert float(w[0]) == pytest.approx(1.8181818181818183)
    assert float(w[-1]) == pytest.approx(0.18181818181818182)
    assert (np.diff(w) < 0).all(), "earlier tournament layers must weigh more"


def test_depth_weights_reject_nonsense() -> None:
    with pytest.raises(ValueError, match="depth m must be"):
        depth_weights(0)


# ---------------------------------------------------------------------------
# the decomposition — §4.6 mechanism 1
# ---------------------------------------------------------------------------


def test_heat_is_the_all_ones_and_all_zeros_extremes() -> None:
    m = 30
    assert float(heat_values(np.ones((1, m), dtype=np.uint8))[0]) == pytest.approx(1.0)
    assert float(heat_values(np.zeros((1, m), dtype=np.uint8))[0]) == pytest.approx(0.0)
    half = np.zeros((1, m), dtype=np.uint8)
    half[0, ::2] = 1
    assert 0.0 < float(heat_values(half)[0]) < 1.0


def test_contributions_reconstruct_the_score_exactly() -> None:
    g = compute_g_values(IDS50)
    result = weighted_mean_score(g)
    assert float(result.contributions.sum()) == pytest.approx(result.score, abs=1e-15)
    assert np.array_equal(contributions(g), result.contributions)


def test_weighted_mean_matches_the_deepmind_reference_formula() -> None:
    """Transcribed from `google-deepmind/synthid-text`'s `detector_mean.py`::

        weights = linspace(10, 1, depth); weights *= depth / sum(weights)
        g *= weights
        score = sum(g * mask[..., None]) / (depth * sum(mask))

    That repository's *g-values* are the bit-incompatible variant (§4.1), but
    §4.2 states its `mean_score` / `weighted_mean_score` are identical to HF's —
    so it is a genuine independent authority on the scorer, and this asserts it
    by a different arithmetic path than the implementation takes.
    """
    frame = compute_frame(REPEATED_IDS)
    g = frame.g.astype(np.float64)
    mask = frame.mask.astype(np.float64)
    depth = g.shape[1]
    weights = np.linspace(10.0, 1.0, depth)
    weights *= depth / weights.sum()
    reference = float((g * weights).sum(axis=1) @ mask) / (depth * mask.sum())
    assert weighted_mean_score(frame.g, frame.mask).score == pytest.approx(reference, abs=1e-15)


def test_masked_rows_contribute_nothing_to_either_side_of_the_fraction() -> None:
    """§4.4(c): a masked position is removed from numerator AND denominator.

    Dropping the masked rows entirely must give the identical score — that
    equality is what makes "make a context echo an earlier one" a real strategy
    rather than a rounding artifact.
    """
    frame = compute_frame(REPEATED_IDS)
    full = weighted_mean_score(frame.g, frame.mask)
    kept = weighted_mean_score(frame.g[frame.mask])
    assert full.score == pytest.approx(kept.score, abs=1e-15)
    assert full.n_scored == kept.n_scored
    assert float(full.contributions[~frame.mask].sum()) == 0.0


def test_empty_and_fully_masked_read_neutral_not_zero() -> None:
    """A needle pinned at maximum evidence AGAINST the watermark would be a lie
    about a text the detector has no opinion on."""
    empty = weighted_mean_score(np.zeros((0, 30), dtype=np.uint8))
    assert empty.score == 0.5
    assert empty.n_scored == 0

    g = compute_g_values(IDS50)
    all_masked = weighted_mean_score(g, np.zeros(g.shape[0], dtype=np.bool_))
    assert all_masked.score == 0.5
    assert all_masked.n_scored == 0
    assert float(all_masked.contributions.sum()) == 0.0
    assert closed_form_calibration().z(all_masked.score, 0) == 0.0


def test_mask_length_mismatch_is_an_error_not_a_broadcast() -> None:
    g = compute_g_values(IDS50)
    with pytest.raises(ValueError, match="one entry per g-value row"):
        weighted_mean_score(g, np.ones(g.shape[0] - 1, dtype=np.bool_))


# ---------------------------------------------------------------------------
# golden case #6
# ---------------------------------------------------------------------------


def test_end_to_end_score_on_ids50() -> None:
    reading = detect_ids(IDS50, calibration=closed_form_calibration())
    assert reading.score == pytest.approx(GOLDEN_SCORE, abs=TOL)
    assert reading.z == pytest.approx(GOLDEN_Z_CLOSED_FORM, abs=TOL)
    assert reading.n_scored == GOLDEN_N_SCORED
    assert reading.n_tokens == 50
    assert reading.masked_fraction == 0.0
    assert [float(x) for x in reading.result.heat[:3]] == pytest.approx(GOLDEN_HEAT_HEAD, abs=TOL)
    assert reading.cleared is True  # random ids carry no watermark
    assert reading.calibration_source == "closed_form"


def test_end_to_end_score_with_masked_rows() -> None:
    reading = detect_ids(REPEATED_IDS, calibration=closed_form_calibration())
    assert reading.score == pytest.approx(GOLDEN_MASKED_SCORE, abs=TOL)
    assert reading.n_scored == GOLDEN_MASKED_N_SCORED
    assert reading.frame.n_rows == GOLDEN_MASKED_ROWS
    assert reading.masked_fraction == pytest.approx(2 / 11)


def test_token_heat_addresses_the_current_token() -> None:
    reading = detect_ids(REPEATED_IDS, calibration=closed_form_calibration())
    heat = reading.token_heat()
    assert len(heat) == reading.frame.n_rows
    assert [i for i, _h, _m in heat] == list(range(4, 15))
    assert [m for _i, _h, m in heat] == [False] * 8 + [True, True, False]
    assert all(0.0 <= h <= 1.0 for _i, h, _m in heat)


# ---------------------------------------------------------------------------
# properties
# ---------------------------------------------------------------------------


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    ids=st.lists(st.integers(min_value=0, max_value=262143), min_size=0, max_size=90),
    eos=st.booleans(),
)
def test_heat_in_unit_interval_and_contributions_reconstruct_the_score(
    ids: list[int], eos: bool
) -> None:
    frame = compute_frame(ids, eos_token_id=1 if eos else None)
    result = weighted_mean_score(frame.g, frame.mask)

    assert result.heat.shape == (frame.n_rows,)
    assert result.contributions.shape == (frame.n_rows,)
    if frame.n_rows:
        assert float(result.heat.min()) >= 0.0
        assert float(result.heat.max()) <= 1.0
    assert 0.0 <= result.score <= 1.0
    if result.n_scored:
        assert float(result.contributions.sum()) == pytest.approx(result.statistic, abs=1e-12)
        assert float(result.contributions[~frame.mask].sum()) == 0.0
    assert 0.0 <= result.masked_fraction <= 1.0


@settings(max_examples=40, deadline=None)
@given(rows=st.integers(min_value=1, max_value=40), seed=st.integers(0, 2**32 - 1))
def test_score_is_the_mask_weighted_mean_of_heat(rows: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    g = rng.integers(0, 2, size=(rows, 30), dtype=np.uint8)
    mask = rng.integers(0, 2, size=rows, dtype=np.uint8).astype(bool)
    result = weighted_mean_score(g, mask)
    if result.n_scored:
        assert result.score == pytest.approx(float(result.heat[mask].mean()), abs=1e-12)


# ---------------------------------------------------------------------------
# the Protocol, and the drop-in behind it
# ---------------------------------------------------------------------------


def test_both_detectors_satisfy_the_protocol() -> None:
    detectors: list[Detector] = [WeightedMeanDetector(), BayesianDetector()]
    g = compute_g_values(IDS50)
    for det in detectors:
        result = det.score(g)
        assert isinstance(result, DetectorScore)
        assert result.detector == det.name
        assert 0.0 <= result.score <= 1.0
        assert result.heat.shape == (g.shape[0],)
        assert float(result.contributions.sum()) == pytest.approx(result.statistic, abs=1e-9)


def test_bayesian_statistic_is_log_odds_and_is_additive() -> None:
    g = compute_g_values(IDS50)
    result = bayesian_score(g)
    assert result.detector == "bayesian"
    assert float(result.contributions.sum()) == pytest.approx(result.statistic, abs=1e-12)
    # score is the posterior, i.e. sigmoid(prior_log_odds + statistic)
    assert result.score == pytest.approx(1.0 / (1.0 + np.exp(-result.statistic)), abs=1e-12)
    assert (result.heat >= 0.0).all() and (result.heat <= 1.0).all()


def test_bayesian_separates_watermarked_from_null() -> None:
    """An all-ones g-matrix is maximal evidence; all-zeros is maximal counter-evidence."""
    ones = np.ones((40, 30), dtype=np.uint8)
    zeros = np.zeros((40, 30), dtype=np.uint8)
    assert bayesian_score(ones).statistic > 0.0
    assert bayesian_score(zeros).statistic < 0.0
    assert bayesian_score(ones).score > 0.99
    assert bayesian_score(zeros).score < 0.01


def test_bayesian_params_can_be_fitted_and_defaults_are_only_a_shape() -> None:
    rng = np.random.default_rng(17)
    p_true = np.linspace(0.62, 0.51, 30)
    g = (rng.random((4000, 30)) < p_true).astype(np.uint8)
    fitted = fit_params(g)
    assert fitted.source.startswith("mle:")
    assert np.allclose(np.array(fitted.p), p_true, atol=0.03)

    # The default profile is an unmeasured hypothesis with the right SHAPE only.
    default = default_params(30)
    assert default.source == "hypothesis"
    assert all(0.5 < v < 0.55 for v in default.p)
    assert list(default.p) == sorted(default.p, reverse=True)


def test_bayesian_rejects_impossible_parameters() -> None:
    with pytest.raises(ValueError, match=r"must lie strictly inside"):
        default_params(4).__class__(p=(0.5, 1.0), prior_odds=1.0)
    with pytest.raises(ValueError, match="params cover depth"):
        bayesian_score(compute_g_values(IDS50), params=default_params(9))


def test_the_two_detectors_rank_the_same_way() -> None:
    """§4.2: 'weighted mean and Bayesian are within noise'. They must at least
    agree on ordering, or the drop-in is not a drop-in."""
    rng = np.random.default_rng(1234)
    pairs = []
    for p in (0.5, 0.52, 0.55, 0.6):
        g = (rng.random((60, 30)) < p).astype(np.uint8)
        pairs.append((weighted_mean_score(g).score, bayesian_score(g).statistic))
    wm = [a for a, _ in pairs]
    bayes = [b for _, b in pairs]
    assert wm == sorted(wm)
    assert bayes == sorted(bayes)
