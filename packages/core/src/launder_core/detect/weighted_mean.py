"""The shipped detector: weighted mean over the 30 tournament layers.

    w        = linspace(10, 1, m);  w *= m / sum(w)      # so sum(w) == m
    heat[i]  = sum_L w[L]*g[i][L] / m                    # in [0,1], 0.5 neutral
    score    = sum_i mask[i]*heat[i] / sum_i mask[i]
    n_scored = sum_i mask[i]

Why this detector (TECH_PLAN.md §4.2): zero training, zero negative corpus,
**honest per-token attribution** — the score *is* a sum of per-token
contributions — and a closed-form null shape. The independent analysis
(arXiv:2603.03410) shows TPR under the mean score is unimodal in `m` and peaks
at `m ~= 28`; the stock config's `m = 30` sits essentially at the peak, so
weighted mean and Bayesian are within noise.

The per-token decomposition is not a visualization convenience, it is
mechanism 1 of §4.6 — the structural reason the game can never teach "spot the
fancy AI word". `contributions[i]` is literally the amount `score` would lose if
row `i` stopped contributing, and `sum(contributions) == score` is asserted as a
property test. Heat is the live detector contribution or it is nothing: it is a
function of token ids and the published key, recomputed every keystroke. It is
not entropy, not perplexity, not rarity, and the browser could not colour by
those if it wanted to, because it does not have the numbers.

`DetectorScore` is declared here rather than in `__init__` because the shipped
detector owns the result shape and `bayesian.py` conforms to it, which keeps the
import graph acyclic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

__all__ = [
    "DetectorScore",
    "WeightedMeanDetector",
    "contributions",
    "depth_weights",
    "weighted_mean_score",
]

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
ByteArray = npt.NDArray[np.uint8]

#: Neutral score: what an unwatermarked text reads, and what an empty text must
#: report. Returning 0.0 for "nothing to score" would render as a needle pinned
#: at maximum evidence *against* the watermark, which is a lie about a text the
#: detector has no opinion on.
NEUTRAL: Final[float] = 0.5


@dataclass(frozen=True)
class DetectorScore:
    """One detector's reading of one exact token sequence.

    Fields that every `Detector` must fill, and the contract on each:

    * `score` — in `[0, 1]`, `0.5` neutral. What the API returns. For the
      weighted mean this is the mean g-value; for a Bayesian detector it is a
      posterior.
    * `statistic` — the raw *additive* quantity the detector accumulates.
      `sum(contributions) == statistic` for every detector. For the weighted
      mean `statistic == score`; for the Bayesian detector it is the log-odds,
      because a posterior is not additive over tokens and pretending otherwise
      would make the mirror lie.
    * `heat` — length `R`, in `[0, 1]`, `0.5` neutral. THE mirror value.
    * `contributions` — length `R`, sums to `statistic`. Masked rows are exactly
      `0.0`, which is the honest rendering of "this position is free".
    """

    detector: str
    score: float
    statistic: float
    n_scored: int
    n_rows: int
    heat: FloatArray
    contributions: FloatArray

    @property
    def masked_fraction(self) -> float:
        if self.n_rows == 0:
            return 0.0
        return 1.0 - self.n_scored / self.n_rows


def depth_weights(m: int) -> FloatArray:
    """`linspace(10, 1, m)` renormalized so `sum(w) == m`.

    Earlier tournament layers carry more signal, so they weigh 10x the last.
    The renormalization is what keeps `heat` in `[0, 1]` and keeps the score on
    the same scale as an unweighted mean, so the `0.5` neutral point and the
    published `tau` threshold still mean what they say.
    """
    if m < 1:
        raise ValueError(f"depth m must be >= 1, got {m}")
    if m == 1:
        return np.ones(1, dtype=np.float64)
    w = np.linspace(10.0, 1.0, m, dtype=np.float64)
    w *= m / w.sum()
    return w


def _prepare(
    g: ByteArray,
    mask: BoolArray | None,
    weights: FloatArray | None,
) -> tuple[FloatArray, BoolArray, FloatArray]:
    gm = np.ascontiguousarray(g, dtype=np.float64)
    if gm.ndim != 2:
        raise ValueError(f"g must be a 2-D (rows, depth) matrix, got shape {gm.shape}")
    rows, m = gm.shape
    w = depth_weights(m) if weights is None else np.ascontiguousarray(weights, dtype=np.float64)
    if w.shape != (m,):
        raise ValueError(f"weights have shape {w.shape}, expected ({m},)")
    if mask is None:
        msk = np.ones(rows, dtype=np.bool_)
    else:
        msk = np.ascontiguousarray(mask, dtype=np.bool_)
        if msk.shape != (rows,):
            raise ValueError(
                f"mask has shape {msk.shape}, expected ({rows},) — one entry per g-value row. "
                "A mask computed for a different text is the one way to get a plausible "
                "number that means nothing."
            )
    return gm, msk, w


def _heat_from(gm: FloatArray, w: FloatArray) -> FloatArray:
    """`heat[i] = sum_L w[L]*g[i][L] / m`, CLIPPED into `[0, 1]`.

    THE CLIP IS NOT COSMETIC. `depth_weights` renormalizes so that `w.sum()`
    is exactly `m`, but `gm @ w` is a different summation order, and for an
    all-ones row it lands on `30.000000000000004` — so `heat` came out as
    `1.0000000000000002`. That breaks the contract `DetectorScore` states in
    its own docstring, and it breaks it where it costs the most: `TokenHeat.heat`
    is `Field(ge=0.0, le=1.0)`, so `/api/detect` raised a pydantic
    ValidationError — a 500 on every keystroke — for any text containing one
    n-gram whose thirty tournament layers all read 1. That is roughly 1e-9 per
    row and therefore certain to happen eventually, and when it does it is
    permanent for that passage rather than intermittent.

    Clipping rather than rescaling: the true value IS 1 (or 0) at those two
    extremes, and 2e-16 is float noise, not evidence. The clip can only ever
    move a value that was already outside a closed interval it is proved to be
    inside.
    """
    m = gm.shape[1]
    out: FloatArray = (gm @ w) / m
    np.clip(out, 0.0, 1.0, out=out)
    return out


def heat_values(
    g: ByteArray,
    weights: FloatArray | None = None,
) -> FloatArray:
    """`heat[i] = sum_L w[L]*g[i][L] / m`, for every row including masked ones.

    Masked rows still get a heat value: the mirror renders them grey *and* the
    solver needs to know what they would have been worth. Whether a position
    counts is the mask's job, not heat's.
    """
    gm, _msk, w = _prepare(g, None, weights)
    return _heat_from(gm, w)


def contributions(
    g: ByteArray,
    mask: BoolArray | None = None,
    weights: FloatArray | None = None,
) -> FloatArray:
    """Per-row contribution to the score. Sums to the score exactly (§4.6 #1)."""
    return weighted_mean_score(g, mask, weights).contributions


def weighted_mean_score(
    g: ByteArray,
    mask: BoolArray | None = None,
    weights: FloatArray | None = None,
) -> DetectorScore:
    """Score a g-value matrix. `mask` defaults to "every row counts"."""
    gm, msk, w = _prepare(g, mask, weights)
    rows, _m = gm.shape

    heat: FloatArray = _heat_from(gm, w)
    n_scored = int(msk.sum())

    if n_scored == 0:
        return DetectorScore(
            detector="weighted_mean",
            score=NEUTRAL,
            statistic=NEUTRAL,
            n_scored=0,
            n_rows=rows,
            heat=heat,
            contributions=np.zeros(rows, dtype=np.float64),
        )

    contrib: FloatArray = np.where(msk, heat, 0.0) / n_scored
    # Clamped for the same reason `heat` is, and it is a SEPARATE failure: a
    # mean of clipped heats is still summed in floating point, so `n_scored`
    # copies of `1/n_scored` add up to 1.0000000000000002 for 73 of the first
    # 400 values of `n_scored`. `DetectorReading.score` and `DetectResponse.score`
    # are both `Field(ge=0.0, le=1.0)`. The clamp moves the value by at most one
    # ulp, so `sum(contributions) == statistic` still holds to 1e-15 — far inside
    # the tolerance the §4.6 decomposition property is asserted at.
    score = min(1.0, max(0.0, float(contrib.sum())))
    return DetectorScore(
        detector="weighted_mean",
        score=score,
        statistic=score,
        n_scored=n_scored,
        n_rows=rows,
        heat=heat,
        contributions=contrib,
    )


class WeightedMeanDetector:
    """The `Detector` implementation that ships. Stateless; the weights are cached.

    Structurally typed against `launder_core.detect.Detector` — it does not
    inherit from the Protocol, and `test_detect.py` asserts conformance, so the
    Bayesian drop-in cannot diverge in shape.
    """

    name: Final[str] = "weighted_mean"

    def __init__(self, weights: FloatArray | None = None) -> None:
        self._weights = weights

    def score(self, g: ByteArray, mask: BoolArray | None = None) -> DetectorScore:
        return weighted_mean_score(g, mask, self._weights)
