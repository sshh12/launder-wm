"""The Bayesian drop-in. Real, tested, and unused in v1 — by decision, not by default.

TECH_PLAN.md §4.2 rules for the weighted mean and keeps this behind the same
`Detector` Protocol: "Bayesian stays as a 4 KB drop-in behind the same `Detector`
Protocol, unused in v1. The Bayesian *posterior* is unusable on a needle — it
saturates at `1.0000000` for eight edits and then falls off a cliff — which is
an argument for that ruling, not against it."

This is the training-free likelihood ratio, not HF's learned
`BayesianDetectorModel`. Under the null every g-value is a fair coin; under the
watermark, depth `L` has `P(g=1) = p[L] > 1/2`. The evidence is additive::

    llr[i] = sum_L [ g log(p_L / 0.5) + (1-g) log((1-p_L) / 0.5) ]
    statistic = sum_i mask[i] * llr[i]                 # log-odds of evidence
    score     = sigmoid(prior_log_odds + statistic)    # the posterior

`statistic`, not `score`, is what `contributions` sums to — a posterior is not
additive over tokens, and reporting one as if it were is precisely the kind of
plausible-looking lie §4.3 is about. `heat[i] = sigmoid(llr[i])` keeps the
Protocol's "0.5 is neutral, range [0,1]" contract while remaining this
detector's *own* honest decomposition rather than a borrowed one.

**The default `p_L` are an unmeasured hypothesis and say so.** `forge calibrate`
would fit them with `fit_params()` over watermarked output; until then
`default_params()` returns a monotone decreasing profile whose only defence is
that it has the right qualitative shape (early tournament layers carry more
signal). Do not read a number out of this file and treat it as measured. The
ranking between passages is nearly insensitive to the profile; the absolute
posterior is not, which is the saturation problem in one sentence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np

from launder_core.detect.weighted_mean import BoolArray, ByteArray, DetectorScore, FloatArray

__all__ = [
    "BayesianDetector",
    "BayesianParams",
    "bayesian_score",
    "default_params",
    "fit_params",
]

#: Depth-0 excess over a fair coin, and the per-depth decay. HYPOTHESES.
_DEFAULT_DELTA0: Final[float] = 0.03
_DEFAULT_DECAY: Final[float] = 0.97
_EPS: Final[float] = 1e-6


@dataclass(frozen=True)
class BayesianParams:
    """Per-depth `P(g=1 | watermarked)` plus the prior odds of "watermarked"."""

    p: tuple[float, ...]
    prior_odds: float = 1.0
    source: str = "hypothesis"

    def __post_init__(self) -> None:
        if not self.p:
            raise ValueError("BayesianParams.p must have one entry per tournament depth")
        for i, v in enumerate(self.p):
            if not 0.0 < v < 1.0:
                raise ValueError(f"p[{i}] = {v} must lie strictly inside (0, 1)")
        if self.prior_odds <= 0.0:
            raise ValueError(f"prior_odds must be > 0, got {self.prior_odds}")

    @property
    def depth(self) -> int:
        return len(self.p)

    @property
    def prior_log_odds(self) -> float:
        return math.log(self.prior_odds)


def default_params(depth: int = 30, prior_odds: float = 1.0) -> BayesianParams:
    """`p[L] = 0.5 + 0.03 * 0.97**L`. Shape only — nobody has measured these."""
    return BayesianParams(
        p=tuple(0.5 + _DEFAULT_DELTA0 * _DEFAULT_DECAY**level for level in range(depth)),
        prior_odds=prior_odds,
        source="hypothesis",
    )


def fit_params(
    g: ByteArray,
    mask: BoolArray | None = None,
    prior_odds: float = 1.0,
) -> BayesianParams:
    """Maximum-likelihood `p[L]` from a watermarked corpus: the per-depth mean g.

    This is what makes the detector real rather than decorative — point it at
    `forge`'s watermarked candidate pool and the parameters stop being a guess.
    Clipped away from 0 and 1 so a depth that happens to be constant in a small
    sample cannot produce an infinite log-odds.
    """
    gm = np.ascontiguousarray(g, dtype=np.float64)
    if gm.ndim != 2:
        raise ValueError(f"g must be a 2-D (rows, depth) matrix, got shape {gm.shape}")
    if mask is not None:
        msk = np.ascontiguousarray(mask, dtype=np.bool_)
        if msk.shape != (gm.shape[0],):
            raise ValueError(f"mask has shape {msk.shape}, expected ({gm.shape[0]},)")
        gm = gm[msk]
    if gm.shape[0] == 0:
        raise ValueError("cannot fit Bayesian parameters from zero scored rows")
    means = np.clip(gm.mean(axis=0), _EPS, 1.0 - _EPS)
    return BayesianParams(
        p=tuple(float(v) for v in means),
        prior_odds=prior_odds,
        source=f"mle:{gm.shape[0]}rows",
    )


def _sigmoid(x: FloatArray) -> FloatArray:
    out: FloatArray = 0.5 * (1.0 + np.tanh(0.5 * x))
    return out


def bayesian_score(
    g: ByteArray,
    mask: BoolArray | None = None,
    params: BayesianParams | None = None,
) -> DetectorScore:
    """Log-odds of the watermark, decomposed per row. Same result shape as the weighted mean."""
    gm = np.ascontiguousarray(g, dtype=np.float64)
    if gm.ndim != 2:
        raise ValueError(f"g must be a 2-D (rows, depth) matrix, got shape {gm.shape}")
    rows, depth = gm.shape
    par = default_params(depth) if params is None else params
    if par.depth != depth:
        raise ValueError(f"params cover depth {par.depth} but g has depth {depth}")

    if mask is None:
        msk = np.ones(rows, dtype=np.bool_)
    else:
        msk = np.ascontiguousarray(mask, dtype=np.bool_)
        if msk.shape != (rows,):
            raise ValueError(f"mask has shape {msk.shape}, expected ({rows},)")

    p = np.asarray(par.p, dtype=np.float64)
    w_one = np.log(p / 0.5)
    w_zero = np.log((1.0 - p) / 0.5)

    llr: FloatArray = gm @ w_one + (1.0 - gm) @ w_zero
    contrib: FloatArray = np.where(msk, llr, 0.0)
    statistic = float(contrib.sum())
    posterior = float(_sigmoid(np.array([par.prior_log_odds + statistic]))[0])

    return DetectorScore(
        detector="bayesian",
        score=posterior,
        statistic=statistic,
        n_scored=int(msk.sum()),
        n_rows=rows,
        heat=_sigmoid(llr),
        contributions=contrib,
    )


class BayesianDetector:
    """`Detector` implementation. Structurally identical to `WeightedMeanDetector`.

    Swapping detectors is therefore a one-line change at the call site, which is
    the entire reason this file exists: the ruling in §4.2 is reversible without
    touching the API, the wire schema or the renderer.
    """

    name: Final[str] = "bayesian"

    def __init__(self, params: BayesianParams | None = None) -> None:
        self._params = params

    def score(self, g: ByteArray, mask: BoolArray | None = None) -> DetectorScore:
        return bayesian_score(g, mask, self._params)
