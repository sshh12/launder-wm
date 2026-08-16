"""sigma_null(T) -> z. The quantity the needle actually shows.

**The needle displays `z`, not `score`** (TECH_PLAN.md §4.2). The weighted-mean
score lives in about `[0.500, 0.555]` and its threshold moves with length; a
needle showing that is a needle showing noise. `z` is *standard deviations of
watermark evidence above the human baseline*::

    z = (score - 0.5) / sigma_null(n_scored)

with the notch at a **constant** `z* = 2.3263` (FPR 1%). Smooth, monotone,
length-aware, per-token decomposable, and it turns the intro slider's lesson
(`z` grows like `sqrt(T)`) into a literal instrument reading.

The two-part calibration, which is §14.2 item 7 and is the whole point of this
module:

* **The closed form is the SHAPE.** `tau(eps) = 1/2 + Phi_inv(1-eps)/(2*sqrt(m*T))`,
  i.e. `sigma_closed(T) = 1/(2*sqrt(m*T))`.
  Sanity check that it is at least the right *shape*: the paper reports a
  watermarked mean of 0.548 against a 1%-FPR threshold of 0.510, which this
  formula solves at `T ~= 450`, matching their ~400-token texts. That check is
  a test in `test_calibration.py`.

* **`kappa` HAS AN ALGEBRAIC PART AND AN EMPIRICAL PART, AND THE ALGEBRAIC PART
  IS MOST OF IT.** This is the sentence the module got wrong for a while, and
  the correction matters because it changes what invalidates the file.

  The closed form above is the sd of an UNWEIGHTED mean of `m*T` independent
  Bernoulli(1/2)s. The shipped detector takes a WEIGHTED mean,
  `w = linspace(10,1,m)` renormalized so `sum(w) == m` (`detect.weighted_mean`),
  and for independent g-values that has sd

      sigma_weighted(T) = sqrt(sum(w^2)) / (2 * m * sqrt(T))
                        = sqrt(sum(w^2)/m) * sigma_closed(T)

  so `kappa` picks up a CONSTANT factor `WEIGHTING_KAPPA = sqrt(sum(w^2)/m)`
  = 1.1128924 for the shipped weights, with no correlation involved at all. All
  seven shipped buckets measure 0.947-0.998 of exactly that number. A direct
  Monte-Carlo over independent Bernoulli g-values reproduces it to 0.5%
  (`test_calibration.py`).

  **Consequence for whoever edits the weight vector:** changing `w` changes
  `sigma_null` by a factor you can compute in closed form, and every threshold
  and every packed `expected_z` must be regenerated — for algebraic reasons,
  before any question of correlation arises.

  The residual (the 0.947-0.998, i.e. the buckets sitting slightly BELOW the
  analytic prediction) is the empirical part: sampling noise plus whatever
  row-to-row structure real text has. Measured lag-1 autocorrelation of row heat
  on synthetic sequences is about -0.025, i.e. no measurable correlation — which
  is expected, because random token ids have no repeated n-grams. **§14.2 item 7's
  real question — how correlated are the rows in ENGLISH — is still unmeasured,
  and random ids cannot answer it.**

* **The empirical percentile is the LEVEL.** `forge calibrate --fpr 1e-2
  --n 20000 --buckets 40,60,80,120,180,260,400` writes
  `data/assets/thresholds.v1.json`; each bucket carries the measured
  `score_at_fpr` over >= 20,000 null passages. We convert that to an empirical
  sigma and store the ratio `kappa(T) = sigma_empirical / sigma_closed`.
  `sigma(T) = kappa(T) * sigma_closed(T)`, with `kappa` interpolated linearly in
  `log T` and clamped outside the measured range.

  Negatives in that sweep must include **both** human prose **and** Gemma-3
  output generated without the watermark and with a different key. Otherwise
  the calibration encodes "is this Gemma-shaped text", not "does this carry key
  K", and the game's central claim collapses.

Until that file exists we fall back to the closed form, once, with a loud
`logging.warning` naming the command that fixes it. That fallback carries
`WEIGHTING_KAPPA` rather than `kappa = 1`: the weighting factor is *known*, not
measured, and shipping `kappa = 1` overstated every `z` by 5-11% — a 1%-FPR
notch actually running at 2-3% FPR. What remains unverified in the fallback is
the residual, not the algebra. Falling back *silently* would ship a needle whose
numbers are wrong by an unknown factor and look completely reasonable.

scipy is NOT imported here. `launder-core` must import with only pydantic,
blake3 and numpy (§2.1), so `Phi_inv` is implemented locally — Acklam's rational
approximation plus one Halley refinement against `math.erfc`, which lands within
about 1e-15 of scipy's `norm.ppf` across the range we use. `test_calibration.py`
checks it against exact quantiles.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

from launder_core.watermark.config import CANONICAL_CONFIG, data_dir

__all__ = [
    "DEFAULT_FPR",
    "THRESHOLDS_PATH",
    "WEIGHTING_KAPPA",
    "Z_STAR",
    "Calibration",
    "CalibrationBucket",
    "closed_form_calibration",
    "load_thresholds",
    "parse_thresholds",
    "phi",
    "phi_inv",
    "sigma_closed_form",
    "sigma_weighted_closed_form",
    "weighting_kappa",
    "z_from_score",
]

_log = logging.getLogger(__name__)

#: FPR 1%. The number the Nature paper headlines, so "you are beating a detector
#: calibrated at Google's published 1% false-positive rate" is a sentence this
#: project can actually say. 0.1% raises the notch and makes clearing harder for
#: no rhetorical gain (§14.4 decision 2).
DEFAULT_FPR: Final[float] = 0.01

#: `Phi_inv(1 - 0.01)`, rounded as it appears throughout the plan and shipped in
#: `thresholds.v1.json`. The file's value wins if it disagrees; this is the
#: default for the no-file fallback.
Z_STAR: Final[float] = 2.3263

#: Path relative to `data/`.
THRESHOLDS_PATH: Final[str] = "assets/thresholds.v1.json"

# --------------------------------------------------------------------------
# the normal distribution, without scipy
# --------------------------------------------------------------------------

_ACKLAM_A: Final[tuple[float, ...]] = (
    -3.969683028665376e01,
    2.209460984245205e02,
    -2.759285104469687e02,
    1.383577518672690e02,
    -3.066479806614716e01,
    2.506628277459239e00,
)
_ACKLAM_B: Final[tuple[float, ...]] = (
    -5.447609879822406e01,
    1.615858368580409e02,
    -1.556989798598866e02,
    6.680131188771972e01,
    -1.328068155288572e01,
)
_ACKLAM_C: Final[tuple[float, ...]] = (
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e00,
    -2.549732539343734e00,
    4.374664141464968e00,
    2.938163982698783e00,
)
_ACKLAM_D: Final[tuple[float, ...]] = (
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e00,
    3.754408661907416e00,
)
_P_LOW: Final[float] = 0.02425


def phi(x: float) -> float:
    """Standard normal CDF, via `math.erfc`. Exact to double precision."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def phi_inv(p: float) -> float:
    """Standard normal quantile `Phi^-1(p)` for `0 < p < 1`.

    Acklam's rational approximation (relative error < 1.15e-9) refined by one
    Halley step against `erfc`, which takes it to full double precision. This
    exists so `launder-core` does not depend on scipy; `forge` may use scipy and
    `test_calibration.py` pins the two together where scipy is available.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"phi_inv requires 0 < p < 1, got {p}")

    if p < _P_LOW:
        q = math.sqrt(-2.0 * math.log(p))
        x = (
            (
                (((_ACKLAM_C[0] * q + _ACKLAM_C[1]) * q + _ACKLAM_C[2]) * q + _ACKLAM_C[3]) * q
                + _ACKLAM_C[4]
            )
            * q
            + _ACKLAM_C[5]
        ) / ((((_ACKLAM_D[0] * q + _ACKLAM_D[1]) * q + _ACKLAM_D[2]) * q + _ACKLAM_D[3]) * q + 1.0)
    elif p <= 1.0 - _P_LOW:
        q = p - 0.5
        r = q * q
        x = (
            (
                (
                    (((_ACKLAM_A[0] * r + _ACKLAM_A[1]) * r + _ACKLAM_A[2]) * r + _ACKLAM_A[3]) * r
                    + _ACKLAM_A[4]
                )
                * r
                + _ACKLAM_A[5]
            )
            * q
        ) / (
            (
                (((_ACKLAM_B[0] * r + _ACKLAM_B[1]) * r + _ACKLAM_B[2]) * r + _ACKLAM_B[3]) * r
                + _ACKLAM_B[4]
            )
            * r
            + 1.0
        )
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -(
            (
                (((_ACKLAM_C[0] * q + _ACKLAM_C[1]) * q + _ACKLAM_C[2]) * q + _ACKLAM_C[3]) * q
                + _ACKLAM_C[4]
            )
            * q
            + _ACKLAM_C[5]
        ) / ((((_ACKLAM_D[0] * q + _ACKLAM_D[1]) * q + _ACKLAM_D[2]) * q + _ACKLAM_D[3]) * q + 1.0)

    # One Halley step on f(x) = Phi(x) - p.
    err = phi(x) - p
    pdf = math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
    if pdf > 0.0:
        u = err / pdf
        x = x - u / (1.0 + x * u / 2.0)
    return x


# --------------------------------------------------------------------------
# the curve
# --------------------------------------------------------------------------


@lru_cache(maxsize=4)
def weighting_kappa(depth: int = 30) -> float:
    """`sqrt(sum(w^2)/m)` for the shipped tournament weights — 1.1128924 at m=30.

    The exact factor by which the WEIGHTED mean's null sd exceeds the unweighted
    closed form, for independent g-values. No correlation, no measurement: it is
    the Cauchy-Schwarz gap of `linspace(10,1,m)` renormalized to sum to `m`.

    Computed from `depth_weights` rather than hard-coded, so editing the weight
    vector moves the fallback with it instead of leaving a stale literal behind.
    """
    from launder_core.detect.weighted_mean import depth_weights

    w = depth_weights(depth)
    return float(math.sqrt(float((w * w).sum()) / depth))


#: The shipped value, for reading in a traceback. `weighting_kappa()` is the API.
WEIGHTING_KAPPA: Final[float] = weighting_kappa(CANONICAL_CONFIG.depth)


def sigma_weighted_closed_form(n_scored: int | float, depth: int = 30) -> float:
    """`sqrt(sum(w^2)) / (2*m*sqrt(T))` — the closed form for the SHIPPED detector.

    `sigma_closed_form` is the unweighted quantity §4.2 states and the one every
    `kappa` in `thresholds.v1.json` is expressed against; this is what the
    weighted mean actually has under independence. They differ by exactly
    `weighting_kappa(depth)`.
    """
    closed = sigma_closed_form(n_scored, depth)
    if not math.isfinite(closed):
        return math.inf
    return weighting_kappa(depth) * closed


def sigma_closed_form(n_scored: int | float, depth: int = 30) -> float:
    """`1 / (2*sqrt(m*T))` — the independence-assuming null sigma. The SHAPE.

    Equivalent to `tau(eps) = 1/2 + Phi_inv(1-eps) * sigma_closed(T)`, which is
    the form §4.2 and §14.2 state. Returns `inf` for `T <= 0`, so a text with
    nothing scored reads `z = 0` rather than dividing by zero.
    """
    t = float(n_scored)
    if t <= 0.0 or depth <= 0:
        return math.inf
    return 1.0 / (2.0 * math.sqrt(depth * t))


@dataclass(frozen=True)
class CalibrationBucket:
    """One measured point of the null distribution.

    Exactly one of `sigma` or `score_at_fpr` must be present in the JSON; the
    other is derived. `score_at_fpr` is what `forge calibrate` naturally
    produces (the `(1-fpr)` percentile of the null score distribution), and
    `sigma = (score_at_fpr - 0.5) / Phi_inv(1-fpr)` converts it.

    `depth` is carried on the bucket because `kappa` is a RATIO against the
    closed form, and the closed form has an `m` in it. It used to be defaulted
    to 30 here while `Calibration.sigma` divided by `sigma_closed_form(T,
    self.depth)`, so the two `m`s only cancelled while the thresholds file said
    `depth = 30`: a file declaring any other depth reported `sigma` off by
    `sqrt(30/depth)` — a MEASURED sigma silently replaced by a different number
    — with nothing raising. `parse_thresholds` now threads the file's own depth
    through, so the identity `Calibration.sigma(b.n_scored) == b.sigma` holds at
    every depth, which is the only thing a measured bucket can mean.
    """

    n_scored: int
    sigma: float
    n_samples: int = 0
    score_at_fpr: float | None = None
    depth: int = 30

    @property
    def kappa(self) -> float:
        """How many times wider the truth is than the closed form, at this `T`."""
        closed = sigma_closed_form(self.n_scored, self.depth)
        if not math.isfinite(closed) or closed <= 0.0:
            return 1.0
        return self.sigma / closed


@dataclass(frozen=True)
class Calibration:
    """sigma(T) and z*, from measured buckets where available and the formula elsewhere.

    `source` is either `"closed_form"` or `"empirical:<path>"` and is meant to be
    surfaced — a deployment running on the fallback is running on an unverified
    level, and that should be visible rather than inferred.
    """

    depth: int = 30
    z_star: float = Z_STAR
    fpr: float = DEFAULT_FPR
    name: str = "default"
    source: str = "closed_form"
    buckets: tuple[CalibrationBucket, ...] = field(default_factory=tuple)
    wm_config_id: str | None = None
    #: `kappa` when there are no measured buckets. NOT 1.0 for the shipped
    #: detector: `sigma_closed_form` is the UNWEIGHTED closed form and the
    #: weighted mean's null sd is `weighting_kappa(depth)` times it, exactly and
    #: analytically. Defaulting this to 1.0 overstated every fallback `z` by
    #: 5-11% and ran a 1%-FPR notch at a true 2-3%.
    #:
    #: ...and it WAS still defaulted to 1.0, three lines under that sentence.
    #: Both constructors in this module passed `weighting_kappa` explicitly, so
    #: the shipped paths were right and the default was a loaded gun for the
    #: next caller: `Calibration(depth=30)` — the obvious way to write "the
    #: fallback curve" — silently reinstated the bug the comment describes.
    #: `None` now means "the analytic weighting factor for MY depth", which is
    #: the only correct answer, and an explicit float still wins.
    base_kappa: float | None = None

    def __post_init__(self) -> None:
        if self.depth < 1:
            raise ValueError(f"depth must be >= 1, got {self.depth}")
        ns = [b.n_scored for b in self.buckets]
        if ns != sorted(ns):
            raise ValueError(f"calibration buckets must be sorted by n_scored, got {ns}")
        if len(set(ns)) != len(ns):
            raise ValueError(f"duplicate n_scored in calibration buckets: {ns}")

    @property
    def empirical(self) -> bool:
        return bool(self.buckets)

    def kappa(self, n_scored: int | float) -> float:
        """Correction factor at `T`: linear in `log T`, clamped outside the buckets.

        Clamping rather than extrapolating is deliberate. Beyond the measured
        range the closed form's *shape* is the only thing we trust, so we keep
        the nearest measured *level* and let `sqrt(T)` do the rest. Extrapolating
        a fitted slope past 400 tokens would invent precision nobody measured.
        """
        if not self.buckets:
            return weighting_kappa(self.depth) if self.base_kappa is None else self.base_kappa
        t = max(float(n_scored), 1.0)
        if len(self.buckets) == 1 or t <= self.buckets[0].n_scored:
            return self.buckets[0].kappa
        if t >= self.buckets[-1].n_scored:
            return self.buckets[-1].kappa
        lo = self.buckets[0]
        for hi in self.buckets[1:]:
            if t <= hi.n_scored:
                span = math.log(hi.n_scored) - math.log(lo.n_scored)
                frac = 0.0 if span == 0.0 else (math.log(t) - math.log(lo.n_scored)) / span
                return lo.kappa + frac * (hi.kappa - lo.kappa)
            lo = hi
        return self.buckets[-1].kappa  # pragma: no cover - unreachable, bounded above

    def sigma(self, n_scored: int | float) -> float:
        """The null standard deviation of `score` at `n_scored` scored positions."""
        closed = sigma_closed_form(n_scored, self.depth)
        if not math.isfinite(closed):
            return math.inf
        return self.kappa(n_scored) * closed

    def z(self, score: float, n_scored: int | float) -> float:
        """`(score - 0.5) / sigma(n_scored)`. `0.0` when nothing was scored.

        A text with no scored positions carries no evidence either way, and
        `z = 0` is exactly that statement. Returning `inf` or `nan` here would
        propagate into the needle, the gate and the leaderboard.
        """
        s = self.sigma(n_scored)
        if not math.isfinite(s) or s <= 0.0:
            return 0.0
        return (score - 0.5) / s

    def threshold_score(self, n_scored: int | float, fpr: float | None = None) -> float:
        """`tau(eps) = 1/2 + Phi_inv(1-eps)*sigma(T)` — the score the notch sits at.

        With the default `fpr` this is `0.5 + z_star*sigma(T)`; passing an
        explicit `fpr` recomputes the multiplier, which is what makes the
        "sanity check the shape" test in `test_calibration.py` expressible.
        """
        s = self.sigma(n_scored)
        if not math.isfinite(s):
            return 0.5
        mult = self.z_star if fpr is None else phi_inv(1.0 - fpr)
        return 0.5 + mult * s

    def cleared(self, score: float, n_scored: int | float) -> bool:
        """True when the passage reads *below* the notch — the win condition."""
        return self.z(score, n_scored) < self.z_star


def closed_form_calibration(depth: int | None = None, z_star: float = Z_STAR) -> Calibration:
    """The no-file fallback: correct shape, correct WEIGHTING, unverified residual.

    Carries `base_kappa = weighting_kappa(depth)`. That factor is algebra, not a
    measurement — the weighted mean's null sd under independence — and shipping
    `kappa = 1` here made every fallback `z` 5-11% too large, i.e. a notch
    labelled 1% FPR running at roughly 2-3%. What is genuinely unverified
    without `thresholds.v1.json` is the residual on top of it.
    """
    m = CANONICAL_CONFIG.depth if depth is None else depth
    return Calibration(
        depth=m,
        z_star=z_star,
        source="closed_form",
        base_kappa=weighting_kappa(m),
    )


def _bucket_from_json(raw: dict[str, Any], fpr: float, depth: int) -> CalibrationBucket:
    n_scored = int(raw["n_scored"])
    score_at_fpr = raw.get("score_at_fpr")
    sigma = raw.get("sigma")
    if sigma is None:
        if score_at_fpr is None:
            raise ValueError(
                f"calibration bucket n_scored={n_scored} has neither 'sigma' nor "
                "'score_at_fpr'; one of them is what makes it a measurement"
            )
        mult = phi_inv(1.0 - fpr)
        sigma = (float(score_at_fpr) - 0.5) / mult
    sigma = float(sigma)
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError(f"calibration bucket n_scored={n_scored} has non-positive sigma {sigma}")
    return CalibrationBucket(
        n_scored=n_scored,
        sigma=sigma,
        n_samples=int(raw.get("n_samples", 0)),
        score_at_fpr=None if score_at_fpr is None else float(score_at_fpr),
        depth=depth,
    )


def _assert_no_shadow(group: dict[str, Any], key: str, top: float, name: str) -> None:
    """A calibration group may repeat `z_star`/`fpr`, never contradict it."""
    if key not in group:
        return
    value = float(group[key])
    if value != top:
        raise ValueError(
            f"thresholds file: calibrations.{name}.{key} = {value!r} contradicts the "
            f"top-level {key} = {top!r}. The top-level value is THE knob (TECH_PLAN.md "
            f"§12 row 8: 'edit {key}, redeploy'); a group key that quietly overrode it "
            "made the documented edit inert and hid the real control one level down. "
            f"Delete calibrations.{name}.{key}, or change both."
        )


def parse_thresholds(raw: dict[str, Any], name: str, source: str) -> Calibration:
    """Build a `Calibration` from the parsed contents of `thresholds.v1.json`.

    File shape (written by `forge calibrate`)::

        {
          "schema": "launder.thresholds/1",
          "wm_config_id": "wm1:...",
          "z_star": 2.3263,
          "fpr": 0.01,
          "depth": 30,
          "calibrations": {
            "default": {"buckets": [
               {"n_scored": 40,  "score_at_fpr": 0.5213, "n_samples": 20000},
               ...
            ]},
            "code": {"buckets": [...]}         # L5: code is its own bucket (§7.6)
          }
        }

    A flat top-level `"buckets"` list is also accepted and treated as
    `"default"`, because that is the smaller thing to hand-write when checking a
    hypothesis.

    **THE TOP-LEVEL `z_star` AND `fpr` ARE THE KNOBS, AND THEY ARE NOT
    SHADOWABLE.** §12 row 8 documents the notch as "edit `z_star` in
    `data/assets/thresholds.v1.json`, redeploy". A group key of the same name
    used to override it silently, so the documented edit did nothing and the
    real control was an undocumented nested key. A group may REPEAT the values
    (the shipped file does, and `forge calibrate` writes them) but disagreeing
    with the top level is a load failure, here and in
    `web/src/detector/calibration.ts`, which enforces the same rule.

    What IS legitimately per-group is `buckets`: L5 fits its own sigma(T) curve
    because code has far fewer scored tokens (§7.6 deviation #2). The notch is
    an FPR, and an FPR is global.
    """
    fpr = float(raw.get("fpr", DEFAULT_FPR))
    z_star = float(raw.get("z_star", Z_STAR))
    depth = int(raw.get("depth", CANONICAL_CONFIG.depth))

    groups = raw.get("calibrations")
    if isinstance(groups, dict):
        if name not in groups:
            raise KeyError(
                f"calibration bucket set {name!r} not found in thresholds file; "
                f'it has {sorted(groups)}. Levels select one via `calibration = "..."` '
                "in data/config/levels.toml."
            )
        group = groups[name]
        entries = group.get("buckets", [])
        _assert_no_shadow(group, "fpr", fpr, name)
        _assert_no_shadow(group, "z_star", z_star, name)
    else:
        if name != "default":
            raise KeyError(
                f"thresholds file has a flat 'buckets' list and therefore only a 'default' "
                f"calibration, but {name!r} was requested"
            )
        entries = raw.get("buckets", [])

    buckets = tuple(
        sorted(
            (_bucket_from_json(e, fpr, depth) for e in entries),
            key=lambda b: b.n_scored,
        )
    )
    return Calibration(
        depth=depth,
        z_star=z_star,
        fpr=fpr,
        name=name,
        source=source if buckets else "closed_form",
        buckets=buckets,
        wm_config_id=raw.get("wm_config_id"),
        # Only consulted when `buckets` is empty, i.e. when this file degenerated
        # to the fallback anyway.
        base_kappa=weighting_kappa(depth),
    )


@lru_cache(maxsize=8)
def load_thresholds(
    path: Path | str | None = None,
    *,
    name: str = "default",
    required: bool = False,
) -> Calibration:
    """Load `data/assets/thresholds.v1.json`, or fall back to the closed form.

    `required=True` turns the missing file into an error instead of a warning —
    what `forge verify` and any calibrated production boot should pass, so a
    deploy cannot quietly ship the unverified level.

    Result is cached: this is read on every `/api/detect` and the file never
    changes within a process.
    """
    p = Path(path) if path is not None else data_dir() / THRESHOLDS_PATH
    if not p.is_file():
        msg = (
            f"calibration file {p} does not exist. Falling back to the CLOSED-FORM null "
            "sigma = 1/(2*sqrt(m*T)), which assumes all m*T g-values are independent. The 30 "
            "tournament layers are strongly correlated, so the true null sigma is several "
            "times larger and every z this process reports will be OVERSTATED by an unknown "
            "factor (TECH_PLAN.md sec 14.2 item 7). The shape is right; the level is not "
            "measured. Fix: uv run forge calibrate --fpr 1e-2 --n 20000 "
            "--buckets 40,60,80,120,180,260,400 --out data/assets/thresholds.v1.json"
        )
        if required:
            raise FileNotFoundError(msg)
        _log.warning("%s", msg)
        return closed_form_calibration()

    raw: dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
    schema = raw.get("schema")
    if schema is not None and schema != "launder.thresholds/1":
        raise ValueError(f"{p} declares schema {schema!r}, expected 'launder.thresholds/1'")
    return parse_thresholds(raw, name=name, source=f"empirical:{p}")


def z_from_score(
    score: float,
    n_scored: int | float,
    calibration: Calibration | None = None,
) -> float:
    """`z` for a score. Loads the shipped calibration when none is passed."""
    cal = load_thresholds() if calibration is None else calibration
    return cal.z(score, n_scored)
