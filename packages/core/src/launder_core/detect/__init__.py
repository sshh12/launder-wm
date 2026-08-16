"""The detector: score -> per-token heat -> z.

One Protocol, two implementations, one calibration. `detect_ids()` is the whole
pipeline in one call and is what `/api/detect` and `forge verify` both use, so
the server and the golden vectors cannot drift apart by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy.typing as npt

from launder_core.detect.bayesian import (
    BayesianDetector,
    BayesianParams,
    bayesian_score,
    default_params,
    fit_params,
)
from launder_core.detect.calibration import (
    DEFAULT_FPR,
    Z_STAR,
    Calibration,
    CalibrationBucket,
    closed_form_calibration,
    load_thresholds,
    phi,
    phi_inv,
    sigma_closed_form,
    z_from_score,
)
from launder_core.detect.weighted_mean import (
    BoolArray,
    ByteArray,
    DetectorScore,
    WeightedMeanDetector,
    contributions,
    depth_weights,
    heat_values,
    weighted_mean_score,
)
from launder_core.watermark.config import (
    CANONICAL_CONFIG,
    SCORING_EOS_TOKEN_ID,
    SynthIDConfig,
)
from launder_core.watermark.gvalues import WatermarkFrame, compute_frame

__all__ = [
    "DEFAULT_FPR",
    "Z_STAR",
    "BayesianDetector",
    "BayesianParams",
    "Calibration",
    "CalibrationBucket",
    "Detector",
    "DetectorScore",
    "Reading",
    "WeightedMeanDetector",
    "bayesian_score",
    "closed_form_calibration",
    "contributions",
    "default_params",
    "depth_weights",
    "detect_frame",
    "detect_ids",
    "fit_params",
    "heat_values",
    "load_thresholds",
    "phi",
    "phi_inv",
    "sigma_closed_form",
    "weighted_mean_score",
    "z_from_score",
]


class Detector(Protocol):
    """What a detector must be to be swappable.

    Structural, not nominal: neither `WeightedMeanDetector` nor
    `BayesianDetector` inherits from this, and `test_detect.py` asserts both
    satisfy it. `DetectorScore`'s contract — `score` in `[0,1]` with `0.5`
    neutral, `heat` in `[0,1]` per row, `contributions` summing to `statistic` —
    is the part that actually has to hold.
    """

    # Read-only on purpose: implementations declare it `Final`, and a Protocol
    # spelled `name: str` would demand a *settable* attribute and reject them.
    @property
    def name(self) -> str: ...

    def score(
        self,
        g: ByteArray,
        mask: BoolArray | None = None,
    ) -> DetectorScore: ...


@dataclass(frozen=True)
class Reading:
    """A complete detector reading of one token sequence: the needle's whole input."""

    frame: WatermarkFrame
    result: DetectorScore
    z: float
    z_star: float
    calibration_source: str

    @property
    def score(self) -> float:
        return self.result.score

    @property
    def n_scored(self) -> int:
        return self.result.n_scored

    @property
    def n_tokens(self) -> int:
        return self.frame.n_tokens

    @property
    def masked_fraction(self) -> float:
        return self.frame.masked_fraction

    @property
    def cleared(self) -> bool:
        """Below the notch. The win condition, and the only place it is defined."""
        return self.z < self.z_star

    def token_heat(self) -> list[tuple[int, float, bool]]:
        """`(token_index, heat, masked)` per scored row, in token order.

        The first `ngram_len - 1` tokens are the final token of no window and so
        appear in no row: they carry no heat and are rendered neutral. Char
        offsets — what `TokenHeat` actually carries on the wire — are the
        tokenizer's business and belong to the caller that owns it.
        """
        n = self.frame.ngram_len
        return [
            (i + n - 1, float(self.result.heat[i]), not bool(self.frame.mask[i]))
            for i in range(self.frame.n_rows)
        ]


def detect_frame(
    frame: WatermarkFrame,
    detector: Detector | None = None,
    calibration: Calibration | None = None,
) -> Reading:
    """Score an already-computed frame. Use this when the frame is cached."""
    det: Detector = detector if detector is not None else WeightedMeanDetector()
    cal = load_thresholds() if calibration is None else calibration
    result = det.score(frame.g, frame.mask)
    return Reading(
        frame=frame,
        result=result,
        z=cal.z(result.score, result.n_scored),
        z_star=cal.z_star,
        calibration_source=cal.source,
    )


def detect_ids(
    ids: npt.ArrayLike,
    config: SynthIDConfig = CANONICAL_CONFIG,
    *,
    detector: Detector | None = None,
    calibration: Calibration | None = None,
    eos_token_id: int | None = SCORING_EOS_TOKEN_ID,
    table: ByteArray | None = None,
) -> Reading:
    """Token ids in, complete reading out. The one call the server makes.

    `eos_token_id` defaults to the ONE shipped policy
    (`launder_core.watermark.config.SCORING_EOS_TOKEN_ID`), which the browser's
    `detect()` also defaults to. Do not leave it implicit in a NEW call site
    expecting a different value: the two runtimes disagreeing about this default
    is what once produced z 1.971 and z 0.109 for the same sentence.
    """
    frame = compute_frame(ids, config, table=table, eos_token_id=eos_token_id)
    return detect_frame(frame, detector, calibration)
