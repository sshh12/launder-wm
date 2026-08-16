"""Check — "too clean is also a tell".

The win condition (`detector_threshold`) is a ceiling: get the reading at or
below the notch. This is the matching FLOOR, and together they make the level a
window rather than a direction.

**Why this is honest and not an arbitrary difficulty knob.** `z` is standardized
against the null distribution of un-watermarked text, so human prose scatters
around zero and a reading of -3 is exactly as improbable under that null as +3.
A submission that lands far below zero is not "clean"; it is anti-correlated
with the key, which is itself evidence that somebody knew the key and steered
away from it. Real detectors are two-sided for this reason. Telling the player
"you scrubbed so hard you look guilty" is a true statement about the statistic,
which is the bar every piece of copy in this product has to clear.

**What it does to the game.** Every automated attack on this game is a hill
climb that minimizes z and stops at the first value under the line. A floor
turns that into a targeting problem: overshoot and you fail, so "delete words
until it goes quiet" stops being a strategy and precision starts being one.

`min_z` is expressed RELATIVE TO THE NOTCH, exactly like `detector_threshold`'s
`max_z`, so a level's two bounds are read off the same origin and a change to
the calibration moves both together. A level clears when
`min_z <= z - z_star <= max_z`.

ORDER THIS BEFORE `detector_threshold` in levels.toml — `validate_level`
requires the threshold to sit IMMEDIATELY before `llm_gate`, which is what keeps
submissions that have not beaten the watermark away from the paid judge, so
nothing may be slipped between them. That ordering means the floor can fail
before the threshold ever runs, which is why this check reports the full reading
in its `meta` and `/api/submit` accepts either check as the source: a player
rejected for over-scrubbing must be shown the reading that rejected them, not a
zero.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from launder_core.gates.registry import GateContext, GateDependencyError, Phase, register
from launder_core.readout import format_points
from launder_core.schemas import CheckResult

__all__ = ["DetectorFloor"]


@register
class DetectorFloor:
    name: ClassVar[str] = "detector_floor"
    phase: ClassVar[int] = Phase.DETECTOR
    fail_open: ClassVar[bool] = False
    config_params: ClassVar[frozenset[str]] = frozenset({"min_z", "calibration"})
    required_params: ClassVar[frozenset[str]] = frozenset({"min_z"})
    template_params: ClassVar[frozenset[str]] = frozenset(
        {"z_display", "z_star_display", "z_floor_display"}
    )
    required_deps: ClassVar[frozenset[str]] = frozenset({"detector"})

    def __call__(self, ctx: GateContext, params: Mapping[str, Any]) -> CheckResult:
        detector = ctx.deps.detector
        if detector is None:
            raise GateDependencyError(
                "detector_floor ran with no detector in Deps. Passing it by default would "
                "silently remove the floor and turn the level back into a one-sided race to "
                "zero. Wire launder_core.detect into Deps.detector."
            )
        min_z = float(params["min_z"])
        calibration = params.get("calibration")
        # RAW, not normalized — the same rule the threshold follows, and for the
        # same reason: whitespace folding changes tokenization, and the floor
        # has to be measured on the text the player was watching.
        reading = detector.read(
            ctx.raw, ctx.passage, str(calibration) if calibration is not None else None
        )
        # Points, for the same reason `detector_threshold` renders points: the
        # message sits under a meter that prints them.
        rendered = {
            "z_display": format_points(reading.z, reading.z_star),
            "z_star_display": format_points(reading.z_star, reading.z_star),
            "z_floor_display": format_points(reading.z_star + min_z, reading.z_star),
        }
        meta = {
            "score": reading.score,
            "z": reading.z,
            "z_star": reading.z_star,
            "n_scored": reading.n_scored,
            "masked_fraction": reading.masked_fraction,
            "z_floor": reading.z_star + min_z,
        }
        if reading.z - reading.z_star < min_z:
            return CheckResult(
                status="fail", check=self.name, code=self.name, params=rendered, meta=meta
            )
        return CheckResult(status="pass", check=self.name, params=rendered, meta=meta)
