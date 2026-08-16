"""Check 7 — TECH_PLAN.md §7.1 row 7. **The actual win condition.**

Recomputed server-side, always. The client's needle is a preview and is never
trusted — which is precisely what makes publishing the watermark keys costless
to integrity (§1): a client that lies about its needle is lying only to itself.

Its position in the pipeline is the single largest cost saver in the product.
Only submissions that ALREADY beat the watermark ever reach the paid judge, so
the common failure — a rewrite that reads beautifully and is still detected —
costs about three milliseconds of CPU and nothing else.

`max_z` is relative to the notch: the level clears at `z - z_star <= max_z`,
and levels.toml ships `max_z = 0.0`, i.e. "at or below z*".
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from launder_core.gates.registry import GateContext, GateDependencyError, Phase, register
from launder_core.schemas import CheckResult

__all__ = ["DetectorThreshold", "format_z"]


def format_z(z: float) -> str:
    """The one place a z becomes a string a player reads.

    One decimal below 10, none above, because "the needle reads 41.7" and "the
    needle reads 42" are both readable but "41.72834" is a number nobody can
    place (§10.7). The same function renders the threshold, so the two can
    never be formatted differently on the same line.
    """
    return f"{z:.0f}" if abs(z) >= 10.0 else f"{z:.1f}"


@register
class DetectorThreshold:
    name: ClassVar[str] = "detector_threshold"
    phase: ClassVar[int] = Phase.DETECTOR
    fail_open: ClassVar[bool] = False
    config_params: ClassVar[frozenset[str]] = frozenset({"max_z", "calibration"})
    required_params: ClassVar[frozenset[str]] = frozenset({"max_z"})
    template_params: ClassVar[frozenset[str]] = frozenset(
        {"z_display", "z_star_display", "z_target_display"}
    )
    required_deps: ClassVar[frozenset[str]] = frozenset({"detector"})

    def __call__(self, ctx: GateContext, params: Mapping[str, Any]) -> CheckResult:
        detector = ctx.deps.detector
        if detector is None:
            raise GateDependencyError(
                "detector_threshold ran with no detector in Deps. This check IS the win "
                "condition; passing it by default would clear every submission on every "
                "level. Wire launder_core.detect into Deps.detector."
            )
        max_z = float(params["max_z"])
        # L5 gets its own calibration bucket: code has far fewer scored tokens
        # and far lower optionality, so its sigma(T) curve is fit separately
        # (§7.6, deviation #2).
        calibration = params.get("calibration")
        # RAW, not `ctx.normalized` — the same rule as `/api/detect` (see the
        # long note there). Whitespace folding changes the tokenization, so
        # scoring the normalized form would make the gate's win condition
        # disagree with the needle the player was watching while they edited.
        reading = detector.read(
            ctx.raw, ctx.passage, str(calibration) if calibration is not None else None
        )
        # THE EFFECTIVE BAR IS `z_star + max_z`, and that is the number the
        # player has to be told. The rejection used to render `z_star_display`
        # while the test was `z - z_star > max_z`, so on any level that tunes
        # max_z the message named a number that would still fail — and with a
        # negative max_z it was self-contradictory: "reads 1.31, has to reach
        # 2.33" when the real bar was -0.67 and 2.33 would not clear.
        # `z_star_display` is kept because it is z*, the published notch, and
        # the checklist blurb refers to it.
        rendered = {
            "z_display": format_z(reading.z),
            "z_star_display": format_z(reading.z_star),
            "z_target_display": format_z(reading.z_star + max_z),
        }
        meta = {
            "score": reading.score,
            "z": reading.z,
            "z_star": reading.z_star,
            "n_scored": reading.n_scored,
            "masked_fraction": reading.masked_fraction,
        }
        if reading.z - reading.z_star > max_z:
            return CheckResult(
                status="fail", check=self.name, code=self.name, params=rendered, meta=meta
            )
        return CheckResult(status="pass", check=self.name, params=rendered, meta=meta)
