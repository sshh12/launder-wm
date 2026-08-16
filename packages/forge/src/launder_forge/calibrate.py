"""`forge calibrate` — the null-distribution sweep that produces `thresholds.v1.json`.

TECH_PLAN.md §6.5, and it is worth restating the ruling exactly:

> The closed form `tau(eps) = 1/2 + Phi^-1(1-eps)/(2*sqrt(mT))` assumes all
> `mT` g-values are independent; the 30 tournament layers are strongly
> correlated and the paper's own empirics show true variance is several times
> naive. **Use the formula for the interpolation SHAPE and the 20,000-sample
> percentile for the LEVEL.**

That is implemented literally. For each length bucket we measure the empirical
sd of the weighted-mean score on negatives and divide by the closed-form sd.
The resulting per-bucket **inflation factor** is the only thing the shipped
curve interpolates; sigma itself keeps its exact `1/sqrt(n)` shape, so the
needle behaves sensibly at lengths nobody sampled.

    sigma(n) = 1 / (2*sqrt(m*n)) * kappa(n)
    z        = (score - 0.5) / sigma(n_scored)
    z*       = Phi^-1(1 - fpr)              # constant, 2.3263 at fpr = 1%

The closed form is spelled exactly as `launder_core.detect.calibration` spells
it. Carrying the depth weights through the variance would give
`sqrt(sum(w^2)/(4 m^2 n))`, a constant 1.1129x larger — and that constant is
precisely what `kappa` is for. Measured confirmation: on uniform random token
ids (independent rows by construction) the sweep returns kappa = 1.10-1.11 at
every length, i.e. the weighting factor alone. On 208 real unwatermarked
Gemma-3 passages an earlier measurement returned 1.17-1.27; the excess over
1.1129 is the correlation the plan warns about.

`z*` stays a constant on purpose (§4.2): a needle whose notch moves with length
is a needle showing noise.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from launder_forge.corpus import Corpus
from launder_forge.numerics import (
    compute_context_hashes,
    compute_context_repetition_mask,
    compute_g_values,
    depth_weights,
    weighted_mean_score,
)

__all__ = [
    "DEFAULT_BUCKETS",
    "BucketStats",
    "ThresholdCurve",
    "calibrate",
    "inflation_at",
    "sigma_closed_form",
    "write_thresholds",
    "z_star_for",
]

#: Token-length buckets from the §6.5 invocation.
DEFAULT_BUCKETS: tuple[int, ...] = (40, 60, 80, 120, 180, 260, 400)


def z_star_for(fpr: float) -> float:
    """`Phi^-1(1 - fpr)`. 2.3263478740408408 at fpr = 1e-2."""
    if not 0.0 < fpr < 1.0:
        raise ValueError(f"fpr must be in (0,1), got {fpr}")
    # math.erfc is exact enough and avoids dragging scipy into the hot path;
    # the inverse is found by bisection on a monotone function.
    lo, hi = -10.0, 10.0
    target = 1.0 - fpr
    for _ in range(200):
        mid = (lo + hi) / 2
        cdf = 0.5 * math.erfc(-mid / math.sqrt(2.0))
        if cdf < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def sigma_closed_form(n_scored: int, depth: int) -> float:
    """`1 / (2*sqrt(m*n))` — the independence assumption, stated.

    Identical to `launder_core.detect.calibration.sigma_closed_form`. See the
    note on `launder_forge.numerics.sigma_null` for why the weight-exact
    variant is deliberately not used: its constant factor belongs in `kappa`.
    """
    return 1.0 / (2.0 * math.sqrt(depth * n_scored))


@dataclass(slots=True)
class BucketStats:
    """Measured null statistics at one target length."""

    target_tokens: int
    n_samples: int
    mean_n_scored: float
    mean_score: float
    sd_empirical: float
    sd_closed_form: float
    inflation: float
    p_fpr_score: float
    """The empirical (1-fpr) percentile of the score. THE LEVEL."""
    tau_closed_form: float
    """What the closed form alone would have used. Recorded to show the gap."""
    mean_masked_fraction: float


@dataclass(slots=True)
class ThresholdCurve:
    fpr: float
    z_star: float
    depth: int
    buckets: list[BucketStats] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

    def sigma_at_fpr(self, b: BucketStats) -> float:
        """The bucket's sigma as the LOADER derives it: `(score_at_fpr - 0.5) / z*`.

        This is `launder_core.detect.calibration._bucket_from_json`, spelled the
        same way, and it is deliberately NOT `b.sd_empirical`. The percentile is
        the level (§14.2 item 7); the sd is a description of the same sample and
        the two differ by the null distribution's non-normality — 1.089 vs 1.110
        at n=36 on the synthetic sweep. Writing one number and interpolating the
        other would make the file's stated model disagree with the file's own
        loader, which is exactly the class of divergence this project exists to
        catch.
        """
        return (b.p_fpr_score - 0.5) / z_star_for(self.fpr)

    def knots(self) -> list[dict[str, float]]:
        """Interpolation knots for `kappa(n)` — what the loader actually uses.

        `kappa = sigma_at_fpr / sigma_closed_form(n_scored)`, at the same
        `n_scored` the buckets are keyed on (integers, as the loader reads
        them), so `inflation_at()` here and `Calibration.kappa` in core are the
        same function of the same numbers.
        """
        out: list[dict[str, float]] = []
        for b in sorted(self.buckets, key=lambda b: b.mean_n_scored):
            n = round(b.mean_n_scored)
            closed = sigma_closed_form(n, self.depth)
            out.append(
                {
                    "n_scored": float(n),
                    "inflation": self.sigma_at_fpr(b) / closed if closed > 0 else 1.0,
                }
            )
        return out

    def to_json(self, *, wm_config_id: str, name: str = "default") -> dict[str, Any]:
        """The exact shape `launder_core.detect.calibration.parse_thresholds` reads.

        `calibrations.<name>.buckets[*]` needs `n_scored` and `score_at_fpr`;
        core derives `sigma = (score_at_fpr - 0.5) / Phi_inv(1-fpr)` from them,
        and `kappa = sigma / sigma_closed_form(n_scored)`. Everything under
        `measurements` and `provenance` is forge's own record and is ignored by
        the loader — but it is the part a reviewer reads.

        `sigma` is ALSO written explicitly, even though core would derive it.
        The browser reads this same file (§5.4 `calibration_url`) and a second
        implementation of `Phi^-1` in TypeScript is a silent-divergence trap for
        no gain; `web/src/detector/calibration.ts` therefore requires `sigma`
        and refuses a bucket that only carries `score_at_fpr`. Core keeps
        preferring `sigma` when present, so the two paths read one number.
        """
        return {
            "schema": "launder.thresholds/1",
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "wm_config_id": wm_config_id,
            "fpr": self.fpr,
            "z_star": self.z_star,
            "depth": self.depth,
            "calibrations": {
                name: {
                    "fpr": self.fpr,
                    "z_star": self.z_star,
                    "buckets": [
                        {
                            "n_scored": round(b.mean_n_scored),
                            "score_at_fpr": b.p_fpr_score,
                            "sigma": self.sigma_at_fpr(b),
                            "n_samples": b.n_samples,
                        }
                        for b in sorted(self.buckets, key=lambda b: b.mean_n_scored)
                    ],
                }
            },
            "sigma_model": {
                # Spelled out so the TS port has no room to guess.
                "closed_form": "1 / (2*sqrt(m*n_scored))",
                "inflation": {
                    "kind": "piecewise_linear_in_log_n",
                    "clamp": "constant outside the knot range",
                    "knots": self.knots(),
                },
                "z": "(score - 0.5) / (closed_form(n_scored) * kappa(n_scored))",
                "z_star_is_constant": True,
            },
            "measurements": [asdict(b) for b in self.buckets],
            "provenance": self.provenance,
        }


def inflation_at(n_scored: float, knots: list[dict[str, float]]) -> float:
    """Piecewise linear in `log(n)`, clamped flat outside the sampled range."""
    if not knots:
        return 1.0
    xs = [math.log(max(k["n_scored"], 1.0)) for k in knots]
    ys = [k["inflation"] for k in knots]
    x = math.log(max(n_scored, 1.0))
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            t = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
            return ys[i - 1] + t * (ys[i] - ys[i - 1])
    return ys[-1]


def calibrate(
    corpus: Corpus,
    *,
    table: np.ndarray,
    keys: tuple[int, ...],
    ngram_len: int,
    context_history_size: int,
    buckets: tuple[int, ...] = DEFAULT_BUCKETS,
    n: int = 20000,
    fpr: float = 1e-2,
    seed: int = 12345,
    progress: Any = None,
) -> ThresholdCurve:
    """Score `n` negatives per bucket and build the curve.

    The corpus decides length by being *asked* for one bucket at a time, so a
    corpus that can only emit whatever lengths it has (a prose directory) still
    contributes to the bucket nearest its natural length.
    """
    depth = len(keys)
    per_bucket = max(1, n // len(buckets))
    curve = ThresholdCurve(fpr=fpr, z_star=z_star_for(fpr), depth=depth)

    for bi, target in enumerate(buckets):
        scores: list[float] = []
        n_scored_all: list[int] = []
        masked: list[float] = []
        for sample in corpus.samples(per_bucket, [target, target], seed + 1000 * bi):
            ids = np.asarray(sample.token_ids, dtype=np.int64)
            if ids.shape[0] < ngram_len:
                continue
            g = compute_g_values(ids, keys=keys, ngram_len=ngram_len, table=table)
            ctx = compute_context_hashes(ids, ngram_len=ngram_len)
            mask = compute_context_repetition_mask(ctx, context_history_size=context_history_size)
            score, n_scored = weighted_mean_score(g, mask, depth_weights(depth))
            if n_scored == 0:
                continue
            scores.append(score)
            n_scored_all.append(n_scored)
            masked.append(1.0 - n_scored / mask.shape[0])
            if progress is not None:
                progress()
        if not scores:
            raise RuntimeError(
                f"the corpus produced no usable sequences at target length {target}. "
                "Every sequence was shorter than ngram_len or fully masked."
            )
        arr = np.asarray(scores, dtype=np.float64)
        mean_n = float(np.mean(n_scored_all))
        closed = sigma_closed_form(max(1, round(mean_n)), depth)
        sd = float(arr.std(ddof=1)) if arr.shape[0] > 1 else 0.0
        curve.buckets.append(
            BucketStats(
                target_tokens=target,
                n_samples=int(arr.shape[0]),
                mean_n_scored=mean_n,
                mean_score=float(arr.mean()),
                sd_empirical=sd,
                sd_closed_form=closed,
                inflation=(sd / closed) if closed > 0 else 1.0,
                p_fpr_score=float(np.quantile(arr, 1.0 - fpr)),
                tau_closed_form=0.5 + curve.z_star * closed,
                mean_masked_fraction=float(np.mean(masked)),
            )
        )
    return curve


def write_thresholds(
    path: Path, curve: ThresholdCurve, *, wm_config_id: str, name: str = "default"
) -> dict[str, Any]:
    """Write the curve into `path`, KEEPING every other calibration in the file.

    `to_json` builds `{"calibrations": {name: ...}}` — a fresh one-entry map —
    and this used to write it straight over the file. So the one documented way
    to produce L5's bucket, `forge calibrate --name code --force` (levels.toml:
    "L5 also gets its own calibration bucket ... its sigma(T) curve and par must
    be fit separately"), DELETED `calibrations.default`: every other level then
    booted into `KeyError: calibration bucket set 'default' not found`, and
    thresholds.v1.json is one of the four inputs to `asset_bundle_id`, so the
    same command invalidated every packed passage. Bucket sets are additive.

    The scalars every group shares — `fpr`, `z_star`, `depth`, `wm_config_id` —
    live at the top level and are asserted rather than overwritten: two curves
    measured against different constants are not two buckets of one file, and
    core's `_assert_no_shadow` treats a per-group override of them as a config
    error rather than a knob.
    """
    payload = curve.to_json(wm_config_id=wm_config_id, name=name)
    if path.exists():
        existing: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        for key, mine in (
            ("fpr", payload["fpr"]),
            ("z_star", payload["z_star"]),
            ("depth", payload["depth"]),
            ("wm_config_id", payload["wm_config_id"]),
        ):
            theirs = existing.get(key)
            if theirs is not None and theirs != mine:
                raise ValueError(
                    f"{path} records {key} = {theirs!r} and this run measured {mine!r}. "
                    f"Every calibration in one thresholds file shares these, so writing "
                    f"{name!r} here would silently restate {key} for the bucket sets already "
                    f"in it ({sorted(existing.get('calibrations', {}))}). Re-run the others "
                    "against the same constants, or write this one to a different file."
                )
        merged = dict(existing.get("calibrations") or {})
        merged.update(payload["calibrations"])
        payload["calibrations"] = merged
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
