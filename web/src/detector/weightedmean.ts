/**
 * The shipped scorer. TS port of `launder_core.detect.weighted_mean`.
 *
 * ---------------------------------------------------------------------------
 * DIVERGENCE TRAP #4 (§4.3): rounding drift.
 * ---------------------------------------------------------------------------
 * Trivial next to the other three, but pinned anyway by golden case #6 at
 * `tol = 1e-9`. Both sides are float64 — JS `number` IS float64, and Python
 * uses `numpy.float64` — so the only way to diverge is to change the *order* of
 * operations, e.g. dividing per row instead of once at the end. Do not
 * "optimise" the accumulation into a running mean.
 *
 *     w      = linspace(10, 1, m); w *= m / sum(w)        # so sum(w) == m
 *     score  = SUM_{i,L} mask[i]*w[L]*g[i][L] / (m * SUM_i mask[i])
 *     heat[i]= SUM_L w[L]*g[i][L] / m                     # in [0,1], 0.5 neutral
 *     n_scored = SUM_i mask[i]
 */

/**
 * `w = linspace(10, 1, m)`, rescaled so `sum(w) == m` exactly in the limit.
 *
 * `linspace` endpoints are exact: `w[0] = 10`, `w[m-1] = 1`. The interior uses
 * `start + k*(stop-start)/(m-1)`, which is numpy's formula, not repeated
 * addition.
 */
export function tournamentWeights(m: number): Float64Array {
  if (m < 2) throw new Error(`tournament depth must be >= 2, got ${m}`);
  const w = new Float64Array(m);
  const start = 10;
  const stop = 1;
  const step = (stop - start) / (m - 1);
  for (let k = 0; k < m; k++) w[k] = start + k * step;
  w[m - 1] = stop; // numpy pins the endpoint exactly
  let s = 0;
  for (let k = 0; k < m; k++) s += w[k] as number;
  const scale = m / s;
  for (let k = 0; k < m; k++) w[k] = (w[k] as number) * scale;
  return w;
}

export interface WeightedMeanResult {
  /** The weighted mean g-value over unmasked rows. ~[0.500, 0.555]. */
  readonly score: number;
  /** Per-row heat in [0,1], 0.5 neutral. THE heat-mirror value. */
  readonly heat: Float64Array;
  /** Number of unmasked rows. */
  readonly nScored: number;
  /** Fraction of rows removed by the mask — instrumented per §4.4(c). */
  readonly maskedFraction: number;
}

/**
 * @param g    row-major `(rows x m)` 0/1 matrix
 * @param mask `rows` 0/1 flags; masked rows leave numerator AND denominator
 */
/**
 * `[0, 1]`, which both quantities are PROVED to be inside and neither reliably
 * lands inside.
 *
 * `tournamentWeights` renormalizes so `sum(w) == m`, but every consumer sums the
 * same numbers in a different order, and float addition is not associative. On
 * an all-ones row `rowSum` is a subset sum of `w` that reaches its maximum at
 * the full sum, and that full sum is not `m`: it is `m + 2.2e-16` at depths 9,
 * 12, 16, 17, 19, 22, 23, 28, 32, 42, 45, 47, 55-61 and `m - 7.1e-15` at the
 * shipped depth of 30. So `heat` here is one ulp OVER 1 at nineteen of the first
 * sixty-three depths and one ulp under it at ours — latent, and latent only
 * because of the constant.
 *
 * IT WAS NOT LATENT IN PYTHON. `heat_values` computes the same quantity as
 * `gm @ w`, whose pairwise summation overshoots at m = 30 — `30.000000000000004`
 * — and `TokenHeat.heat` and `DetectorReading.score` are both
 * `Field(ge=0.0, le=1.0)`. `/api/detect` therefore raised a pydantic
 * ValidationError, a 500 on every keystroke, for any text containing one n-gram
 * whose thirty tournament layers all read 1: about 1e-9 per row, so certain
 * eventually, and permanent for that passage once it happened rather than
 * intermittent. `_heat_from` now clips and so does `weighted_mean_score`.
 *
 * Nothing validates the range in the browser, so the same value is silent here
 * — which is exactly why it has to be written down. The divergence is <= 2.2e-16
 * and the parity gate's tolerance is 1e-9, three orders coarser, so the gate
 * passes either way and cannot be the thing that keeps these two honest.
 *
 * CLIP, NOT RESCALE. The true value at those two extremes IS 0 or 1; 2e-16 is
 * float noise, not evidence. This can only ever move a value that was already
 * outside a closed interval it is proved to be inside, so it cannot change the
 * heat mirror, the ripple, `z`, or the §4.6 decomposition property.
 */
const unit = (v: number): number => Math.min(1, Math.max(0, v));

export function weightedMean(
  g: Uint8Array,
  mask: Uint8Array,
  rows: number,
  m: number,
): WeightedMeanResult {
  const w = tournamentWeights(m);
  const heat = new Float64Array(rows);
  let total = 0;
  let nScored = 0;

  for (let i = 0; i < rows; i++) {
    const base = i * m;
    let rowSum = 0;
    for (let d = 0; d < m; d++) {
      if ((g[base + d] as number) !== 0) rowSum += w[d] as number;
    }
    // Heat is computed for EVERY row, masked or not: the mirror needs a value
    // to grey out, and a masked row's g-values are still real.
    heat[i] = unit(rowSum / m);
    if ((mask[i] as number) !== 0) {
      total += rowSum;
      nScored += 1;
    }
  }

  // Clamped SEPARATELY, because it is a separate accumulation: `total` is a
  // running sum over rows, so it carries its own rounding on top of whatever
  // each `rowSum` already had, and no clamp on `heat` can bound it.
  const score = nScored === 0 ? 0.5 : unit(total / (m * nScored));
  return {
    score,
    heat,
    nScored,
    maskedFraction: rows === 0 ? 0 : (rows - nScored) / rows,
  };
}
