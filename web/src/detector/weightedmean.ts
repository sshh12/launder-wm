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
    heat[i] = rowSum / m;
    if ((mask[i] as number) !== 0) {
      total += rowSum;
      nScored += 1;
    }
  }

  const score = nScored === 0 ? 0.5 : total / (m * nScored);
  return {
    score,
    heat,
    nScored,
    maskedFraction: rows === 0 ? 0 : (rows - nScored) / rows,
  };
}
