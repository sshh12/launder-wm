/**
 * THE SHIPPED SCORER'S RANGE CONTRACT.
 *
 * `tests/parity.test.ts` proves this function agrees with Python to 1e-9 over
 * the golden vectors, and that is the property that matters for the numbers on
 * screen. It is three orders of magnitude too coarse to see the property below,
 * which is that `heat` and `score` are `[0, 1]` — a closed interval, stated in
 * `WeightedMeanResult`'s own doc comment, relied on by `TokenHeat.heat` and
 * `DetectorReading.score` (both `Field(ge=0.0, le=1.0)` on the Python side, where
 * a one-ulp overshoot was a 500 on every keystroke) and by the heat ramp's
 * `color-mix` percentage.
 *
 * `sum(w) == m` holds by construction; `sum over d of w[d]` does not, because
 * float addition is not associative and the renormalization summed those numbers
 * in a different order than every consumer does. So the guarantee has to be
 * asserted, not assumed, and it has to be asserted at more than one depth —
 * the shipped `m = 30` is one of the depths where the error happens to land on
 * the safe side, which is precisely how it stayed invisible.
 */
import { describe, expect, it } from "vitest";

import { tournamentWeights, weightedMean } from "../src/detector/weightedmean";

/** Every row all ones: the maximum the scorer can produce, and the only place
 *  the upper bound can be violated. */
function allOnes(rows: number, m: number): Uint8Array {
  return new Uint8Array(rows * m).fill(1);
}

const DEPTHS = Array.from({ length: 63 }, (_, i) => i + 2); // 2..64
const SHIPPED_DEPTH = 30;

/** The raw accumulation the scorer performs, unclamped. */
function rawRowSum(m: number): number {
  const w = tournamentWeights(m);
  let sum = 0;
  for (let d = 0; d < m; d++) sum += w[d] as number;
  return sum;
}

/** The depths at which that accumulation overshoots the total it was
 *  renormalized to. Derived, never listed: the set is a property of float
 *  addition and would go stale the moment the weight vector changed. */
const OVERSHOOTS = DEPTHS.filter((m) => rawRowSum(m) / m > 1);

describe("the weighted mean stays inside [0, 1]", () => {
  it("has depths where the unclamped sum overshoots, and ours is not one of them", () => {
    // Nineteen of these sixty-three depths exceed `m` by one or two ulps. The
    // SHIPPED depth is not one of them — it lands one ulp low instead, which is
    // exactly why this could never be found by playing the game, and why the
    // assertions below have to sweep depths rather than test the one we run.
    expect(OVERSHOOTS.length).toBeGreaterThan(0);
    expect(OVERSHOOTS).toContain(17);
    expect(OVERSHOOTS).not.toContain(SHIPPED_DEPTH);
  });

  it("gives an all-ones row a heat of exactly 1 where the sum overshoots", () => {
    // Not `toBeCloseTo`. The Python test that should have caught this asserted
    // `pytest.approx(1.0)`, which is true of 1.0000000000000002 and is the
    // reason a value outside the declared range shipped.
    for (const m of OVERSHOOTS) {
      expect(rawRowSum(m) / m, `unclamped heat at depth ${m}`).toBeGreaterThan(1);
      const out = weightedMean(allOnes(1, m), Uint8Array.of(1), 1, m);
      expect(out.heat[0], `heat at depth ${m}`).toBe(1);
    }
  });

  it("never lets heat leave the interval, at any depth", () => {
    for (const m of DEPTHS) {
      const out = weightedMean(allOnes(1, m), Uint8Array.of(1), 1, m);
      expect(out.heat[0], `heat at depth ${m}`).toBeLessThanOrEqual(1);
      expect(out.heat[0], `heat at depth ${m}`).toBeGreaterThanOrEqual(0);
    }
    // The clip is ONE-SIDED, matching `_heat_from`'s `np.clip`: an undershoot
    // is already inside the interval, so nothing lifts it and nothing should.
    // At the shipped depth that leaves an all-ones row at 1 - 2.2e-16.
    const shipped = weightedMean(allOnes(1, SHIPPED_DEPTH), Uint8Array.of(1), 1, SHIPPED_DEPTH);
    expect(shipped.heat[0]).toBe(rawRowSum(SHIPPED_DEPTH) / SHIPPED_DEPTH);
    expect(shipped.heat[0]).toBeLessThan(1);
  });

  it("never lets the score leave the interval, at any depth or length", () => {
    for (const m of DEPTHS) {
      for (const rows of [1, 4, 50]) {
        const out = weightedMean(allOnes(rows, m), new Uint8Array(rows).fill(1), rows, m);
        expect(out.score, `score at depth ${m}, ${rows} rows`).toBeLessThanOrEqual(1);
        expect(out.score, `score at depth ${m}, ${rows} rows`).toBeGreaterThanOrEqual(0);
        expect(out.nScored).toBe(rows);
      }
    }
    // `total` is a running sum over rows, so it carries rounding no clamp on
    // `heat` could bound — hence the separate clamp, asserted separately.
    for (const rows of [1, 4, 50]) {
      const out = weightedMean(allOnes(rows, 17), new Uint8Array(rows).fill(1), rows, 17);
      expect(out.score, `score at depth 17, ${rows} rows`).toBe(1);
    }
  });

  it("gives an all-zeros matrix exactly 0, and never a negative", () => {
    for (const m of DEPTHS) {
      const out = weightedMean(new Uint8Array(4 * m), new Uint8Array(4).fill(1), 4, m);
      expect(out.score, `score at depth ${m}`).toBe(0);
      for (const h of out.heat) expect(h).toBe(0);
    }
  });

  it("clamps only the ends: a mixed matrix is untouched", () => {
    // The clamp must be provably inert on every value that was already inside
    // the interval — it is a bound, not a transform.
    const m = SHIPPED_DEPTH;
    const g = new Uint8Array(3 * m);
    for (let i = 0; i < 3; i++) for (let d = 0; d < m; d += 2) g[i * m + d] = 1;
    const out = weightedMean(g, Uint8Array.of(1, 1, 1), 3, m);
    const w = tournamentWeights(m);
    let expected = 0;
    for (let d = 0; d < m; d += 2) expected += w[d] as number;
    expect(out.heat[0]).toBe(expected / m);
    expect(out.score).toBeGreaterThan(0);
    expect(out.score).toBeLessThan(1);
  });

  it("still reports the neutral score when nothing is scored", () => {
    const out = weightedMean(allOnes(3, 17), new Uint8Array(3), 3, 17);
    expect(out.nScored).toBe(0);
    expect(out.score).toBe(0.5);
    expect(out.maskedFraction).toBe(1);
    // Heat is still computed for masked rows — the mirror greys them, it does
    // not blank them — and it is clamped for them too, because the mirror reads
    // it either way.
    expect(out.heat[0]).toBe(1);
  });
});
