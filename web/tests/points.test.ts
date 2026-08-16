/**
 * The 0-100 readout (§11).
 *
 * The instrument keeps working in z; only the displayed number changes. That
 * makes `points()` a relabelling, and a relabelling has exactly one way to be
 * wrong that matters: PRINTING A NUMBER THAT DISAGREES WITH THE VERDICT. z* is
 * 2.3263 and lands on 36.05, so every z in the sliver between 36.0 and 36.5
 * rounds to the same integer the line itself does — a player one edit above the
 * line would read the same "36" as a player who had cleared it. That is the
 * two-decimal-z bug ("2.3 on both sides of the line") arriving again through a
 * coarser scale, and the tests below are all about that sliver.
 */

import { describe, expect, it } from "vitest";

import { POINTS_MAX, POINTS_MIN, formatPoints, points, type PointsScale } from "../src/game/needle";

/** The production numbers: SynthID's z* against the -2..10 printed face. */
const CFG: PointsScale = { zStar: 2.3263, scale: { min: -2, max: 10 } };

/** raw(z) with no correction and no clamp — what a naive readout would print. */
function naive(z: number, cfg: PointsScale = CFG): number {
  return Math.round(((z - cfg.scale.min) / (cfg.scale.max - cfg.scale.min)) * 100);
}

describe("points", () => {
  it("relabels the printed scale end to end", () => {
    expect(points(-2, CFG)).toBe(POINTS_MIN);
    expect(points(10, CFG)).toBe(POINTS_MAX);
    expect(points(4, CFG)).toBe(50);
    expect(points(1, CFG)).toBe(25);
    expect(points(7, CFG)).toBe(75);
  });

  it("puts z* itself at 36", () => {
    expect(points(CFG.zStar, CFG)).toBe(36);
    expect(naive(CFG.zStar)).toBe(36);
  });

  it("clamps a reading that runs off either end of the face", () => {
    // z is unbounded; the face is not. A -40 must not print a negative number
    // and a 400 must not print one longer than the readout can hold.
    expect(points(-40, CFG)).toBe(POINTS_MIN);
    expect(points(400, CFG)).toBe(POINTS_MAX);
  });

  it("SIDE OF THE LINE WINS: a z above z* never prints z*'s number", () => {
    // 2.33 through 2.38 all round to 36, the same integer as the line. Every
    // one of them is above the line and the gate would reject them, so the
    // readout is forced up.
    for (const z of [2.3264, 2.33, 2.35, 2.379]) {
      expect(naive(z), `${z} should be the collision case`).toBe(36);
      expect(points(z, CFG), `${z} is above the line`).toBe(37);
    }
  });

  it("never prints the line's own number on the failing side of the line", () => {
    // The property the two corrections exist to guarantee, swept across the
    // whole face at a resolution far finer than the readout: every cleared
    // reading is at or below 36 and every failing one is strictly above it.
    // The boundary is `z <= z*` because that is how the gate decides.
    const pStar = points(CFG.zStar, CFG);
    const wrong: string[] = [];
    for (let i = -3000; i <= 11000; i++) {
      const z = i / 1000;
      const p = points(z, CFG);
      const ok = z > CFG.zStar ? p > pStar : p <= pStar;
      if (!ok) wrong.push(`z=${z} -> ${p}`);
    }
    expect(wrong.slice(0, 5)).toEqual([]);
  });

  it("is monotone across the line, so the number never moves the wrong way", () => {
    let previous = -1;
    for (let z = -3; z <= 11; z += 0.01) {
      const p = points(z, CFG);
      expect(p).toBeGreaterThanOrEqual(previous);
      previous = p;
    }
  });

  it("prints a bare integer — never a decimal, never a percent sign", () => {
    // A number that looks like a percentage would read as "% AI", which is the
    // one claim the detector cannot support.
    for (const z of [-9, -2, 0, 2.3263, 2.4, 6.44, 10, 99]) {
      expect(formatPoints(z, CFG)).toMatch(/^\d{1,3}$/);
    }
  });

  it("survives a degenerate scale instead of printing NaN", () => {
    const flat: PointsScale = { zStar: 0, scale: { min: 5, max: 5 } };
    expect(Number.isFinite(points(3, flat))).toBe(true);
    expect(Number.isFinite(points(7, flat))).toBe(true);
  });
});
