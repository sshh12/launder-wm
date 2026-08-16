/**
 * The window — `detector_floor` (levels 7 and 8).
 *
 * Two levels give the printed scale a bottom as well as a top, and the whole
 * risk lives in one line: `min_z` is RELATIVE TO THE NOTCH, exactly like
 * `detector_threshold`'s `max_z`. Read as an absolute, the shipped -1.2 draws a
 * floor at 8 on a face whose notch is 36 — a bottom of the window sitting well
 * below the top of it, which is not a window and is not wrong in any way the
 * player could see until they tripped it.
 *
 * The second half of the file is the other thing a mark can get wrong: standing
 * somewhere other than the number it claims. A printed limit is placed with
 * `points()`, the same relabelling the readout prints, so "26" and the mark are
 * the same 26.
 */

import { describe, expect, it } from "vitest";

import { floorZFromChecks, points, type PointsScale } from "../src/game/needle";
import type { CheckSpecWire } from "../src/state";

/** The production numbers: SynthID's z* against the -2..10 printed face. */
const CFG: PointsScale = { zStar: 2.3263, scale: { min: -2, max: 10 } };

/** L8 and L9 as they ship in data/config/levels.toml. */
const SHIPPED: CheckSpecWire[] = [
  { check: "unicode_sanitation" },
  { check: "word_floor" },
  { check: "detector_floor", params: { min_z: -1.2 } },
  { check: "detector_threshold" },
];

/** A level with a top and no bottom — every other level in the campaign. */
const NO_FLOOR: CheckSpecWire[] = [
  { check: "unicode_sanitation" },
  { check: "edit_budget", params: { max_word_distance: 12 } },
  { check: "detector_threshold", params: { max_z: 0 } },
];

describe("floorZFromChecks", () => {
  it("reads min_z RELATIVE TO THE NOTCH, like max_z", () => {
    expect(floorZFromChecks(SHIPPED, CFG.zStar)).toBeCloseTo(CFG.zStar - 1.2, 10);
    // The absolute reading, which is the bug this test exists for.
    expect(floorZFromChecks(SHIPPED, CFG.zStar)).not.toBeCloseTo(-1.2, 3);
  });

  it("is null on a level that does not run the check", () => {
    expect(floorZFromChecks(NO_FLOOR, CFG.zStar)).toBeNull();
    expect(floorZFromChecks([], CFG.zStar)).toBeNull();
    expect(floorZFromChecks(undefined, CFG.zStar)).toBeNull();
  });

  it("draws nothing rather than inventing a bound the server never agreed to", () => {
    // A check present but carrying no usable `min_z`. A default here would be a
    // line the player can fail against that no gate is enforcing.
    expect(floorZFromChecks([{ check: "detector_floor" }], CFG.zStar)).toBeNull();
    expect(floorZFromChecks([{ check: "detector_floor", params: {} }], CFG.zStar)).toBeNull();
    for (const min_z of ["-1.2", null, Number.NaN, Number.POSITIVE_INFINITY]) {
      expect(floorZFromChecks([{ check: "detector_floor", params: { min_z } }], CFG.zStar)).toBeNull();
    }
  });

  it("tracks z*, because the floor is measured from it", () => {
    const shifted = floorZFromChecks(SHIPPED, 4);
    expect(shifted).toBeCloseTo(2.8, 10);
  });
});

describe("where the mark stands", () => {
  it("puts the shipped floor at 26 on a notch of 36", () => {
    const floor = floorZFromChecks(SHIPPED, CFG.zStar);
    expect(floor).not.toBeNull();
    expect(points(floor as number, CFG)).toBe(26);
    expect(points(CFG.zStar, CFG)).toBe(36);
  });

  it("leaves a window ten points wide — narrow, and the whole twist", () => {
    const floor = points(floorZFromChecks(SHIPPED, CFG.zStar) as number, CFG);
    const notch = points(CFG.zStar, CFG);
    expect(notch - floor).toBe(10);
    // The floor is INSIDE the window: the gate's rule is "at or above".
    expect(floor).toBeLessThan(notch);
  });

  it("is placed by the same function the readout prints, to the point", () => {
    // A limit placed by the raw ratio instead stands at 26.05 while its number
    // reads 26 — invisible here, and a disagreement the moment either moves.
    const floor = floorZFromChecks(SHIPPED, CFG.zStar) as number;
    const raw = ((floor - CFG.scale.min) / (CFG.scale.max - CFG.scale.min)) * 100;
    expect(points(floor, CFG)).toBe(Math.round(raw));
  });

  it("a min_z the right way up puts the floor at or above the notch", () => {
    // The degenerate input the instrument refuses outright rather than drawing:
    // a "window" whose bottom is not below its top is not a window, and the
    // needle must not announce a bound the face is not showing.
    for (const min_z of [0, 0.5, 3]) {
      const z = floorZFromChecks([{ check: "detector_floor", params: { min_z } }], CFG.zStar);
      expect(points(z as number, CFG)).toBeGreaterThanOrEqual(points(CFG.zStar, CFG));
    }
  });

  it("clamps a floor that would fall off the left end of the face", () => {
    const deep = floorZFromChecks([{ check: "detector_floor", params: { min_z: -40 } }], CFG.zStar);
    expect(points(deep as number, CFG)).toBe(0);
  });
});
