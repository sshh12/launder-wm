/**
 * The live half of the gate.
 *
 * These assertions are the CONTRACT WITH THE SERVER, not a description of the
 * UI: every one of them mirrors a statement in the corresponding Python check,
 * and where the two could plausibly differ the test names the difference. A
 * live rule that is stricter than the gate is the failure mode that matters —
 * the player is shown a red rule and then clears the level anyway — so the
 * boundary cases (exactly at the budget, exactly at the floor, exactly at the
 * notch) are asserted on the passing side.
 *
 * `packages/core/tests/test_checks_region_and_floor.py` and `test_gates.py` are
 * the other half of each pair.
 */
import { describe, expect, it } from "vitest";

import { LIVE_CHECKS, LivePreview } from "../src/game/live";
import type { CheckSpecWire, Reading } from "../src/state";

// Twelve words per line, so word indices are easy to reason about — the same
// fixture shape test_checks_region_and_floor.py uses.
const ORIGINAL =
  "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima " +
  "mike november oscar papa quebec romeo sierra tango uniform victor whiskey xray";

const POINTS = { zStar: 2.3263, scale: { min: -2, max: 10 } };

function reading(z: number): Reading {
  return {
    textHash: "",
    seq: 1,
    z,
    zStar: 2.3263,
    score: 0.5,
    nScored: 120,
    nTokens: 140,
    tokens: [],
    previewDistance: 0,
  };
}

function live(text: string, opts: { phrases?: string[]; z?: number } = {}): LivePreview {
  const preview = new LivePreview({
    original: ORIGINAL,
    lockedPhrases: opts.phrases ?? [],
    points: POINTS,
  });
  preview.setText(text);
  if (opts.z !== undefined) preview.setReading(reading(opts.z));
  return preview;
}

const spec = (check: string, params: Record<string, unknown> = {}): CheckSpecWire => ({
  check,
  params,
});

const editWord = (index: number, to: string): string => {
  const w = ORIGINAL.split(" ");
  w[index] = to;
  return w.join(" ");
};

// ---------------------------------------------------------------------------
describe("which checks are answerable in the browser", () => {
  it("answers the seven closed-form rules and refuses the other three", () => {
    expect([...LIVE_CHECKS].sort()).toEqual([
      "detector_floor",
      "detector_threshold",
      "edit_budget",
      "edit_region",
      "locked_phrase",
      "unicode_sanitation",
      "word_floor",
    ]);
  });

  it("returns null for anything it cannot honestly answer", () => {
    const p = live(ORIGINAL);
    // Not "pass". A green pip for a paid model that has not run is a lie, and
    // the pending state is the whole reason `gate.pending_label` exists.
    expect(p.evaluate(spec("llm_gate"))).toBeNull();
    expect(p.evaluate(spec("unit_test"))).toBeNull();
    expect(p.evaluate(spec("close_paraphrase"))).toBeNull();
    expect(p.evaluate(spec("a_check_that_does_not_exist"))).toBeNull();
  });
});

// ---------------------------------------------------------------------------
describe("word_floor", () => {
  it("counts the same words the scorer counts", () => {
    const out = live("one two three").evaluate(spec("word_floor", { min_words: 3 }));
    expect(out).toEqual({ status: "pass", params: { n_words: 3, min_words: 3 } });
  });

  it("fails below the minimum and passes exactly at it", () => {
    expect(live("one two").evaluate(spec("word_floor", { min_words: 3 }))?.status).toBe("fail");
    expect(live("one two three").evaluate(spec("word_floor", { min_words: 3 }))?.status).toBe(
      "pass",
    );
  });

  it("collapses whitespace before counting, as normalize() does", () => {
    const out = live("  one   two \n three  ").evaluate(spec("word_floor", { min_words: 3 }));
    expect(out?.params.n_words).toBe(3);
  });
});

// ---------------------------------------------------------------------------
describe("edit_budget", () => {
  it("is the SAME number the counter under the textarea shows", () => {
    // The property `edit_budget.py` exists to guarantee: "the budget and the
    // scoreboard are one number computed once, so they cannot disagree."
    const p = live(editWord(2, "CHANGED"));
    const out = p.evaluate(spec("edit_budget", { max_word_distance: 6 }));
    expect(out?.params.distance).toBe(p.distance());
    expect(p.distance()).toBe(1);
  });

  it("passes AT the budget and fails one past it", () => {
    let text = ORIGINAL;
    for (let i = 0; i < 6; i++) text = text.replace(ORIGINAL.split(" ")[i] as string, `w${i}`);
    const at = live(text).evaluate(spec("edit_budget", { max_word_distance: 6 }));
    expect(at).toEqual({ status: "pass", params: { distance: 6, max_word_distance: 6 } });
    expect(live(text).evaluate(spec("edit_budget", { max_word_distance: 5 }))?.status).toBe("fail");
  });

  it("recomputes when the text changes", () => {
    const p = live(ORIGINAL);
    expect(p.distance()).toBe(0);
    p.setText(editWord(0, "A"));
    expect(p.distance()).toBe(1);
    p.setText(editWord(0, "A") + " extra");
    expect(p.distance()).toBe(2);
  });
});

// ---------------------------------------------------------------------------
describe("locked_phrase", () => {
  const phrases = ["golf hotel india"];

  it("passes while the phrase survives and fails the moment it does not", () => {
    expect(live(ORIGINAL, { phrases }).evaluate(spec("locked_phrase"))).toEqual({
      status: "pass",
      params: { phrase: "golf hotel india" },
    });
    expect(live(editWord(7, "HOTEL"), { phrases }).evaluate(spec("locked_phrase"))).toEqual({
      status: "fail",
      params: { phrase: "golf hotel india" },
    });
  });

  it("is whitespace-insensitive by default and exact on demand", () => {
    const spaced = ORIGINAL.replace("golf hotel india", "golf  hotel\nindia");
    expect(live(spaced, { phrases }).evaluate(spec("locked_phrase"))?.status).toBe("pass");
    expect(
      live(spaced, { phrases }).evaluate(spec("locked_phrase", { match: "exact" }))?.status,
    ).toBe("fail");
  });

  it("declines to answer when the passage declares no phrases", () => {
    // Server-side this is a GateDataError — "the check would be vacuous and the
    // level would be a lie". A green pip here would be the browser telling that
    // lie on its behalf.
    expect(live(ORIGINAL, { phrases: [] }).evaluate(spec("locked_phrase"))).toBeNull();
  });

  it("requires EVERY phrase, and names the first one missing", () => {
    const two = ["golf hotel india", "sierra tango uniform"];
    const broken = live(editWord(19, "TANGO"), { phrases: two });
    expect(broken.evaluate(spec("locked_phrase"))).toEqual({
      status: "fail",
      params: { phrase: "sierra tango uniform" },
    });
  });
});

// ---------------------------------------------------------------------------
describe("edit_region", () => {
  const rule = spec("edit_region", { editable_prefix_words: 6 });

  it("is about position, not size", () => {
    let big = ORIGINAL.split(" ");
    for (let i = 0; i < 6; i++) big[i] = `w${i}`;
    expect(live(big.join(" ")).evaluate(rule)?.status).toBe("pass");
    expect(live(editWord(23, "XRAY")).evaluate(rule)?.status).toBe("fail");
  });

  it("reports the offending word 1-based, because a person counts them", () => {
    const out = live(editWord(15, "PAPA")).evaluate(rule);
    expect(out?.params).toEqual({
      editable_prefix_words: 6,
      n_outside: 1,
      first_bad_word_index: 16,
    });
  });

  it("names the FIRST word outside the window when several are", () => {
    let w = ORIGINAL.split(" ");
    w[20] = "UNIFORM";
    w[10] = "KILO";
    const out = live(w.join(" ")).evaluate(rule);
    expect(out?.params.n_outside).toBe(2);
    expect(out?.params.first_bad_word_index).toBe(11);
  });

  it("passes an untouched passage", () => {
    expect(live(ORIGINAL).evaluate(rule)).toEqual({
      status: "pass",
      params: { editable_prefix_words: 6, n_outside: 0, first_bad_word_index: 0 },
    });
  });

  it("declines a zero-width window rather than making the level unwinnable", () => {
    expect(live(ORIGINAL).evaluate(spec("edit_region", { editable_prefix_words: 0 }))).toBeNull();
  });
});

// ---------------------------------------------------------------------------
describe("unicode_sanitation", () => {
  const rule = spec("unicode_sanitation", {
    reject_categories: ["zero_width", "soft_hyphen", "bidi_control", "control_char"],
    homoglyph_policy: "reject_unless_in_original",
    max_combining_marks: 2,
  });

  it("passes the pristine passage", () => {
    expect(live(ORIGINAL).evaluate(rule)).toEqual({ status: "pass", params: { count: 0 } });
  });

  it("catches a paste that brought invisible characters with it", () => {
    expect(live(ORIGINAL + " a​b").evaluate(rule)).toEqual({
      status: "fail",
      params: { count: 1 },
    });
  });

  it("keeps homoglyphs live even though the level does not list them", () => {
    // The Python check adds `homoglyph` and `combining_marks` to the declared
    // set unconditionally, because they are governed by their own params.
    expect(live(ORIGINAL.replace("alpha", "аlpha")).evaluate(rule)?.status).toBe("fail");
  });
});

// ---------------------------------------------------------------------------
describe("the two detector bounds", () => {
  it("says nothing before the first reading lands", () => {
    const p = live(ORIGINAL);
    expect(p.evaluate(spec("detector_threshold", { max_z: 0 }))).toBeNull();
    expect(p.evaluate(spec("detector_floor", { min_z: -1.2 }))).toBeNull();
  });

  it("clears AT the notch, not merely under it", () => {
    const rule = spec("detector_threshold", { max_z: 0 });
    expect(live(ORIGINAL, { z: 2.3263 }).evaluate(rule)?.status).toBe("pass");
    expect(live(ORIGINAL, { z: 2.4 }).evaluate(rule)?.status).toBe("fail");
  });

  it("reads max_z off the level rather than assuming the notch", () => {
    // Every shipped level runs max_z = 0.0, and the browser used to hardcode
    // `z <= z_star` — which silently becomes the wrong bar the day a level
    // tunes it, in the direction that tells the player they have cleared.
    const tuned = spec("detector_threshold", { max_z: -1.0 });
    expect(live(ORIGINAL, { z: 2.0 }).evaluate(tuned)?.status).toBe("fail");
    expect(live(ORIGINAL, { z: 1.3 }).evaluate(tuned)?.status).toBe("pass");
  });

  it("rejects over-scrubbing, with both bounds relative to the notch", () => {
    const floor = spec("detector_floor", { min_z: -1.2 });
    expect(live(ORIGINAL, { z: 1.5 }).evaluate(floor)?.status).toBe("pass");
    expect(live(ORIGINAL, { z: -2.0 }).evaluate(floor)?.status).toBe("fail");
    expect(live(ORIGINAL, { z: 2.3263 - 1.2 }).evaluate(floor)?.status).toBe("pass");
  });

  it("renders POINTS, so a live rule and the meter above it agree", () => {
    // Raw z here would put two number systems on one screen — the bug §11 was
    // written to close ("reads 4.0, has to reach 2.3" under a needle showing
    // 50 and a notch at 36).
    const out = live(ORIGINAL, { z: 4.0 }).evaluate(spec("detector_threshold", { max_z: 0 }));
    expect(out?.params.z_display).toBe("50");
    expect(out?.params.z_star_display).toBe("36");
    expect(out?.params.z_target_display).toBe("36");
  });
});
