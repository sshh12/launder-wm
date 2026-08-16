/**
 * The browser's half of `unicode_sanitation`.
 *
 * The DATA is pinned from the other side —
 * `packages/core/tests/test_unicode_ts_tables.py` re-derives the homoglyph
 * domain by scanning all of Unicode in Python and compares it to the constant
 * this file imports. So what is asserted here is the BEHAVIOUR that data is
 * wired into: which categories are counted, what each policy means, and the
 * combining-mark run rule.
 */
import { describe, expect, it } from "vitest";

import {
  HOMOGLYPH_RANGES,
  REJECT_CATEGORIES,
  classify,
  decodeRanges,
  isHomoglyph,
  offendingCount,
} from "../src/scoring/unicode";

const ALL = new Set<string>(REJECT_CATEGORIES);

function counts(
  text: string,
  opts: Partial<Parameters<typeof classify>[1]> = {},
): Record<string, number> {
  return classify(text, {
    rejectCategories: ALL,
    homoglyphPolicy: "allow",
    maxCombiningMarks: 2,
    allowedChars: new Set<string>(),
    ...opts,
  });
}

describe("the homoglyph table", () => {
  it("decodes to ascending, non-adjacent, non-empty ranges", () => {
    const ranges = decodeRanges(HOMOGLYPH_RANGES);
    expect(ranges.length).toBeGreaterThan(50);
    let prevEnd = -2;
    for (const [start, end] of ranges) {
      expect(end).toBeGreaterThanOrEqual(start);
      // Non-adjacent: two ranges that touch would mean the encoder emitted a
      // split it did not need, which is the shape a hand-edit leaves behind.
      expect(start).toBeGreaterThan(prevEnd + 1);
      prevEnd = end;
    }
    expect(prevEnd).toBeLessThanOrEqual(0x10ffff);
  });

  it("covers both mechanisms and nothing a reader would call ordinary", () => {
    expect(isHomoglyph("Ａ".codePointAt(0) as number)).toBe(true); // NFKC: fullwidth
    expect(isHomoglyph("𝐀".codePointAt(0) as number)).toBe(true); // NFKC: math bold
    expect(isHomoglyph("а".codePointAt(0) as number)).toBe(true); // table: Cyrillic
    expect(isHomoglyph("ο".codePointAt(0) as number)).toBe(true); // table: Greek
    expect(isHomoglyph("a".codePointAt(0) as number)).toBe(false);
    expect(isHomoglyph("é".codePointAt(0) as number)).toBe(false);
    expect(isHomoglyph("–".codePointAt(0) as number)).toBe(false);
  });

  it("excludes the circled numbers that fold to two ASCII digits", () => {
    // U+2469 -> "10". Python's isdigit() is False for it, so the gate does not
    // reject it, and a browser that did would paint a rule red on text that
    // clears. This is the boundary the table exists to hold.
    expect(isHomoglyph(0x2460)).toBe(true); // ① -> "1"
    expect(isHomoglyph(0x2469)).toBe(false); // ⑩ -> "10"
    expect(isHomoglyph(0x2473)).toBe(false); // ⑳ -> "20"
  });
});

describe("classify", () => {
  it("says nothing about ordinary prose", () => {
    expect(counts("The committee met on Thursday to consider the proposal.")).toEqual({});
  });

  it("finds the invisible families by name", () => {
    expect(counts("zero​width")).toEqual({ zero_width: 1 });
    expect(counts("soft­hyphen")).toEqual({ soft_hyphen: 1 });
    expect(counts("bidi‮trick")).toEqual({ bidi_control: 1 });
    expect(counts("unitseparator")).toEqual({ control_char: 1 });
    expect(counts("var️sel")).toEqual({ variation_selector: 1 });
    expect(counts("privuse")).toEqual({ private_use: 1 });
  });

  it("counts only the categories the level declared", () => {
    // `word_floor`'s level does not list `control_char`; a check that counted
    // it anyway would enforce a rule the level did not ask for.
    const declared = new Set(["zero_width", "homoglyph", "combining_marks"]);
    const text = "a​bc";
    expect(classify(text, {
      rejectCategories: declared,
      homoglyphPolicy: "allow",
      maxCombiningMarks: 2,
      allowedChars: new Set<string>(),
    })).toEqual({ zero_width: 1 });
  });

  it("honours the three homoglyph policies", () => {
    const cyrillic = "the prоposal"; // Cyrillic о
    expect(counts(cyrillic, { homoglyphPolicy: "allow" })).toEqual({});
    expect(counts(cyrillic, { homoglyphPolicy: "reject_always" })).toEqual({ homoglyph: 1 });
    // The shipped policy: a passage that legitimately contains a lookalike must
    // not become unplayable.
    expect(
      counts(cyrillic, {
        homoglyphPolicy: "reject_unless_in_original",
        allowedChars: new Set<string>(),
      }),
    ).toEqual({ homoglyph: 1 });
    expect(
      counts(cyrillic, {
        homoglyphPolicy: "reject_unless_in_original",
        allowedChars: new Set(["о"]),
      }),
    ).toEqual({});
    // `reject_always` ignores the original entirely — that is what "always"
    // means, and L5 (code) ships it.
    expect(
      counts(cyrillic, { homoglyphPolicy: "reject_always", allowedChars: new Set(["о"]) }),
    ).toEqual({ homoglyph: 1 });
  });

  it("allows a run of combining marks up to the limit and not past it", () => {
    const base = "e";
    expect(counts(base + "́̂")).toEqual({}); // 2, at the limit
    expect(counts(base + "́̂̃")).toEqual({ combining_marks: 1 });
    // The run RESETS at the next ordinary character, so two legal clusters are
    // not silently added together.
    expect(counts("é̂ á̂")).toEqual({});
  });

  it("counts every offender, which is what {count} renders", () => {
    expect(offendingCount(counts("a​b​c­d"))).toBe(3);
    expect(offendingCount({})).toBe(0);
  });

  it("iterates by codepoint, not by UTF-16 unit", () => {
    // A private-use character above the BMP is a surrogate PAIR in JS. Looping
    // with charCodeAt would see two lone surrogates and classify neither.
    expect(counts("emoji 🧼 and \u{F0000}")).toEqual({ private_use: 1 });
  });
});
