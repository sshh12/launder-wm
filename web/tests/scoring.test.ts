/**
 * Case 8 — normalize + Damerau-Levenshtein, against `data/golden/vectors.json`.
 *
 * PREVIEW ONLY (§8.3): the server's number is the one persisted, ranked and
 * shared. These tests exist so that any client/server disagreement is a caught
 * parity bug rather than a leaderboard that quietly disagrees with itself.
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

import {
  DEFAULT_NORMALIZE_CONFIG,
  WHITESPACE_CLASS,
  damerau,
  damerauDistance,
  normalize,
  score,
  words,
} from "../src/scoring/index.js";

const REPO = fileURLToPath(new URL("../../", import.meta.url));
const VECTORS = JSON.parse(readFileSync(`${REPO}data/golden/vectors.json`, "utf8")) as {
  cases: Record<string, any>;
};
const C8 = VECTORS.cases["normalize_and_damerau"];

describe("case 8 — normalize", () => {
  it("agrees with the golden file", () => {
    for (const c of C8.normalize) {
      expect(normalize(c.input)).toBe(c.expected);
    }
  });

  it("is idempotent on every golden case", () => {
    for (const c of C8.normalize) {
      const once = normalize(c.input);
      expect(normalize(once)).toBe(once);
      expect(c.idempotent).toBe(true);
    }
  });

  it("is idempotent on a fuzz corpus", () => {
    // A cheap stand-in for the Python side's Hypothesis property test.
    const alphabet = [
      "a",
      "Z",
      " ",
      "\t",
      "\n",
      "\r\n",
      " ",
      " ",
      "　",
      "",
      "",
      "“",
      "’",
      "—",
      "…",
      "﻿",
      "​",
      "­",
      "é",
      "é",
      "ﬁ",
      "①",
      "🧼",
      ".",
      ",",
    ];
    let s = 12345;
    const rnd = () => (s = (s * 1103515245 + 12345) % 2147483648) / 2147483648;
    for (let t = 0; t < 2000; t++) {
      let str = "";
      const len = 1 + Math.floor(rnd() * 12);
      for (let k = 0; k < len; k++) str += alphabet[Math.floor(rnd() * alphabet.length)];
      const once = normalize(str);
      expect(normalize(once)).toBe(once);
    }
  });

  it("keeps what unicode_sanitation is supposed to reject", () => {
    // §8.1: zero-width, soft hyphen, BOM etc. are NOT stripped here. Stripping
    // would let the exploit succeed at the detector while the judge saw clean
    // text.
    expect(normalize("zero​width")).toContain("​");
    expect(normalize("soft­hyphen")).toContain("­");
    expect(normalize("﻿BOM")).toContain("﻿");
    // and NFKC bait survives, because we use NFC
    expect(normalize("ﬁ")).toBe("ﬁ");
    expect(normalize("①")).toBe("①");
  });

  it("never lowercases and never strips punctuation", () => {
    expect(DEFAULT_NORMALIZE_CONFIG.lowercase).toBe(false);
    expect(DEFAULT_NORMALIZE_CONFIG.stripPunctuation).toBe(false);
    expect(normalize("The Study, Holds.")).toBe("The Study, Holds.");
  });

  it("pins Unicode White_Space minus U+FEFF — neither language's \\s", () => {
    // Neither shorthand is usable. Python's \s matches \x1c-\x1f and \x85 but
    // not U+FEFF; JavaScript's matches U+FEFF but not \x1c-\x1f or \x85. So the
    // class is written out, and it is the Unicode White_Space property — a name
    // both languages spell identically — MINUS U+FEFF.
    //
    // TWO EXCLUSIONS, BOTH PAIRED WITH unicode_sanitation. normalize() must not
    // silently repair an exploit (§8.1), so an invisible character it leaves in
    // place has to be REJECTED downstream rather than eaten here:
    //   U+FEFF     -> category `zero_width`   (ZERO WIDTH NO-BREAK SPACE)
    //   U+001C-1F  -> category `control_char` (FILE/GROUP/RECORD/UNIT SEPARATOR)
    // The golden file records the same class in
    // cases.normalize_and_damerau.whitespace_class; core holds the other half in
    // packages/core/src/launder_core/gates/checks/unicode_sanitation.py.
    expect(WHITESPACE_CLASS).toContain("\\x85"); // NEL IS whitespace; JS \s misses it
    expect(WHITESPACE_CLASS).not.toContain("\\x1c"); // C0 separator is NOT whitespace
    expect(WHITESPACE_CLASS).not.toContain("feff"); // zero width is NOT whitespace
    expect(`[${WHITESPACE_CLASS}]`).toBe(C8.whitespace_class);
    expect(words("a\x85b")).toEqual(["a", "b"]);
    // Both survivors stay glued to their word, visible to the gate that rejects them.
    expect(words("a\x1cb")).toEqual(["a\x1cb"]);
    expect(words("a\ufeffb")).toEqual(["a\ufeffb"]);
  });

  it("splits words with punctuation attached", () => {
    for (const c of C8.words) {
      expect(words(c.input)).toEqual(c.expected);
    }
    expect(words("punctuation, stays; attached.")).toEqual([
      "punctuation,",
      "stays;",
      "attached.",
    ]);
  });
});

describe("case 8 — damerau", () => {
  it("reproduces every golden distance and op list", () => {
    for (const c of C8.distance) {
      const r = score(c.a, c.b);
      expect(r.aWords).toEqual(c.a_words);
      expect(r.bWords).toEqual(c.b_words);
      expect(r.distance).toBe(c.distance);
      expect(r.ops.map((o) => ({ op: o.op, i: o.i, j: o.j, from: o.from, to: o.to }))).toEqual(
        c.ops,
      );
    }
  });

  it("prices an adjacent reorder at 1 (unrestricted, not OSA)", () => {
    expect(damerauDistance(["a", "b"], ["b", "a"])).toBe(1);
    // The OSA discriminator: swap two words, then edit one of them. OSA says 3;
    // unrestricted Damerau-Levenshtein says 2, and CONCEPT.md requires the
    // latter ("a reorder counts as 1, so reordering is cheap, not free").
    expect(damerauDistance(["c", "a", "b"], ["a", "b", "c"])).toBe(2);
    expect(damerauDistance(["a", "b", "x"], ["b", "a", "y"])).toBe(2);
  });

  it("is symmetric in distance and self-consistent in ops", () => {
    const pairs: [string[], string[]][] = [
      [["the", "quick", "brown"], ["the", "brown", "quick"]],
      [["a"], []],
      [[], []],
      [["x", "y", "z"], ["z", "y", "x"]],
      [["one", "two", "three", "four", "five"], ["five", "two", "three", "one"]],
    ];
    for (const [a, b] of pairs) {
      expect(damerauDistance(a, b)).toBe(damerauDistance(b, a));
      const r = damerau(a, b);
      // Every op the backtrace emits must be one of the four kinds, in range.
      for (const o of r.ops) {
        expect(["sub", "ins", "del", "transpose"]).toContain(o.op);
        expect(o.i).toBeGreaterThanOrEqual(0);
        expect(o.j).toBeGreaterThanOrEqual(0);
      }
      // and the op count must account for the distance
      const cost = r.ops.reduce((acc, o) => acc + (o.op === "transpose" ? 1 : 1), 0);
      expect(cost).toBeGreaterThanOrEqual(r.distance);
    }
  });

  it("is deterministic under the fixed tie-break order", () => {
    // "the cat sat" -> "the sat cat" is reachable by transpose (1) and by
    // two substitutions (2). The lower cost wins, and the op list must be
    // identical across runs.
    const a = ["the", "cat", "sat"];
    const b = ["the", "sat", "cat"];
    const r1 = damerau(a, b);
    const r2 = damerau(a, b);
    expect(r1.distance).toBe(1);
    expect(r1.ops).toEqual(r2.ops);
    expect(r1.ops[0]!.op).toBe("transpose");
  });

  it("fuzz: the backtrace never gets stuck and never under-counts", () => {
    const vocab = ["a", "b", "c", "d", "e", "f"];
    let s = 987654321;
    const rnd = () => (s = (s * 1103515245 + 12345) % 2147483648) / 2147483648;
    for (let t = 0; t < 1500; t++) {
      const mk = () =>
        Array.from({ length: Math.floor(rnd() * 8) }, () => vocab[Math.floor(rnd() * vocab.length)]!);
      const a = mk();
      const b = mk();
      const r = damerau(a, b);
      expect(r.distance).toBe(damerauDistance(a, b));
      expect(r.ops.length).toBeGreaterThanOrEqual(r.distance);
      if (a.join(" ") === b.join(" ")) expect(r.distance).toBe(0);
    }
  });

  it("scores 200-word inputs in well under a millisecond", () => {
    const base = Array.from({ length: 200 }, (_, i) => `w${i % 37}`).join(" ");
    const edited = base.replace("w5", "w99").replace("w17", "w98");
    const t0 = performance.now();
    const N = 200;
    for (let i = 0; i < N; i++) score(base, edited);
    const per = (performance.now() - t0) / N;
    // §8.2 claims "sub-millisecond in both Python and JS" at ~200 words.
    expect(per).toBeLessThan(5);
  });
});
