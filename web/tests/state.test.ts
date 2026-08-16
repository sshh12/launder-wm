/**
 * The one reconciliation rule, the copy layer, and the diff exhibit.
 *
 * §10.2: apply a Reading only if `reading.seq >= state.lastSeq` AND
 * `reading.textHash === sha256(state.text)`. Everything else is discarded
 * silently. Out-of-order responses are the #1 source of needle flicker (§5.4),
 * so this is the single most load-bearing branch in the client and it is worth
 * more tests than it has lines.
 */

import { describe, expect, it } from "vitest";

import { diffSegments, words } from "../src/game/diff";
import { Copy, Store, readingFromWire, sha256Hex, type DetectResponseWire } from "../src/state";

function wire(over: Partial<DetectResponseWire> = {}): DetectResponseWire {
  return {
    seq: 1,
    text_hash: "",
    score: 0.53,
    z: 6.44,
    z_star: 2.3263,
    n_scored: 171,
    n_tokens: 194,
    tokens: [{ s: 0, e: 3, heat: 0.61, masked: false }],
    preview_distance: 4,
    ...over,
  };
}

describe("Store reconciliation", () => {
  it("applies a fresh reading and advances lastSeq", () => {
    const store = new Store("hello");
    expect(store.applyReading(readingFromWire(wire({ seq: 3 })), "hello")).toBe(true);
    expect(store.get().lastSeq).toBe(3);
    expect(store.get().applied?.z).toBe(6.44);
    expect(store.isStale()).toBe(false);
  });

  it("DROPS a reading whose seq is below the last applied one", () => {
    const store = new Store("hello");
    store.applyReading(readingFromWire(wire({ seq: 7 })), "hello");
    const late = store.applyReading(readingFromWire(wire({ seq: 6, z: 0.1 })), "hello");
    expect(late).toBe(false);
    expect(store.get().applied?.z).toBe(6.44);
  });

  it("DROPS a reading computed for text the player has already moved past", () => {
    const store = new Store("hello");
    store.setText("hello there");
    expect(store.applyReading(readingFromWire(wire({ seq: 9 })), "hello")).toBe(false);
    expect(store.get().applied).toBeNull();
    expect(store.isStale()).toBe(true);
  });

  it("DROPS a reading whose text_hash does not match the current text", async () => {
    const store = new Store("hello");
    // NOT `setTimeout(0)`: crypto.subtle.digest is thread-pool backed and can
    // resolve after a macrotask under load, which made this test flaky.
    await store.hashSettled();
    const bad = readingFromWire(wire({ seq: 2, text_hash: `sha256:${"0".repeat(64)}` }));
    expect(store.applyReading(bad, "hello")).toBe(false);
  });

  it("DROPS a reading with no sourceText while the digest is still in flight", () => {
    // §10.2's rule needs PROOF the reading describes the current text: either
    // the exact source string or a matching digest. Between a keystroke and the
    // digest landing there is neither, and this used to APPLY — installing a
    // reading for superseded text and then reporting isStale() === false.
    const store = new Store("first text");
    store.setText("second text"); // no await: textHash is null right now
    const stale = readingFromWire(wire({ seq: 5, z: 7.7, text_hash: `sha256:${"a".repeat(64)}` }));
    expect(store.applyReading(stale)).toBe(false);
    expect(store.get().applied).toBeNull();
    expect(store.isStale()).toBe(true);
  });

  it("ACCEPTS a reading whose text_hash matches", async () => {
    const store = new Store("hello");
    await store.hashSettled();
    const hash = await sha256Hex("hello");
    expect(hash).toMatch(/^sha256:[0-9a-f]{64}$/);
    const good = readingFromWire(wire({ seq: 2, text_hash: hash ?? "" }));
    expect(store.applyReading(good, "hello")).toBe(true);
  });

  it("hands seq numbers out monotonically across both detector paths", () => {
    const store = new Store("x");
    expect([store.nextSeq(), store.nextSeq(), store.nextSeq()]).toEqual([1, 2, 3]);
  });

  it("drops a submission outcome the moment the text changes under it", () => {
    const store = new Store("hello");
    store.setOutcome({
      cleared: true,
      provisional: false,
      score: { distance: 4, ops: [] },
      detector: { score: 0.5, z: 1.7, z_star: 2.3263, n_scored: 171 },
      trace: [],
    });
    expect(store.get().outcome).not.toBeNull();
    store.setText("hello!");
    expect(store.get().outcome).toBeNull();
  });

  it("notifies subscribers with the change kind", () => {
    const store = new Store("a");
    const seen: string[] = [];
    const off = store.subscribe((_s, change) => seen.push(change));
    store.setText("b");
    store.setMode("LOADING");
    store.setSubmitting(true);
    store.applyReading(readingFromWire(wire({ seq: 1 })), "b");
    off();
    store.setText("c");
    expect(seen).toEqual(["text", "mode", "submitting", "reading"]);
  });
});

describe("Copy", () => {
  const copy = new Copy({
    readout: { threshold_label: "Under {z_star_display} to clear", above: "AI detected" },
    check: { edit_budget: { label: "Budget {max_word_distance} words" } },
  });

  it("fills placeholders from params", () => {
    expect(copy.t("readout.threshold_label", { z_star_display: "2.33" })).toBe(
      "Under 2.33 to clear",
    );
    expect(copy.t("check.edit_budget.label", { max_word_distance: 12 })).toBe("Budget 12 words");
  });

  it("leaves an unresolved placeholder VISIBLE rather than blanking it", () => {
    expect(copy.t("readout.threshold_label")).toBe("Under {z_star_display} to clear");
  });

  it("renders a missing key as the key itself: loud, not blank", () => {
    expect(copy.t("screen.does_not_exist")).toBe("screen.does_not_exist");
    expect(copy.has("screen.does_not_exist")).toBe(false);
    expect(copy.has("readout.above")).toBe(true);
  });
});

describe("diff exhibit", () => {
  it("renders a substitution as del+ins", () => {
    const a = words("the audit found seventeen versions");
    const b = words("the audit found eighteen versions");
    const segs = diffSegments(a, b, [{ op: "sub", i: 3, j: 3, from: "seventeen", to: "eighteen" }]);
    expect(segs.map((s) => `${s.kind}:${s.words.join(" ")}`)).toEqual([
      "eq:the audit found",
      "del:seventeen",
      "ins:eighteen",
      "eq:versions",
    ]);
  });

  it("renders insertions and deletions", () => {
    const a = words("one two three");
    const b = words("one three four");
    const segs = diffSegments(a, b, [
      { op: "del", i: 1, j: 1, from: "two", to: "" },
      { op: "ins", i: 3, j: 2, from: "", to: "four" },
    ]);
    expect(segs.map((s) => s.kind)).toEqual(["eq", "del", "eq", "ins"]);
    expect(segs[3]?.words).toEqual(["four"]);
  });

  it("renders a transposition as the pair that moved", () => {
    const a = words("alpha beta gamma");
    const b = words("beta alpha gamma");
    const segs = diffSegments(a, b, [{ op: "transpose", i: 0, j: 0, from: "alpha", to: "beta" }]);
    expect(segs[0]?.kind).toBe("del");
    expect(segs[0]?.words).toEqual(["alpha", "beta"]);
    expect(segs[1]?.words).toEqual(["beta", "alpha"]);
    expect(segs[2]?.words).toEqual(["gamma"]);
  });

  it("never loses or duplicates a word when the op list is empty", () => {
    const a = words("a b c");
    const segs = diffSegments(a, a, []);
    expect(segs).toEqual([{ kind: "eq", words: ["a", "b", "c"] }]);
  });
});
