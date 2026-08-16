/**
 * MIRROR ALIGNMENT IS THE PROJECT'S #4 RISK (TECH_PLAN.md §14.1).
 *
 * There are two halves to testing it and this file is the half that can run in
 * CI on a machine with no browser:
 *
 *   1. STRUCTURE. The mirror must contain every character of the source exactly
 *      once, in order, and must reconcile rather than replace — a renderer that
 *      rebuilds innerHTML looks right in a screenshot and is dead in motion,
 *      because brand-new elements start at their final computed value and CSS
 *      transitions never run (§10.4).
 *   2. THE CSS CONTRACT. §10.3 lists the properties that must match EXACTLY on
 *      the textarea and the mirror. Three of them are the killers (kerning /
 *      ligatures, integer-px line-height, text-size-adjust) and all three fail
 *      only on a real device, at the right end of a long line. So the rule is
 *      enforced textually, here, where it cannot regress unnoticed.
 *
 * The other half — do the two elements actually put glyphs in the same place —
 * needs a layout engine and lives in tests/e2e/mirror-alignment.spec.ts.
 *
 * There is no jsdom in this project's devDependencies, so the structural half
 * runs against a small DOM stand-in below. It implements exactly the surface
 * Mirror touches. That is a deliberate limitation and it is why the Playwright
 * harness exists.
 */

import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

import {
  Mirror,
  layoutFromTokens,
  layoutFromWords,
  reanchorTokens,
  RIPPLE_CLAMP,
} from "../src/game/mirror";
import type { TokenHeat } from "../src/state";

/* ------------------------------------------------------------------ *
 * A DOM stand-in: exactly the surface Mirror touches, nothing else.
 * ------------------------------------------------------------------ */

class FakeStyle {
  readonly props = new Map<string, string>();
  transform = "";
  setProperty(name: string, value: string): void {
    this.props.set(name, value);
  }
  removeProperty(name: string): void {
    this.props.delete(name);
  }
}

class FakeNode {
  nodeType: number;
  nodeName: string;
  data = "";
  className = "";
  childNodes: FakeNode[] = [];
  parentNode: FakeNode | null = null;
  readonly attrs = new Map<string, string>();
  readonly style = new FakeStyle();
  ownerDocument: FakeDocument;

  constructor(doc: FakeDocument, nodeName: string, nodeType = 1) {
    this.ownerDocument = doc;
    this.nodeName = nodeName;
    this.nodeType = nodeType;
  }

  get firstChild(): FakeNode | null {
    return this.childNodes[0] ?? null;
  }

  get textContent(): string {
    if (this.nodeType === 3) return this.data;
    return this.childNodes.map((c) => c.textContent).join("");
  }

  set textContent(value: string) {
    if (this.nodeType === 3) {
      this.data = value;
      return;
    }
    this.childNodes = [];
    if (value !== "") this.appendChild(this.ownerDocument.createTextNode(value));
  }

  appendChild(node: FakeNode): FakeNode {
    node.parentNode?.removeChild(node);
    node.parentNode = this;
    this.childNodes.push(node);
    return node;
  }

  insertBefore(node: FakeNode, ref: FakeNode | null): FakeNode {
    node.parentNode?.removeChild(node);
    node.parentNode = this;
    const at = ref === null ? -1 : this.childNodes.indexOf(ref);
    if (at < 0) this.childNodes.push(node);
    else this.childNodes.splice(at, 0, node);
    return node;
  }

  removeChild(node: FakeNode): FakeNode {
    const at = this.childNodes.indexOf(node);
    if (at >= 0) this.childNodes.splice(at, 1);
    node.parentNode = null;
    return node;
  }

  setAttribute(name: string, value: string): void {
    this.attrs.set(name, value);
  }

  removeAttribute(name: string): void {
    this.attrs.delete(name);
  }

  getAttribute(name: string): string | null {
    return this.attrs.get(name) ?? null;
  }
}

class FakeDocument {
  readonly defaultView = null;
  createElement(name: string): FakeNode {
    return new FakeNode(this, name.toUpperCase(), 1);
  }
  createTextNode(data: string): FakeNode {
    const n = new FakeNode(this, "#text", 3);
    n.data = data;
    return n;
  }
}

function makeMirror(): { mirror: Mirror; root: FakeNode } {
  const doc = new FakeDocument();
  const root = doc.createElement("div");
  return { mirror: new Mirror(root as unknown as HTMLElement), root };
}

function tokensOf(root: FakeNode): FakeNode[] {
  return root.childNodes.filter((c) => c.className === "tk");
}

const TEXT = "The committee met on Thursday to consider the proposal.";

function wordTokens(text: string, heat: (i: number) => number): TokenHeat[] {
  return layoutFromWords(text).spans.map((sp, i) => ({
    s: sp.s,
    e: sp.e,
    heat: heat(i),
    masked: false,
  }));
}

/* ------------------------------------------------------------------ *
 * 1. Layout invariants
 * ------------------------------------------------------------------ */

describe("layout", () => {
  it("covers every character of the source exactly once, in order", () => {
    const layout = layoutFromTokens(TEXT, wordTokens(TEXT, () => 0));
    let joined = "";
    for (let i = 0; i < layout.toks.length; i++) {
      joined += layout.gaps[i] ?? "";
      joined += layout.toks[i] ?? "";
    }
    joined += layout.gaps[layout.toks.length] ?? "";
    expect(joined).toBe(TEXT);
    expect(layout.gaps).toHaveLength(layout.toks.length + 1);
  });

  it("keeps the invariant when the tokenizer emits leading-space tokens", () => {
    // Gemma-3 pieces carry their own leading space; the gaps are then empty.
    const text = "one two three";
    const tokens: TokenHeat[] = [
      { s: 0, e: 3, heat: 0.2, masked: false },
      { s: 3, e: 7, heat: 0.9, masked: false },
      { s: 7, e: 13, heat: 0.0, masked: true },
    ];
    const layout = layoutFromTokens(text, tokens);
    expect(layout.toks).toEqual(["one", " two", " three"]);
    expect(layout.gaps.join("")).toBe("");
    expect(layout.masked).toEqual([false, false, true]);
  });

  it("drops overlapping or reversed spans rather than duplicating text", () => {
    const text = "abcdef";
    const layout = layoutFromTokens(text, [
      { s: 0, e: 3, heat: 0, masked: false },
      { s: 1, e: 4, heat: 0, masked: false }, // overlaps: dropped
      { s: 4, e: 6, heat: 0, masked: false },
      { s: 5, e: 5, heat: 0, masked: false }, // empty: dropped
    ]);
    expect(layout.toks).toEqual(["abc", "ef"]);
    expect(layout.gaps.join("") + layout.toks.join("")).toHaveLength(text.length);
  });

  it("layoutFromWords covers the source including leading and trailing space", () => {
    const text = "  hello   world \n";
    const layout = layoutFromWords(text);
    let joined = "";
    for (let i = 0; i < layout.toks.length; i++) {
      joined += (layout.gaps[i] ?? "") + (layout.toks[i] ?? "");
    }
    joined += layout.gaps[layout.toks.length] ?? "";
    expect(joined).toBe(text);
  });
});

/* ------------------------------------------------------------------ *
 * 2. Carrying tokens across a keystroke
 * ------------------------------------------------------------------ */

describe("reanchorTokens", () => {
  const old = "the cat sat on the mat";
  const tokens = wordTokens(old, (i) => (i + 1) / 10);

  it("keeps prefix heat, shifts suffix offsets and zeroes only the dirty span", () => {
    const next = "the cat slept on the mat";
    const out = reanchorTokens(tokens, old, next);
    // still sorted and non-overlapping
    for (let i = 1; i < out.length; i++) {
      expect((out[i]?.s ?? 0) >= (out[i - 1]?.e ?? 0)).toBe(true);
    }
    // the layout still reproduces the new text exactly
    const layout = layoutFromTokens(next, out);
    let joined = "";
    for (let i = 0; i < layout.toks.length; i++) {
      joined += (layout.gaps[i] ?? "") + (layout.toks[i] ?? "");
    }
    expect(joined + (layout.gaps[layout.toks.length] ?? "")).toBe(next);
    // "the" and "cat" are untouched, and their real heat survives
    expect(out[0]).toEqual(tokens[0]);
    expect(out[1]).toEqual(tokens[1]);
    // the changed region carries NO heat: absence, never an invented value
    const dirty = out.filter((t) => t.heat === 0);
    expect(dirty).toHaveLength(1);
    expect(next.slice(dirty[0]?.s ?? 0, dirty[0]?.e ?? 0)).toContain("l");
    expect(out.filter((t) => t.heat !== 0)).toHaveLength(tokens.length - 1);
    // the tail keeps its measured heat, shifted by the length delta
    const tail = out[out.length - 1];
    expect(tail?.heat).toBe(tokens[tokens.length - 1]?.heat);
    expect(next.slice(tail?.s ?? 0, tail?.e ?? 0)).toBe("mat");
  });

  it("handles pure insertion at the end and pure deletion at the start", () => {
    const appended = `${old} today`;
    const grown = reanchorTokens(tokens, old, appended);
    expect(layoutFromTokens(appended, grown).toks.join("").length).toBeGreaterThan(0);
    const shortened = "cat sat on the mat";
    const shrunk = reanchorTokens(tokens, old, shortened);
    const layout = layoutFromTokens(shortened, shrunk);
    let joined = "";
    for (let i = 0; i < layout.toks.length; i++) {
      joined += (layout.gaps[i] ?? "") + (layout.toks[i] ?? "");
    }
    expect(joined + (layout.gaps[layout.toks.length] ?? "")).toBe(shortened);
  });

  it("is a no-op when the text did not change", () => {
    expect(reanchorTokens(tokens, old, old)).toEqual(tokens);
  });

  /* --- astral characters: the scan counts CODE UNITS ---------------- *
   * The prefix/suffix scan compares UTF-16 code units, and a boundary that
   * lands between the two halves of a surrogate pair splits a character the
   * renderer then puts in two different elements. Surrogate halves do not
   * shape across an element boundary, so the mirror draws two replacement
   * boxes where the textarea draws one glyph, and everything to the right of
   * it on that line is off by the difference. Every emoji in U+1F600..U+1F63F
   * shares the high surrogate D83D, so swapping one for another is the case,
   * not an exotic one.
   * ------------------------------------------------------------------ */
  const codePointBoundaries = (text: string): Set<number> => {
    const out = new Set<number>();
    for (let i = 0; i < text.length; ) {
      out.add(i);
      i += (text.codePointAt(i) ?? 0) > 0xffff ? 2 : 1;
    }
    out.add(text.length);
    return out;
  };

  const assertNoSplitPairs = (text: string, out: readonly TokenHeat[]): void => {
    const legal = codePointBoundaries(text);
    for (const t of out) {
      expect(legal.has(t.s)).toBe(true);
      expect(legal.has(t.e)).toBe(true);
    }
  };

  it("never cuts a surrogate pair when the two emoji share a high surrogate", () => {
    // U+1F600 and U+1F601: D83D DE00 -> D83D DE01. The common prefix matches
    // the first unit and stops on the second.
    const before = "a \u{1F600} b";
    const after = "a \u{1F601} b";
    const out = reanchorTokens(wordTokens(before, () => 0.5), before, after);
    assertNoSplitPairs(after, out);
    // and the character is inside the dirty span, so it renders as one glyph
    const dirty = out.find((t) => t.heat === 0);
    expect(after.slice(dirty?.s ?? 0, dirty?.e ?? 0)).toContain("\u{1F601}");
  });

  it("never cuts a surrogate pair when the two emoji share a low surrogate", () => {
    // U+1F600 and U+1FA00: D83D DE00 -> D83E DE00. Here it is the common
    // SUFFIX that starts mid-pair.
    const before = "a \u{1F600} b";
    const after = "a \u{1FA00} b";
    assertNoSplitPairs(after, reanchorTokens(wordTokens(before, () => 0.5), before, after));
  });

  it("still reproduces the text exactly across an astral edit", () => {
    const before = "one \u{1F600}\u{1F601} three";
    const after = "one \u{1F601}\u{1F601} three";
    const out = reanchorTokens(wordTokens(before, () => 0.5), before, after);
    const layout = layoutFromTokens(after, out);
    let joined = "";
    for (let i = 0; i < layout.toks.length; i++) {
      joined += (layout.gaps[i] ?? "") + (layout.toks[i] ?? "");
    }
    expect(joined + (layout.gaps[layout.toks.length] ?? "")).toBe(after);
    assertNoSplitPairs(after, out);
  });
});

/* ------------------------------------------------------------------ *
 * 3. The renderer reconciles; it never replaces
 * ------------------------------------------------------------------ */

describe("Mirror", () => {
  it("renders gap, token, gap, ... and always a trailing <br>", () => {
    const { mirror, root } = makeMirror();
    mirror.setText(layoutFromTokens(TEXT, wordTokens(TEXT, () => 0)));
    const kinds = root.childNodes.map((c) => (c.nodeName === "BR" ? "br" : c.className || "gap"));
    expect(kinds[kinds.length - 1]).toBe("br");
    expect(kinds.filter((k) => k === "tk")).toHaveLength(9);
    // strict alternation gap, tk, gap, tk, ..., gap, br
    for (let i = 0; i < kinds.length - 1; i++) {
      expect(kinds[i]).toBe(i % 2 === 0 ? "gap" : "tk");
    }
    expect(mirror.renderedText()).toBe(TEXT);
  });

  it("REUSES the same span objects across an update (this is the ripple)", () => {
    const { mirror, root } = makeMirror();
    mirror.setText(layoutFromTokens(TEXT, wordTokens(TEXT, () => 0)));
    const before = tokensOf(root);
    const next = TEXT.replace("proposal", "proposals");
    mirror.setText(layoutFromTokens(next, wordTokens(next, () => 0)));
    const after = tokensOf(root);
    expect(after).toHaveLength(before.length);
    after.forEach((span, i) => expect(span).toBe(before[i]));
    expect(mirror.renderedText()).toBe(next);
  });

  it("writes only paint-only properties, and --d is clamped", () => {
    const { mirror, root } = makeMirror();
    const heat = [0.1, 0.9, 0.0, 0.4];
    const text = "aa bb cc dd";
    mirror.setText(layoutFromTokens(text, wordTokens(text, (i) => heat[i] ?? 0)));
    mirror.setHeat(heat, { editIndex: 1, prev: [0.1, 0.1, 0.0, 0.4] });
    const spans = tokensOf(root);
    for (const span of spans) {
      // ONLY these three; anything else changes line breaking (§10.4)
      expect([...span.style.props.keys()].sort()).toEqual(["--a", "--d", "--h"]);
    }
    expect(spans[1]?.getAttribute("data-dir")).toBe("heating");
    expect(spans[1]?.getAttribute("data-impulse")).toBe("1");
    expect(spans[0]?.getAttribute("data-dir")).toBeNull();
    expect(spans[3]?.style.props.get("--d")).toBe("2");
    expect(root.getAttribute("data-heat")).toBe("live");
  });

  it("clamps the ripple stagger so a long passage does not read as a loading bar", () => {
    const { mirror, root } = makeMirror();
    const many = Array.from({ length: 40 }, (_, i) => `w${i}`).join(" ");
    mirror.setText(layoutFromTokens(many, wordTokens(many, () => 0)));
    mirror.setHeat(new Array<number>(40).fill(0.5), { editIndex: 0 });
    const spans = tokensOf(root);
    expect(spans[39]?.style.props.get("--d")).toBe(String(RIPPLE_CLAMP));
  });

  it("marks a heat-less render pending rather than claiming zero heat", () => {
    const { mirror, root } = makeMirror();
    const layout = layoutFromWords(TEXT);
    mirror.hardRepaint(layout, new Array<number>(layout.toks.length).fill(0), true);
    expect(root.getAttribute("data-heat")).toBe("pending");
    // data-noanim is set and then cleared two frames later; with no
    // requestAnimationFrame in this stand-in it is cleared synchronously.
    expect(root.getAttribute("data-noanim")).toBeNull();
  });

  it("grows and shrinks the pool and still reproduces the text", () => {
    const { mirror, root } = makeMirror();
    const a = "one two";
    mirror.setText(layoutFromTokens(a, wordTokens(a, () => 0)));
    const b = "one two three four five";
    mirror.setText(layoutFromTokens(b, wordTokens(b, () => 0)));
    expect(tokensOf(root)).toHaveLength(5);
    expect(mirror.renderedText()).toBe(b);
    mirror.setText(layoutFromTokens(a, wordTokens(a, () => 0)));
    expect(tokensOf(root)).toHaveLength(2);
    expect(mirror.renderedText()).toBe(a);
    expect(root.childNodes[root.childNodes.length - 1]?.nodeName).toBe("BR");
  });

  it("marks masked tokens, which render grey rather than cold", () => {
    const { mirror, root } = makeMirror();
    const text = "one two";
    mirror.setText(
      layoutFromTokens(text, [
        { s: 0, e: 3, heat: 0.4, masked: false },
        { s: 4, e: 7, heat: 0, masked: true },
      ]),
    );
    expect(tokensOf(root)[1]?.getAttribute("data-masked")).toBe("1");
    expect(tokensOf(root)[0]?.getAttribute("data-masked")).toBeNull();
  });
});

/* ------------------------------------------------------------------ *
 * 4. The CSS contract (§10.3)
 * ------------------------------------------------------------------ */

const PASSAGE_CSS = readFileSync(new URL("../src/styles/passage.css", import.meta.url), "utf8");

/** Every property §10.3 says must match EXACTLY on both elements. */
const MUST_MATCH = [
  "box-sizing",
  "width",
  "margin",
  "border",
  "padding",
  "font-family",
  "font-size",
  "font-weight",
  "font-style",
  "line-height",
  "letter-spacing",
  "word-spacing",
  "text-indent",
  "text-transform",
  "text-rendering",
  "font-kerning",
  "font-variant-ligatures",
  "font-feature-settings",
  "-webkit-font-smoothing",
  "text-size-adjust",
  "white-space",
  "overflow-wrap",
  "word-break",
  "hyphens",
  "tab-size",
  "direction",
  "unicode-bidi",
];

/** Properties that must NEVER be set on one of the two boxes alone. */
const NEVER_ALONE = [
  "box-sizing",
  "font-family",
  "font-size",
  "font-weight",
  "font-style",
  "line-height",
  "letter-spacing",
  "word-spacing",
  "text-indent",
  "padding",
  "margin",
  "border",
  "white-space",
  "tab-size",
  "text-transform",
];

interface Rule {
  selector: string;
  body: string;
}

function rules(css: string): Rule[] {
  const out: Rule[] = [];
  const re = /([^{}]+)\{([^{}]*)\}/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(css)) !== null) {
    const selector = (m[1] ?? "").trim();
    if (selector.startsWith("@")) continue;
    out.push({ selector, body: m[2] ?? "" });
  }
  return out;
}

function declaredProps(body: string): Set<string> {
  const names = new Set<string>();
  for (const decl of body.split(";")) {
    const name = decl.split(":")[0]?.trim();
    if (name && !name.startsWith("/*")) names.add(name);
  }
  return names;
}

describe("the mirror/textarea CSS contract (§10.3)", () => {
  const shared = rules(PASSAGE_CSS).find(
    (r) => r.selector.includes(".mirror") && r.selector.includes(".raw"),
  );

  it("declares one shared block for both boxes", () => {
    expect(shared).toBeDefined();
  });

  it("sets every property that must match exactly", () => {
    const props = declaredProps(shared?.body ?? "");
    for (const name of MUST_MATCH) {
      const present =
        props.has(name) || props.has(`-webkit-${name}`) || props.has(name.replace(/^-webkit-/, ""));
      expect(present, `${name} is missing from the shared .mirror/.raw block`).toBe(true);
    }
  });

  it("uses an integer-px line-height, because textarea line boxes round differently", () => {
    const lh = /line-height:\s*([^;]+);/.exec(shared?.body ?? "")?.[1]?.trim();
    expect(lh).toMatch(/^\d+px$/);
  });

  it("uses a font-size of at least 16px, or iOS zooms on focus and never resets", () => {
    const fs = /font-size:\s*(\d+)px/.exec(shared?.body ?? "")?.[1];
    expect(Number(fs)).toBeGreaterThanOrEqual(16);
  });

  it("kills kerning and ligatures on both boxes", () => {
    const body = shared?.body ?? "";
    expect(body).toMatch(/font-kerning:\s*none/);
    expect(body).toMatch(/font-variant-ligatures:\s*none/);
    expect(body).toMatch(/"liga"\s*0/);
  });

  it("never sets a layout-affecting property on only one of the two", () => {
    for (const rule of rules(PASSAGE_CSS)) {
      const hasMirror = rule.selector.includes(".mirror");
      const hasRaw = rule.selector.includes(".raw");
      if (hasMirror === hasRaw) continue; // both, or neither
      for (const name of declaredProps(rule.body)) {
        expect(
          NEVER_ALONE.includes(name),
          `${rule.selector} sets ${name} on only one of the two boxes`,
        ).toBe(false);
      }
    }
  });

  it("holds the pair rule across EVERY stylesheet, not just this one", () => {
    // The responsive padding at 37.5em lives in state.css and sets both boxes;
    // that is fine. What is not fine is any rule anywhere that moves one of
    // them without the other. `.pr-demo .mirror` is exempt because the primer's
    // demo strip has no textarea over it at all.
    const EXEMPT = [".pr-demo .mirror"];
    for (const file of [
      "passage.css",
      "rail.css",
      "gate.css",
      "sheets.css",
      "intro.css",
      "result.css",
      "state.css",
    ]) {
      const css = readFileSync(new URL(`../src/styles/${file}`, import.meta.url), "utf8");
      for (const rule of rules(css)) {
        if (EXEMPT.some((sel) => rule.selector.includes(sel))) continue;
        const hasMirror = rule.selector.includes(".mirror");
        const hasRaw = rule.selector.includes(".raw");
        if (hasMirror === hasRaw) continue;
        for (const name of declaredProps(rule.body)) {
          expect(
            NEVER_ALONE.includes(name),
            `${file}: ${rule.selector} sets ${name} on only one of the two boxes`,
          ).toBe(false);
        }
      }
    }
  });

  it("caps the heat wash so the real text keeps its contrast", () => {
    const tokens = readFileSync(new URL("../src/styles/tokens.css", import.meta.url), "utf8");
    const light = /--cap:\s*(\d+)/.exec(tokens)?.[1];
    expect(Number(light)).toBeLessThanOrEqual(36);
    const dark = [...tokens.matchAll(/--cap:\s*(\d+)/g)].map((m) => Number(m[1]));
    for (const cap of dark) expect(cap).toBeLessThanOrEqual(45);
  });
});
