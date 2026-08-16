/**
 * The heat mirror — TECH_PLAN.md §10.3 and §10.4.
 *
 * A RECONCILING renderer. It owns the span pool and writes only paint-only
 * custom properties (`--h`, `--a`, `--d`) plus two data attributes.
 *
 * THE TRAP THIS MODULE EXISTS TO AVOID: rebuilding `mirror.innerHTML` on every
 * re-score creates brand-new elements, which start at their final computed
 * value, so CSS transitions never run. It looks correct in a screenshot and
 * dead in motion. There is therefore no `innerHTML` anywhere in this file, and
 * `web/tests/mirror.test.ts` asserts that span identity survives an update.
 *
 * The mirror is in normal flow and DEFINES the passage box height; the textarea
 * is absolutely positioned and stretched to it. `setText` is synchronous for
 * that reason — the box must never wrap into a 100ms-stale height.
 *
 * The mirror contains every character of the source exactly once, in order:
 * token spans plus gap spans for inter-token text. Nothing here is escaped by
 * hand because nothing here is ever parsed as HTML — every character reaches
 * the DOM through `Text.data` / `textContent`.
 */

import type { TokenHeat } from "../state";

export interface MirrorLayout {
  /** inter-token text; always `toks.length + 1` entries, some empty */
  gaps: string[];
  toks: string[];
  /** char offsets of each token into the source text */
  spans: { s: number; e: number }[];
  masked: boolean[];
}

/** Ripple propagation is clamped at 16 tokens -> 224ms (§10.4). Unclamped
 *  across 300 tokens takes 4s and reads as a loading bar. */
export const RIPPLE_CLAMP = 16;

/**
 * Build a layout from the detector's CHAR OFFSETS. This is why the mirror needs
 * no client tokenizer in SERVER mode (§9.2) — and why the local detector's
 * output can be rendered by the very same code.
 *
 * Tokens that overlap or run backwards are dropped rather than trusted: the
 * "every character exactly once" invariant is what keeps the mirror aligned
 * with the textarea, and it outranks any individual token's heat.
 */
export function layoutFromTokens(text: string, tokens: readonly TokenHeat[]): MirrorLayout {
  const gaps: string[] = [];
  const toks: string[] = [];
  const spans: { s: number; e: number }[] = [];
  const masked: boolean[] = [];
  let cursor = 0;
  for (const t of tokens) {
    const s = Math.max(0, Math.min(text.length, t.s));
    const e = Math.max(0, Math.min(text.length, t.e));
    if (s < cursor || e <= s) continue;
    gaps.push(text.slice(cursor, s));
    toks.push(text.slice(s, e));
    spans.push({ s, e });
    masked.push(t.masked);
    cursor = e;
  }
  gaps.push(text.slice(cursor));
  return { gaps, toks, spans, masked };
}

/**
 * Layout with no reading behind it: whitespace-delimited words, purely as a
 * geometry scaffold so the passage box has its final height and the alignment
 * harness has something to measure before the first /api/detect answers.
 *
 * These spans carry NO heat and the mirror is marked `data-heat="pending"`.
 * Inventing heat the detector did not compute is forbidden (§10.4).
 */
export function layoutFromWords(text: string): MirrorLayout {
  const gaps: string[] = [];
  const toks: string[] = [];
  const spans: { s: number; e: number }[] = [];
  const masked: boolean[] = [];
  const re = /\S+/g;
  let cursor = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    gaps.push(text.slice(cursor, m.index));
    toks.push(m[0]);
    spans.push({ s: m.index, e: m.index + m[0].length });
    masked.push(false);
    cursor = m.index + m[0].length;
  }
  gaps.push(text.slice(cursor));
  return { gaps, toks, spans, masked };
}

/**
 * Carry a token list across a keystroke.
 *
 * The mirror must contain the CURRENT text on the same frame the player typed —
 * it defines the passage box's height, and a stale mirror clips the last line
 * of a textarea that cannot scroll. But the tokens describing that text do not
 * arrive until the detector answers (one RTT in SERVER mode).
 *
 * So: keep every token that lies entirely inside the unchanged prefix (offsets
 * and heat unchanged, because a token's g-value depends only on the tokens
 * BEFORE it), shift every token inside the unchanged suffix by the length
 * delta, and collapse the changed span into ONE token with no heat.
 *
 * The dirty span renders with no wash, which reads as "not yet measured" and is
 * the only honest thing to show: absence is not the same as a heat value, and
 * inventing one for the word the player is mid-way through typing is exactly
 * the invention §10.4 forbids.
 */
/** `NaN` (an out-of-range index) is neither, which is what makes the boundary
 *  checks below safe at position 0 and at the end of the string. */
const isHighSurrogate = (u: number): boolean => u >= 0xd800 && u <= 0xdbff;
const isLowSurrogate = (u: number): boolean => u >= 0xdc00 && u <= 0xdfff;

export function reanchorTokens(
  tokens: readonly TokenHeat[],
  oldText: string,
  newText: string,
): TokenHeat[] {
  if (oldText === newText) return tokens.slice();
  const maxCommon = Math.min(oldText.length, newText.length);
  let p = 0;
  while (p < maxCommon && oldText.charCodeAt(p) === newText.charCodeAt(p)) p += 1;
  let s = 0;
  while (
    s < maxCommon - p &&
    oldText.charCodeAt(oldText.length - 1 - s) === newText.charCodeAt(newText.length - 1 - s)
  ) {
    s += 1;
  }
  // THE SCAN COUNTS UTF-16 CODE UNITS, AND A CODE POINT CAN BE TWO OF THEM.
  //
  // Every emoji in U+1F600..U+1F63F shares the high surrogate D83D, so replacing
  // one with another matched the first unit and stopped on the second: the
  // common prefix ended BETWEEN the halves of an astral character. The dirty
  // span then began mid-pair, `layoutFromTokens` sliced there, and the mirror
  // rendered a lone high surrogate in the gap span and a lone low surrogate in
  // the token — two replacement boxes where the textarea draws one emoji. The
  // "every character exactly once" invariant survives that (the concatenation is
  // unchanged) but the GEOMETRY does not, and the mirror's geometry is the only
  // reason it exists. Surrogate halves never shape across an element boundary,
  // so the fix is to refuse a boundary that splits a pair: back the prefix off
  // its trailing high surrogate and the suffix off its leading low surrogate.
  // Both only ever shrink, so `p + s <= maxCommon` still holds.
  if (
    isHighSurrogate(oldText.charCodeAt(p - 1)) &&
    (isLowSurrogate(oldText.charCodeAt(p)) || isLowSurrogate(newText.charCodeAt(p)))
  ) {
    p -= 1;
  }
  if (
    isLowSurrogate(oldText.charCodeAt(oldText.length - s)) &&
    (isHighSurrogate(oldText.charCodeAt(oldText.length - s - 1)) ||
      isHighSurrogate(newText.charCodeAt(newText.length - s - 1)))
  ) {
    s -= 1;
  }
  const delta = newText.length - oldText.length;
  const oldSuffixStart = oldText.length - s;
  const out: TokenHeat[] = [];
  for (const t of tokens) {
    if (t.e <= p) out.push(t);
  }
  const dirtyStart = p;
  const dirtyEnd = newText.length - s;
  if (dirtyEnd > dirtyStart) {
    out.push({ s: dirtyStart, e: dirtyEnd, heat: 0, masked: false });
  }
  for (const t of tokens) {
    if (t.s >= oldSuffixStart && t.s >= p) {
      out.push({ s: t.s + delta, e: t.e + delta, heat: t.heat, masked: t.masked });
    }
  }
  return out;
}

export interface HeatOptions {
  /** token index the caret was in when this edit happened, or null */
  editIndex?: number | null;
  /** the previous heat array, so heating/cooling can be told apart */
  prev?: readonly number[] | null;
  masked?: readonly boolean[];
  /** true while no real reading describes the current text */
  pending?: boolean;
}

export class Mirror {
  readonly el: HTMLElement;
  private tk: HTMLSpanElement[] = [];
  private ws: HTMLSpanElement[] = [];
  private readonly br: HTMLBRElement;
  private layout: MirrorLayout = { gaps: [""], toks: [], spans: [], masked: [] };
  private heat: number[] = [];
  private readonly reduceMotion: MediaQueryList | null;

  constructor(el: HTMLElement) {
    this.el = el;
    // ALWAYS append a trailing <br> (§10.3): a trailing newline occupies a line
    // box in a textarea but not in a pre-wrap div, so without it the box loses
    // a line exactly when the player presses Enter at the end.
    this.br = el.ownerDocument.createElement("br");
    el.appendChild(this.br);
    this.reduceMotion =
      typeof el.ownerDocument.defaultView?.matchMedia === "function"
        ? el.ownerDocument.defaultView.matchMedia("(prefers-reduced-motion: reduce)")
        : null;
    this.grow(0);
  }

  get tokenCount(): number {
    return this.tk.length;
  }

  /** Exposed for the alignment harness and the reconciliation test. */
  tokenSpans(): readonly HTMLSpanElement[] {
    return this.tk;
  }

  tokenOffsets(): readonly { s: number; e: number }[] {
    return this.layout.spans;
  }

  currentHeat(): readonly number[] {
    return this.heat;
  }

  /** Token index containing `char`, or the token just before it. */
  tokenIndexAtChar(char: number): number | null {
    const spans = this.layout.spans;
    if (spans.length === 0) return null;
    for (let i = 0; i < spans.length; i++) {
      const sp = spans[i];
      if (sp !== undefined && char <= sp.e) return i;
    }
    return spans.length - 1;
  }

  private grow(n: number): void {
    const doc = this.el.ownerDocument;
    while (this.ws.length < n + 1) {
      const w = doc.createElement("span");
      this.el.insertBefore(w, this.br);
      this.ws.push(w);
      if (this.tk.length < n) {
        const t = doc.createElement("span");
        t.className = "tk";
        this.el.insertBefore(t, this.br);
        this.tk.push(t);
      }
    }
    while (this.tk.length > n) {
      const t = this.tk.pop();
      if (t) this.el.removeChild(t);
    }
    while (this.ws.length > n + 1) {
      const w = this.ws.pop();
      if (w) this.el.removeChild(w);
    }
    // Re-thread the pool into gap, token, gap, token, ..., gap, <br> order.
    let ref: Node = this.br;
    for (let i = n; i >= 0; i--) {
      const t = this.tk[i];
      if (t) {
        this.el.insertBefore(t, ref);
        ref = t;
      }
      const w = this.ws[i];
      if (w) {
        this.el.insertBefore(w, ref);
        ref = w;
      }
    }
  }

  private static write(span: HTMLSpanElement, text: string): void {
    const first = span.firstChild;
    if (first !== null && first.nodeType === 3) {
      if ((first as Text).data !== text) (first as Text).data = text;
    } else {
      span.textContent = text;
    }
  }

  /** Synchronous, so the box never wraps into a stale height. */
  setText(layout: MirrorLayout): void {
    this.layout = layout;
    const n = layout.toks.length;
    if (this.tk.length !== n) this.grow(n);
    for (let k = 0; k < n; k++) {
      const t = this.tk[k];
      const w = this.ws[k];
      if (t === undefined || w === undefined) continue;
      Mirror.write(t, layout.toks[k] ?? "");
      Mirror.write(w, layout.gaps[k] ?? "");
      if (layout.masked[k] === true) t.setAttribute("data-masked", "1");
      else t.removeAttribute("data-masked");
    }
    const tail = this.ws[n];
    if (tail !== undefined) Mirror.write(tail, layout.gaps[n] ?? "");
  }

  /**
   * Write heat. For every surviving span this touches exactly two registered
   * custom properties and two data attributes — all paint-only. Never a layout
   * property: a span that changed `padding` or `letter-spacing` would change
   * line breaking and desynchronise the mirror from the textarea (§10.4).
   *
   * The animation supplies TIMING ONLY, never magnitude. If the real g-values
   * downstream barely move, the ripple is barely visible, and that is correct.
   */
  setHeat(heat: readonly number[], opts: HeatOptions = {}): void {
    const editIndex = opts.editIndex ?? null;
    const prev = opts.prev ?? null;
    const next: number[] = [];
    for (let k = 0; k < this.tk.length; k++) {
      const t = this.tk[k];
      const h = heat[k] ?? 0;
      next.push(h);
      if (t === undefined) continue;
      const ph = prev?.[k] ?? h;
      if (h > ph + 0.001) t.setAttribute("data-dir", "heating");
      else t.removeAttribute("data-dir");
      if (editIndex !== null && k === editIndex) t.setAttribute("data-impulse", "1");
      else t.removeAttribute("data-impulse");
      const d = editIndex === null ? 0 : Math.max(0, Math.min(RIPPLE_CLAMP, k - editIndex));
      t.style.setProperty("--d", String(d));
      t.style.setProperty("--a", h.toFixed(3));
      t.style.setProperty("--h", h.toFixed(3));
    }
    this.heat = next;
    this.el.setAttribute("data-heat", opts.pending === true ? "pending" : "live");
    if (editIndex !== null) this.impulse(editIndex);
  }

  private impulse(index: number): void {
    const t = this.tk[index];
    if (!t || this.reduceMotion?.matches === true) return;
    // jsdom has no WAAPI; guard rather than assume the DOM lib's optimism.
    const animate = (t as { animate?: Element["animate"] }).animate;
    if (typeof animate !== "function") return;
    try {
      animate.call(
        t,
        [
          { boxShadow: "inset 0 0 0 1.5px var(--hot)" },
          { boxShadow: "inset 0 0 0 0 transparent" },
        ],
        { duration: 180, easing: "ease-out" },
      );
    } catch {
      /* WAAPI is a nicety; its absence must not break the render */
    }
  }

  /**
   * Load-time paint. A reload with the same token count reuses pooled spans, so
   * without suppressing transitions for two frames you get a 340ms heat
   * animation on load — heat arriving as an event that never happened.
   */
  hardRepaint(layout: MirrorLayout, heat: readonly number[], pending = false): void {
    this.el.setAttribute("data-noanim", "1");
    this.setText(layout);
    this.setHeat(heat, { prev: heat, pending });
    const el = this.el;
    const raf = el.ownerDocument.defaultView?.requestAnimationFrame;
    if (typeof raf === "function") {
      raf.call(el.ownerDocument.defaultView, () => {
        raf.call(el.ownerDocument.defaultView, () => el.removeAttribute("data-noanim"));
      });
    } else {
      el.removeAttribute("data-noanim");
    }
  }

  /** The concatenation the alignment invariant is stated over. */
  renderedText(): string {
    let out = "";
    for (const node of Array.from(this.el.childNodes)) {
      if (node.nodeName === "BR") continue;
      out += node.textContent ?? "";
    }
    return out;
  }
}
