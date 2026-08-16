/**
 * The intro — TECH_PLAN.md §10.1, §10.6, CONCEPT.md "Intro".
 *
 * A length slider over `passage.intro.prefix_z`: the CUMULATIVE PREFIX z OF A
 * REAL PASSAGE, computed by the forge and shipped in public.json. Drag it and
 * the needle climbs, because more text really is more evidence. Nothing here is
 * simulated and nothing here is a curve — if the numbers were invented the
 * lesson would be a lie, and this is the one screen whose entire job is to
 * teach that the instrument is honest.
 *
 * The slider TEACHES; prose does not (§10.6). Its three strings come from
 * copy.toml's [intro] block and nothing else is written on the screen.
 *
 * Clipping the heat to the prefix is exact rather than approximate: a token's
 * g-value depends only on the n-1 tokens BEFORE it, so the first k tokens of a
 * passage score identically whether or not the rest of the passage exists.
 */

import { Mirror, layoutFromTokens, layoutFromWords } from "./mirror";
import type { Needle } from "./needle";
import type { Copy, IntroWire, TokenHeat } from "../state";

export interface IntroElements {
  wrap: HTMLElement;
  mirror: HTMLElement;
  slider: HTMLInputElement;
  scale: HTMLElement;
  wordCount: HTMLElement;
  line: HTMLElement;
  start: HTMLButtonElement;
}

export interface IntroOptions {
  els: IntroElements;
  copy: Copy;
  intro: IntroWire;
  text: string;
  tokens: readonly TokenHeat[];
  minWords: number;
  needle: Needle;
  onStart: () => void;
}

interface WordSpan {
  s: number;
  e: number;
}

function wordSpans(text: string): WordSpan[] {
  const out: WordSpan[] = [];
  const re = /\S+/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) out.push({ s: m.index, e: m.index + m[0].length });
  return out;
}

export class Intro {
  private readonly mirror: Mirror;
  private readonly wordSpans: WordSpan[];
  private readonly steps: number[];

  constructor(private readonly o: IntroOptions) {
    this.mirror = new Mirror(o.els.mirror);
    this.wordSpans = wordSpans(o.text);
    // word_index aligns 1:1 with prefix_z when present; otherwise prefix_z is
    // one entry per word.
    this.steps =
      o.intro.word_index.length === o.intro.prefix_z.length
        ? o.intro.word_index.slice()
        : o.intro.prefix_z.map((_, i) => i + 1);

    const min = this.steps[0] ?? 1;
    const max = this.steps[this.steps.length - 1] ?? this.wordSpans.length;
    o.els.slider.min = String(min);
    o.els.slider.max = String(max);
    o.els.slider.value = String(Math.min(max, Math.max(min, Math.round(max * 0.14))));
    o.els.slider.setAttribute("aria-label", o.copy.t("intro.slider_label"));
    o.els.start.textContent = o.copy.t("intro.hand_off");

    this.buildScale(min, max);
    o.els.slider.addEventListener("input", () => this.render());
    o.els.start.addEventListener("click", () => o.onStart());
    this.render();
  }

  private buildScale(min: number, max: number): void {
    const doc = this.o.els.scale.ownerDocument;
    const span = max - min || 1;
    for (let w = Math.ceil(min / 10) * 10; w <= max; w += 10) {
      const tick = doc.createElement("i");
      tick.style.left = `${((w - min) / span) * 100}%`;
      this.o.els.scale.appendChild(tick);
    }
    const floor = this.o.minWords;
    if (floor > min && floor < max) {
      const mark = doc.createElement("i");
      mark.setAttribute("data-floor", "1");
      mark.style.left = `${((floor - min) / span) * 100}%`;
      this.o.els.scale.appendChild(mark);
      const label = this.o.copy.has("check.word_floor.label")
        ? this.o.copy.t("check.word_floor.label", { min_words: floor })
        : null;
      if (label !== null) {
        const u = doc.createElement("u");
        u.textContent = label;
        u.style.left = `${((floor - min) / span) * 100}%`;
        u.style.transform = "translateX(-50%)";
        this.o.els.scale.appendChild(u);
      }
    }
  }

  private zFor(nWords: number): number {
    const { prefix_z } = this.o.intro;
    let best = 0;
    for (let i = 0; i < this.steps.length; i++) {
      if ((this.steps[i] ?? Infinity) <= nWords) best = i;
      else break;
    }
    return prefix_z[best] ?? prefix_z[prefix_z.length - 1] ?? 0;
  }

  render(): void {
    const n = Math.max(1, Number.parseInt(this.o.els.slider.value, 10) || 1);
    const cut = this.wordSpans[Math.min(n, this.wordSpans.length) - 1]?.e ?? this.o.text.length;
    const text = this.o.text.slice(0, cut);
    const visible = this.o.tokens.filter((t) => t.e <= cut);
    const layout =
      visible.length > 0 ? layoutFromTokens(text, visible) : layoutFromWords(text);
    const heat = visible.length > 0 ? visible.map((t) => t.heat) : layout.toks.map(() => 0);
    this.mirror.setText(layout);
    this.mirror.setHeat(heat, { prev: heat, pending: visible.length === 0 });
    this.o.needle.setTarget(this.zFor(n));

    const label = this.o.copy.has("intro.word_count_label")
      ? this.o.copy.t("intro.word_count_label", { n_words: n })
      : String(n);
    this.o.els.wordCount.textContent = label;
    // beat_2 is the second lesson and only lands once the player has tried to
    // shorten their way out of the puzzle.
    this.o.els.line.textContent =
      n < this.o.minWords ? this.o.copy.t("intro.beat_2") : this.o.copy.t("intro.beat_1");
  }
}
