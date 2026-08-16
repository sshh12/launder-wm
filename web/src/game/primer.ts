/**
 * The primer and the sheet vocabulary — TECH_PLAN.md §10.7.
 *
 * Two on-demand explainers, both sheets, both invoked by the same glyph (`?`).
 * ONE explain-affordance in the whole product; learn it once.
 *
 * The primer follows Wordle's contract exactly: it opens once on first land,
 * keyed on localStorage["launderwm.primer.v1"], and thereafter only from the
 * `?` in the rail. It never explains the ripple — the mirror teaches that
 * wordlessly and prose would spoil it.
 *
 * The stained demo sentence is GENERATED, never hand-painted: `boot.primer_demo`
 * carries char-offset heat produced by the production detector over the real
 * key, so the cold slots are cold because the POSITION had no options. A
 * hand-authored stain would be the one surface in the product free to teach the
 * false heuristic, so if the server ships no demo reading this renders the
 * sentence UNSTAINED rather than inventing one.
 *
 * The sheet controller lives here because the primer is the canonical sheet;
 * gate.ts drives its own sheets through the same object.
 */

import { Mirror, layoutFromTokens, layoutFromWords } from "./mirror";
import type { Boot, Copy } from "../state";

export const PRIMER_SEEN_KEY = "launderwm.primer.v1";

export class Sheets {
  private invoker: Element | null = null;
  constructor(
    private readonly doc: Document,
    private readonly scrim: HTMLElement,
    private readonly inertRoots: HTMLElement[],
    private readonly ids: string[],
  ) {
    for (const b of Array.from(doc.querySelectorAll("[data-close]"))) {
      b.addEventListener("click", () => this.closeAll());
    }
    scrim.addEventListener("click", () => this.closeAll());
    doc.addEventListener("keydown", (e) => {
      if ((e as KeyboardEvent).key === "Escape") this.closeAll();
    });
  }

  isAnyOpen(): boolean {
    return this.ids.some((id) => this.doc.getElementById(id)?.getAttribute("data-open") === "1");
  }

  open(id: string): void {
    this.closeAll();
    this.invoker = this.doc.activeElement;
    const sheet = this.doc.getElementById(id);
    if (sheet === null) return;
    sheet.setAttribute("data-open", "1");
    this.scrim.setAttribute("data-open", "1");
    for (const r of this.inertRoots) r.inert = true;
    const x = sheet.querySelector<HTMLElement>(".x");
    x?.focus();
  }

  closeAll(): void {
    let any = false;
    for (const id of this.ids) {
      const el = this.doc.getElementById(id);
      if (el === null) continue;
      if (el.getAttribute("data-open") === "1") any = true;
      el.setAttribute("data-open", "0");
    }
    this.scrim.setAttribute("data-open", "0");
    for (const r of this.inertRoots) r.inert = false;
    if (any && this.invoker instanceof HTMLElement) {
      try {
        this.invoker.focus();
      } catch {
        /* the invoker may have been re-rendered away */
      }
    }
    this.invoker = null;
  }
}

export interface PrimerElements {
  sheet: HTMLElement;
  demo: HTMLElement;
  whatButton: HTMLElement;
  playButton: HTMLElement;
}

export class Primer {
  private readonly mirror: Mirror;

  constructor(
    private readonly els: PrimerElements,
    private readonly sheets: Sheets,
    private readonly copy: Copy,
    private readonly boot: Boot,
  ) {
    this.mirror = new Mirror(els.demo);
    this.paintDemo();
    els.whatButton.addEventListener("click", () => this.open());
    els.playButton.addEventListener("click", () => this.sheets.closeAll());
  }

  private paintDemo(): void {
    const demo = this.boot.primer_demo;
    if (demo !== null && demo.tokens.length > 0) {
      const layout = layoutFromTokens(
        demo.text,
        demo.tokens.map((t) => ({ s: t.s, e: t.e, heat: t.heat, masked: t.masked === true })),
      );
      this.mirror.hardRepaint(
        layout,
        demo.tokens.map((t) => t.heat),
        false,
      );
      return;
    }
    // No generated reading: show the sentence with no stain at all. Painting a
    // plausible-looking one here would teach "spot the fancy word", which is
    // the single thing CONCEPT.md forbids.
    const text = this.copy.t("primer.demo");
    const layout = layoutFromWords(text);
    this.mirror.hardRepaint(layout, new Array<number>(layout.toks.length).fill(0), true);
  }

  open(): void {
    this.sheets.open(this.els.sheet.id);
  }

  /** Wordle's contract: once on first land, then only when asked for. Called
   *  after the screen has painted, so closing it reveals a finished screen and
   *  not one assembling itself. */
  openOnFirstLand(storage: Storage | null = safeStorage()): void {
    let seen = false;
    try {
      seen = storage?.getItem(PRIMER_SEEN_KEY) === "1";
    } catch {
      seen = false;
    }
    if (seen) return;
    try {
      storage?.setItem(PRIMER_SEEN_KEY, "1");
    } catch {
      /* private mode: the primer will open again next time, which is fine */
    }
    this.open();
  }
}

/** localStorage throws on access in some privacy modes, not just on write. */
export function safeStorage(): Storage | null {
  try {
    return globalThis.localStorage ?? null;
  } catch {
    return null;
  }
}
