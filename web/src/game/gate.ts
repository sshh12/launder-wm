/**
 * The gate — TECH_PLAN.md §10.1, §10.5, §10.7, §9.3.
 *
 * Owns: the pips, the tappable constraint disclosures, the submit button, and
 * the rendering of a rejection. NEVER computes a verdict — every status here is
 * either the server's `trace` or the live detector reading the needle is
 * already showing. `trace` always returns, pass or fail, and it is what makes
 * the gate legible rather than oracular.
 *
 * EVERY STRING COMES FROM data/config/copy.toml (§10.7). There are no
 * player-facing words in this file. Where copy.toml has no key for a piece of
 * furniture, the element is omitted rather than filled with an invented string
 * — see `optional()`.
 *
 * The care-symbol vocabulary is the graft from direction C: a failed check is a
 * STRUCK CROSS, so the state survives colour blindness and greyscale. The path
 * data below is geometry, not copy.
 */

import { renderDiff, words } from "./diff";
import { formatPoints, type PointsScale } from "./needle";
import type { Sheets } from "./primer";
import type {
  CheckResultWire,
  CheckSpecWire,
  Copy,
  CopyParams,
  LevelWire,
  Reading,
  SubmitResponseWire,
} from "../state";

const SVG_NS = "http://www.w3.org/2000/svg";

/** Laundry care symbols, drawn as one system on a 24x24 grid. */
const SYM_PATHS: Record<string, string> = {
  wash:
    "M3.2 8.4h17.6l-1.5 10.2a2.2 2.2 0 0 1-2.2 1.9H6.9a2.2 2.2 0 0 1-2.2-1.9Z" +
    "|M4.4 12.2c1.7-1.5 3.2 1.2 4.9 0s3.2 1.2 4.9 0 3.2 1.2 4.9 0" +
    "|M4.6 8.4C6.5 5 9.2 3.6 12 3.6s5.5 1.4 7.4 4.8",
  bleach: "M12 3.6 21 20.4H3Z",
  nobleach: "M12 3.6 21 20.4H3Z|M10 11v6|M14 11v6",
  drum: "M3.4 3.6h17.2v16.8H3.4Z|M12 7.2a4.8 4.8 0 1 1 0 9.6 4.8 4.8 0 0 1 0-9.6",
  iron: "M2.8 18.2h18.4l-2.4-8.6H8.4L6 14.2H2.8Z|M8.4 9.6C9 6.4 10.4 5 12.8 5h5",
  test: "M12 3.4a8.6 8.6 0 1 1 0 17.2 8.6 8.6 0 0 1 0-17.2|M7 12h10",
  dryclean: "M12 3.4a8.6 8.6 0 1 1 0 17.2 8.6 8.6 0 0 1 0-17.2|M9.6 15.6V8.4h2.9a2.2 2.2 0 0 1 0 4.4H9.6",
  floor: "M3.4 20.2h17.2|M12 16.4V4.8|M7.4 9.4 12 4.8l4.6 4.6",
};
const CROSS = "M4.4 4.4 19.6 19.6|M19.6 4.4 4.4 19.6";
/** Unknown check name -> the bleach triangle, so a new REGISTRY key still
 *  draws something rather than an empty box. `forge lint-copy` is what stops it
 *  shipping without its own symbol and its own copy. */
const FALLBACK_SYMBOL = "bleach";

/** One symbol per REGISTRY key (§7.6). Adding a check adds a row here. */
export const CHECK_SYMBOL: Record<string, string> = {
  unicode_sanitation: "bleach",
  word_floor: "floor",
  edit_budget: "dryclean",
  locked_phrase: "iron",
  close_paraphrase: "nobleach",
  unit_test: "test",
  detector_threshold: "drum",
  llm_gate: "wash",
};

export type RowStatus = "pass" | "fail" | "error" | "pending";

export function symbolSvg(doc: Document, kind: string, status: RowStatus): SVGSVGElement {
  const svg = doc.createElementNS(SVG_NS, "svg");
  svg.setAttribute("class", "sym");
  svg.setAttribute("data-s", status);
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("fill", "none");
  svg.setAttribute("stroke", "currentColor");
  svg.setAttribute("stroke-width", "1.35");
  svg.setAttribute("stroke-linejoin", "round");
  svg.setAttribute("stroke-linecap", "round");
  svg.setAttribute("aria-hidden", "true");
  const d = SYM_PATHS[kind] ?? SYM_PATHS[FALLBACK_SYMBOL] ?? "";
  for (const p of d.split("|")) {
    const path = doc.createElementNS(SVG_NS, "path");
    path.setAttribute("d", p);
    svg.appendChild(path);
  }
  if (status === "fail") {
    for (const p of CROSS.split("|")) {
      const path = doc.createElementNS(SVG_NS, "path");
      path.setAttribute("d", p);
      path.setAttribute("stroke-width", "1.9");
      svg.appendChild(path);
    }
  }
  return svg;
}

export interface CheckRow {
  check: string;
  status: RowStatus;
  label: string;
  blurb: string;
  why: string;
}

export interface GateElements {
  pips: HTMLElement[];
  checklistSheet: HTMLElement;
  checklistBody: HTMLElement;
  checklistHint: HTMLElement;
  sideChecks: HTMLElement | null;
  gateSheet: HTMLElement;
  verdict: HTMLElement;
  reason: HTMLElement;
  gateNote: HTMLElement;
  gateChecks: HTMLElement;
  resultSheet: HTMLElement;
  resultTop: HTMLElement;
  resultBig: HTMLElement;
  resultSub: HTMLElement;
  resultSyms: HTMLElement;
  resultDiff: HTMLElement;
  resultShare: HTMLElement;
  resultActions: HTMLElement;
  allClear: HTMLElement;
  nextButton: HTMLElement;
  copyButton: HTMLElement;
  checkButton: HTMLButtonElement;
  changed: HTMLElement[];
  parLine: HTMLElement;
}

export interface GateOptions {
  doc: Document;
  els: GateElements;
  copy: Copy;
  level: LevelWire;
  par: number | null;
  originalText: string;
  sheets: Sheets;
  /** the z -> 0..100 relabelling, so the result's big number and the needle
   *  the player was just watching cannot print different figures */
  points: PointsScale;
  /** 1-based campaign position, and the campaign's length */
  levelN: number;
  levelCount: number;
  /** advance to the next level: write the cookie and the local record, then
   *  navigate. Owned by main.ts, which owns the persistence. */
  onNext: (res: SubmitResponseWire) => void;
  /** the end-of-campaign share, built at show time from the whole progress
   *  record — including the level that was cleared moments ago */
  campaignShare: () => string;
}

interface Templated {
  spec: CheckSpecWire;
  symbol: string;
}

export class Gate {
  private readonly specs: Templated[];
  private rows: CheckRow[] = [];
  private pipSignature = "";
  private trace: CheckResultWire[] | null = null;
  private reading: Reading | null = null;
  /** the response the open result sheet describes; the advance button needs
   *  its distance and it must not be read from a later submission. */
  private cleared: SubmitResponseWire | null = null;

  constructor(private readonly o: GateOptions) {
    this.specs = o.level.checks.map((spec) => ({
      spec,
      symbol: CHECK_SYMBOL[spec.check] ?? FALLBACK_SYMBOL,
    }));
    const par = this.optional("screen.par_label", { par: o.par ?? 0 });
    if (par !== null && o.par !== null) o.els.parLine.textContent = par;
    else o.els.parLine.hidden = true;

    const hint = this.optional("gate.disclosure_hint");
    if (hint !== null) o.els.checklistHint.textContent = hint;

    for (const pip of o.els.pips) {
      pip.addEventListener("click", () => this.openChecklist());
    }
    o.els.copyButton.addEventListener("click", () => void this.copyShare());
    o.els.nextButton.addEventListener("click", () => {
      const res = this.cleared;
      if (res !== null) o.onNext(res);
    });

    // ONE delegated handler covers every .checks container — the gate sheet,
    // the checklist sheet and the desktop rail, including the ones re-rendered
    // mid-play (§10.7).
    o.doc.addEventListener("click", (e) => {
      const target = e.target;
      if (!(target instanceof Element)) return;
      const row = target.closest<HTMLElement>(".checkrow[aria-controls]");
      if (row === null) return;
      const body = o.doc.getElementById(row.getAttribute("aria-controls") ?? "");
      if (body === null) return;
      const open = row.getAttribute("aria-expanded") === "true";
      row.setAttribute("aria-expanded", open ? "false" : "true");
      body.hidden = open;
    });

    this.rebuild();
    // The live words-changed count is one of the four things that never
    // collapse at any width (§10.5), so it is on screen before the first
    // reading rather than blank until one arrives.
    this.renderChanged();
  }

  private optional(path: string, params?: CopyParams): string | null {
    return this.o.copy.has(path) ? this.o.copy.t(path, params) : null;
  }

  private paramsFor(spec: CheckSpecWire, extra?: Record<string, unknown>): CopyParams {
    const merged: CopyParams = {};
    for (const [k, v] of Object.entries(spec.params ?? {})) {
      if (typeof v === "string" || typeof v === "number") merged[k] = v;
    }
    for (const [k, v] of Object.entries(extra ?? {})) {
      if (typeof v === "string" || typeof v === "number") merged[k] = v;
    }
    return merged;
  }

  /** The reading the needle is showing; the only live input to the pips. */
  setReading(reading: Reading | null): void {
    this.reading = reading;
    this.rebuild();
    this.renderChanged();
  }

  /** A submission's trace describes ONE text. The moment the player types
   *  again it is history, so main.ts clears it on the next keystroke. */
  setOutcome(outcome: SubmitResponseWire | null): void {
    this.trace = outcome?.trace ?? null;
    this.rebuild();
  }

  private statusFor(spec: CheckSpecWire): { status: RowStatus; result: CheckResultWire | null } {
    const fromTrace = this.trace?.find((r) => r.check === spec.check) ?? null;
    if (fromTrace !== null) return { status: fromTrace.status, result: fromTrace };
    // detector_threshold is the one row the client may show live: it is the
    // needle's own state under a different glyph, not a verdict of our own.
    if (spec.check === "detector_threshold" && this.reading !== null) {
      return { status: this.reading.z <= this.reading.zStar ? "pass" : "fail", result: null };
    }
    return { status: "pending", result: null };
  }

  private rebuild(): void {
    this.rows = this.specs.map(({ spec }) => {
      const { status, result } = this.statusFor(spec);
      const params = this.paramsFor(spec, result?.params);
      const whyKey =
        status === "pass"
          ? "gate.pass_label"
          : status === "fail"
            ? "gate.fail_label"
            : status === "error"
              ? "gate.error_label"
              : "gate.pending_label";
      return {
        check: spec.check,
        status,
        label: this.o.copy.t(`check.${spec.check}.label`, params),
        blurb: this.o.copy.t(`check.${spec.check}.blurb`),
        why: this.o.copy.t(whyKey),
      };
    });
    this.renderPips();
    if (this.o.els.sideChecks !== null) {
      this.renderChecks(this.o.els.sideChecks, this.rows);
    }
  }

  private renderPips(): void {
    const sig = this.rows.map((r) => `${r.check}:${r.status}`).join(",");
    if (sig === this.pipSignature) return;
    this.pipSignature = sig;
    const label = this.rows.map((r) => `${r.label}: ${r.why}`).join(", ");
    for (const box of this.o.els.pips) {
      while (box.firstChild) box.removeChild(box.firstChild);
      for (const r of this.rows) {
        box.appendChild(symbolSvg(this.o.doc, CHECK_SYMBOL[r.check] ?? FALLBACK_SYMBOL, r.status));
      }
      box.setAttribute("aria-label", label);
    }
  }

  /** Live words-changed count. `preview_distance` is the SERVER's number
   *  (§9.2) — the client asserts nothing about distance. */
  private renderChanged(): void {
    const d = this.reading?.previewDistance ?? 0;
    const text =
      d === 0
        ? this.o.copy.t("readout.distance_zero")
        : this.o.copy.t("readout.distance_label", { distance: d });
    for (const el of this.o.els.changed) el.textContent = text;
  }

  /** Every row is a disclosure button: tap it and the constraint explains
   *  itself in place (aria-expanded + a hidden region). */
  renderChecks(box: HTMLElement, rows: readonly CheckRow[]): void {
    const doc = this.o.doc;
    while (box.firstChild) box.removeChild(box.firstChild);
    rows.forEach((r, index) => {
      const wrap = doc.createElement("div");
      wrap.className = "check";
      const id = `${box.id}-x${index}`;
      const hasBlurb = r.blurb !== "";
      const row = doc.createElement(hasBlurb ? "button" : "div");
      row.className = "checkrow";
      if (hasBlurb && row instanceof HTMLButtonElement) {
        wrap.setAttribute("data-disclosure", "1");
        row.type = "button";
        row.setAttribute("aria-expanded", "false");
        row.setAttribute("aria-controls", id);
      }
      row.appendChild(symbolSvg(doc, CHECK_SYMBOL[r.check] ?? FALLBACK_SYMBOL, r.status));
      const label = doc.createElement("span");
      label.textContent = r.label;
      row.appendChild(label);
      const why = doc.createElement("span");
      why.className = "why";
      why.textContent = r.why;
      row.appendChild(why);
      if (hasBlurb) {
        const more = doc.createElement("span");
        more.className = "cmore";
        more.setAttribute("aria-hidden", "true");
        more.textContent = "?";
        row.appendChild(more);
      }
      wrap.appendChild(row);
      if (hasBlurb) {
        const body = doc.createElement("div");
        body.className = "cx";
        body.id = id;
        body.hidden = true;
        body.textContent = r.blurb;
        wrap.appendChild(body);
      }
      box.appendChild(wrap);
    });
  }

  openChecklist(): void {
    this.renderChecks(this.o.els.checklistBody, this.rows);
    this.o.sheets.open(this.o.els.checklistSheet.id);
  }

  setSubmitting(on: boolean): void {
    const btn = this.o.els.checkButton;
    btn.disabled = on;
    if (on) btn.setAttribute("aria-busy", "true");
    else btn.removeAttribute("aria-busy");
  }

  /** A rejection is a game outcome, not an error (§9.3). */
  showRejection(res: SubmitResponseWire): void {
    const { els, copy } = this.o;
    const failure = res.failure ?? null;
    els.verdict.setAttribute("data-v", "fail");
    els.verdict.textContent = copy.t("gate.fail_label");
    let message = failure?.message ?? "";
    if (message === "" && failure !== null) {
      const params = this.paramsFor({ check: failure.check }, failure.params);
      message = copy.t(`check.${failure.check}.reject`, params);
    }
    els.reason.textContent = message;
    els.gateNote.textContent = res.provisional ? copy.t("readout.provisional_note") : "";
    els.gateNote.hidden = !res.provisional;
    this.renderChecks(els.gateChecks, this.rows);
    this.o.sheets.open(els.gateSheet.id);
  }

  /** Client-side failures: no connection, 429, 413, unknown passage (§10.6:
   *  errors are specific and never apologize). */
  showError(copyKey: string, params?: CopyParams): void {
    const { els, copy } = this.o;
    els.verdict.setAttribute("data-v", "fail");
    els.verdict.textContent = copy.t("gate.error_label");
    els.reason.textContent = copy.t(copyKey, params);
    els.gateNote.hidden = true;
    this.renderChecks(els.gateChecks, this.rows);
    this.o.sheets.open(els.gateSheet.id);
  }

  showResult(res: SubmitResponseWire, submittedText: string): void {
    const { els, copy } = this.o;
    this.cleared = res;
    const distance = res.score.distance;
    els.resultTop.textContent = copy.t("readout.cleared_toast", { distance });
    els.resultBig.textContent = formatPoints(res.detector.z, this.o.points);
    const sub = copy.t("readout.distance_label", { distance });
    const par = this.optional("screen.par_label", { par: this.o.par ?? 0 });
    els.resultSub.textContent = par !== null && this.o.par !== null ? `${sub} · ${par}` : sub;

    while (els.resultSyms.firstChild) els.resultSyms.removeChild(els.resultSyms.firstChild);
    for (const { spec } of this.specs) {
      els.resultSyms.appendChild(
        symbolSvg(this.o.doc, CHECK_SYMBOL[spec.check] ?? FALLBACK_SYMBOL, "pass"),
      );
    }

    renderDiff(els.resultDiff, words(this.o.originalText), words(submittedText), res.score.ops);

    // The end of the campaign is a different screen, not a different sheet: the
    // advance button has nowhere to go, so it is replaced by the last line the
    // game has to say and by the share that carries every level's score.
    const last = this.o.levelN >= this.o.levelCount;
    const share = last ? this.o.campaignShare() : (res.share ?? "");
    els.nextButton.hidden = last;
    els.allClear.hidden = !last;
    // A copy button with nothing under it is a button that does nothing when
    // pressed, so it follows the share box. On the last level that can leave
    // the action bar holding no button at all — and an empty sticky bar still
    // draws its rule and its 48px of padding.
    els.copyButton.hidden = share === "";
    els.resultActions.hidden = last && share === "";
    els.resultShare.textContent = share;
    els.resultShare.hidden = share === "";
    this.o.sheets.open(els.resultSheet.id);
  }

  private async copyShare(): Promise<void> {
    const text = this.o.els.resultShare.textContent ?? "";
    if (text === "") return;
    try {
      await navigator.clipboard?.writeText(text);
    } catch {
      /* clipboard permission is not a game failure */
    }
  }

  currentRows(): readonly CheckRow[] {
    return this.rows;
  }
}
