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
import type { LivePreview } from "./live";
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
  // The editable window: a garment panel with only its opening marked out.
  region: "M3.4 4.4h17.2v15.2H3.4Z|M9.8 4.4v15.2|M5.8 8.6h2|M5.8 12h2|M5.8 15.4h2",
  // Two rails, not one: `detector_floor` is a bound on BOTH sides, and it sits
  // beside `detector_threshold`'s drum on the campaign's last level.
  window: "M3.4 3.6h17.2v16.8H3.4Z|M6.4 8.2h11.2|M6.4 15.8h11.2",
};
const CROSS = "M4.4 4.4 19.6 19.6|M19.6 4.4 4.4 19.6";
/** Unknown check name -> the bleach triangle, so a new REGISTRY key still
 *  draws something rather than an empty box. `forge lint-copy` is what stops it
 *  shipping without its own symbol and its own copy. */
const FALLBACK_SYMBOL = "bleach";

/** One symbol per REGISTRY key (§7.6). Adding a check adds a row here.
 *
 *  A MISSING ROW IS NOT COSMETIC. `FALLBACK_SYMBOL` is the bleach triangle,
 *  which is also `unicode_sanitation`'s own glyph — so the campaign's last
 *  level, which runs both of the checks added with the eight-level rewrite,
 *  drew that triangle three times for three different rules. The pips are the
 *  whole legibility story on a phone; two rules sharing a glyph erases it. */
export const CHECK_SYMBOL: Record<string, string> = {
  unicode_sanitation: "bleach",
  word_floor: "floor",
  edit_budget: "dryclean",
  edit_region: "region",
  locked_phrase: "iron",
  close_paraphrase: "nobleach",
  unit_test: "test",
  detector_floor: "window",
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
  /** The rule's own rejection sentence, rendered from the LIVE numbers, and
   *  only while the rule is failing. Empty otherwise.
   *
   *  A red glyph says a rule is broken; it does not say which phrase went
   *  missing or which word fell outside the opening. That sentence already
   *  exists in copy.toml — it is what the gate sheet prints after a submission
   *  — and there is no reason the player should have to spend a submission to
   *  read it when the browser computed the same numbers a keystroke ago. */
  detail: string;
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
  /** the provisional-clear note; see showResult on why it lives here and not
   *  only on the gate sheet */
  resultNote: HTMLElement;
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
  /** the closed-form checks, recomputed on every keystroke. A PREVIEW: the
   *  moment a `trace` arrives it wins on every row (`statusFor`). */
  live: LivePreview;
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

/** How long the copy button holds its confirmation. Long enough to read at a
 *  glance, short enough that the button is back to being a button before
 *  anybody wants to press it a second time. */
const COPIED_MS = 1600;

export class Gate {
  private readonly specs: Templated[];
  private rows: CheckRow[] = [];
  /** the copy button's resting label, captured the first time it is replaced */
  private copyLabel: string | null = null;
  private copyTimer: ReturnType<typeof setTimeout> | null = null;
  private pipSignature = "";
  private trace: CheckResultWire[] | null = null;
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

  /** The reading the needle is showing; the detector rows' live input. */
  setReading(reading: Reading | null): void {
    this.o.live.setReading(reading);
    this.rebuild();
    this.renderChanged();
  }

  /** The player's current text. Every closed-form rule re-answers from it, so
   *  "Real characters", "At least 50 words", the budget, the opening window and
   *  the verbatim phrase all move while typing instead of waiting for Check. */
  setText(text: string): void {
    this.o.live.setText(text);
    this.rebuild();
    this.renderChanged();
  }

  /** A submission's trace describes ONE text. The moment the player types
   *  again it is history, so main.ts clears it on the next keystroke. */
  setOutcome(outcome: SubmitResponseWire | null): void {
    this.trace = outcome?.trace ?? null;
    this.rebuild();
  }

  private statusFor(spec: CheckSpecWire): {
    status: RowStatus;
    result: CheckResultWire | null;
    live: Record<string, string | number> | null;
  } {
    // THE SERVER'S TRACE WINS, ALWAYS. It describes the exact text that was
    // submitted, it is the thing that decided the outcome, and it is the only
    // source that can speak for `llm_gate`. `main.ts` drops it on the next
    // keystroke, at which point the live answer takes over again.
    const fromTrace = this.trace?.find((r) => r.check === spec.check) ?? null;
    if (fromTrace !== null) return { status: fromTrace.status, result: fromTrace, live: null };
    // Otherwise: a closed-form answer if this check has one, and the honest
    // "Not checked yet" if it does not (`llm_gate`, `unit_test`,
    // `close_paraphrase` — see live.ts on why that list is what it is).
    const preview = this.o.live.evaluate(spec);
    if (preview !== null) return { status: preview.status, result: null, live: preview.params };
    return { status: "pending", result: null, live: null };
  }

  private rebuild(): void {
    this.rows = this.specs.map(({ spec }) => {
      const { status, result, live } = this.statusFor(spec);
      const params = this.paramsFor(spec, result?.params ?? live ?? undefined);
      const whyKey =
        status === "pass"
          ? "gate.pass_label"
          : status === "fail"
            ? "gate.fail_label"
            : status === "error"
              ? "gate.error_label"
              : "gate.pending_label";
      // Only for a LIVE failure. A server rejection already has a home — the
      // gate sheet prints it, chosen by `code` (close_paraphrase alone fails
      // five ways), and repeating a guess at it here would sooner or later
      // print a different sentence than the one that actually rejected them.
      const detail =
        status === "fail" && result === null
          ? (this.optional(`check.${spec.check}.reject`, params) ?? "")
          : "";
      return {
        check: spec.check,
        status,
        label: this.o.copy.t(`check.${spec.check}.label`, params),
        blurb: this.o.copy.t(`check.${spec.check}.blurb`),
        why: this.o.copy.t(whyKey),
        detail,
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

  /** Live words-changed count.
   *
   *  It comes off the SAME backtrace `edit_budget` reads, not off the reading's
   *  `preview_distance`. Two reasons. It is instant rather than debounced
   *  behind a detector round trip; and on a budgeted level the counter and the
   *  budget pip are now one number computed once, so the screen can never show
   *  "5 changed" beside a red "Budget 6 words". `edit_budget.py` was written to
   *  guarantee exactly that property server-side — "if this check ever grows
   *  its own distance function, the first bug report will be 'it says 12 but it
   *  rejected me at 12'" — and the browser had a second number all along.
   *
   *  §8.3 is unchanged: the distance that is persisted, ranked and shared is
   *  the server's recomputation, and it arrives in `SubmitResponse.score`. */
  private renderChanged(): void {
    const d = this.o.live.distance();
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
      const hasBlurb = r.blurb !== "" || r.detail !== "";
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
        // The live reason FIRST, then the standing explanation. Somebody who
        // opened a red rule wants to know what they just did, not what the
        // rule is for; somebody who opened a green one only ever sees the
        // second paragraph, which is the row this markup has always drawn.
        if (r.detail !== "") {
          const why = doc.createElement("p");
          why.className = "cxwhy";
          why.textContent = r.detail;
          body.appendChild(why);
        }
        if (r.blurb !== "") {
          const blurb = doc.createElement("p");
          blurb.textContent = r.blurb;
          body.appendChild(blurb);
        }
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
    // A sheet that opens showing "Copied" is describing a copy that has not
    // happened. Reachable whenever a level is cleared inside the confirmation
    // window of the previous one.
    if (this.copyTimer !== null) {
      clearTimeout(this.copyTimer);
      this.copyTimer = null;
      els.copyButton.textContent = this.copyLabel ?? els.copyButton.textContent;
      els.copyButton.removeAttribute("data-done");
    }
    const distance = res.score.distance;
    els.resultTop.textContent = copy.t("readout.cleared_toast", { distance });
    els.resultBig.textContent = formatPoints(res.detector.z, this.o.points);
    const sub = copy.t("readout.distance_label", { distance });
    const par = this.optional("screen.par_label", { par: this.o.par ?? 0 });
    els.resultSub.textContent = par !== null && this.o.par !== null ? `${sub} · ${par}` : sub;

    // "Gate unavailable — this one does not count." THIS is the sheet where
    // `provisional` can actually be true: `run_gate` returns `provisional=False`
    // on every rejection, so the copy of this note on #gatesheet is unreachable
    // and this was the only outcome that could need it — with nowhere to put it.
    // A provisional clear therefore looked exactly like a real one while the
    // client skipped `markCleared` and the cookie behind the player's back, and
    // they returned to find the level uncleared and unexplained.
    els.resultNote.textContent = res.provisional ? copy.t("readout.provisional_note") : "";
    els.resultNote.hidden = !res.provisional;

    // EACH ROW'S OWN STATUS, not a hardcoded "pass". A clear is not proof that
    // every check passed: `run_gate` clears PROVISIONALLY when a fail-open check
    // returns `status="error"` — the judge unreachable, or the daily spend cap
    // hit — and the errored row is exactly the one the care label must not draw
    // as a tick. `setOutcome` has already merged the server's trace into
    // `this.rows` by the time this runs (main.ts calls it first), and `.sym`
    // has a `data-s="error"` treatment for precisely this state.
    while (els.resultSyms.firstChild) els.resultSyms.removeChild(els.resultSyms.firstChild);
    for (const row of this.rows) {
      els.resultSyms.appendChild(
        symbolSvg(this.o.doc, CHECK_SYMBOL[row.check] ?? FALLBACK_SYMBOL, row.status),
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

  /**
   * Copy the share, and SAY SO.
   *
   * A button that does its whole job silently reads as a broken button, and
   * this one is pressed at the only moment the game asks for something back —
   * the end of the campaign, where the four-line score card is the entire
   * reward. The button's own label carries the confirmation rather than a
   * toast: it is already the thing under the player's finger, it needs no new
   * furniture at the bottom of a sheet that is short on room on a phone, and it
   * cannot be missed by someone who is looking at what they just tapped.
   *
   * On failure the text is SELECTED instead. A denied clipboard permission is
   * not a game failure and must not be reported as one, but leaving the player
   * with nothing is how a share gets abandoned; a selection is one keystroke
   * from the same result.
   */
  private async copyShare(): Promise<void> {
    const text = this.o.els.resultShare.textContent ?? "";
    if (text === "") return;
    try {
      await navigator.clipboard?.writeText(text);
      this.flashCopied();
    } catch {
      this.selectShare();
    }
  }

  private flashCopied(): void {
    const button = this.o.els.copyButton;
    const done = this.optional("screen.copied_label");
    if (done === null) return; // no key, no invented string (§10.7)
    if (this.copyLabel === null) this.copyLabel = button.textContent ?? "";
    button.textContent = done;
    button.setAttribute("data-done", "1");
    if (this.copyTimer !== null) clearTimeout(this.copyTimer);
    this.copyTimer = setTimeout(() => {
      button.textContent = this.copyLabel ?? "";
      button.removeAttribute("data-done");
      this.copyTimer = null;
    }, COPIED_MS);
  }

  private selectShare(): void {
    const range = this.o.doc.createRange();
    range.selectNodeContents(this.o.els.resultShare);
    const selection = this.o.doc.defaultView?.getSelection();
    selection?.removeAllRanges();
    selection?.addRange(range);
  }

  currentRows(): readonly CheckRow[] {
    return this.rows;
  }
}
