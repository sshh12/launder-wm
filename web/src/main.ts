/**
 * Boot and wiring — TECH_PLAN.md §5.4, §10.
 *
 * The order of events here is the product's central claim: the passage and the
 * needle are already correct when this file starts running, because the server
 * inlined them (§5.4 step 1). Everything below is an upgrade path, and every
 * one of those upgrades is allowed to fail without taking the game with it.
 *
 *   SERVER   immediate default. Debounced POST /api/detect.
 *   LOADING  the worker is building the tokenizer. No progress bar for a
 *            capability the user already appears to have.
 *   LOCAL    the worker scores locally. The handover is invisible because both
 *            paths are the same algorithm on the same ids, so the needle's
 *            target has nothing to travel.
 *
 * Failure is permanent and quiet: if the worker cannot build, stay in SERVER
 * forever and never retry in a loop. The game is fully playable there; only
 * latency differs.
 */

import "./styles/index.css";

import { Gate, type GateElements } from "./game/gate";
import { Intro } from "./game/intro";
import {
  Mirror,
  layoutFromTokens,
  layoutFromWords,
  reanchorTokens,
} from "./game/mirror";
import { Needle, formatZ } from "./game/needle";
import { Primer, Sheets, safeStorage } from "./game/primer";
import { DetectClient } from "./net/detect";
import { sessionId, submitText } from "./net/submit";
import {
  Copy,
  Store,
  readBoot,
  readingFromWire,
  type Boot,
  type Reading,
  type TokenHeat,
} from "./state";
// Owned by the detector milestone. Importing it is free in SERVER mode: the
// worker is only constructed if the boot payload names its assets.
import { DetectorClient } from "./worker/client";
import type { ScoreResponse } from "./worker/protocol";

/* ------------------------------------------------------------------ *
 * DOM helpers
 * ------------------------------------------------------------------ */

function must<T extends HTMLElement>(id: string): T {
  const el = document.getElementById(id);
  if (el === null) throw new Error(`index.html is missing #${id}`);
  return el as T;
}

/**
 * Fill every copy slot. `data-copy` and `data-copy-aria` are required keys;
 * the `-opt` spellings are furniture that copy.toml does not describe yet, and
 * those elements are REMOVED rather than filled with a string invented in
 * TypeScript (§10.7: player-facing strings live in data/config/copy.toml).
 */
function applyCopySlots(root: ParentNode, copy: Copy): void {
  for (const el of Array.from(root.querySelectorAll<HTMLElement>("[data-copy]"))) {
    el.textContent = copy.t(el.dataset.copy ?? "");
  }
  for (const el of Array.from(root.querySelectorAll<HTMLElement>("[data-copy-aria]"))) {
    el.setAttribute("aria-label", copy.t(el.dataset.copyAria ?? ""));
  }
  for (const el of Array.from(root.querySelectorAll<HTMLElement>("[data-copy-opt]"))) {
    const key = el.dataset.copyOpt ?? "";
    if (copy.has(key)) el.textContent = copy.t(key);
    else el.remove();
  }
  for (const el of Array.from(root.querySelectorAll<HTMLElement>("[data-copy-aria-opt]"))) {
    const key = el.dataset.copyAriaOpt ?? "";
    if (copy.has(key)) el.setAttribute("aria-label", copy.t(key));
  }
}

/* ------------------------------------------------------------------ *
 * Boot
 * ------------------------------------------------------------------ */

function heatOf(tokens: readonly TokenHeat[]): number[] {
  return tokens.map((t) => t.heat);
}

function start(boot: Boot): void {
  const copy = new Copy(boot.copy);
  applyCopySlots(document, copy);

  const raw = must<HTMLTextAreaElement>("raw");
  const originalText = raw.value;
  const store = new Store(originalText);

  /* ---- rail furniture that is data, not copy ---- */
  must("daily").textContent = `#${boot.puzzle_number}`;
  must("lvlid").textContent = boot.level.id;
  must("lvlname").textContent = boot.level.name;

  const needle = new Needle(
    {
      meter: must("meter"),
      face: must("face"),
      needle: must("needle"),
      fill: must("fill"),
      notch: must("notch"),
      peak: must("peak"),
      sweep: must("sweep"),
      tri: must("tri"),
      floorLabel: must("floorlbl"),
      scaleRow: must("scalerow"),
      num: must("num"),
      stateWord: must("stateword"),
      live: must("live"),
    },
    { zStar: boot.detector.z_star, scale: boot.detector.scale, copy },
    boot.detector.expected_z,
  );
  must("meter").setAttribute("aria-valuemin", String(boot.detector.scale.min));
  must("meter").setAttribute("aria-valuemax", String(boot.detector.scale.max));
  // "Under 2.33 to clear" — NOT "floor", which already means the word minimum.
  must("floorlbl").textContent = copy.t("readout.threshold_label", {
    z_star_display: formatZ(boot.detector.z_star),
  });

  // data-cleared means ONE thing: the needle is under the line. It is not the
  // gate's verdict — clearing the detector and clearing the gate are different
  // events and the player must not learn to read one as the other (§10.7,
  // "one word, one meaning").
  needle.onCross = (below: boolean): void => {
    document.body.setAttribute("data-cleared", below ? "1" : "0");
  };

  const mirror = new Mirror(must("mirror"));
  const sheets = new Sheets(document, must("scrim"), [must("app")], [
    "primersheet",
    "gatesheet",
    "checksheet",
    "resultsheet",
  ]);

  const gateEls: GateElements = {
    pips: [must("pips"), must("pipsStrip")],
    checklistSheet: must("checksheet"),
    checklistBody: must("listchecks"),
    checklistHint: must("checkhint"),
    sideChecks: document.getElementById("sidechecks"),
    gateSheet: must("gatesheet"),
    verdict: must("gv"),
    reason: must("greason"),
    gateNote: must("gnote"),
    gateChecks: must("gatechecks"),
    resultSheet: must("resultsheet"),
    resultTop: must("rtop"),
    resultBig: must("rbig"),
    resultSub: must("rsub"),
    resultSyms: must("rsyms"),
    resultDiff: must("rdiff"),
    resultShare: must("rshare"),
    copyButton: document.getElementById("copyres") ?? document.createElement("button"),
    checkButton: must<HTMLButtonElement>("check"),
    changed: [must("changed"), must("stripcount")],
    parLine: must("parline"),
  };
  const gate = new Gate({
    doc: document,
    els: gateEls,
    copy,
    level: boot.level,
    par: boot.par,
    originalText,
    sheets,
  });

  const primer = new Primer(
    {
      sheet: must("primersheet"),
      demo: must("mirrorPrimer"),
      whatButton: must("whatbtn"),
      playButton: must("prplay"),
    },
    sheets,
    copy,
    boot,
  );

  /* ---- the mirror's view of the world -------------------------------- */

  let tokens: TokenHeat[] =
    boot.reading !== null
      ? readingFromWire(boot.reading).tokens
      : layoutFromWords(originalText).spans.map((sp) => ({ ...sp, heat: 0, masked: false }));
  let mirrorText = originalText;
  let lastHeat: number[] = heatOf(tokens);
  let pendingHeat = boot.reading === null;

  function paintMirror(text: string, next: readonly TokenHeat[], editIndex: number | null): void {
    const layout = layoutFromTokens(text, next);
    mirror.setText(layout);
    mirror.setHeat(heatOf(next), {
      editIndex,
      prev: lastHeat,
      pending: pendingHeat,
    });
    lastHeat = heatOf(next);
    mirrorText = text;
    tokens = next.slice();
  }

  mirror.hardRepaint(layoutFromTokens(originalText, tokens), heatOf(tokens), pendingHeat);

  if (boot.reading !== null) {
    const first = readingFromWire(boot.reading);
    store.applyReading(first, originalText);
  }

  /* ---- the detector paths -------------------------------------------- */

  const detect = new DetectClient({
    store,
    passageId: boot.passage_id,
    // A 5xx and a dropped connection are the same event to the player: the
  // needle stopped updating. copy.toml has one string for that today.
  onError: () => showNotice("errors.network"),
    onApplied: () => hideNotice(),
  });

  const notice = must("notice");
  const noticeText = must("noticetxt");
  function showNotice(key: string): void {
    noticeText.textContent = copy.t(key);
    notice.setAttribute("data-tone", "warn");
    notice.hidden = false;
  }
  function hideNotice(): void {
    notice.hidden = true;
    notice.removeAttribute("data-tone");
  }

  let caretChar = 0;

  store.subscribe((state, change) => {
    if (change === "reading" && state.applied !== null) {
      applyReadingToView(state.applied);
    }
  });

  /** Index of the token containing `char` in the reading's own token list —
   *  the ripple's origin has to be an index into the NEW spans, not the old. */
  function indexAtChar(list: readonly TokenHeat[], char: number): number | null {
    for (let i = 0; i < list.length; i++) {
      const t = list[i];
      if (t !== undefined && char <= t.e) return i;
    }
    return list.length > 0 ? list.length - 1 : null;
  }

  let touched = false;

  function applyReadingToView(reading: Reading): void {
    pendingHeat = false;
    const editIndex = touched ? indexAtChar(reading.tokens, caretChar) : null;
    paintMirror(store.get().text, reading.tokens, editIndex);
    needle.setTarget(reading.z);
    gate.setReading(reading);
  }

  /* ---- typing --------------------------------------------------------- */

  const body = document.body;
  function onInput(): void {
    const text = raw.value;
    touched = true;
    // The caret is exact and free; diffing token arrays would report a paste at
    // the start of the changed run instead of where the player actually is.
    caretChar = raw.selectionStart ?? text.length;
    store.setText(text);
    // Text NOW: the mirror defines the box height and the textarea cannot
    // scroll, so a stale mirror clips the line the player is typing on.
    pendingHeat = true;
    paintMirror(text, reanchorTokens(tokens, mirrorText, text), mirror.tokenIndexAtChar(caretChar));
    needle.setPending(true);
    gate.setOutcome(null);
    if (local.ready) local.score(text);
    else detect.schedule(text);
  }

  raw.addEventListener("input", (e) => {
    if ((e as InputEvent).isComposing === true) return;
    onInput();
  });
  raw.addEventListener("compositionend", () => onInput());
  raw.addEventListener("focus", () => {
    body.setAttribute("data-typing", "1");
    sheets.closeAll();
  });
  raw.addEventListener("blur", () => body.setAttribute("data-typing", "0"));
  // The box does not normally scroll (the mirror sizes it), but iOS will scroll
  // a textarea to keep the caret visible — keep the heat under the words.
  raw.addEventListener("scroll", () => {
    mirror.el.style.transform = `translateY(${-raw.scrollTop}px)`;
  });

  /* ---- submit --------------------------------------------------------- */

  const started = Date.now();
  const storage = safeStorage();
  gateEls.checkButton.addEventListener("click", () => {
    void runSubmit();
  });

  async function runSubmit(): Promise<void> {
    if (store.get().submitting) return;
    raw.blur();
    store.setSubmitting(true);
    gate.setSubmitting(true);
    const outcome = await submitText({
      passageId: boot.passage_id,
      levelId: boot.level.id,
      text: raw.value,
      detector: store.get().mode === "LOCAL" ? "local" : "server",
      elapsedMs: Date.now() - started,
      assetBundleId: boot.asset_bundle_id || null,
      sessionId: sessionId(storage),
    });
    store.setSubmitting(false);
    gate.setSubmitting(false);
    if (!outcome.ok) {
      gate.showError(outcome.copyKey, outcome.params);
      return;
    }
    store.setOutcome(outcome.response);
    gate.setOutcome(outcome.response);
    if (outcome.response.cleared) {
      gate.showResult(outcome.response, outcome.text);
    } else {
      gate.showRejection(outcome.response);
    }
  }

  /* ---- the intro ------------------------------------------------------ */

  if (boot.intro !== null && boot.intro.prefix_z.length >= 2) {
    body.setAttribute("data-screen", "intro");
    must("introwrap").hidden = false;
    must("playwrap").hidden = true;
    const minWords = Number(
      boot.level.checks.find((c) => c.check === "word_floor")?.params?.min_words ?? 50,
    );
    new Intro({
      els: {
        wrap: must("introwrap"),
        mirror: must("mirrorIntro"),
        slider: must<HTMLInputElement>("slider"),
        scale: must("slidescale"),
        wordCount: must("wordcount"),
        line: must("introline"),
        start: must<HTMLButtonElement>("begin"),
      },
      copy,
      intro: boot.intro,
      text: originalText,
      tokens,
      minWords,
      needle,
      onStart: () => {
        body.setAttribute("data-screen", "play");
        must("introwrap").hidden = true;
        must("playwrap").hidden = false;
        needle.setTarget(boot.detector.expected_z);
      },
    });
  }

  /* ---- viewport, audio, primer ---------------------------------------- */

  const vv = window.visualViewport;
  if (vv) {
    const publish = (): void => {
      document.documentElement.style.setProperty("--vvh", `${vv.height}px`);
    };
    vv.addEventListener("resize", publish);
    vv.addEventListener("scroll", publish);
    publish();
  }

  const arm = (): void => needle.armAudio();
  window.addEventListener("pointerdown", arm, { once: true });
  window.addEventListener("keydown", arm, { once: true });

  requestAnimationFrame(() => primer.openOnFirstLand(storage));

  /* ---- SERVER -> LOADING -> LOCAL (§5.4) ------------------------------ *
   * The worker is only built if the boot payload names its assets. With no
   * assets this is M2: the server-detect version of the game, playable and
   * shippable, with the LOCAL path simply absent rather than broken.
   *
   * The worker keeps its own seq counter, so every request is stamped with a
   * STORE seq at send time and the reply is re-stamped on the way back. Two
   * independent counters feeding one monotonic guard would let a worker reply
   * numbered 1 outrank a server reply numbered 40.
   * -------------------------------------------------------------------- */

  const local = {
    client: null as DetectorClient | null,
    ready: false,
    sent: new Map<number, { text: string; storeSeq: number }>(),
    score(text: string): void {
      const client = this.client;
      if (client === null) return;
      const storeSeq = store.nextSeq();
      const workerSeq = client.score(text, originalText);
      this.sent.set(workerSeq, { text, storeSeq });
    },
  };

  function applyLocalResult(r: ScoreResponse): void {
    const sent = local.sent.get(r.seq);
    local.sent.delete(r.seq);
    // The worker hashes to bare hex; the wire contract (and the store) uses the
    // `sha256:` prefixed spelling of launder_core.schemas.watermark.
    const hash = r.text_hash.startsWith("sha256:") ? r.text_hash : `sha256:${r.text_hash}`;
    store.applyReading(
      readingFromWire({
        seq: sent?.storeSeq ?? store.nextSeq(),
        text_hash: hash,
        score: r.score,
        z: r.z,
        z_star: r.z_star,
        n_scored: r.n_scored,
        n_tokens: r.n_tokens,
        tokens: r.tokens.map((t) => ({ s: t.s, e: t.e, heat: t.heat, masked: t.masked })),
        preview_distance: r.preview_distance,
      }),
      sent?.text,
    );
  }

  async function upgrade(): Promise<void> {
    const assets = boot.assets;
    // Truthiness, not `=== null`: a server that omits the key entirely means
    // the same thing as one that sends null, and an undefined URL here would
    // surface as a fetch failure instead of as "this build has no local
    // detector".
    if (!assets?.tokenizer_url || !assets.sampling_table_url) return;
    store.setMode("LOADING");
    // No progress bar for a capability the user already appears to have — at
    // most a subtle affordance (§5.4 step 3).
    showNotice("errors.detector_offline");
    try {
      const [blob, table, calibration] = await Promise.all([
        fetch(assets.tokenizer_url).then((r) => r.arrayBuffer()),
        fetch(assets.sampling_table_url).then((r) => r.arrayBuffer()),
        assets.calibration_url === undefined
          ? Promise.resolve(undefined)
          : fetch(assets.calibration_url).then((r) => r.json() as Promise<unknown>),
      ]);
      const client = new DetectorClient({
        onResult: applyLocalResult,
        onReady: () => {
          local.ready = true;
          store.setMode("LOCAL");
          hideNotice();
          // The handover: score the CURRENT exact text and let the store's one
          // rule decide. Both paths are the same algorithm on the same ids, so
          // the needle's target has nothing to travel and the switch is
          // invisible.
          local.score(raw.value);
          detect.stop();
        },
        onFatal: (message: string) => fail(message),
      });
      local.client = client;
      client.init(blob, table, calibration, `tok:${boot.asset_bundle_id}`);
    } catch (err) {
      fail(String(err));
    }
  }

  /** Permanent and quiet (§5.4 step 5). Never retried in a loop. */
  function fail(message: string): void {
    if (message !== "") console.warn(`local detector unavailable: ${message}`);
    local.ready = false;
    local.client?.terminate();
    local.client = null;
    store.setMode("SERVER");
    hideNotice();
  }

  void upgrade();

  // First reading for the pristine passage when the server did not inline one.
  if (boot.reading === null) detect.flush(originalText);
}

const boot = readBoot();
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", () => start(boot));
} else {
  start(boot);
}
