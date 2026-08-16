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
import { LivePreview } from "./game/live";
import {
  Mirror,
  layoutFromTokens,
  layoutFromWords,
  reanchorTokens,
} from "./game/mirror";
import { Needle, formatPoints } from "./game/needle";
import { Primer, Sheets, safeStorage } from "./game/primer";
import {
  markCleared,
  mergeCleared,
  perLevelLine,
  readProgress,
  reconcile,
  safeSession,
  totalChanged,
  writeLevelCookie,
} from "./game/progress";
import { DetectClient } from "./net/detect";
import { sessionId, submitText } from "./net/submit";
import {
  Copy,
  Store,
  readBoot,
  readingFromWire,
  type Boot,
  type ProgressWire,
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
  const storage = safeStorage();
  const session = safeSession();
  const nav = { search: location.search, replace: (url: string) => location.replace(url) };

  /* ---- rail furniture that is data, not copy ---- */
  // "Level 3 of 15". The ruleset id ("L1") is wire and DB vocabulary; the
  // player is never shown it, because two numbering systems on one screen is
  // how "level 3" and "level L1" ended up meaning different things.
  must("levelno").textContent = copy.t("screen.level_label", {
    n: boot.level_n,
    total: boot.level_count,
  });
  must("lvlname").textContent = boot.level.name;

  /* ---- which level should be on screen (§7) --------------------------- *
   * The server picked this page from the `launder_level` cookie, and the
   * cookie can be missing (first visit), stale (cleared on another device) or
   * unwritable. Reconciling can end in a reload, and we deliberately DO NOT
   * bail out of start() when it does: the passage and the needle are already
   * painted from the inlined boot payload, the reload takes a moment to
   * commit, and a page that is dead for that moment is worse than one that
   * finishes building and is then thrown away. If the navigation silently
   * fails, the game is still fully playable on the level that is on screen.
   * -------------------------------------------------------------------- */
  reconcile({
    storage,
    session,
    jar: document,
    nav,
    levelN: boot.level_n,
    levelCount: boot.level_count,
  });

  // ONE relabelling, shared by the needle, the threshold label and the result
  // sheet's big number: three places printing the same reading from three
  // copies of the scale is three chances to disagree.
  const pointsCfg = { zStar: boot.detector.z_star, scale: boot.detector.scale };

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
    { ...pointsCfg, copy },
    boot.detector.expected_z,
  );
  // "Under 36 to clear" — NOT "floor", which already means the word minimum.
  must("floorlbl").textContent = copy.t("readout.threshold_label", {
    z_star_display: formatPoints(boot.detector.z_star, pointsCfg),
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
    resultNote: must("rnote"),
    resultActions: must("ractions"),
    allClear: must("rallclear"),
    nextButton: must("nextlvl"),
    // `must`, not a detached fallback: #copyres carries `data-copy-opt`, so it
    // only survives applyCopySlots while copy.toml defines `screen.copy_label`.
    // A silent stand-in element let that key go missing and took the share's
    // copy button with it — a listener bound to nothing, and no error anywhere.
    copyButton: must("copyres"),
    checkButton: must<HTMLButtonElement>("check"),
    changed: [must("changed"), must("stripcount")],
    parLine: must("parline"),
  };

  /** The whole campaign in one paste: every level's best distance, their sum,
   *  and the origin the player is actually on — never a hardcoded domain, so a
   *  preview deployment shares its own URL rather than advertising production. */
  function campaignShare(): string {
    const progress = readProgress(storage);
    return copy.t("readout.share_all_template", {
      level_count: boot.level_count,
      total: totalChanged(progress, boot.level_count),
      per_level: perLevelLine(progress, boot.level_count),
      url: location.origin,
    });
  }

  // Every closed-form rule, answered from the text in the box. Constructed with
  // the pristine passage, so before the player types anything the checklist
  // already reads as it should: nothing changed, nothing invisible, the phrase
  // still there, and only the detector above the line.
  const live = new LivePreview({
    original: originalText,
    lockedPhrases: boot.locked_phrases ?? [],
    points: pointsCfg,
  });

  const gate = new Gate({
    doc: document,
    els: gateEls,
    copy,
    level: boot.level,
    par: boot.par,
    originalText,
    sheets,
    live,
    points: pointsCfg,
    levelN: boot.level_n,
    levelCount: boot.level_count,
    campaignShare,
    onNext: (res) => {
      // Both stores, in this order: the cookie decides what `GET /` renders and
      // the record decides what the cookie should say next time. `assign`, not
      // `replace` — the result sheet the player just read is a reasonable place
      // for the back button to return to.
      //
      // Unconditional, unlike the submit path: a player who tapped a button
      // labelled "next level" must arrive at the next level. If the gate was
      // down and the clear was provisional, this is the one place that lets
      // them past it — the SERVER's record still does not have the level, so
      // nothing they carry to another device claims a clear that never ran.
      markCleared(storage, boot.level_n, res.score.distance);
      writeLevelCookie(document, Math.min(boot.level_n + 1, boot.level_count));
      location.assign("/");
    },
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

  /**
   * Paint the mirror, and let it name the ripple's origin ITSELF.
   *
   * `caret` is a character offset into `text`, not a token index, and the
   * translation happens AFTER `setText` — against the layout that is about to be
   * painted. It used to be translated by the caller, against whatever layout the
   * mirror still held from the previous frame, and on the typing path that
   * layout describes the text as it was BEFORE the keystroke. Typing a character
   * at the end of the passage put the impulse (and the `--d` propagation origin)
   * on the second-to-last span rather than on the one just typed, and a paste of
   * N characters moved it N characters' worth of tokens to the right of the
   * edit. `layoutFromTokens` may also DROP a malformed token, so the layout's
   * own span list is the only list whose indices match the spans on screen.
   */
  function paintMirror(text: string, next: readonly TokenHeat[], caret: number | null): void {
    const layout = layoutFromTokens(text, next);
    mirror.setText(layout);
    mirror.setHeat(heatOf(next), {
      editIndex: caret === null ? null : mirror.tokenIndexAtChar(caret),
      prev: lastHeat,
      pending: pendingHeat,
    });
    lastHeat = heatOf(next);
    mirrorText = text;
    tokens = next.slice();
  }

  mirror.hardRepaint(layoutFromTokens(originalText, tokens), heatOf(tokens), pendingHeat);

  /* ---- the detector paths -------------------------------------------- */

  const detect = new DetectClient({
    store,
    passageId: boot.passage_id,
    // A 5xx and a dropped connection ARE the same event to the player: the
    // needle stopped updating, and "No connection. Your text is safe" is true
    // of both. A 404 is not — it means this page is playing a level the server
    // does not have (a tab left open across a deploy, or a hand-typed
    // `?level=`), and no amount of waiting fixes it; the fix is a reload, which
    // is what `errors.unknown_passage` says. `DetectClient` has always passed
    // the status and this callback has always dropped it, under a comment
    // claiming copy.toml had one string for the case — it has had two since the
    // campaign shipped, and `net/submit.ts` has been mapping them all along.
    onError: (_kind, status) =>
      showNotice(status === 404 ? "errors.unknown_passage" : "errors.network"),
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

  let touched = false;

  function applyReadingToView(reading: Reading): void {
    pendingHeat = false;
    paintMirror(store.get().text, reading.tokens, touched ? caretChar : null);
    needle.setTarget(reading.z);
    gate.setReading(reading);
  }

  /* ---- the reading the SERVER already inlined (§5.4 step 1) ------------ *
   * Applied HERE, below the subscription, and that position is the whole
   * point. It used to run beside `hardRepaint`, before `store.subscribe` had
   * been called — so the one `emit("reading")` the pristine passage ever
   * produces reached no listener, and `gate.setReading` was never called with
   * it. The mirror was fine (`hardRepaint` paints the same tokens directly),
   * which is what hid it: what was missing was the LIVE detector rows. The two
   * checks that read the needle — `detector_threshold`, the win condition, and
   * `detector_floor` — stayed on "Not checked yet" until the player's first
   * keystroke came back from the detector, on a page that was holding a real
   * reading the entire time. The level's own rule read as unanswered.
   * -------------------------------------------------------------------- */
  if (boot.reading !== null) {
    store.applyReading(readingFromWire(boot.reading), originalText);
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
    paintMirror(text, reanchorTokens(tokens, mirrorText, text), caretChar);
    needle.setPending(true);
    gate.setOutcome(null);
    // The closed-form rules re-answer NOW, from this exact string — they do not
    // wait for the detector's round trip, because none of them needs it.
    gate.setText(text);
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
      // A PROVISIONAL clear is one the gate could not fully run, and it does not
      // count — the server does not record it either, so neither do we. Recorded
      // BEFORE the sheet opens, because the last level's share is built from
      // this record and has to include the clear that just happened.
      if (!outcome.response.provisional) {
        markCleared(storage, boot.level_n, outcome.response.score.distance);
        writeLevelCookie(document, Math.min(boot.level_n + 1, boot.level_count));
      }
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

  /* ---- the server's half of the progress record (§7 step 4) ----------- *
   * localStorage is per-browser; the server's record is per-session-id, and
   * the two disagree the moment somebody clears a phone's site data or opens
   * the game in a second browser they had already used. Union the server's
   * list in and run the SAME reconciliation again, sharing the same
   * one-reload-per-tab budget so a disagreement can never cost two reloads.
   *
   * This is an upgrade, not a requirement: every failure path — no session id,
   * offline, a 5xx, a body that is not what we expect — leaves the local
   * record exactly as it was and the game exactly as playable.
   * -------------------------------------------------------------------- */
  async function syncProgress(): Promise<void> {
    const sid = sessionId(storage);
    if (sid === null) return;
    try {
      const res = await fetch(`/api/progress?session_id=${encodeURIComponent(sid)}`, {
        cache: "no-store",
      });
      if (!res.ok) return;
      const wire = (await res.json()) as ProgressWire;
      if (!Array.isArray(wire.cleared)) return;
      mergeCleared(storage, wire.cleared);
      reconcile({
        storage,
        session,
        jar: document,
        nav,
        levelN: boot.level_n,
        levelCount: boot.level_count,
      });
    } catch {
      /* the campaign is playable from localStorage alone */
    }
  }
  void syncProgress();

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
    // The store's guard compares against the `sha256:`-prefixed spelling of
    // `launder_core.schemas.watermark`, and an unprefixed digest would never
    // match — it would silently drop every local reading.
    //
    // `detector.worker.ts:123` already posts the prefixed form, so this is
    // belt and braces rather than a conversion. The comment here used to say
    // the opposite ("the worker hashes to bare hex"), which would send the next
    // reader to fix the wrong side of the boundary.
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
