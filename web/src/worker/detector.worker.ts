/**
 * The detector worker. Owns the tokenizer and the detector; the main thread
 * never blocks (§5.5: "All tokenization and scoring is in the worker. The main
 * thread only writes two CSS custom properties per span.").
 *
 * Every result carries a `performance.now()` measurement. That is not
 * telemetry-for-its-own-sake: §5.5 states the worker score-to-postMessage
 * budget (<= 12 ms p95 on a mid-range Android) and enforces it with a Playwright
 * assertion on a 4x-throttled Chromium. The number has to be on the wire for
 * that gate to exist.
 */

/// <reference lib="webworker" />

import { type DetectResult, detect } from "../detector/index.js";
import { SCORING_EOS_TOKEN_ID } from "../detector/config.js";
import { assertSamplingTableDigest, unpackSamplingTable } from "../detector/gvalues.js";
import { type Calibration, DEFAULT_CALIBRATION, parseCalibration } from "../detector/calibration.js";
import { buildTokenizerFromBlob, encodeForScoring } from "../tokenizer/tokenizer.js";
import { mergesToPairs, storeMerges } from "../tokenizer/idb.js";
import { previewDistance } from "../scoring/index.js";
import type { ScoreMessage, WorkerRequest, WorkerResponse } from "./protocol.js";

declare const self: DedicatedWorkerGlobalScope;

interface State {
  readonly encode: (text: string) => ReturnType<typeof encodeForScoring>;
  readonly table: Uint8Array;
  readonly calibration: Calibration;
}

let state: State | null = null;

function post(msg: WorkerResponse, transfer: Transferable[] = []): void {
  self.postMessage(msg, transfer);
}

async function sha256Hex(s: string): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(s));
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join("");
}

async function handleInit(msg: Extract<WorkerRequest, { type: "init" }>): Promise<void> {
  const t0 = performance.now();

  // VERIFY THE KEY MATERIAL BEFORE SCORING WITH IT.
  //
  // `unpackSamplingTable` checks the ones count, and a ones count is invariant
  // under every permutation — including the MSB/LSB bit-order reversal that
  // gvalues.ts names as "exactly as undetectable and exactly as fatal as trap
  // #3". These bytes arrived over the network; Python has verified both digests
  // since day one and this runtime verified neither. `crypto.subtle` is already
  // imported here for the text hash, so it costs one await on 8 KB. A failure
  // throws out of `handleInit` and lands as a FATAL error, which §5.4 step 5
  // turns into a permanent, quiet fall back to SERVER mode.
  const packed = new Uint8Array(msg.samplingTable);
  const table = unpackSamplingTable(packed);
  await assertSamplingTableDigest(packed, table);

  const tUnpack = performance.now();
  const built = buildTokenizerFromBlob(new Uint8Array(msg.blob));
  const tTok = performance.now();

  const calibration =
    msg.calibration === undefined ? DEFAULT_CALIBRATION : parseCalibration(msg.calibration);

  state = {
    encode: (text: string) => encodeForScoring(built.tokenizer, text),
    table,
    calibration,
  };

  post({
    type: "ready",
    timings: {
      unpackMs: tUnpack - t0,
      tokenizerMs: tTok - tUnpack,
      totalMs: tTok - t0,
      mergesFromCache: false,
    },
    vocabSize: built.blob.pieces.length,
    mergeCount: built.blob.merges.length,
  });

  // Persist the derived merges for the next cold load (§5.3 layer 3). Fire and
  // forget: `storeMerges` never throws and never retries, so a private-mode
  // browser costs latency on the next visit and nothing else.
  if (msg.cacheKey !== undefined) {
    const vocab = new Map<string, number>();
    for (let i = 0; i < built.blob.pieces.length; i++) {
      vocab.set(built.blob.pieces[i] as string, i);
    }
    const pairs = mergesToPairs(built.blob.merges, vocab);
    if (pairs !== null) void storeMerges(msg.cacheKey, pairs);
  }
}

function handleScore(msg: ScoreMessage): void {
  const s = state;
  if (s === null) {
    post({ type: "error", seq: msg.seq, message: "worker not initialised", fatal: false });
    return;
  }
  const t0 = performance.now();
  const enc = s.encode(msg.text);
  const t1 = performance.now();
  // eos EXPLICITLY. `launder_serve.engine.CoreDetector` passes the same
  // constant; leaving both sides on their own defaults is what once made the
  // browser read z 1.971 where the server read z 0.109 on the same text.
  const result: DetectResult = detect(enc.ids, enc.spans, s.table, {
    calibration: s.calibration,
    eosTokenId: SCORING_EOS_TOKEN_ID,
  });
  const t2 = performance.now();
  const dist = msg.original === undefined ? 0 : previewDistance(msg.original, msg.text);
  const t3 = performance.now();

  // The hash is async; everything else is done, so resolve it and post once.
  void sha256Hex(msg.text).then((hash) => {
    post({
      type: "result",
      seq: msg.seq,
      text_hash: `sha256:${hash}`,
      score: result.score,
      z: result.z,
      z_star: result.z_star,
      n_scored: result.n_scored,
      n_tokens: result.n_tokens,
      masked_fraction: result.masked_fraction,
      tokens: result.tokens,
      preview_distance: dist,
      elapsed_ms: performance.now() - t0,
      timings: { encodeMs: t1 - t0, detectMs: t2 - t1, distanceMs: t3 - t2 },
    });
  });
}

self.onmessage = (ev: MessageEvent<WorkerRequest>): void => {
  const msg = ev.data;
  try {
    if (msg.type === "init") {
      void handleInit(msg).catch((err: unknown) => {
        post({
          type: "error",
          seq: null,
          message: err instanceof Error ? err.message : String(err),
          fatal: true, // §5.4: build failure is permanent and quiet
        });
      });
    } else {
      handleScore(msg);
    }
  } catch (err: unknown) {
    post({
      type: "error",
      seq: msg.type === "score" ? msg.seq : null,
      message: err instanceof Error ? err.message : String(err),
      fatal: msg.type === "init",
    });
  }
};
