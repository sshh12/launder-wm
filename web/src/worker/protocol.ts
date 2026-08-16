/**
 * The worker wire protocol.
 *
 * Imported by BOTH `detector.worker.ts` and the main thread, so the two cannot
 * drift. The result shape is deliberately `DetectResponse` (§9.2) plus a
 * timing field: one renderer serves the SERVER and LOCAL paths, and the
 * handover is a no-op in the view layer.
 */

import type { TokenHeat } from "../detector/index.js";

export interface InitMessage {
  readonly type: "init";
  /** Decompressed tokenizer blob bytes (transferable). */
  readonly blob: ArrayBuffer;
  /** The 8,192-byte packed sampling table (transferable). */
  readonly samplingTable: ArrayBuffer;
  /** `thresholds.v1.json`, already parsed. Omit to use the built-in default. */
  readonly calibration?: unknown;
  /** Persist the derived merges under this key. Omit to skip IndexedDB. */
  readonly cacheKey?: string;
}

export interface ScoreMessage {
  readonly type: "score";
  /** monotonic; the main thread drops any reply whose seq is below the last applied */
  readonly seq: number;
  readonly text: string;
  /** original passage text, for the live distance preview. Optional. */
  readonly original?: string;
}

export type WorkerRequest = InitMessage | ScoreMessage;

export interface ReadyResponse {
  readonly type: "ready";
  /** `performance.now()` deltas for each cold-start phase, in ms (§5.3). */
  readonly timings: {
    readonly unpackMs: number;
    readonly tokenizerMs: number;
    readonly totalMs: number;
    readonly mergesFromCache: boolean;
  };
  readonly vocabSize: number;
  readonly mergeCount: number;
}

export interface ScoreResponse {
  readonly type: "result";
  readonly seq: number;
  /** sha256 of the exact text this reading describes (§9.2 `text_hash`). */
  readonly text_hash: string;
  readonly score: number;
  readonly z: number;
  readonly z_star: number;
  readonly n_scored: number;
  readonly n_tokens: number;
  readonly masked_fraction: number;
  readonly tokens: TokenHeat[];
  readonly preview_distance: number;
  /**
   * §5.5 REQUIRES this: "Worker score-to-postMessage: <= 12 ms p95 on a
   * mid-range Android. Enforced by a `performance.now()` measurement posted
   * with every result and a Playwright perf assertion in CI on a CPU-throttled
   * Chromium (4x slowdown)." Removing it removes the CI gate.
   */
  readonly elapsed_ms: number;
  /** Sub-timings, so a regression localises without a profiler. */
  readonly timings: {
    readonly encodeMs: number;
    readonly detectMs: number;
    readonly distanceMs: number;
  };
}

export interface ErrorResponse {
  readonly type: "error";
  readonly seq: number | null;
  readonly message: string;
  /**
   * §5.4 item 5: "Failure is permanent and quiet." When true, the main thread
   * must fall back to SERVER mode forever and never retry in a loop.
   */
  readonly fatal: boolean;
}

export type WorkerResponse = ReadyResponse | ScoreResponse | ErrorResponse;
