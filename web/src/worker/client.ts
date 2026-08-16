/**
 * Main-thread handle for the detector worker.
 *
 * The game layer imports THIS, never the worker file directly, so the
 * seq-guard and the permanent-failure rule live in one place.
 *
 * §5.4: the state machine is `SERVER -> LOADING -> LOCAL` with `SERVER` as the
 * immediate default, and "failure is permanent and quiet" — a worker that fails
 * to build leaves the game in `SERVER` mode forever and is never retried in a
 * loop. `onFatal` fires exactly once.
 */

import type { ScoreResponse, WorkerRequest, WorkerResponse } from "./protocol.js";

export interface DetectorClientOptions {
  /** Applied only if `seq` is at least the last applied seq (§5.4 item 2). */
  readonly onResult: (r: ScoreResponse) => void;
  readonly onReady?: (r: Extract<WorkerResponse, { type: "ready" }>) => void;
  /** Called at most once. After this, stay in SERVER mode forever. */
  readonly onFatal?: (message: string) => void;
}

export class DetectorClient {
  private readonly worker: Worker;
  private lastApplied = -1;
  private dead = false;
  private seq = 0;

  constructor(private readonly opts: DetectorClientOptions) {
    this.worker = new Worker(new URL("./detector.worker.ts", import.meta.url), { type: "module" });
    this.worker.onmessage = (ev: MessageEvent<WorkerResponse>) => this.receive(ev.data);
    this.worker.onerror = (ev: ErrorEvent) => this.die(ev.message || "worker failed to load");
  }

  /** Hand the worker its assets. Buffers are TRANSFERRED, not copied. */
  init(blob: ArrayBuffer, samplingTable: ArrayBuffer, calibration?: unknown, cacheKey?: string) {
    const msg: WorkerRequest = {
      type: "init",
      blob,
      samplingTable,
      ...(calibration === undefined ? {} : { calibration }),
      ...(cacheKey === undefined ? {} : { cacheKey }),
    };
    this.worker.postMessage(msg, [blob, samplingTable]);
  }

  /** Request a score. Returns the seq assigned, for the caller's own bookkeeping. */
  score(text: string, original?: string): number {
    const seq = ++this.seq;
    const msg: WorkerRequest = {
      type: "score",
      seq,
      text,
      ...(original === undefined ? {} : { original }),
    };
    this.worker.postMessage(msg);
    return seq;
  }

  terminate(): void {
    this.dead = true;
    this.worker.terminate();
  }

  private die(message: string): void {
    if (this.dead) return;
    this.dead = true;
    this.worker.terminate();
    this.opts.onFatal?.(message);
  }

  private receive(msg: WorkerResponse): void {
    if (this.dead) return;
    switch (msg.type) {
      case "ready":
        this.opts.onReady?.(msg);
        return;
      case "result":
        // The seq guard. Out-of-order results are the #1 source of needle
        // flicker; the caller additionally keys on `text_hash` (§5.4 item 2).
        if (msg.seq < this.lastApplied) return;
        this.lastApplied = msg.seq;
        this.opts.onResult(msg);
        return;
      case "error":
        if (msg.fatal) this.die(msg.message);
        return;
    }
  }
}
