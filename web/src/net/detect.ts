/**
 * POST /api/detect — TECH_PLAN.md §5.4 step 2, §9.2.
 *
 * Fetch with `seq`, abort on supersede. It does not interpret results: the
 * store owns the one reconciliation rule and this module only feeds it.
 *
 * Detection is tokenize + 5-gram hash + weighted mean — single-digit
 * milliseconds of CPU — and the endpoint is deliberately NOT rate limited, so
 * the debounce here is about network churn, not about protecting the server.
 *
 * This is the SERVER path, which is the immediate default and is kept forever:
 * if client/server parity ever proves impossible, deleting the local detector
 * leaves a fully playable game (§5.4 step 5).
 */

import { readingFromWire, type DetectResponseWire, type Store } from "../state";

export type DetectErrorKind = "network" | "http";

export interface DetectOptions {
  store: Store;
  passageId: string;
  endpoint?: string;
  /** §5.4 step 2: debounce `input` at 120ms. */
  debounceMs?: number;
  onError?: (kind: DetectErrorKind, status?: number) => void;
  onApplied?: () => void;
  fetchImpl?: typeof fetch;
}

export class DetectClient {
  private timer: ReturnType<typeof setTimeout> | null = null;
  private inflight: AbortController | null = null;
  private pendingText: string | null = null;
  private readonly endpoint: string;
  private readonly debounceMs: number;
  private readonly fetchImpl: typeof fetch;
  private stopped = false;

  constructor(private readonly o: DetectOptions) {
    this.endpoint = o.endpoint ?? "/api/detect";
    this.debounceMs = o.debounceMs ?? 120;
    this.fetchImpl = o.fetchImpl ?? globalThis.fetch.bind(globalThis);
  }

  schedule(text: string): void {
    if (this.stopped) return;
    this.pendingText = text;
    if (this.timer !== null) clearTimeout(this.timer);
    this.timer = setTimeout(() => {
      this.timer = null;
      void this.send();
    }, this.debounceMs);
  }

  /** Send now, skipping the debounce (used on submit and on handover). */
  flush(text?: string): void {
    if (text !== undefined) this.pendingText = text;
    if (this.timer !== null) {
      clearTimeout(this.timer);
      this.timer = null;
    }
    void this.send();
  }

  stop(): void {
    this.stopped = true;
    if (this.timer !== null) clearTimeout(this.timer);
    this.timer = null;
    this.inflight?.abort();
    this.inflight = null;
  }

  private async send(): Promise<void> {
    const text = this.pendingText;
    if (text === null) return;
    this.pendingText = null;
    // A superseded request is wasted server work and a source of out-of-order
    // responses; abort it rather than let it race.
    this.inflight?.abort();
    const ctrl = new AbortController();
    this.inflight = ctrl;
    const seq = this.o.store.nextSeq();
    try {
      const res = await this.fetchImpl(this.endpoint, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ passage_id: this.o.passageId, text, seq }),
        signal: ctrl.signal,
        cache: "no-store",
      });
      if (!res.ok) {
        this.o.onError?.("http", res.status);
        return;
      }
      const wire = (await res.json()) as DetectResponseWire;
      // The store drops it if seq went backwards or the text moved on.
      if (this.o.store.applyReading(readingFromWire(wire), text)) this.o.onApplied?.();
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      this.o.onError?.("network");
    } finally {
      if (this.inflight === ctrl) this.inflight = null;
    }
  }
}
