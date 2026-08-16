/**
 * POST /api/submit — TECH_PLAN.md §9.3.
 *
 * NOTE THE REQUEST SHAPE: no scores, no z, no distance. The client asserts
 * nothing. Distance, the detector reading and every verdict are computed
 * server-side, which is what makes publishing the watermark keys costless to
 * integrity (§1, §12). `launder_core.schemas.api.SubmitRequest` rejects a body
 * carrying any of those fields by name, so adding one here fails loudly rather
 * than quietly becoming trusted.
 *
 * A rejection is a game outcome and arrives as 200. Real HTTP errors are
 * reserved for real errors, and each maps to one specific string in
 * data/config/copy.toml — never to an apology (§10.6).
 */

import type { CopyParams, SubmitResponseWire } from "../state";

export type SubmitOutcome =
  | { ok: true; response: SubmitResponseWire; text: string }
  | { ok: false; copyKey: string; params?: CopyParams; text: string };

export interface SubmitOptions {
  passageId: string;
  levelId: string;
  text: string;
  detector: "local" | "server";
  elapsedMs: number;
  assetBundleId?: string | null;
  sessionId?: string | null;
  endpoint?: string;
  signal?: AbortSignal;
  fetchImpl?: typeof fetch;
}

const STATUS_COPY: Record<number, string> = {
  404: "errors.unknown_passage",
  413: "errors.too_long",
  429: "errors.rate_limited",
};

export async function submitText(o: SubmitOptions): Promise<SubmitOutcome> {
  const doFetch = o.fetchImpl ?? globalThis.fetch.bind(globalThis);
  const body: Record<string, unknown> = {
    passage_id: o.passageId,
    level_id: o.levelId,
    text: o.text,
    client: {
      detector: o.detector,
      elapsed_ms: Math.max(0, Math.round(o.elapsedMs)),
      ...(o.assetBundleId ? { asset_bundle_id: o.assetBundleId } : {}),
    },
  };
  if (o.sessionId) body.session_id = o.sessionId;

  try {
    const res = await doFetch(o.endpoint ?? "/api/submit", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
      ...(o.signal ? { signal: o.signal } : {}),
      cache: "no-store",
    });
    if (res.status === 429) {
      const retry = Number.parseInt(res.headers.get("retry-after") ?? "", 10);
      return {
        ok: false,
        copyKey: "errors.rate_limited",
        params: { retry_after_s: Number.isFinite(retry) ? retry : 60 },
        text: o.text,
      };
    }
    if (!res.ok) {
      return { ok: false, copyKey: STATUS_COPY[res.status] ?? "errors.network", text: o.text };
    }
    return { ok: true, response: (await res.json()) as SubmitResponseWire, text: o.text };
  } catch {
    // "No connection. Your text is safe; try Check again." — the reassurance is
    // load-bearing: the player's edits live in the textarea, not on the server.
    return { ok: false, copyKey: "errors.network", text: o.text };
  }
}

/** localStorage UUID. NOT identity, NOT trusted; used only for streaks. */
export function sessionId(storage: Storage | null): string | null {
  if (storage === null) return null;
  const key = "launderlm.session.v1";
  try {
    const existing = storage.getItem(key);
    if (existing !== null && existing !== "") return existing;
    const fresh =
      typeof crypto.randomUUID === "function"
        ? crypto.randomUUID()
        : `s_${Date.now().toString(36)}${Math.random().toString(36).slice(2, 10)}`;
    storage.setItem(key, fresh);
    return fresh;
  } catch {
    return null;
  }
}
