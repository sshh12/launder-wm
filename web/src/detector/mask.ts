/**
 * The scoring mask. TS port of `launder_core.watermark.gvalues`'s mask half.
 *
 * `mask[i] = rep_mask[i] * eos_mask[i]` over the `T - n + 1` n-gram rows
 * (§4.2). A masked row contributes nothing to the numerator OR the denominator,
 * which is what makes context repetition a real strategy rather than a bug
 * (§4.4(c)).
 *
 * §4.4(b) is the reason this module exists as a whole-array recompute and not
 * an incremental update: `rep_mask[i]` depends on whether `ctx[i]` appeared
 * among the previous 1,024 contexts, so a single edit can flip the mask
 * arbitrarily far to the right (never to the left — the history is causal).
 * Recomputing all of it is O(T) with a hash map and costs microseconds. Only
 * g-values may be incrementally cached, and only in the solver.
 */

import type { SynthIDConfig } from "./config.js";

/**
 * Causal context-repetition mask over the row-aligned context hashes.
 *
 * The membership test happens BEFORE insertion, and the hash is pushed even
 * when it was a repeat — both details are load-bearing and both are cheap to
 * get wrong. The ring buffer starts full of `0n`, matching the reference
 * (`history = ringbuffer(1024) filled with 0`), so a genuine context hash of
 * exactly zero would be masked; that is the reference behaviour, not an
 * oversight.
 */
export function repetitionMask(ctx: BigInt64Array, historySize: number): Uint8Array {
  const rows = ctx.length;
  const mask = new Uint8Array(rows);

  const ring = new BigInt64Array(historySize); // filled with 0n
  const counts = new Map<bigint, number>();
  counts.set(0n, historySize);
  let head = 0;

  for (let i = 0; i < rows; i++) {
    const h = ctx[i] as bigint;
    mask[i] = (counts.get(h) ?? 0) > 0 ? 0 : 1;

    // push, evicting the oldest — even when this row was a repeat
    const evicted = ring[head] as bigint;
    const ec = (counts.get(evicted) ?? 0) - 1;
    if (ec <= 0) counts.delete(evicted);
    else counts.set(evicted, ec);
    ring[head] = h;
    counts.set(h, (counts.get(h) ?? 0) + 1);
    head = (head + 1) % historySize;
  }
  return mask;
}

/**
 * Row-aligned eos mask: the first `eosTokenId` position and everything after it
 * is zeroed over the token axis, then sliced `[n-1:]` to line up with rows.
 *
 * Row `i`'s current token is `ids[i + n - 1]`, so row `i` survives iff the
 * first eos is strictly after that position.
 */
export function eosMask(
  ids: ArrayLike<number>,
  cfg: SynthIDConfig,
  eosTokenId: number | null,
): Uint8Array {
  const n = cfg.ngramLen;
  const rows = Math.max(0, ids.length - n + 1);
  const mask = new Uint8Array(rows).fill(1);
  if (eosTokenId === null) return mask;

  let firstEos = -1;
  for (let p = 0; p < ids.length; p++) {
    if (ids[p] === eosTokenId) {
      firstEos = p;
      break;
    }
  }
  if (firstEos < 0) return mask;

  for (let i = 0; i < rows; i++) {
    if (i + n - 1 >= firstEos) mask[i] = 0;
  }
  return mask;
}

/** `mask[i] = rep_mask[i] * eos_mask[i]`. */
export function combineMasks(rep: Uint8Array, eos: Uint8Array): Uint8Array {
  const out = new Uint8Array(rep.length);
  for (let i = 0; i < rep.length; i++) out[i] = (rep[i] as number) & (eos[i] as number);
  return out;
}
