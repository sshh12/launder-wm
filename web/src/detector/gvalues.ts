/**
 * Sampling table + g-values. TS port of `launder_core.watermark.{table,gvalues}`.
 *
 * ---------------------------------------------------------------------------
 * DIVERGENCE TRAP #1 (§4.3): negative modulo.
 * ---------------------------------------------------------------------------
 * `torch.remainder(-12345678901234567, 65536) == 46201`. JS `BigInt %` gives
 * `-19335`. Half of all depth keys are negative, so a naive `%` sends half the
 * lookups to a negative index, `Uint8Array[-19335]` is `undefined`, `undefined`
 * coerces to `NaN`/0, and the detector reads ~0.5 on absolutely everything
 * while looking completely healthy. `sampleIndex` is the only place a table
 * index is ever computed, and it is the fixed form `((k % N) + N) % N`.
 *
 * ---------------------------------------------------------------------------
 * DIVERGENCE TRAP #3 (§4.3): device-dependent sampling table.
 * ---------------------------------------------------------------------------
 * HF builds the table with `torch.Generator(device=device).manual_seed(0)`;
 * CPU is MT19937 and CUDA is Philox, so a table built on the GPU that generated
 * the passage would be *uncorrelated* with a table built on the CPU that scores
 * it — a needle that moves, looks fine, and measures nothing. The mitigation is
 * that nobody ever generates the table: it is built once on CPU, committed as
 * `data/assets/sampling_table.v1.bin`, and every consumer LOADS it. There is
 * deliberately no table-construction function in this file, in any form.
 */

import { HASH_IV, accumulateHash, accumulateStep, contextHash } from "./hash.js";
import { type SynthIDConfig, depth } from "./config.js";

/** sha256 of the 8,192 packed bytes of `data/assets/sampling_table.v1.bin`. */
export const SAMPLING_TABLE_SHA256_PACKED =
  "5151fe795a218d1adf0b7fd707204de23867ef2653fdebb8c9ff28d93f6aa55b";

/** sha256 of the 65,536 unpacked uint8 values. */
export const SAMPLING_TABLE_SHA256_UNPACKED =
  "a3e9e18ea34546849c5136cf9b06c7bfd03e50e601da14af47cd926f9e623a85";

/** Measured count of 1s in the canonical table. A cheap, always-on tripwire. */
export const SAMPLING_TABLE_ONES = 32743;

/** Hex sha256 of `bytes`, via `crypto.subtle` (present in the worker). */
async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const buf = bytes.buffer.slice(
    bytes.byteOffset,
    bytes.byteOffset + bytes.byteLength,
  ) as ArrayBuffer;
  const digest = await crypto.subtle.digest("SHA-256", buf);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join("");
}

/**
 * **THE BROWSER'S ONLY REAL INTEGRITY CHECK ON ITS KEY MATERIAL.** Call this on
 * the fetched bytes before scoring anything with them.
 *
 * The ones count in `unpackSamplingTable` is not enough and never was: it is
 * INVARIANT UNDER EVERY PERMUTATION, including the exact MSB/LSB bit-order
 * reversal the header comment names as "exactly as undetectable and exactly as
 * fatal as trap #3" — both orders of the committed file count 32,743 ones while
 * disagreeing on 32,930 of the 65,536 values. Python has verified both digests
 * plus the count since day one (`launder_core.watermark.table._load`); the
 * browser declared these two constants and read neither, so the one runtime
 * that fetches the table over a network was the one runtime that did not check
 * it. Verifying costs one `await` on 8 KB.
 *
 * Rejects loudly. `main.ts` treats a throw here as a permanent, quiet fall back
 * to SERVER mode (§5.4 step 5) — a detector reading plausible numbers off the
 * wrong table is worse than no local detector.
 */
export async function assertSamplingTableDigest(
  packed: Uint8Array,
  unpacked?: Uint8Array,
): Promise<void> {
  const gotPacked = await sha256Hex(packed);
  if (gotPacked !== SAMPLING_TABLE_SHA256_PACKED) {
    throw new Error(
      `sampling table: packed sha256 ${gotPacked} != ${SAMPLING_TABLE_SHA256_PACKED}. ` +
        "The table is KEY MATERIAL (TECH_PLAN.md §4.3 #3): a different table gives " +
        "g-values uncorrelated with the watermark — a needle that moves, looks fine and " +
        "measures nothing. Refusing to score locally.",
    );
  }
  if (unpacked !== undefined) {
    // The unpacked digest is what catches a bit-order mistake, which the packed
    // digest and the ones count both survive unchanged.
    const gotUnpacked = await sha256Hex(unpacked);
    if (gotUnpacked !== SAMPLING_TABLE_SHA256_UNPACKED) {
      throw new Error(
        `sampling table: unpacked sha256 ${gotUnpacked} != ${SAMPLING_TABLE_SHA256_UNPACKED}. ` +
          "The packed bytes were right, so this is the BIT ORDER: numpy's packbits is " +
          "MSB-first, and the reversed reading is a permutation of the correct table with " +
          "an identical ones count. Refusing to score locally.",
      );
    }
  }
}

/**
 * Unpack the committed bitmap into one byte per value.
 *
 * Format: `np.packbits` with numpy's default MSB-first bit order — byte `k`
 * holds values `8k .. 8k+7` with value `8k` in bit 7. Getting the bit order
 * backwards produces a table that is a *permutation* of the right one, which
 * is exactly as undetectable and exactly as fatal as trap #3.
 */
export function unpackSamplingTable(packed: Uint8Array, expectedValues = 65536): Uint8Array {
  if (packed.length * 8 !== expectedValues) {
    throw new Error(
      `sampling table: expected ${expectedValues / 8} packed bytes for ${expectedValues} values, ` +
        `got ${packed.length}. This file is key material and its size is fixed; a short read ` +
        `here means the asset was truncated or served with the wrong Content-Encoding.`,
    );
  }
  const out = new Uint8Array(expectedValues);
  let ones = 0;
  for (let i = 0; i < packed.length; i++) {
    const b = packed[i] as number;
    for (let bit = 0; bit < 8; bit++) {
      const v = (b >> (7 - bit)) & 1; // MSB first
      out[i * 8 + bit] = v;
      ones += v;
    }
  }
  if (ones !== SAMPLING_TABLE_ONES) {
    throw new Error(
      `sampling table: ${ones} ones, expected ${SAMPLING_TABLE_ONES}. The table is key material ` +
        `built once on CPU (TECH_PLAN.md §4.3 #3); a different table means g-values uncorrelated ` +
        `with the watermark. Refusing to score.`,
    );
  }
  return out;
}

/**
 * Trap #1, in isolation. `((k % N) + N) % N` — Python/torch remainder
 * semantics, always non-negative. Pinned by golden case #2.
 */
export function sampleIndex(h: bigint, tableSize: number): number {
  const n = BigInt(tableSize);
  return Number(((h % n) + n) % n);
}

/**
 * The `(rows x m)` g-value matrix, row-major, as 0/1 bytes.
 *
 * `rows = T - n + 1`, i.e. one row per full n-gram window. Row `i` covers
 * `ids[i .. i+n-1]`; its "current token" is `ids[i+n-1]`.
 */
export function gValues(
  ids: ArrayLike<number>,
  cfg: SynthIDConfig,
  table: Uint8Array,
): { g: Uint8Array; rows: number; m: number } {
  const n = cfg.ngramLen;
  const m = depth(cfg);
  const rows = Math.max(0, ids.length - n + 1);
  const g = new Uint8Array(rows * m);
  const keys = cfg.keys;

  for (let i = 0; i < rows; i++) {
    // Hash the full n-gram once...
    let shared = HASH_IV;
    for (let k = 0; k < n; k++) {
      shared = accumulateStep(shared, BigInt(ids[i + k] as number));
    }
    // ...then fold in each watermark key as one more accumulate step.
    const base = i * m;
    for (let d = 0; d < m; d++) {
      const hL = accumulateStep(shared, BigInt(keys[d] as number));
      g[base + d] = table[sampleIndex(hL, cfg.samplingTableSize)] as number;
    }
  }
  return { g, rows, m };
}

/**
 * The per-row context hashes: `ctx[i] = accumulate(1, ids[i .. i+n-2])`.
 * These feed the repetition mask, never the g-values.
 *
 * The staging window was an `Int32Array`, which silently truncated any id
 * outside int32 while `gValues` (above) read `ids[i+k]` untruncated — two
 * functions in the bit-exactness file disagreeing about the same array. Gemma-3
 * tops out at 262,143 so no tokenizer output could reach it, but "latent" is not
 * "absent", and the asymmetry fired the moment anyone unit-tested this with
 * synthetic ids. `contextHash` takes an `ArrayLike<number>`, so the window is
 * now a plain array and the two functions read ids identically.
 */
export function contextHashes(ids: ArrayLike<number>, cfg: SynthIDConfig): BigInt64Array {
  const n = cfg.ngramLen;
  const rows = Math.max(0, ids.length - n + 1);
  const out = new BigInt64Array(rows);
  const window = new Array<number>(n - 1);
  for (let i = 0; i < rows; i++) {
    for (let k = 0; k < n - 1; k++) window[k] = ids[i + k] as number;
    out[i] = contextHash(window);
  }
  return out;
}

/** Re-exported so callers that only need the primitive do not reach into hash.ts. */
export { accumulateHash };
