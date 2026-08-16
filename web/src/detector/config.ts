/**
 * The SynthID configuration, mirrored from `data/config/watermark.toml`.
 *
 * CHANGING ANY OF THESE SIX FIELDS INVALIDATES EVERY PASSAGE. They are the only
 * inputs to `wm_config_id`, which the client asserts against every passage on
 * load (§4.5 "runtime tripwire"). The constants below are a *cache* of the TOML;
 * `wmConfigId()` recomputes the digest so a drift between this file and the
 * shipped `wm_config.v1.json` is a loud failure, not a silent one.
 */

export interface SynthIDConfig {
  readonly ngramLen: number;
  readonly contextHistorySize: number;
  readonly samplingTableSize: number;
  readonly samplingTableSeed: number;
  readonly skipFirstNgramCalls: boolean;
  readonly keys: readonly number[];
}

/**
 * The canonical published 30-key set (§4.2). DeepMind's `synthid_mixin`, the HF
 * research-projects config and both published Hub detector `config.json` files
 * agree byte-for-byte.
 *
 * `keys.length === 30` IS the tournament depth `m`. Do NOT copy the
 * transformers docstring's 9-key example.
 */
export const CANONICAL_KEYS: readonly number[] = [
  654, 400, 836, 123, 340, 443, 597, 160, 57, 29, 590, 639, 13, 715, 468, 990,
  966, 226, 324, 585, 118, 504, 421, 521, 129, 669, 732, 225, 90, 960,
];

export const WATERMARK_CONFIG: SynthIDConfig = {
  ngramLen: 5,
  contextHistorySize: 1024,
  samplingTableSize: 65536,
  samplingTableSeed: 0,
  skipFirstNgramCalls: false,
  keys: CANONICAL_KEYS,
};

/** Gemma-3's `<eos>`. `<pad>`=0, `<eos>`=1, `<bos>`=2, `<unk>`=3. */
export const GEMMA3_EOS_TOKEN_ID = 1;

/**
 * **THE SHIPPED SCORING eos POLICY.** Mirrors
 * `launder_core.watermark.config.SCORING_EOS_TOKEN_ID`, and
 * `data/config/watermark.toml [scoring]` records the same two values so a boot
 * assertion catches drift on the Python side.
 *
 * `null` — no eos mask, in either runtime. `compute_eos_token_mask` zeroes the
 * first `eos_token_id` AND EVERYTHING AFTER IT, and `add_special_tokens=false`
 * still maps the literal string `<eos>` to id 1: with the mask on, typing it
 * near the start of the passage masks every downstream row, `n_scored`
 * collapses, `sigmaNull` explodes and `z` drops under the notch for free. §6.2's
 * canonical scoring unit is the passage text alone and carries no eos, so the
 * mask could only ever fire on that exploit.
 *
 * This constant exists because the two runtimes silently disagreed: this file's
 * detector defaulted to `1` while `compute_frame` defaulted to `None`, and the
 * same sentence read z 1.971 in the browser and z 0.109 on the server. Both
 * call sites now pass it by name.
 */
export const SCORING_EOS_TOKEN_ID: number | null = null;

/** `wm_config_id` recorded in `data/config/watermark.toml`. */
export const EXPECTED_WM_CONFIG_ID =
  "wm1:4f5f7fa87a12a49e36c0f83eeb4c28ac3b6f4957c4dcbe54a50514eff170a591";

/** The tournament depth `m`. */
export function depth(cfg: SynthIDConfig): number {
  return cfg.keys.length;
}

/**
 * The exact bytes hashed into `wm_config_id`: JSON with keys sorted
 * alphabetically and no whitespace. Six fields, nothing else — adding a field
 * here silently invalidates every passage on disk.
 */
export function wmConfigCanonicalJson(cfg: SynthIDConfig): string {
  return JSON.stringify({
    context_history_size: cfg.contextHistorySize,
    keys: cfg.keys,
    ngram_len: cfg.ngramLen,
    sampling_table_seed: cfg.samplingTableSeed,
    sampling_table_size: cfg.samplingTableSize,
    skip_first_ngram_calls: cfg.skipFirstNgramCalls,
  });
}

/**
 * `wm1:<sha256 of the canonical JSON>`.
 *
 * Async because `crypto.subtle` is async everywhere it exists (workers, main
 * thread, and node >= 19's `globalThis.crypto`).
 */
export async function wmConfigId(cfg: SynthIDConfig): Promise<string> {
  const bytes = new TextEncoder().encode(wmConfigCanonicalJson(cfg));
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  const hex = Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join("");
  return `wm1:${hex}`;
}
