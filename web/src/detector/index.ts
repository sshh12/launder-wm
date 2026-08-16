/**
 * The detector, assembled. Emits exactly the `/api/detect` response shape
 * (§9.2) so one renderer serves both the SERVER and LOCAL paths and the
 * handover is a no-op in the view layer.
 */

import {
  type SynthIDConfig,
  SCORING_EOS_TOKEN_ID,
  WATERMARK_CONFIG,
  depth,
} from "./config.js";
import { contextHashes, gValues } from "./gvalues.js";
import { combineMasks, eosMask, repetitionMask } from "./mask.js";
import { weightedMean } from "./weightedmean.js";
import { type Calibration, DEFAULT_CALIBRATION, zScore } from "./calibration.js";

export interface CharSpan {
  /** start char offset, inclusive (UTF-16 code units, as the DOM counts) */
  readonly s: number;
  /** end char offset, exclusive */
  readonly e: number;
}

export interface TokenHeat extends CharSpan {
  readonly heat: number;
  readonly masked: boolean;
}

export interface DetectResult {
  readonly score: number;
  readonly z: number;
  readonly z_star: number;
  readonly n_scored: number;
  readonly n_tokens: number;
  readonly masked_fraction: number;
  readonly tokens: TokenHeat[];
  /** row-major g-matrix, kept for `gDigest` and the ripple diff */
  readonly g: Uint8Array;
  readonly rows: number;
  readonly m: number;
}

export interface DetectOptions {
  readonly cfg?: SynthIDConfig;
  readonly calibration?: Calibration;
  /**
   * Omit to get `SCORING_EOS_TOKEN_ID` — THE shipped policy, the same constant
   * `launder_core` passes to `compute_frame`. Pass a number only to reproduce
   * the transformers detector's masking behaviour in a test; passing one in
   * product code re-opens the divergence this constant closed.
   */
  readonly eosTokenId?: number | null;
}

/**
 * Score a token sequence.
 *
 * `spans` must be char offsets for `ids`, one per token, in order. They are
 * carried through untouched — the detector never looks at text.
 *
 * Row `i` covers `ids[i .. i+n-1]`; its heat is attributed to the CURRENT token
 * `i + n - 1` (§4.4: "expressed as current-token indices"). The leading `n-1`
 * tokens are in no window's current position, so they get neutral heat and are
 * reported `masked: true` — they contribute nothing to the score, which is
 * exactly what `masked` means on the wire.
 */
export function detect(
  ids: ArrayLike<number>,
  spans: readonly CharSpan[],
  table: Uint8Array,
  opts: DetectOptions = {},
): DetectResult {
  const cfg = opts.cfg ?? WATERMARK_CONFIG;
  const cal = opts.calibration ?? DEFAULT_CALIBRATION;
  const eos = opts.eosTokenId === undefined ? SCORING_EOS_TOKEN_ID : opts.eosTokenId;
  const n = cfg.ngramLen;
  const m = depth(cfg);

  const { g, rows } = gValues(ids, cfg, table);
  const mask =
    rows === 0
      ? new Uint8Array(0)
      : combineMasks(
          repetitionMask(contextHashes(ids, cfg), cfg.contextHistorySize),
          eosMask(ids, cfg, eos),
        );
  const wm = weightedMean(g, mask, rows, m);
  const z = zScore(wm.score, wm.nScored, cal);

  const tokens: TokenHeat[] = [];
  for (let p = 0; p < spans.length; p++) {
    const span = spans[p] as CharSpan;
    const row = p - (n - 1);
    if (row < 0 || row >= rows) {
      tokens.push({ s: span.s, e: span.e, heat: 0.5, masked: true });
    } else {
      tokens.push({
        s: span.s,
        e: span.e,
        heat: wm.heat[row] as number,
        masked: (mask[row] as number) === 0,
      });
    }
  }

  return {
    score: wm.score,
    z,
    z_star: cal.z_star,
    n_scored: wm.nScored,
    n_tokens: ids.length,
    masked_fraction: wm.maskedFraction,
    tokens,
    g,
    rows,
    m,
  };
}

/**
 * The ripple blast radius (§4.4), as an executable fact rather than a comment.
 *
 * > Editing token `j` changes the g-values of exactly `ngram_len` rows: the
 * > rows whose window ends at `j, j+1, ..., j+ngram_len-1`.
 *
 * Given two token arrays, returns the half-open CURRENT-TOKEN index range whose
 * heat can have changed. Computed by longest common prefix/suffix (never by
 * assuming a 1-word edit is a 1-token edit — Gemma-3 is BPE with byte fallback
 * and leading-space merges, §4.4(a)), then extended rightward by `n-1`.
 *
 * The mask is NOT covered by this span: §4.4(b), the repetition mask has
 * unbounded rightward reach, which is why the whole mask is recomputed every
 * keystroke. This function describes the g-value ripple only, and the mirror
 * uses it for animation, never for correctness.
 */
export function rippleSpan(
  before: ArrayLike<number>,
  after: ArrayLike<number>,
  cfg: SynthIDConfig = WATERMARK_CONFIG,
): { start: number; end: number } {
  const n = cfg.ngramLen;
  const la = before.length;
  const lb = after.length;
  let pre = 0;
  while (pre < la && pre < lb && before[pre] === after[pre]) pre++;
  if (pre === la && la === lb) return { start: 0, end: 0 }; // identical

  let suf = 0;
  while (suf < la - pre && suf < lb - pre && before[la - 1 - suf] === after[lb - 1 - suf]) suf++;

  const start = pre;
  const changedEnd = Math.max(lb - suf, pre + 1); // last differing index in `after`, exclusive
  return { start, end: Math.min(lb, changedEnd + n - 1) };
}

export {
  WATERMARK_CONFIG,
  CANONICAL_KEYS,
  EXPECTED_WM_CONFIG_ID,
  GEMMA3_EOS_TOKEN_ID,
  SCORING_EOS_TOKEN_ID,
  depth,
  wmConfigId,
  wmConfigCanonicalJson,
} from "./config.js";
export type { SynthIDConfig } from "./config.js";
export {
  unpackSamplingTable,
  assertSamplingTableDigest,
  SAMPLING_TABLE_ONES,
  SAMPLING_TABLE_SHA256_PACKED,
  SAMPLING_TABLE_SHA256_UNPACKED,
  sampleIndex,
  gValues,
  contextHashes,
} from "./gvalues.js";
export { repetitionMask, eosMask, combineMasks } from "./mask.js";
export { weightedMean, tournamentWeights } from "./weightedmean.js";
export {
  type Calibration,
  CLOSED_FORM_CALIBRATION,
  DEFAULT_CALIBRATION,
  WEIGHTING_KAPPA,
  Z_STAR,
  kappa,
  sigmaNull,
  weightingKappa,
  zScore,
  parseCalibration,
} from "./calibration.js";
export { accumulateHash, ngramDepthKeys, contextHash, wrap64 } from "./hash.js";
