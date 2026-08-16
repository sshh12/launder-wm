/**
 * The edit-distance preview. PREVIEW ONLY — §8.3 makes the server authority for
 * every number that is persisted, ranked or shared. This exists so the player
 * sees "N words changed" while typing, off the identical algorithm.
 */

import { DEFAULT_NORMALIZE_CONFIG, type NormalizeConfig, words } from "./normalize.js";
import { type DistanceResult, damerau } from "./damerau.js";

/** `score(original, submission, cfg)` — the same entry point as the Python. */
export function score(
  original: string,
  submission: string,
  cfg: NormalizeConfig = DEFAULT_NORMALIZE_CONFIG,
): DistanceResult {
  return damerau(words(original, cfg), words(submission, cfg));
}

/** Distance only — the hot path for the live preview. */
export function previewDistance(
  original: string,
  submission: string,
  cfg: NormalizeConfig = DEFAULT_NORMALIZE_CONFIG,
): number {
  return score(original, submission, cfg).distance;
}

export {
  normalize,
  words,
  DEFAULT_NORMALIZE_CONFIG,
  WHITESPACE_CLASS,
  type NormalizeConfig,
} from "./normalize.js";
export { damerau, damerauDistance, type EditOp, type EditOpKind, type DistanceResult } from "./damerau.js";
