/**
 * Text normalization for the edit-distance PREVIEW. TS port of
 * `launder_core.scoring.normalize`.
 *
 * **THE SERVER IS AUTHORITY** (§8.3). This runs live so the player sees
 * "N words changed" while typing; the number that is persisted, ranked and
 * shared is recomputed server-side. Any disagreement between the two is a
 * parity bug and is logged as one — golden case #8 exists for exactly this.
 *
 * NFC, NOT NFKC (§8.1). NFKC folds `ﬁ` -> `fi` and `①` -> `1`, changing
 * tokenization in ways the player did not author, and the judge eval set
 * contains NFKC-bait as a MUST-FAIL case. Folding it here would let that
 * exploit through. Zero-width characters, soft hyphens, bidi controls,
 * variation selectors and private-use codepoints are NOT stripped either — they
 * are REJECTED by the `unicode_sanitation` gate. Stripping would let the
 * exploit succeed at the detector while the judge saw clean text.
 */

export interface NormalizeConfig {
  readonly nfc: boolean;
  readonly collapseWhitespace: boolean;
  readonly foldSmartQuotes: boolean;
  readonly foldDashes: boolean;
  readonly foldEllipsis: boolean;
  readonly nbspToSpace: boolean;
  /** NEVER true — casing is evidence for the judge. */
  readonly lowercase: false;
  /** NEVER true — punctuation is sentence structure. */
  readonly stripPunctuation: false;
}

export const DEFAULT_NORMALIZE_CONFIG: NormalizeConfig = {
  nfc: true,
  collapseWhitespace: true,
  foldSmartQuotes: true,
  foldDashes: true,
  foldEllipsis: true,
  nbspToSpace: true,
  lowercase: false,
  stripPunctuation: false,
};

/**
 * The whitespace class, written out rather than spelled `\s`.
 *
 * This is a real cross-language trap, and neither language's shorthand is the
 * answer. Python's `re` `\s` on `str` matches `\x1c-\x1f` and `\x85` but NOT
 * `﻿`; JavaScript's `\s` matches `﻿` but NOT `\x1c-\x1f` or
 * `\x85`. Using the bare shorthand on either side gives a text containing a
 * BOM, a NEL or a field separator a different word count client- and
 * server-side, which surfaces as a phantom parity bug in the leaderboard.
 *
 * The class below is **Unicode `White_Space` minus U+FEFF** — a named,
 * portable property both languages can spell identically, and it is
 * character-for-character `launder_core.scoring.normalize.WHITESPACE_CODEPOINTS`.
 *
 * Two deliberate exclusions, for the same reason:
 *
 * - `﻿` (U+FEFF) is ZERO WIDTH NO-BREAK SPACE. `unicode_sanitation`
 *   REJECTS it; eating it here would let that exploit through.
 * - `\x1c-\x1f` (FILE/GROUP/RECORD/UNIT SEPARATOR) are C0 controls, not
 *   Unicode whitespace. Python's `\s` matching them is an `re` quirk, not a
 *   spec. Collapsing them would mean `normalize()` silently deleting invisible
 *   control characters from the player's text — precisely the "silently repair
 *   an exploit" behaviour §8.1 forbids ("zero-width, soft hyphen, bidi
 *   controls ... are not stripped here; they are rejected by
 *   `unicode_sanitation`"). They are rejected there, under `control_char`.
 */
export const WHITESPACE_CLASS =
  "\\t\\n\\v\\f\\r \\x85\\u00a0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";

const WS_RUN = new RegExp(`[${WHITESPACE_CLASS}]+`, "gu");
const WS_TRIM = new RegExp(`^[${WHITESPACE_CLASS}]+|[${WHITESPACE_CLASS}]+$`, "gu");

const SMART_DOUBLE = /[“”„‟«»]/gu;
const SMART_SINGLE = /[‘’‚‛‹›]/gu;
const DASHES = /[‐‑‒–—―]/gu;
const ELLIPSIS = /…/gu;
const NBSP = /[  ]/gu;

/**
 * MUST be idempotent: `normalize(normalize(x)) === normalize(x)`, asserted by a
 * property test on both sides. It is also the same function that produces the
 * text sent to the LLM, so the thing hashed for the judge cache key is exactly
 * the thing judged.
 *
 * Order matters. NFC first (so folding sees composed characters), then the
 * character-level folds, then whitespace collapse LAST — because
 * `fold_ellipsis` can emit `...` and `fold_dashes` can emit `-`, neither of
 * which is whitespace, but `nbsp_to_space` emits a space that the collapse must
 * then be able to absorb.
 */
export function normalize(s: string, cfg: NormalizeConfig = DEFAULT_NORMALIZE_CONFIG): string {
  let out = s;
  if (cfg.nfc) out = out.normalize("NFC");
  if (cfg.nbspToSpace) out = out.replace(NBSP, " ");
  if (cfg.foldSmartQuotes) out = out.replace(SMART_DOUBLE, '"').replace(SMART_SINGLE, "'");
  if (cfg.foldDashes) out = out.replace(DASHES, "-");
  if (cfg.foldEllipsis) out = out.replace(ELLIPSIS, "...");
  if (cfg.collapseWhitespace) {
    out = out.replace(WS_RUN, " ");
    out = out.replace(WS_TRIM, "");
  }
  return out;
}

/**
 * `normalize` -> split on whitespace. Punctuation STAYS ATTACHED to its word.
 *
 * Deliberate (§8.1): `"study."` -> `"study,"` costs 1, because it is one thing
 * a player did. Comparing on stripped words would make punctuation free and
 * invite a degenerate "repunctuate everything" move that the tokenizer *does*
 * notice.
 */
export function words(s: string, cfg: NormalizeConfig = DEFAULT_NORMALIZE_CONFIG): string[] {
  const n = normalize(s, cfg);
  if (n.length === 0) return [];
  return n.split(new RegExp(`[${WHITESPACE_CLASS}]+`, "u"));
}
