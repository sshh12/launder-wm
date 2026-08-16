/**
 * Character classification for the LIVE preview. TS port of
 * `launder_core.gates.checks.unicode_sanitation.classify`.
 *
 * **THE SERVER IS AUTHORITY** (§8.3), exactly as for `damerau.ts`. This runs on
 * every keystroke so the "Real characters" rule can go red the moment a paste
 * brings a zero-width joiner in, rather than at the next Check. The verdict that
 * decides the level is still the server's.
 *
 * Two of the four mechanisms port by transcription — the codepoint sets below
 * are the same literals as the Python, and `\p{M}` is exactly the `Mn`/`Mc`/`Me`
 * test `unicodedata.category` performs. The third, the combining-mark run
 * counter, is nine lines of the same loop.
 *
 * **The fourth is the homoglyph predicate, and it does NOT port by
 * transcription.** Python asks `ch.isalpha() or ch.isdigit()` before consulting
 * NFKC, and neither half has an exact JavaScript spelling: `str.isdigit()` is
 * `Numeric_Type ∈ {Digit, Decimal}`, which no `\p{...}` escape names, and the
 * nearest approximations disagree on 41 real codepoints (the circled numbers
 * ⑩–㋅ fold to ASCII digit PAIRS, so Python's `isdigit()` says no and `\p{No}`
 * says yes). A client that is stricter than the gate is worse than no client
 * check at all: the player is shown a red rule and then submits successfully.
 *
 * So the predicate ships as DATA rather than as an algorithm — the exact domain
 * of `homoglyph_target`, 980 codepoints in 117 ranges, computed by scanning all
 * of Unicode in Python. `test_cross_language_contract.py` re-scans and compares
 * against the string below, so the two cannot drift; the encoding is chosen to
 * be diffable by that test rather than to be read by hand.
 */

/** Name -> membership test. The keys are the strings `reject_categories` uses. */
export const REJECT_CATEGORIES = [
  "zero_width",
  "soft_hyphen",
  "bidi_control",
  "control_char",
  "variation_selector",
  "private_use",
  "homoglyph",
  "combining_marks",
] as const;

export type RejectCategory = (typeof REJECT_CATEGORIES)[number];

export type HomoglyphPolicy = "reject_always" | "reject_unless_in_original" | "allow";

const ZERO_WIDTH = new Set([
  0x200b, 0x200c, 0x200d, 0x2060, 0x2061, 0x2062, 0x2063, 0x2064, 0xfeff, 0x180e,
]);
const SOFT_HYPHEN = new Set([0x00ad]);
const BIDI_CONTROL = new Set([
  0x061c, 0x200e, 0x200f, 0x202a, 0x202b, 0x202c, 0x202d, 0x202e, 0x2066, 0x2067, 0x2068, 0x2069,
]);

/**
 * C0 and C1 controls that are NOT whitespace, plus DEL.
 *
 * `\t`, `\n`, `\v`, `\f`, `\r` and U+0085 are excluded: they ARE whitespace,
 * `normalize()` collapses them, and a player who pressed Enter has not exploited
 * anything. U+001C–U+001F are here because `normalize()` deliberately does not
 * collapse them (see `normalize.ts` on the whitespace class) — the two halves of
 * that decision have to stay paired.
 */
function isControlChar(cp: number): boolean {
  if (cp < 0x20) return cp !== 0x09 && cp !== 0x0a && cp !== 0x0b && cp !== 0x0c && cp !== 0x0d;
  if (cp === 0x7f) return true;
  return cp >= 0x80 && cp < 0xa0 && cp !== 0x85;
}

function isVariationSelector(cp: number): boolean {
  return (cp >= 0xfe00 && cp <= 0xfe0f) || (cp >= 0xe0100 && cp <= 0xe01ef);
}

function isPrivateUse(cp: number): boolean {
  return (
    (cp >= 0xe000 && cp <= 0xf8ff) ||
    (cp >= 0xf0000 && cp <= 0xffffd) ||
    (cp >= 0x100000 && cp <= 0x10fffd)
  );
}

/**
 * The domain of `launder_core...unicode_sanitation.homoglyph_target`, as
 * delta-encoded ranges: space-separated `gap:span`, both hex, where `gap` is the
 * number of codepoints skipped since the previous range ended and `span` is the
 * range's length minus one. It covers BOTH mechanisms the Python uses — the
 * NFKC-decomposable lookalikes (fullwidth, mathematical alphanumerics,
 * ligatures, circled letters) and the hand-written `CONFUSABLES` table of
 * cross-script lookalikes that have no decomposition at all.
 *
 * REGENERATE, NEVER HAND-EDIT. `test_homoglyph_ranges_match_the_typescript_table`
 * prints the replacement string when it fails.
 */
export const HOMOGLYPH_RANGES =
  "aa:0 7:1 5:1 76:2 4b:0 40:0 6:5 24:2 43:0 78:0 1:1 3:1 28:2 ad:1 2:2 1:1 1:1 1:0 1:0 " +
  "2:1 1:0 9:0 3:0 3:1 2:0 1:0 1:0 2:1 1:0 3d:1 1:0 7:0 1:0 2:0 4:0 1:2 1:2 2:0 a:0 4:0 " +
  "8:0 1:1 1:0 1:0 f:1 1:0 55:0 c:0 13:0 31:0 76:0 4:0 7:0 17a6:0 1:0 1:1 1:7 1:0 1:5 3:2 " +
  "3:0 1:1 1:0 3:2 2:0 6:3 36:0 3:0 1a:0 2b4:1 2:5 5:a 6:3 1:7 65:0 7:4 1:3 1:0 3:4 6:0 " +
  "3:0 1:0 1:1 1:2 1:1 4:0 b:4 316:8 81:0 791:1 7b74:2 530b:6 409:9 7:19 6:19 84a:0 " +
  "cc5a:54 1:46 1:1 2:0 2:1 2:3 1:b 1:0 1:6 1:40 1:3 2:7 1:6 1:1b 1:3 1:4 1:0 3:6 1:151 " +
  "12a:31 23f0:9";

/** `[start, end]` inclusive pairs, ascending and non-adjacent. */
export function decodeRanges(encoded: string): [number, number][] {
  const out: [number, number][] = [];
  let last = -1;
  for (const chunk of encoded.split(" ")) {
    if (chunk === "") continue;
    const [gapHex, spanHex] = chunk.split(":");
    const start = last + 1 + parseInt(gapHex ?? "", 16);
    const end = start + parseInt(spanHex ?? "", 16);
    if (!Number.isFinite(start) || !Number.isFinite(end)) {
      throw new Error(`unicode: HOMOGLYPH_RANGES has an unreadable chunk ${JSON.stringify(chunk)}`);
    }
    out.push([start, end]);
    last = end;
  }
  return out;
}

const HOMOGLYPHS = decodeRanges(HOMOGLYPH_RANGES);

/** True when this codepoint imitates an ASCII letter or digit. */
export function isHomoglyph(cp: number): boolean {
  let lo = 0;
  let hi = HOMOGLYPHS.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    const range = HOMOGLYPHS[mid] as [number, number];
    if (cp < range[0]) hi = mid - 1;
    else if (cp > range[1]) lo = mid + 1;
    else return true;
  }
  return false;
}

const COMBINING = /\p{M}/u;

export interface ClassifyOptions {
  /** The categories `levels.toml` names. `homoglyph` and `combining_marks` are
   *  governed by their own params and are added by the caller, exactly as the
   *  Python check does. */
  readonly rejectCategories: ReadonlySet<string>;
  readonly homoglyphPolicy: HomoglyphPolicy;
  readonly maxCombiningMarks: number;
  /** The character set of the ORIGINAL passage, for
   *  `homoglyph_policy = "reject_unless_in_original"`: a passage that legitimately
   *  contains a Greek letter must not become unplayable. */
  readonly allowedChars: ReadonlySet<string>;
}

/**
 * Count offending characters per category. An empty object means clean.
 *
 * Call it on NORMALIZED text — the Python check classifies `normalize(raw)`, and
 * normalization changes which characters are present (an NBSP becomes a space).
 */
export function classify(text: string, opts: ClassifyOptions): Record<string, number> {
  const counts: Record<string, number> = {};
  const bump = (category: RejectCategory): void => {
    if (opts.rejectCategories.has(category)) {
      counts[category] = (counts[category] ?? 0) + 1;
    }
  };

  let run = 0;
  for (const ch of text) {
    const cp = ch.codePointAt(0) ?? 0;
    if (ZERO_WIDTH.has(cp)) bump("zero_width");
    else if (SOFT_HYPHEN.has(cp)) bump("soft_hyphen");
    else if (BIDI_CONTROL.has(cp)) bump("bidi_control");
    else if (isControlChar(cp)) bump("control_char");
    else if (isVariationSelector(cp)) bump("variation_selector");
    else if (isPrivateUse(cp)) bump("private_use");

    if (
      opts.homoglyphPolicy !== "allow" &&
      isHomoglyph(cp) &&
      (opts.homoglyphPolicy === "reject_always" || !opts.allowedChars.has(ch))
    ) {
      bump("homoglyph");
    }

    if (COMBINING.test(ch)) {
      run += 1;
      if (run > opts.maxCombiningMarks) bump("combining_marks");
    } else {
      run = 0;
    }
  }
  return counts;
}

/** The total the `[check.unicode_sanitation].reject` template renders as `{count}`. */
export function offendingCount(counts: Record<string, number>): number {
  let total = 0;
  for (const n of Object.values(counts)) total += n;
  return total;
}
