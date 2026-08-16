/**
 * Builds the HF tokenizer from the reconstructed blob, and maps its token
 * strings back to CHARACTER offsets in the source text.
 *
 * `@huggingface/tokenizers@0.1.3` returns `{ids, tokens, attention_mask}` and
 * no offsets, but §9.2's wire shape is char-addressed — the mirror colours DOM
 * spans, not token indices. `charSpans()` below reconstructs them exactly, and
 * fails loudly rather than approximately.
 */

import { Tokenizer } from "@vendor/tokenizers";
import {
  GEMMA3_TOKENIZER_CONFIG,
  type UnpackedBlob,
  buildTokenizerJSON,
  unpackBlob,
} from "./blob.js";
import type { CharSpan } from "../detector/index.js";

/** SentencePiece metaspace. The normalizer maps U+0020 -> this. */
export const METASPACE = "▁";

const BYTE_FALLBACK_RE = /^<0x([0-9A-F]{2})>$/;

export interface BuiltTokenizer {
  readonly tokenizer: Tokenizer;
  readonly blob: UnpackedBlob;
}

/**
 * Behavioural proof that no Unicode-folding normalizer slipped in.
 *
 * `blob.ts` checks the spec structurally; this checks the built object's actual
 * behaviour, which is what actually matters. Under NFKC, `ﬁ` folds to `fi` and
 * `①` folds to `1`, so these encodes would collide. Under Gemma-3's real
 * `Replace(" " -> "▁")` they cannot.
 */
function assertNoNfkcBehaviour(tok: Tokenizer): void {
  const probes: [string, string][] = [
    ["ﬁ", "fi"], // U+FB01 LATIN SMALL LIGATURE FI
    ["①", "1"], // U+2460 CIRCLED DIGIT ONE
    ["ａ", "a"], // U+FF41 FULLWIDTH LATIN SMALL LETTER A
  ];
  for (const [bait, folded] of probes) {
    const a = tok.encode(bait, { add_special_tokens: false }).ids;
    const b = tok.encode(folded, { add_special_tokens: false }).ids;
    if (a.length === b.length && a.every((x, i) => x === b[i])) {
      throw new Error(
        `tokenizer: ${JSON.stringify(bait)} and ${JSON.stringify(folded)} encode identically, ` +
          `which means a compatibility normalizer (NFKC/NFKD) is active. Gemma-3 has none ` +
          `(TECH_PLAN.md §14.2 item 8), and §8.1 depends on it: the judge eval set uses ` +
          `NFKC-bait as a must-fail case. Refusing to build.`,
      );
    }
  }
}

/** Build a tokenizer from a decompressed blob. */
export function buildTokenizerFromBlob(raw: Uint8Array): BuiltTokenizer {
  const blob = unpackBlob(raw);
  const spec = buildTokenizerJSON(blob);
  const tokenizer = new Tokenizer(spec, { ...GEMMA3_TOKENIZER_CONFIG });
  assertNoNfkcBehaviour(tokenizer);
  return { tokenizer, blob };
}

/**
 * Build a tokenizer from pre-derived merges (the IndexedDB warm path), skipping
 * the ~350 ms candidate derivation.
 */
export function buildTokenizerFromParts(blob: UnpackedBlob): BuiltTokenizer {
  const spec = buildTokenizerJSON(blob);
  const tokenizer = new Tokenizer(spec, { ...GEMMA3_TOKENIZER_CONFIG });
  assertNoNfkcBehaviour(tokenizer);
  return { tokenizer, blob };
}

/**
 * Map token pieces back to `[start, end)` char offsets in `text`.
 *
 * The trick that makes this exact rather than heuristic: the normalizer is
 * `Replace(" " -> "▁")`, a strict 1-char -> 1-char substitution, so character
 * indices in the normalized string are identical to character indices in the
 * source. We therefore walk the NORMALIZED text's UTF-8 bytes and never try to
 * undo the metaspace substitution — which is the step that would silently drift
 * on a passage containing a literal `▁`.
 *
 * Each token consumes:
 *   - a byte-fallback token `<0xNN>`: exactly one raw byte
 *   - anything else: the UTF-8 bytes of the piece, which must match the source
 *
 * If the pieces do not reassemble the text byte for byte, this throws. That is
 * deliberate: a silently wrong offset map paints heat on the wrong words, which
 * is precisely the "plausible numbers that mean nothing" failure §4.3 is about.
 */
export function charSpans(text: string, tokens: readonly string[]): CharSpan[] {
  const norm = text.replaceAll(" ", METASPACE);
  const enc = new TextEncoder();
  const normBytes = enc.encode(norm);

  // byteOffset -> UTF-16 char index in `norm` (== char index in `text`)
  const byteToChar = new Int32Array(normBytes.length + 1);
  {
    let b = 0;
    for (let c = 0; c < norm.length; ) {
      const cp = norm.codePointAt(c) as number;
      const units = cp > 0xffff ? 2 : 1;
      const nb = cp < 0x80 ? 1 : cp < 0x800 ? 2 : cp < 0x10000 ? 3 : 4;
      for (let k = 0; k < nb; k++) byteToChar[b + k] = c;
      b += nb;
      c += units;
    }
    byteToChar[normBytes.length] = norm.length;
  }

  const spans: CharSpan[] = [];
  let pos = 0;
  for (let t = 0; t < tokens.length; t++) {
    const piece = tokens[t] as string;
    const start = pos;
    const bf = BYTE_FALLBACK_RE.exec(piece);
    if (bf !== null) {
      const want = parseInt(bf[1] as string, 16);
      if (normBytes[pos] !== want) {
        throw new Error(
          `charSpans: byte-fallback token ${piece} at token ${t} expects byte 0x${want
            .toString(16)
            .padStart(2, "0")} but the text has 0x${(normBytes[pos] ?? -1).toString(16)} ` +
            `at byte ${pos}. The token stream does not reassemble the text.`,
        );
      }
      pos += 1;
    } else {
      const pb = enc.encode(piece);
      for (let k = 0; k < pb.length; k++) {
        if (normBytes[pos + k] !== pb[k]) {
          throw new Error(
            `charSpans: token ${t} ${JSON.stringify(piece)} does not match the text at byte ` +
              `${pos}. The token stream does not reassemble the text (an <unk> token, or a ` +
              `normalizer that drops characters, would do this).`,
          );
        }
      }
      pos += pb.length;
    }
    spans.push({ s: byteToChar[start] as number, e: byteToChar[pos] as number });
  }

  if (pos !== normBytes.length) {
    throw new Error(
      `charSpans: consumed ${pos} of ${normBytes.length} bytes. ${normBytes.length - pos} bytes ` +
        `of the text are not covered by any token.`,
    );
  }
  return spans;
}

/**
 * Encode for SCORING.
 *
 * `add_special_tokens: false` is not a preference, it is
 * `data/config/watermark.toml [scoring]`: the canonical scoring unit is
 * `tokenizer(text)` on the PASSAGE TEXT ALONE — no BOS, no chat template, no
 * prompt. A BOS here would shift every n-gram window by one and change every
 * g-value in the passage.
 */
export function encodeForScoring(
  tok: Tokenizer,
  text: string,
): { ids: number[]; tokens: string[]; spans: CharSpan[] } {
  const enc = tok.encode(text, { add_special_tokens: false });
  return { ids: enc.ids, tokens: enc.tokens, spans: charSpans(text, enc.tokens) };
}

export { unpackBlob, buildTokenizerJSON } from "./blob.js";
export type { UnpackedBlob } from "./blob.js";
