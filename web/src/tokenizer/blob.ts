/**
 * Reader for `data/assets/gemma3-tok.v1.bin.br`.
 *
 * Port of the measured reference implementation (`tools/reference/blobio.js`).
 * DO NOT REINVENT THE FORMAT — the packed blob is committed, its blake3 is
 * recorded in `data/config/watermark.toml`, and `forge tok pack` must reproduce
 * it byte for byte or `asset_bundle_id` changes.
 *
 * ---------------------------------------------------------------------------
 * The format (§5.1), single stream, all integers LEB128 varints:
 * ---------------------------------------------------------------------------
 *   v  n                       piece count (262,144)
 *   n x { v len; len bytes }   vocab strings, ID ORDER, UTF-8
 *   v  nnCount                 non-normal piece types, sparse
 *   nnCount x { v deltaId; u8 type }
 *   v  mLen                    merge count (514,906)
 *   mLen x { sv perm[i] - i }  zigzag permutation over the CANONICAL candidate
 *                              list, which the reader re-derives from the vocab
 *   v  aLen                    added tokens
 *   aLen x { v deltaId; u8 flags; [v len; len bytes if flags & 32] }
 *
 * The merge list is not stored. Every one of the 514,906 merges is derivable
 * from the vocab strings alone (measured: `only-in-derived = 0`,
 * `only-in-HF = 0`), so only the tie-ordering among equal-score pieces is
 * irreducible and that is what the zigzag permutation carries. ~79 KiB brotli
 * instead of ~31 MB of JSON.
 */

/** Piece count and merge count of the canonical Gemma-3 blob. Cheap tripwires. */
export const GEMMA3_VOCAB_SIZE = 262_144;
export const GEMMA3_MERGE_COUNT = 514_906;

/**
 * Normalizer / decoder node types that fold Unicode. NONE of these may ever
 * appear in the reconstructed spec.
 *
 * §14.2 item 8: "Gemma-3's tokenizer has no NFKC/`Precompiled` normalizer —
 * just `Replace(" " -> "▁")` plus a no-op `Split`. This is load-bearing (it
 * removes the hardest thing to reimplement in JS) and was read from the real
 * `tokenizer.json`, but re-assert it with a hard check."
 *
 * It is load-bearing twice over: §8.1 requires NFC and explicitly NOT NFKC,
 * because the judge eval set contains NFKC-bait as a MUST-FAIL case. A
 * tokenizer that quietly folded `ﬁ` -> `fi` would let that exploit through at
 * the detector while the judge saw clean text.
 */
const FORBIDDEN_NORMALIZERS = new Set([
  "NFKC",
  "NFKD",
  "NFC",
  "NFD",
  "Precompiled",
  "Nmt",
  "Lowercase",
  "StripAccents",
  "BertNormalizer",
]);

/**
 * Hard check, run on every blob read and re-run by `tools/pack-check.mjs`
 * against the real `data/build/tokenizer.json`.
 */
export function assertNoUnicodeFoldingNormalizer(node: unknown, where = "normalizer"): void {
  if (node === null || node === undefined) return;
  if (Array.isArray(node)) {
    node.forEach((child, i) => assertNoUnicodeFoldingNormalizer(child, `${where}[${i}]`));
    return;
  }
  if (typeof node !== "object") return;
  const o = node as Record<string, unknown>;
  const t = o["type"];
  if (typeof t === "string" && FORBIDDEN_NORMALIZERS.has(t)) {
    throw new Error(
      `tokenizer ${where}: found a Unicode-folding normalizer of type "${t}". Gemma-3 has NONE ` +
        `(TECH_PLAN.md §14.2 item 8) and the project depends on that: §8.1 pins NFC and NOT NFKC ` +
        `because the judge eval set uses NFKC-bait as a must-fail case. Refusing to build.`,
    );
  }
  for (const v of Object.values(o)) {
    if (v !== null && typeof v === "object") assertNoUnicodeFoldingNormalizer(v, where);
  }
}

/** The `post_processor` from the real `tokenizer.json`, verbatim. */
export const GEMMA3_POST_PROCESSOR = {
  type: "TemplateProcessing",
  single: [{ SpecialToken: { id: "<bos>", type_id: 0 } }, { Sequence: { id: "A", type_id: 0 } }],
  pair: [
    { SpecialToken: { id: "<bos>", type_id: 0 } },
    { Sequence: { id: "A", type_id: 0 } },
    { SpecialToken: { id: "<bos>", type_id: 1 } },
    { Sequence: { id: "B", type_id: 1 } },
  ],
  special_tokens: { "<bos>": { id: "<bos>", ids: [2], tokens: ["<bos>"] } },
} as const;

/** The minimal `tokenizer_config.json` the packed blob needs. */
export const GEMMA3_TOKENIZER_CONFIG = {
  add_bos_token: true,
  add_eos_token: false,
  clean_up_tokenization_spaces: false,
  bos_token: "<bos>",
  eos_token: "<eos>",
  unk_token: "<unk>",
  pad_token: "<pad>",
  spaces_between_special_tokens: false,
  tokenizer_class: "GemmaTokenizer",
} as const;

export interface AddedTokenSpec {
  id: number;
  content: string;
  single_word: boolean;
  lstrip: boolean;
  rstrip: boolean;
  normalized: boolean;
  special: boolean;
}

export interface UnpackedBlob {
  /** vocab strings in id order */
  readonly pieces: string[];
  /** SentencePiece piece type per id; 1 = NORMAL */
  readonly types: Uint8Array;
  /** the merge list, `[left, right]` pairs, in HF order */
  readonly merges: [string, string][];
  readonly addedTokens: AddedTokenSpec[];
  /** the raw zigzag deltas, kept so `pack-check` can round-trip without re-deriving */
  readonly permDeltas: Int32Array;
}

/** Varint / zigzag reader over the raw blob bytes. */
class BlobReader {
  private o = 0;
  /**
   * `ignoreBOM: true` is LOAD-BEARING, not a style choice.
   *
   * `TextDecoder` defaults to `ignoreBOM: false`, which silently STRIPS a
   * leading U+FEFF from every decoded string. Node's `Buffer.toString('utf8')`
   * — which the packer uses — does not. Gemma-3's vocab contains pieces that
   * begin with U+FEFF, so without this flag three of the 514,906 merge
   * candidates vanish, the permutation indexes past the end of the candidate
   * list, and the blob reads as "corrupt". Measured: 514,903 vs 514,906.
   *
   * `fatal: true` so invalid UTF-8 throws instead of decoding to U+FFFD, which
   * would corrupt the vocab in exactly the same undetectable way.
   */
  private readonly decoder = new TextDecoder("utf-8", { fatal: true, ignoreBOM: true });
  constructor(private readonly b: Uint8Array) {}

  get offset(): number {
    return this.o;
  }

  /** LEB128 unsigned. Uses `Math.pow(2, s)` not `<<`, because ids exceed 2^31. */
  v(): number {
    let r = 0;
    let s = 0;
    let x: number;
    do {
      if (this.o >= this.b.length) throw new Error("tokenizer blob: truncated varint");
      x = this.b[this.o++] as number;
      r += (x & 0x7f) * Math.pow(2, s);
      s += 7;
    } while (x & 0x80);
    return r;
  }

  /** Zigzag signed. */
  sv(): number {
    const u = this.v();
    return u & 1 ? -((u + 1) / 2) : u / 2;
  }

  u8(): number {
    if (this.o >= this.b.length) throw new Error("tokenizer blob: truncated byte");
    return this.b[this.o++] as number;
  }

  str(n: number): string {
    if (this.o + n > this.b.length) throw new Error("tokenizer blob: truncated string");
    const s = this.decoder.decode(this.b.subarray(this.o, this.o + n));
    this.o += n;
    return s;
  }
}

/**
 * The canonical merge-candidate derivation, byte-identical on both sides of the
 * pack: for each piece in id order, every split point where both halves are in
 * the vocab, split index ascending.
 *
 * `Array.from(string)` iterates CODE POINTS, not UTF-16 units — splitting a
 * surrogate pair would produce two lone surrogates and a candidate list that
 * silently disagrees with the packer's.
 */
export function deriveMergeCandidates(pieces: readonly string[]): [string, string][] {
  const inVocab = new Set(pieces);
  const cand: [string, string][] = [];
  for (let i = 0; i < pieces.length; i++) {
    const cps = Array.from(pieces[i] as string);
    for (let s = 1; s < cps.length; s++) {
      const l = cps.slice(0, s).join("");
      const r = cps.slice(s).join("");
      if (inVocab.has(l) && inVocab.has(r)) cand.push([l, r]);
    }
  }
  return cand;
}

/**
 * Decode the raw (already brotli-decompressed) blob.
 *
 * The caller is responsible for decompression. In the browser that is the
 * network stack: the asset is served as `gemma3-tok.<hash>.bin` with a
 * precompressed `.br` sibling and `Content-Encoding: br`, so `fetch` hands back
 * plain bytes. In node it is `zlib.brotliDecompressSync`.
 */
export function unpackBlob(raw: Uint8Array): UnpackedBlob {
  const r = new BlobReader(raw);

  const n = r.v();
  if (n !== GEMMA3_VOCAB_SIZE) {
    throw new Error(
      `tokenizer blob: leading piece count is ${n}, expected ${GEMMA3_VOCAB_SIZE}. If this is a ` +
        `wildly wrong number the bytes are almost certainly still brotli-compressed — the asset ` +
        `must be served with Content-Encoding: br, or decompressed before it gets here.`,
    );
  }

  const pieces = new Array<string>(n);
  for (let i = 0; i < n; i++) pieces[i] = r.str(r.v());

  const types = new Uint8Array(n).fill(1);
  const nnCount = r.v();
  {
    let prev = 0;
    for (let i = 0; i < nnCount; i++) {
      prev += r.v();
      types[prev] = r.u8();
    }
  }

  const cand = deriveMergeCandidates(pieces);

  const mLen = r.v();
  const merges = new Array<[string, string]>(mLen);
  const permDeltas = new Int32Array(mLen);
  for (let i = 0; i < mLen; i++) {
    const d = r.sv();
    permDeltas[i] = d;
    const c = cand[i + d];
    if (c === undefined) {
      throw new Error(
        `tokenizer blob: merge ${i} points at candidate ${i + d}, outside the ${cand.length} ` +
          `derived candidates. The vocab and the permutation disagree — the blob is corrupt.`,
      );
    }
    merges[i] = c;
  }

  const aLen = r.v();
  const addedTokens = new Array<AddedTokenSpec>(aLen);
  {
    let prev = 0;
    for (let i = 0; i < aLen; i++) {
      prev += r.v();
      const f = r.u8();
      const content = f & 32 ? r.str(r.v()) : (pieces[prev] as string);
      addedTokens[i] = {
        id: prev,
        content,
        single_word: !!(f & 1),
        lstrip: !!(f & 2),
        rstrip: !!(f & 4),
        normalized: !!(f & 8),
        special: !!(f & 16),
      };
    }
  }

  if (r.offset !== raw.length) {
    throw new Error(
      `tokenizer blob: ${raw.length - r.offset} trailing bytes after the added-token table. ` +
        `The blob is a single exact stream; trailing bytes mean a version mismatch.`,
    );
  }

  return { pieces, types, merges, addedTokens, permDeltas };
}

/**
 * Assemble a `tokenizer.json`-equivalent object from the unpacked blob.
 *
 * The four spec nodes below are the real Gemma-3 ones, read from the real
 * `tokenizer.json` and re-asserted by `tools/pack-check.mjs`. Note what is NOT
 * here: no NFKC, no `Precompiled`, no `Metaspace` pre-tokenizer. Just
 * `Replace(" " -> "▁")` and a `Split` that merges with the previous chunk.
 */
export function buildTokenizerJSON(blob: UnpackedBlob): Record<string, unknown> {
  const vocab: Record<string, number> = Object.create(null) as Record<string, number>;
  for (let i = 0; i < blob.pieces.length; i++) vocab[blob.pieces[i] as string] = i;

  const spec: Record<string, unknown> = {
    version: "1.0",
    truncation: null,
    padding: null,
    added_tokens: blob.addedTokens,
    normalizer: { type: "Replace", pattern: { String: " " }, content: "▁" },
    pre_tokenizer: {
      type: "Split",
      pattern: { String: " " },
      behavior: "MergedWithPrevious",
      invert: false,
    },
    post_processor: GEMMA3_POST_PROCESSOR,
    decoder: {
      type: "Sequence",
      decoders: [
        { type: "Replace", pattern: { String: "▁" }, content: " " },
        { type: "ByteFallback" },
        { type: "Fuse" },
      ],
    },
    model: {
      type: "BPE",
      dropout: null,
      unk_token: "<unk>",
      continuing_subword_prefix: null,
      end_of_word_suffix: null,
      fuse_unk: true,
      byte_fallback: true,
      ignore_merges: false,
      vocab,
      merges: blob.merges,
    },
  };

  // The hard check, on the object we are about to hand to the tokenizer.
  assertNoUnicodeFoldingNormalizer(spec["normalizer"], "normalizer");
  assertNoUnicodeFoldingNormalizer(spec["decoder"], "decoder");
  assertNoUnicodeFoldingNormalizer(spec["pre_tokenizer"], "pre_tokenizer");

  return spec;
}

/** Convenience: bytes in, `tokenizer.json`-equivalent out. */
export function rebuildFromBlob(raw: Uint8Array): Record<string, unknown> {
  return buildTokenizerJSON(unpackBlob(raw));
}
