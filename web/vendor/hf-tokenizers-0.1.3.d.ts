/**
 * Hand-written ambient types for the vendored `@huggingface/tokenizers@0.1.3`
 * build (`dist/tokenizers.mjs`, copied verbatim — see the sibling `.sha256`).
 *
 * The upstream package ships `types/index.d.ts`, but we vendor only `dist/`
 * (TECH_PLAN.md §2.2: "pin exactly, vendor the 30 KB"), so the surface we
 * actually call is declared here. Keeping it narrow is deliberate: if an
 * upgrade changes `new Tokenizer(json, config)` or `encode()`'s option bag,
 * this file stops compiling instead of the detector silently reading garbage.
 */
declare module "@vendor/tokenizers" {
  /** A `tokenizer.json`-shaped object. Deliberately loose: `blob.ts` builds it. */
  export type TokenizerJSON = Record<string, unknown>;
  /** A `tokenizer_config.json`-shaped object. */
  export type TokenizerConfigJSON = Record<string, unknown>;

  export interface EncodeOptions {
    text_pair?: string | null;
    add_special_tokens?: boolean;
    return_token_type_ids?: boolean | null;
  }

  export interface Encoding {
    ids: number[];
    tokens: string[];
    attention_mask: number[];
    token_type_ids?: number[];
  }

  export class Tokenizer {
    constructor(tokenizerJSON: TokenizerJSON, tokenizerConfig?: TokenizerConfigJSON);
    encode(text: string, options?: EncodeOptions): Encoding;
    decode(ids: number[], options?: Record<string, unknown>): string;
    token_to_id(token: string): number | undefined;
    id_to_token(id: number): string | undefined;
    get_vocab(withAddedTokens?: boolean): Map<string, number>;
  }
}
