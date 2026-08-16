/**
 * Parity: the TS port against `data/golden/vectors.json`, and the tokenizer
 * against all 4,031 Rust `tokenizers` golden vectors from BOTH the original
 * `tokenizer.json` and the packed blob.
 *
 * The SAME `vectors.json` is read by pytest. §4.5: "Regenerating goldens is
 * therefore always a reviewed diff and never a way to turn a red test green."
 */
import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { brotliDecompressSync } from "node:zlib";
import { beforeAll, describe, expect, it } from "vitest";

import { Tokenizer } from "@vendor/tokenizers";
import {
  CLOSED_FORM_CALIBRATION,
  EXPECTED_WM_CONFIG_ID,
  SAMPLING_TABLE_ONES,
  SAMPLING_TABLE_SHA256_PACKED,
  SAMPLING_TABLE_SHA256_UNPACKED,
  assertSamplingTableDigest,
  SCORING_EOS_TOKEN_ID,
  WEIGHTING_KAPPA,
  WATERMARK_CONFIG,
  accumulateHash,
  combineMasks,
  contextHashes,
  detect,
  eosMask,
  gValues,
  ngramDepthKeys,
  repetitionMask,
  rippleSpan,
  sampleIndex,
  tournamentWeights,
  kappa,
  sigmaNull,
  unpackSamplingTable,
  weightingKappa,
  wmConfigId,
  zScore,
  parseCalibration,
} from "../src/detector/index.js";
import {
  GEMMA3_TOKENIZER_CONFIG,
  assertNoUnicodeFoldingNormalizer,
  buildTokenizerJSON,
  unpackBlob,
} from "../src/tokenizer/blob.js";
import { buildTokenizerFromBlob, charSpans, encodeForScoring } from "../src/tokenizer/tokenizer.js";
import { mergesToPairs, pairsToMerges } from "../src/tokenizer/idb.js";

const REPO = fileURLToPath(new URL("../../", import.meta.url));
const VECTORS = JSON.parse(readFileSync(`${REPO}data/golden/vectors.json`, "utf8")) as Golden;
// The golden calibration block is the shipped thresholds file in the shipped
// shape, so it goes through the SAME parser the worker uses for a fetched
// calibration_url. Reading it as a bare object would let the two paths drift.
const VECTOR_CAL = parseCalibration(VECTORS.calibration);

interface Golden {
  schema: string;
  wm_config: Record<string, unknown>;
  sampling_table: Record<string, string | number>;
  calibration: unknown;
  cases: Record<string, any>;
}

const packedTable = new Uint8Array(readFileSync(`${REPO}data/assets/sampling_table.v1.bin`));
const table = unpackSamplingTable(packedTable);
const cfg = WATERMARK_CONFIG;
const S = (x: bigint): string => x.toString();

// ---------------------------------------------------------------------------
describe("vendored tokenizer", () => {
  it("matches its pinned checksum", () => {
    const p = `${REPO}web/vendor/hf-tokenizers-0.1.3.mjs`;
    const bytes = readFileSync(p);
    const actual = createHash("sha256").update(bytes).digest("hex");
    const recorded = readFileSync(`${p}.sha256`, "utf8").trim().split(/\s+/)[0];
    // §2.2: "v0.1.3 is young: pin exactly, vendor the 30 KB into web/vendor/,
    // let the golden gate catch upgrade regressions." A vendored file whose
    // checksum is not asserted is just a copy.
    expect(actual).toBe(recorded);
    expect(actual).toBe("6d92e25f9576e67124b3a3f910f5cc1df95a42bde4df4f5387f62dee4554f301");
    expect(bytes.length).toBe(81970);
  });
});

// ---------------------------------------------------------------------------
describe("golden file", () => {
  it("is schema launder.golden/1 and matches CHECKSUM", () => {
    expect(VECTORS.schema).toBe("launder.golden/1");
    const digest = createHash("sha256")
      .update(readFileSync(`${REPO}data/golden/vectors.json`))
      .digest("hex");
    const checksum = readFileSync(`${REPO}data/golden/CHECKSUM`, "utf8");
    expect(checksum).toContain(digest);
  });

  it("pins the same watermark config the code carries", async () => {
    expect(VECTORS.wm_config["ngram_len"]).toBe(cfg.ngramLen);
    expect(VECTORS.wm_config["keys"]).toEqual(cfg.keys);
    expect(VECTORS.wm_config["wm_config_id"]).toBe(EXPECTED_WM_CONFIG_ID);
    expect(await wmConfigId(cfg)).toBe(EXPECTED_WM_CONFIG_ID);
  });

  it("pins the sampling table this build loaded", () => {
    const st = VECTORS.sampling_table;
    expect("sha256:" + createHash("sha256").update(packedTable).digest("hex")).toBe(
      st["sha256_packed"],
    );
    expect("sha256:" + createHash("sha256").update(table).digest("hex")).toBe(st["sha256_unpacked"]);
    expect(table.reduce((a, b) => a + b, 0)).toBe(st["ones"]);
  });
});

// ------------------------------------------------------------ case 1 -------
describe("case 1 — accumulate_hash (int64 wraparound)", () => {
  const c = VECTORS.cases["accumulate_hash"];
  it("accumulates", () => {
    for (const a of c.accumulate) {
      expect(S(accumulateHash(BigInt(a.iv), a.data))).toBe(a.expected);
    }
  });
  it("derives depth keys (full n-gram first, then the key)", () => {
    for (const a of c.depth_keys) {
      expect(Array.from(ngramDepthKeys(a.ngram, a.keys)).map(S)).toEqual(a.expected);
    }
  });
  it("reproduces the TECH_PLAN §4.5 known-good depth keys", () => {
    expect(Array.from(ngramDepthKeys([1, 235280, 2121, 576, 573], cfg.keys.slice(0, 5))).map(S)).toEqual([
      "-6504205589568445072",
      "318672039786673866",
      "8070454580555681646",
      "8340369110341966617",
      "5852124156879677502",
    ]);
  });
  it("hashes contexts from the n-1 LEADING tokens", () => {
    for (const a of c.context_hashes) {
      expect(Array.from(contextHashes(a.tokens, cfg)).map(S)).toEqual(a.expected);
    }
  });
});

// ------------------------------------------------------------ case 2 -------
describe("case 2 — sample_index (the negative-modulo trap)", () => {
  const c = VECTORS.cases["sample_index"];
  it("always returns a non-negative index", () => {
    for (const a of c.cases) {
      const got = sampleIndex(BigInt(a.h), a.table_size);
      expect(got).toBe(a.expected);
      expect(got).toBeGreaterThanOrEqual(0);
      expect(got).toBeLessThan(a.table_size);
    }
  });
  it("differs from a bare BigInt %, which is the whole point", () => {
    const h = -12345678901234567n;
    expect(Number(h % 65536n)).toBe(-19335); // the wrong answer
    expect(sampleIndex(h, 65536)).toBe(46201); // torch.remainder
  });
});

// ------------------------------------------------------------ case 4 -------
describe("case 4 — g_values", () => {
  const c = VECTORS.cases["g_values"];
  it("reproduces the full 0/1 matrix", () => {
    const { g, rows, m } = gValues(c.ids, cfg, table);
    expect(rows).toBe(c.rows);
    const asRows: string[] = [];
    for (let i = 0; i < rows; i++) asRows.push(Array.from(g.subarray(i * m, (i + 1) * m)).join(""));
    expect(asRows).toEqual(c.g_rows);
    expect("sha256:" + createHash("sha256").update(g).digest("hex")).toBe(c.g_digest);
  });
});

// ------------------------------------------------------------ case 5 -------
describe("case 5 — repetition_mask", () => {
  const c = VECTORS.cases["repetition_mask"];
  it("masks the repeated 4-gram, causally", () => {
    const ctx = contextHashes(c.ids, cfg);
    expect(Array.from(ctx).map(S)).toEqual(c.context_hashes);
    const rep = repetitionMask(ctx, cfg.contextHistorySize);
    expect(Array.from(rep)).toEqual(c.repetition_mask);
    const eos = eosMask(c.ids, cfg, null);
    expect(Array.from(eos)).toEqual(c.eos_mask);
    expect(Array.from(combineMasks(rep, eos))).toEqual(c.mask);
    expect(Array.from(rep).reduce((a: number, b: number) => a + b, 0)).toBe(c.n_scored);
  });
  it("zeroes the first eos and everything after it", () => {
    for (const e of c.eos_cases) {
      expect(Array.from(eosMask(e.ids, cfg, e.eos_token_id))).toEqual(e.expected);
    }
  });
});

// ------------------------------------------------------------ case 6 -------
describe("case 6 — score (tol 1e-9, float64 both sides)", () => {
  const c = VECTORS.cases["score"];
  it("scores through the SHIPPED eos default, which the golden file records", () => {
    expect(c.eos_token_id ?? null).toBe(SCORING_EOS_TOKEN_ID);
  });
  it("distinguishes the two eos policies where a case carries id 1", () => {
    // The regression this case exists for: a text with the literal string
    // `<eos>` in it. Under the shipped policy (no mask) all rows score; under
    // eos=1 everything from that token on is masked away and z collapses.
    let exercised = 0;
    for (const cc of c.cases) {
      if (!cc.eos_masked_variant) continue;
      exercised++;
      const spans = (cc.ids as number[]).map((_, i) => ({ s: i, e: i + 1 }));
      const masked = detect(cc.ids, spans, table, {
        calibration: VECTOR_CAL,
        eosTokenId: cc.eos_masked_variant.eos_token_id,
      });
      expect(masked.n_scored).toBe(cc.eos_masked_variant.n_scored);
      expect(Math.abs(masked.score - cc.eos_masked_variant.score)).toBeLessThanOrEqual(c.tol);
      expect(masked.n_scored).not.toBe(cc.n_scored);
    }
    expect(exercised).toBeGreaterThan(0);
  });
  it("uses weights that sum to m", () => {
    const w = tournamentWeights(30);
    expect(Array.from(w)).toEqual(c.weights);
    expect(w.reduce((a, b) => a + b, 0)).toBeCloseTo(30, 12);
  });
  it("reproduces score, z, n_scored and the per-token heat array", () => {
    for (const cc of c.cases) {
      const spans = (cc.ids as number[]).map((_, i) => ({ s: i, e: i + 1 }));
      // NO `eosTokenId`: the SHIPPED DEFAULT is the thing under test. Passing
      // the golden file's value explicitly is what let the worker (which
      // omitted it and masked at id 1) and `compute_frame` (which omitted it
      // and masked nothing) diverge by 18x in z without this file noticing.
      const r = detect(cc.ids, spans, table, { calibration: VECTOR_CAL });
      expect(Math.abs(r.score - cc.score)).toBeLessThanOrEqual(c.tol);
      expect(Math.abs(r.z - cc.z)).toBeLessThanOrEqual(c.tol);
      expect(r.n_scored).toBe(cc.n_scored);
      expect(r.n_tokens).toBe(cc.n_tokens);
      expect(r.tokens.length).toBe(cc.heat.length);
      r.tokens.forEach((t, i) => {
        expect(Math.abs(t.heat - cc.heat[i])).toBeLessThanOrEqual(c.tol);
        expect(t.masked ? 1 : 0).toBe(cc.masked[i]);
      });
    }
  });
  it("z is (score - 0.5) / sigma_null and scales like sqrt(T)", () => {
    // §4.2's "the intro slider's lesson (z ∝ √T) is a literal instrument
    // reading": at a fixed score, quadrupling the scored length doubles z.
    const cal = VECTOR_CAL;
    const z1 = zScore(0.52, 100, cal);
    const z4 = zScore(0.52, 400, cal);
    expect(z4 / z1).toBeGreaterThan(1.8);
    expect(z4 / z1).toBeLessThan(2.2);
  });
});

// ------------------------------------------------------------ case 7 -------
describe("case 7 — edit_locality (the ripple is exactly ngram_len wide)", () => {
  const c = VECTORS.cases["edit_locality"];
  it("changes exactly ngram_len g-value rows", () => {
    const gb = gValues(c.ids_before, cfg, table);
    const ga = gValues(c.ids_after, cfg, table);
    const differing: number[] = [];
    for (let i = 0; i < gb.rows; i++) {
      for (let d = 0; d < gb.m; d++) {
        if (gb.g[i * gb.m + d] !== ga.g[i * ga.m + d]) {
          differing.push(i);
          break;
        }
      }
    }
    expect(differing).toEqual(c.differing_g_rows);
    expect(differing.length).toBe(c.expected_row_count);
    expect(differing.length).toBe(cfg.ngramLen);
    expect(differing.map((i) => i + cfg.ngramLen - 1)).toEqual(c.differing_current_token_indices);
  });
  it("leaves rows beyond the ripple BIT-identical", () => {
    const gb = gValues(c.ids_before, cfg, table);
    const ga = gValues(c.ids_after, cfg, table);
    const last = Math.max(...(c.differing_g_rows as number[]));
    for (let i = last + 1; i < gb.rows; i++) {
      for (let d = 0; d < gb.m; d++) {
        expect(ga.g[i * ga.m + d]).toBe(gb.g[i * gb.m + d]);
      }
    }
  });
  it("does not shift downstream g-values on an insertion", () => {
    const ic = c.insertion_case;
    const gb = gValues(c.ids_before, cfg, table);
    const gi = gValues(ic.ids_after, cfg, table);
    for (let i = ic.first_stable_before_row; i < gb.rows; i++) {
      for (let d = 0; d < gb.m; d++) {
        expect(gi.g[(i + 1) * gi.m + d]).toBe(gb.g[i * gb.m + d]);
      }
    }
  });
  it("rippleSpan covers the changed current-token indices", () => {
    const span = rippleSpan(c.ids_before, c.ids_after, cfg);
    expect(span).toEqual(c.ripple_span);
    for (const idx of c.differing_current_token_indices as number[]) {
      expect(idx).toBeGreaterThanOrEqual(span.start);
      expect(idx).toBeLessThan(span.end);
    }
  });
});

// ------------------------------------------------------- blob + tokenizer --
describe("case 3 — tokenize: 4,031 golden vectors", () => {
  const goldenPath = `${REPO}data/golden/tokenizer_golden.json`;
  const tokJsonPath = `${REPO}data/build/tokenizer.json`;
  const golden = JSON.parse(readFileSync(goldenPath, "utf8")) as {
    text: string;
    ids: number[];
    tokens: string[];
  }[];

  let fromBlob: Tokenizer;
  let blobPieces: string[];
  let blobMerges: [string, string][];

  beforeAll(() => {
    const raw = brotliDecompressSync(readFileSync(`${REPO}data/assets/gemma3-tok.v1.bin.br`));
    const built = buildTokenizerFromBlob(new Uint8Array(raw));
    fromBlob = built.tokenizer;
    blobPieces = built.blob.pieces;
    blobMerges = built.blob.merges;
  });

  it("the golden file is the one vectors.json references", () => {
    const c3 = VECTORS.cases["tokenize"];
    expect(golden.length).toBe(c3.count);
    expect("sha256:" + createHash("sha256").update(readFileSync(goldenPath)).digest("hex")).toBe(
      c3.sha256,
    );
    for (const sp of c3.spot_check) {
      expect(golden[sp.index]!.text).toBe(sp.text);
      expect(golden[sp.index]!.ids).toEqual(sp.ids);
    }
  });

  it("packed blob reproduces every one of the 4,031 vectors", () => {
    let pass = 0;
    const misses: string[] = [];
    for (const g of golden) {
      const ids = fromBlob.encode(g.text).ids;
      if (ids.length === g.ids.length && ids.every((x, i) => x === g.ids[i])) pass++;
      else if (misses.length < 5) misses.push(JSON.stringify(g.text.slice(0, 60)));
    }
    expect(misses).toEqual([]);
    expect(pass).toBe(golden.length);
  });

  it.skipIf(!existsSync(tokJsonPath))(
    "the original tokenizer.json reproduces them too, and agrees with the blob",
    () => {
      const hf = JSON.parse(readFileSync(tokJsonPath, "utf8")) as Record<string, unknown>;
      // §14.2 item 8, asserted against the real upstream file.
      assertNoUnicodeFoldingNormalizer(hf["normalizer"], "normalizer");
      assertNoUnicodeFoldingNormalizer(hf["decoder"], "decoder");
      const cfgJson = JSON.parse(
        readFileSync(`${REPO}data/build/tokenizer_config.json`, "utf8"),
      ) as Record<string, unknown>;
      const orig = new Tokenizer(hf, cfgJson);
      let pass = 0;
      let agree = 0;
      for (const g of golden) {
        const a = orig.encode(g.text).ids;
        const b = fromBlob.encode(g.text).ids;
        if (a.length === g.ids.length && a.every((x, i) => x === g.ids[i])) pass++;
        if (a.length === b.length && a.every((x, i) => x === b[i])) agree++;
      }
      expect(pass).toBe(golden.length);
      expect(agree).toBe(golden.length);
    },
  );

  it("rejects an NFKC-folding tokenizer behaviourally", () => {
    // The bait must NOT collide with its NFKC folding.
    expect(fromBlob.encode("ﬁ", { add_special_tokens: false }).ids).not.toEqual(
      fromBlob.encode("fi", { add_special_tokens: false }).ids,
    );
    expect(fromBlob.encode("①", { add_special_tokens: false }).ids).not.toEqual(
      fromBlob.encode("1", { add_special_tokens: false }).ids,
    );
  });

  it("char offsets reassemble the text exactly", () => {
    const passage = readFileSync(`${REPO}data/dev/passage.txt`, "utf8").trim();
    const samples = [
      passage,
      "Hello World",
      "  leading and   trailing  ",
      "emoji 🧼🚀 and CJK 中文字符 and Arabic العربية",
      "zero​width​space",
      "soft­hyphen",
      "literal ▁ metaspace ▁▁▁",
      "  control bytes",
      "𝔘𝔫𝔦𝔠𝔬𝔡𝔢 math",
    ];
    for (const text of samples) {
      const enc = encodeForScoring(fromBlob, text);
      expect(enc.spans.length).toBe(enc.ids.length);
      // spans are contiguous, ascending, and cover the whole string
      let prev = 0;
      for (const sp of enc.spans) {
        expect(sp.s).toBe(prev);
        expect(sp.e).toBeGreaterThanOrEqual(sp.s);
        prev = sp.e;
      }
      expect(prev).toBe(text.length);
    }
  });

  it("char offsets reassemble ALL 4,031 golden texts", () => {
    // The offset map is what the mirror paints heat onto. If it drifts on any
    // input the player sees heat on the wrong words, which is exactly the
    // "plausible numbers that mean nothing" failure §4.3 is about — so this
    // runs over the whole adversarial corpus, not a handful of samples.
    let covered = 0;
    const failures: string[] = [];
    for (const g of golden) {
      try {
        const enc = encodeForScoring(fromBlob, g.text);
        const last = enc.spans.at(-1);
        const end = last === undefined ? 0 : last.e;
        if (end === g.text.length && enc.spans.length === enc.ids.length) covered++;
        else if (failures.length < 5) failures.push(JSON.stringify(g.text.slice(0, 40)));
      } catch (err) {
        if (failures.length < 5) {
          failures.push(`${JSON.stringify(g.text.slice(0, 40))}: ${(err as Error).message}`);
        }
      }
    }
    expect(failures).toEqual([]);
    expect(covered).toBe(golden.length);
  });

  it("charSpans throws rather than guessing when the tokens do not fit", () => {
    expect(() => charSpans("hello", ["hel"])).toThrow(/not covered by any token/);
    expect(() => charSpans("hello", ["hel", "XX"])).toThrow(/does not match the text/);
  });

  it("the IndexedDB merge round-trip is lossless", () => {
    const vocab = new Map<string, number>();
    blobPieces.forEach((p, i) => vocab.set(p, i));
    const pairs = mergesToPairs(blobMerges, vocab);
    expect(pairs).not.toBeNull();
    expect(pairs!.length).toBe(blobMerges.length * 2);
    // §5.3: 514,906 x 2 x 4 = 4,119,248 B
    expect(pairs!.byteLength).toBe(4_119_248);
    const back = pairsToMerges(pairs!, blobPieces);
    expect(back.length).toBe(blobMerges.length);
    for (let i = 0; i < back.length; i += 5000) expect(back[i]).toEqual(blobMerges[i]);
    expect(back[back.length - 1]).toEqual(blobMerges[blobMerges.length - 1]);
  });

  it("the blob reader refuses a spec carrying a folding normalizer", () => {
    const raw = brotliDecompressSync(readFileSync(`${REPO}data/assets/gemma3-tok.v1.bin.br`));
    const spec = buildTokenizerJSON(unpackBlob(new Uint8Array(raw)));
    expect(spec["normalizer"]).toEqual({ type: "Replace", pattern: { String: " " }, content: "▁" });
    expect(() =>
      assertNoUnicodeFoldingNormalizer({ type: "Sequence", normalizers: [{ type: "NFKC" }] }),
    ).toThrow(/NFKC/);
    expect(GEMMA3_TOKENIZER_CONFIG.add_bos_token).toBe(true);
  });
});

// ---------------------------------------------------------------------------
describe("calibration: the algebraic part of kappa, and the knob", () => {
  it("WEIGHTING_KAPPA is sqrt(sum(w^2)/m) for the shipped weights", () => {
    const w = tournamentWeights(30);
    let sq = 0;
    for (const v of w) sq += v * v;
    expect(WEIGHTING_KAPPA).toBeCloseTo(Math.sqrt(sq / 30), 15);
    expect(WEIGHTING_KAPPA).toBeCloseTo(1.1128924007211063, 12);
    expect(weightingKappa(30)).toBe(WEIGHTING_KAPPA);
  });

  it("the no-file fallback carries it instead of kappa = 1", () => {
    // kappa = 1 overstated every fallback z by 5-11%, i.e. ran a notch labelled
    // 1% FPR at a true 2-3%. The factor is algebra, not a measurement.
    expect(kappa(150, CLOSED_FORM_CALIBRATION)).toBeCloseTo(WEIGHTING_KAPPA, 15);
    expect(sigmaNull(150, CLOSED_FORM_CALIBRATION)).toBeCloseTo(
      (WEIGHTING_KAPPA * 1) / (2 * Math.sqrt(30 * 150)),
      18,
    );
  });

  it("every SHIPPED bucket sits within 10% of the analytic factor", () => {
    // If this drifts, either the weight vector changed without a recalibration
    // or the null corpus grew real correlation. Both are reportable.
    for (const b of VECTOR_CAL.buckets) {
      const closed = 1 / (2 * Math.sqrt(VECTOR_CAL.depth * b.n_scored));
      const ratio = b.sigma / closed / weightingKappa(VECTOR_CAL.depth);
      expect(ratio).toBeGreaterThan(0.9);
      expect(ratio).toBeLessThan(1.1);
    }
  });

  it("a group may repeat z_star but may not contradict it", () => {
    // §12 row 8 documents editing the TOP-LEVEL z_star. A group key of the same
    // name used to win silently, making the documented knob inert.
    const doc = {
      schema: "launder.thresholds/1",
      z_star: 2.3263,
      fpr: 0.01,
      depth: 30,
      calibrations: {
        default: { z_star: 2.3263, fpr: 0.01, buckets: [{ n_scored: 100, sigma: 0.01 }] },
      },
    };
    expect(parseCalibration(doc).z_star).toBe(2.3263);
    doc.calibrations.default.z_star = 1.5;
    expect(() => parseCalibration(doc)).toThrow(/contradicts the top-level z_star/);
  });

  it("editing the top-level z_star moves the notch", () => {
    const doc = {
      schema: "launder.thresholds/1",
      z_star: 2.0,
      fpr: 0.01,
      depth: 30,
      calibrations: { default: { buckets: [{ n_scored: 100, sigma: 0.01 }] } },
    };
    expect(parseCalibration(doc).z_star).toBe(2.0);
  });
});

// ---------------------------------------------------------------------------
describe("sampling table integrity (the browser's key material)", () => {
  it("verifies BOTH digests of the committed table", async () => {
    await expect(assertSamplingTableDigest(packedTable, table)).resolves.toBeUndefined();
    expect(createHash("sha256").update(packedTable).digest("hex")).toBe(
      SAMPLING_TABLE_SHA256_PACKED,
    );
    expect(createHash("sha256").update(table).digest("hex")).toBe(SAMPLING_TABLE_SHA256_UNPACKED);
  });

  it("the ones count CANNOT catch a reversed bit order, and the digest can", async () => {
    // gvalues.ts: "Getting the bit order backwards produces a table that is a
    // *permutation* of the right one, which is exactly as undetectable and
    // exactly as fatal as trap #3." The ones count was the only runtime check
    // this runtime had, and a permutation preserves it exactly.
    const lsbFirst = new Uint8Array(65536);
    let ones = 0;
    for (let i = 0; i < packedTable.length; i++) {
      const b = packedTable[i] as number;
      for (let bit = 0; bit < 8; bit++) {
        const v = (b >> bit) & 1; // LSB first — the WRONG order
        lsbFirst[i * 8 + bit] = v;
        ones += v;
      }
    }
    let differing = 0;
    for (let i = 0; i < lsbFirst.length; i++) if (lsbFirst[i] !== table[i]) differing++;
    expect(ones).toBe(SAMPLING_TABLE_ONES); // the tripwire that does NOT fire
    expect(differing).toBeGreaterThan(30000); // ... on a table this wrong
    await expect(assertSamplingTableDigest(packedTable, lsbFirst)).rejects.toThrow(/BIT ORDER/);
  });

  it("rejects tampered packed bytes", async () => {
    const tampered = new Uint8Array(packedTable);
    // Swap two bytes: same ones count, different table.
    const a = tampered[0] as number;
    tampered[0] = tampered[1] as number;
    tampered[1] = a;
    await expect(assertSamplingTableDigest(tampered)).rejects.toThrow(/packed sha256/);
  });
});

// ---------------------------------------------------------------------------
describe("contextHashes and gValues read the same id array", () => {
  it("does not truncate ids outside int32", () => {
    // The staging window was an Int32Array while gValues read ids untruncated.
    // Gemma-3 cannot emit these, but the two functions must not disagree about
    // the same input inside the one file whose job is bit-exactness.
    const ids = [9007199254740991, -9007199254740991, 4503599627370496, 262143, 7, 8, 9, 10];
    const got = Array.from(contextHashes(ids, cfg)).map(S);
    const expected = ids
      .slice(0, ids.length - cfg.ngramLen + 1)
      .map((_, i) => S(accumulateHash(1n, ids.slice(i, i + cfg.ngramLen - 1))));
    expect(got).toEqual(expected);
  });
});
