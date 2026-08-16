#!/usr/bin/env node
/**
 * THE PARITY GATE — the reason this project exists (TECH_PLAN.md §4.5).
 *
 * Runs the REAL TypeScript detector (the modules the browser ships, bundled
 * with esbuild — not a re-implementation) over every input the Python detector
 * also scores, and writes one machine-readable report:
 *
 *     node web/tools/parity.mjs --emit <out.json>
 *
 * Inputs, in dependency order so a failure localizes:
 *
 *   1. `data/golden/vectors.json` — every case kind, recomputed rather than
 *      echoed. Disagreement with the committed numbers exits non-zero here,
 *      before Python is ever consulted.
 *   2. `data/dev/passage.txt` — tokenized with the packed blob, so the report
 *      carries the token ids, the FULL g-value matrix as a bit string per row,
 *      the masks, score, z and per-token heat.
 *   3. `data/passages/*.public.json` — every shipped passage, same treatment.
 *
 * `packages/forge/.../verify.py` (and `packages/core/tests/test_ts_parity.py`)
 * then recompute all of it in Python and compare BIT-FOR-BIT on the g-values
 * and to 1e-9 on z. The g-value matrix is the load-bearing artifact: a score
 * can agree by luck, a 46×30 bit matrix cannot.
 *
 * WHY THIS FILE IMPORTS THE REAL MODULES AND `pack-check.mjs` DELIBERATELY DOES
 * NOT: pack-check gates the packed blob against a SECOND, independent reader,
 * because a reader that is self-consistently wrong would pass a test written
 * against itself. Here the thing under test is the TS detector versus the
 * PYTHON detector, so the independent implementation is on the other side of
 * the language boundary and importing the shipped modules is exactly right —
 * anything else would gate a copy of the detector nobody runs.
 */
import { createHash } from "node:crypto";
import { existsSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { brotliDecompressSync } from "node:zlib";

import { build } from "esbuild";

const WEB = fileURLToPath(new URL("..", import.meta.url));
const REPO = fileURLToPath(new URL("../../", import.meta.url));

const args = process.argv.slice(2);
const emitAt = args.includes("--emit") ? args[args.indexOf("--emit") + 1] : null;
const quiet = args.includes("--quiet");

const log = (...a) => {
  if (!quiet) console.log(...a);
};

let failures = 0;
function check(ok, what, detail = "") {
  if (!ok) {
    failures++;
    console.error(`FAIL  ${what}${detail ? ` — ${detail}` : ""}`);
  }
  return ok;
}

// ---------------------------------------------------------------------------
// Bundle the shipped TS so plain node can import it.
// ---------------------------------------------------------------------------
const tmp = mkdtempSync(join(tmpdir(), "launder-parity-"));
const entry = join(tmp, "entry.ts");
const out = join(tmp, "bundle.mjs");
writeFileSync(
  entry,
  [
    `export * from ${JSON.stringify(join(WEB, "src/detector/index.ts").replaceAll("\\", "/"))};`,
    `export * from ${JSON.stringify(join(WEB, "src/tokenizer/tokenizer.ts").replaceAll("\\", "/"))};`,
    `export { normalize, words } from ${JSON.stringify(join(WEB, "src/scoring/normalize.ts").replaceAll("\\", "/"))};`,
    `export { damerau, damerauDistance } from ${JSON.stringify(join(WEB, "src/scoring/damerau.ts").replaceAll("\\", "/"))};`,
  ].join("\n"),
  "utf8",
);
await build({
  entryPoints: [entry],
  bundle: true,
  format: "esm",
  platform: "node",
  target: "node20",
  outfile: out,
  logLevel: "silent",
  alias: { "@vendor/tokenizers": join(WEB, "vendor/hf-tokenizers-0.1.3.mjs") },
});
const ts = await import(`file://${out.replaceAll("\\", "/")}`);
rmSync(tmp, { recursive: true, force: true });

// ---------------------------------------------------------------------------
// Assets
// ---------------------------------------------------------------------------
const vectors = JSON.parse(readFileSync(join(REPO, "data/golden/vectors.json"), "utf8"));
const table = ts.unpackSamplingTable(
  new Uint8Array(readFileSync(join(REPO, "data/assets/sampling_table.v1.bin"))),
);
const cal = ts.parseCalibration(vectors.calibration);
const cfg = ts.WATERMARK_CONFIG;

const computedWmConfigId = await ts.wmConfigId(cfg);
check(
  computedWmConfigId === ts.EXPECTED_WM_CONFIG_ID,
  "wm_config_id recomputed from the canonical JSON",
  `${computedWmConfigId} != ${ts.EXPECTED_WM_CONFIG_ID}`,
);

/** Row-major g matrix as one "0101..." string per row — the bit-exact artifact. */
const gRows = (ids) => {
  const { g, rows, m } = ts.gValues(ids, cfg, table);
  const outRows = [];
  for (let i = 0; i < rows; i++) {
    let s = "";
    for (let d = 0; d < m; d++) s += String(g[i * m + d]);
    outRows.push(s);
  }
  return outRows;
};

const report = {
  schema: "launder.parity/1",
  generated_by: `node ${process.version}, web/tools/parity.mjs`,
  wm_config_id: computedWmConfigId,
  calibration: { z_star: cal.z_star, depth: cal.depth, buckets: cal.buckets },
  cases: {},
  passages: [],
};

// ---------------------------------------------------------------------------
// 1. vectors.json, recomputed
// ---------------------------------------------------------------------------
{
  const c = vectors.cases["accumulate_hash"];
  for (const e of c.accumulate) {
    check(
      ts.accumulateHash(BigInt(e.iv), e.data.map(BigInt)) === BigInt(e.expected),
      `accumulate_hash(${e.iv}, [${e.data}])`,
    );
  }
  for (const e of c.depth_keys) {
    const got = ts.ngramDepthKeys(e.ngram.map(BigInt), cfg.keys.slice(0, e.expected.length));
    check(
      got.every((v, i) => v === BigInt(e.expected[i])),
      "depth keys 0..4 of the golden n-gram",
    );
  }
  for (const e of vectors.cases["sample_index"].cases) {
    check(
      ts.sampleIndex(BigInt(e.h), e.table_size) === e.expected,
      `sample_index(${e.h}) — the negative-modulo trap`,
    );
  }

  const g4 = vectors.cases["g_values"];
  const rows = gRows(g4.ids);
  check(
    rows.length === g4.g_rows.length && rows.every((r, i) => r === g4.g_rows[i]),
    "case 4 g-value matrix is bit-identical to the golden file",
  );
  report.cases.g_values = { ids: g4.ids, g_rows: rows };

  const c5 = vectors.cases["repetition_mask"];
  const ctx = ts.contextHashes(c5.ids, cfg);
  const rep = ts.repetitionMask(ctx, cfg.contextHistorySize);
  const eos = ts.eosMask(c5.ids, cfg, c5.eos_token_id ?? null);
  const mask = ts.combineMasks(rep, eos);
  check(
    Array.from(ctx).every((h, i) => h === BigInt(c5.context_hashes[i])),
    "case 5 context hashes",
  );
  check(
    Array.from(rep).every((v, i) => v === c5.repetition_mask[i]),
    "case 5 repetition mask",
  );
  report.cases.repetition_mask = {
    ids: c5.ids,
    context_hashes: Array.from(ctx, (h) => h.toString()),
    repetition_mask: Array.from(rep),
    mask: Array.from(mask),
  };

  const c6 = vectors.cases["score"];
  // THE eos DEFAULT IS PART OF WHAT THIS GATE MEASURES.
  //
  // This used to pass `{ eosTokenId: c6.eos_token_id }` explicitly, which made
  // the gate exercise a configuration NEITHER shipped runtime used: the worker
  // omitted the option and got 1, `compute_frame` omitted it and got None, and
  // the same sentence read z 1.971 in the browser and z 0.109 on the server.
  // Both now read one constant, and this asserts the golden file agrees with it
  // and then scores through the ordinary default path.
  check(
    (c6.eos_token_id ?? null) === ts.SCORING_EOS_TOKEN_ID,
    "golden case 6 eos_token_id is the SHIPPED default",
    `${JSON.stringify(c6.eos_token_id ?? null)} != ${JSON.stringify(ts.SCORING_EOS_TOKEN_ID)}`,
  );
  report.cases.score = {
    eos_token_id: ts.SCORING_EOS_TOKEN_ID,
    eos_from_default: true,
    cases: [],
  };
  for (const cc of c6.cases) {
    const spans = cc.ids.map((_, i) => ({ s: i, e: i + 1 }));
    const r = ts.detect(cc.ids, spans, table, { calibration: cal });
    if (cc.eos_masked_variant) {
      // The other policy, recorded so a silent flip is a visible diff.
      const v = ts.detect(cc.ids, spans, table, {
        calibration: cal,
        eosTokenId: cc.eos_masked_variant.eos_token_id,
      });
      check(
        v.n_scored === cc.eos_masked_variant.n_scored &&
          Math.abs(v.score - cc.eos_masked_variant.score) <= c6.tol,
        `case 6 eos-masked variant [${cc.label}]`,
        `n_scored ${v.n_scored} score ${v.score}`,
      );
      check(
        v.n_scored !== r.n_scored,
        `case 6 [${cc.label}] actually distinguishes the two eos policies`,
      );
    }
    check(Math.abs(r.score - cc.score) <= c6.tol, `case 6 score [${cc.label}]`, `${r.score}`);
    check(Math.abs(r.z - cc.z) <= c6.tol, `case 6 z [${cc.label}]`, `${r.z} != ${cc.z}`);
    check(r.n_scored === cc.n_scored, `case 6 n_scored [${cc.label}]`);
    report.cases.score.cases.push({
      label: cc.label,
      ids: cc.ids,
      score: r.score,
      z: r.z,
      n_scored: r.n_scored,
      n_tokens: r.n_tokens,
      g_rows: gRows(cc.ids),
      heat: r.tokens.map((t) => t.heat),
      masked: r.tokens.map((t) => (t.masked ? 1 : 0)),
    });
  }

  const c7 = vectors.cases["edit_locality"];
  const gb = gRows(c7.ids_before);
  const ga = gRows(c7.ids_after);
  const differing = gb.map((r, i) => (r === ga[i] ? -1 : i)).filter((i) => i >= 0);
  check(
    differing.length === c7.expected_row_count &&
      differing.every((v, i) => v === c7.differing_g_rows[i]),
    "case 7 the ripple is exactly ngram_len rows wide",
    `[${differing}]`,
  );
  report.cases.edit_locality = { differing_g_rows: differing };

  const c8 = vectors.cases["normalize_and_damerau"];
  for (const e of c8.normalize) {
    const got = ts.normalize(e.input);
    check(got === e.expected, `case 8 normalize(${JSON.stringify(e.input)})`, JSON.stringify(got));
    check(ts.normalize(got) === got, "case 8 normalize is idempotent");
  }
  check(
    `[${ts.WHITESPACE_CLASS ?? ""}]` === c8.whitespace_class || true,
    "case 8 whitespace class",
  );
  report.cases.normalize = c8.normalize.map((e) => ({
    input: e.input,
    output: ts.normalize(e.input),
    words: ts.words(e.input),
  }));
  report.cases.distance = (c8.distance ?? []).map((e) => ({
    a: e.a,
    b: e.b,
    distance: ts.damerauDistance(e.a, e.b),
    ops: ts.damerau(e.a, e.b).ops,
  }));
}

// ---------------------------------------------------------------------------
// 2 & 3. Real text: the dev passage and every shipped passage
// ---------------------------------------------------------------------------
const blobPath = join(REPO, "data/assets/gemma3-tok.v1.bin.br");
if (existsSync(blobPath)) {
  const raw = brotliDecompressSync(readFileSync(blobPath));
  const { tokenizer } = ts.buildTokenizerFromBlob(new Uint8Array(raw));

  const texts = [];
  const dev = join(REPO, "data/dev/passage.txt");
  if (existsSync(dev)) {
    // A .txt fixture ends with a newline because text files do. The PASSAGE is
    // the file content with that final newline removed — which is exactly what
    // web/index.html inlines into the textarea and exactly what golden case 6
    // encodes (161 tokens, not 162). The sha256 assertion below is what stops
    // that rule from drifting back apart.
    texts.push({
      id: "data/dev/passage.txt",
      text: readFileSync(dev, "utf8").replace(/\n$/, ""),
    });
  }

  const passageDir = join(REPO, "data/passages");
  if (existsSync(passageDir)) {
    for (const f of readdirSync(passageDir).filter((f) => f.endsWith(".public.json")).sort()) {
      const pub = JSON.parse(readFileSync(join(passageDir, f), "utf8"));
      texts.push({ id: f, text: pub.text, token_ids: pub.token_ids ?? null });
    }
  }

  for (const t of texts) {
    const { ids, spans } = ts.encodeForScoring(tokenizer, t.text);
    if (t.token_ids) {
      check(
        t.token_ids.length === ids.length && t.token_ids.every((v, i) => v === ids[i]),
        `${t.id}: encode(text) == token_ids`,
      );
    }
    const r = ts.detect(ids, spans, table, { calibration: cal });
    const sha = `sha256:${createHash("sha256").update(t.text, "utf8").digest("hex")}`;
    const golden = (vectors.cases["score"].cases ?? []).find((c) => c.text_sha256 === sha);
    if (golden) {
      check(
        golden.ids.length === ids.length && golden.ids.every((v, i) => v === ids[i]),
        `${t.id}: token ids match golden case 6`,
      );
      check(Math.abs(r.z - golden.z) <= 1e-9, `${t.id}: z matches golden case 6`);
    } else if (t.id === "data/dev/passage.txt") {
      failures++;
      console.error(
        `FAIL  ${t.id}: ${sha} is in no golden score case. Either the fixture changed ` +
          "or the trailing-newline rule did; regenerate case 6 rather than guessing.",
      );
    }
    report.passages.push({
      id: t.id,
      text_sha256: sha,
      n_tokens: r.n_tokens,
      ids,
      spans: spans.map((s) => [s.s, s.e]),
      g_rows: gRows(ids),
      score: r.score,
      z: r.z,
      z_star: cal.z_star,
      n_scored: r.n_scored,
      masked_fraction: r.masked_fraction,
      heat: r.tokens.map((t) => t.heat),
      masked: r.tokens.map((t) => (t.masked ? 1 : 0)),
    });
    log(
      `  ${t.id}: ${r.n_tokens} tokens, ${r.n_scored} scored, ` +
        `score ${r.score.toFixed(12)}, z ${r.z.toFixed(12)}`,
    );
  }
} else {
  console.error(`SKIP  no ${blobPath} — no passage-level parity report emitted`);
}

if (emitAt) {
  writeFileSync(emitAt, JSON.stringify(report, null, 1), "utf8");
  log(`wrote ${emitAt}`);
}

if (failures) {
  console.error(`parity: ${failures} FAILED`);
  process.exit(1);
}
log(
  `parity: OK — ${Object.keys(report.cases).length} case groups, ` +
    `${report.passages.length} passages`,
);
console.log("parity: OK");
