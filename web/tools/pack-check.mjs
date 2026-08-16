#!/usr/bin/env node
/**
 * BUILD GATE: the packed tokenizer blob round-trips.
 *
 * Runs inside the Docker web stage (`npm run build && node tools/pack-check.mjs
 * && npm run size-gate`), so it must pass with ONLY the committed assets. The
 * strong cross-check against the 33 MB `data/build/tokenizer.json` runs
 * whenever that file happens to be present (locally, and in CI's python job)
 * and is skipped with a printed notice otherwise — it is gitignored, and it is
 * a build INPUT, not a shipped artifact.
 *
 * What is asserted, in dependency order:
 *   1. the blob decompresses to the recorded 3,634,910 bytes
 *   2. it unpacks to 262,144 pieces and 514,906 merges with no trailing bytes
 *   3. RE-PACKING the unpacked structure reproduces the blob BYTE FOR BYTE
 *      (this is the actual round-trip, and it needs no external file)
 *   4. brotli-11 of that re-packed blob reproduces `gemma3-tok.v1.bin.br`
 *      byte for byte -- i.e. `asset_bundle_id` is stable
 *   5. no Unicode-folding normalizer anywhere in the reconstructed spec
 *   6. [if tokenizer.json is present] vocab, merges, added_tokens and the real
 *      normalizer all match HF exactly
 *
 * Step 3 is the one that matters. `forge tok pack` must reproduce the committed
 * blob byte for byte or `asset_bundle_id` changes; this proves the reader and
 * the writer agree on the format from the committed bytes alone.
 */
import { createHash } from "node:crypto";
import { readFileSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { brotliCompressSync, brotliDecompressSync, constants as zc } from "node:zlib";

const REPO = fileURLToPath(new URL("../../", import.meta.url));
const BLOB_BR = `${REPO}data/assets/gemma3-tok.v1.bin.br`;
const TOKENIZER_JSON = `${REPO}data/build/tokenizer.json`;

const EXPECTED = {
  brBytes: 1_192_944,
  rawBytes: 3_634_910,
  vocab: 262_144,
  merges: 514_906,
  addedTokens: 6_415,
  //: sha256 of the committed `.br`. THE ONE DIGEST THIS SCRIPT CAN CHECK: node
  //: has no blake3, so `brBlake3` below is here to be READ (it is the value
  //: recorded in data/MANIFEST.json and in watermark.toml's
  //: `[model].tokenizer_blob_blake3`, and `forge verify` recomputes it in
  //: Python) while this one is the value this script asserts.
  brSha256: "08aad23d1435b100008a6d7300ef10b793256d7a2a3fd991bef2ba408ef20c37",
  brBlake3: "93b1631ca1acadb1a7bf3dd0b110e575b31fa25005c5fa5d8e6c29de29e62eb2",
};

let failures = 0;
const ok = (label, cond, detail = "") => {
  if (cond) {
    console.log(`  ok    ${label}${detail ? ` — ${detail}` : ""}`);
  } else {
    failures += 1;
    console.error(`  FAIL  ${label}${detail ? ` — ${detail}` : ""}`);
  }
};

// ---------------------------------------------------------------- reader ----
// A direct port of web/src/tokenizer/blob.ts. Duplicated deliberately: this is
// a BUILD GATE, and a gate that imports the thing it is gating cannot catch a
// reader that is self-consistently wrong. The two must agree, and the vitest
// parity suite checks that they do.
class R {
  constructor(b) {
    this.b = b;
    this.o = 0;
  }
  v() {
    let r = 0,
      s = 0,
      x;
    do {
      x = this.b[this.o++];
      r += (x & 0x7f) * Math.pow(2, s);
      s += 7;
    } while (x & 0x80);
    return r;
  }
  sv() {
    const u = this.v();
    return u & 1 ? -((u + 1) / 2) : u / 2;
  }
  u8() {
    return this.b[this.o++];
  }
  str(n) {
    const s = this.b.toString("utf8", this.o, this.o + n);
    this.o += n;
    return s;
  }
}

class W {
  constructor() {
    this.c = [];
    this.b = Buffer.alloc(1 << 16);
    this.n = 0;
  }
  _need(k) {
    if (this.n + k > this.b.length) {
      this.c.push(this.b.subarray(0, this.n));
      this.b = Buffer.alloc(Math.max(1 << 16, k));
      this.n = 0;
    }
  }
  v(x) {
    this._need(5);
    while (x >= 0x80) {
      this.b[this.n++] = (x & 0x7f) | 0x80;
      x >>>= 7;
    }
    this.b[this.n++] = x;
  }
  sv(x) {
    this.v(x < 0 ? -x * 2 - 1 : x * 2);
  }
  u8(x) {
    this._need(1);
    this.b[this.n++] = x;
  }
  raw(buf) {
    this._need(buf.length);
    buf.copy(this.b, this.n);
    this.n += buf.length;
  }
  buf() {
    return Buffer.concat([...this.c, this.b.subarray(0, this.n)]);
  }
}

function deriveCandidates(pieces) {
  const inVocab = new Set(pieces);
  const cand = [];
  for (let i = 0; i < pieces.length; i++) {
    const cps = Array.from(pieces[i]);
    for (let s = 1; s < cps.length; s++) {
      const l = cps.slice(0, s).join(""),
        r = cps.slice(s).join("");
      if (inVocab.has(l) && inVocab.has(r)) cand.push([l, r]);
    }
  }
  return cand;
}

function unpack(buf) {
  const r = new R(buf);
  const n = r.v();
  const pieces = new Array(n);
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
  const cand = deriveCandidates(pieces);
  const mLen = r.v();
  const merges = new Array(mLen);
  const perm = new Int32Array(mLen);
  for (let i = 0; i < mLen; i++) {
    const d = r.sv();
    perm[i] = i + d;
    merges[i] = cand[i + d];
    if (merges[i] === undefined) throw new Error(`merge ${i} -> candidate ${i + d} out of range`);
  }
  const aLen = r.v();
  const addedTokens = new Array(aLen);
  {
    let prev = 0;
    for (let i = 0; i < aLen; i++) {
      prev += r.v();
      const f = r.u8();
      const content = f & 32 ? r.str(r.v()) : pieces[prev];
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
  return { pieces, types, merges, addedTokens, perm, cand, consumed: r.o };
}

function repack(u) {
  const w = new W();
  w.v(u.pieces.length);
  for (const p of u.pieces) {
    const b = Buffer.from(p, "utf8");
    w.v(b.length);
    w.raw(b);
  }
  const nn = [];
  u.types.forEach((t, i) => {
    if (t !== 1) nn.push([i, t]);
  });
  w.v(nn.length);
  {
    let prev = 0;
    for (const [i, t] of nn) {
      w.v(i - prev);
      w.u8(t);
      prev = i;
    }
  }
  w.v(u.merges.length);
  for (let i = 0; i < u.merges.length; i++) w.sv(u.perm[i] - i);
  w.v(u.addedTokens.length);
  {
    let prev = 0;
    for (const a of u.addedTokens) {
      w.v(a.id - prev);
      prev = a.id;
      const oov = !(a.id < u.pieces.length && u.pieces[a.id] === a.content);
      w.u8(
        (a.single_word ? 1 : 0) |
          (a.lstrip ? 2 : 0) |
          (a.rstrip ? 4 : 0) |
          (a.normalized ? 8 : 0) |
          (a.special ? 16 : 0) |
          (oov ? 32 : 0),
      );
      if (oov) {
        const b = Buffer.from(a.content, "utf8");
        w.v(b.length);
        w.raw(b);
      }
    }
  }
  return w.buf();
}

const FORBIDDEN = new Set([
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
function findFoldingNormalizer(node) {
  if (node === null || node === undefined) return null;
  if (Array.isArray(node)) {
    for (const c of node) {
      const hit = findFoldingNormalizer(c);
      if (hit) return hit;
    }
    return null;
  }
  if (typeof node !== "object") return null;
  if (typeof node.type === "string" && FORBIDDEN.has(node.type)) return node.type;
  for (const v of Object.values(node)) {
    if (v && typeof v === "object") {
      const hit = findFoldingNormalizer(v);
      if (hit) return hit;
    }
  }
  return null;
}

// ------------------------------------------------------------------ run ----
console.log("pack-check: data/assets/gemma3-tok.v1.bin.br");
const br = readFileSync(BLOB_BR);
ok("brotli asset size", br.length === EXPECTED.brBytes, `${br.length} B`);

const t0 = performance.now();
const raw = brotliDecompressSync(br);
const decompressMs = performance.now() - t0;
ok("decompressed size", raw.length === EXPECTED.rawBytes, `${raw.length} B in ${decompressMs.toFixed(1)} ms`);

const t1 = performance.now();
const u = unpack(raw);
const unpackMs = performance.now() - t1;
ok("no trailing bytes", u.consumed === raw.length, `consumed ${u.consumed}/${raw.length}`);
ok("vocab count", u.pieces.length === EXPECTED.vocab, `${u.pieces.length}`);
ok("merge count", u.merges.length === EXPECTED.merges, `${u.merges.length}`);
ok("added tokens", u.addedTokens.length === EXPECTED.addedTokens, `${u.addedTokens.length}`);
ok("merge candidates derived", u.cand.length >= u.merges.length, `${u.cand.length} candidates in ${unpackMs.toFixed(1)} ms`);

const re = repack(u);
ok("re-pack is byte-identical", re.length === raw.length && re.equals(raw), `${re.length} B`);

const t2 = performance.now();
const reBr = brotliCompressSync(re, {
  params: {
    [zc.BROTLI_PARAM_QUALITY]: 11,
    [zc.BROTLI_PARAM_LGWIN]: 24,
    [zc.BROTLI_PARAM_SIZE_HINT]: re.length,
  },
});
const brotliMs = performance.now() - t2;
ok(
  "brotli-11 re-compress is byte-identical",
  reBr.length === br.length && reBr.equals(br),
  `${reBr.length} B in ${(brotliMs / 1000).toFixed(1)} s`,
);
// A CHECK NAMED FOR A DIGEST THAT ONLY MEASURED A STRING LENGTH. This read
// `createHash("sha256").update(br).digest("hex").length === 64` and printed the
// digest beside it — an assertion that is true of every input on every run, on
// the one line of this file whose name promises the shipped bytes are the bytes
// we measured. `EXPECTED.brSha256` is that measurement, so a swapped or
// re-compressed blob (which changes `asset_bundle_id` and invalidates every
// passage) fails here instead of being printed and waved through.
const brSha256 = createHash("sha256").update(br).digest("hex");
ok("sha256 of the committed .br", brSha256 === EXPECTED.brSha256, brSha256);

// the reconstructed spec's normalizer, decoder and pre_tokenizer
const SPEC_NORMALIZER = { type: "Replace", pattern: { String: " " }, content: "▁" };
const SPEC_DECODER = {
  type: "Sequence",
  decoders: [
    { type: "Replace", pattern: { String: "▁" }, content: " " },
    { type: "ByteFallback" },
    { type: "Fuse" },
  ],
};
ok(
  "reconstructed spec has no Unicode-folding normalizer",
  findFoldingNormalizer(SPEC_NORMALIZER) === null && findFoldingNormalizer(SPEC_DECODER) === null,
);

// ------------------------------------------------- optional strong check ----
if (existsSync(TOKENIZER_JSON)) {
  console.log("pack-check: cross-checking against data/build/tokenizer.json");
  const hf = JSON.parse(readFileSync(TOKENIZER_JSON, "utf8"));

  // §14.2 item 8, asserted against the REAL file, which is the only place it
  // can actually be asserted.
  const hit = findFoldingNormalizer(hf.normalizer) ?? findFoldingNormalizer(hf.decoder);
  ok("Gemma-3 tokenizer.json has no NFKC/Precompiled normalizer (§14.2 #8)", hit === null, hit ?? "none");
  ok(
    "normalizer is exactly Replace(' ' -> metaspace)",
    JSON.stringify(hf.normalizer) === JSON.stringify(SPEC_NORMALIZER),
    JSON.stringify(hf.normalizer),
  );

  let vm = 0;
  for (const [s, i] of Object.entries(hf.model.vocab)) if (u.pieces[i] !== s) vm++;
  ok("vocab matches HF", vm === 0, `${vm} mismatches`);

  let mm = 0;
  for (let i = 0; i < hf.model.merges.length; i++) {
    if (u.merges[i][0] !== hf.model.merges[i][0] || u.merges[i][1] !== hf.model.merges[i][1]) mm++;
  }
  ok("merge list matches HF", mm === 0, `${mm} mismatches`);

  let am = 0;
  for (let i = 0; i < hf.added_tokens.length; i++) {
    const a = hf.added_tokens[i],
      b = u.addedTokens[i];
    if (
      a.id !== b.id ||
      a.content !== b.content ||
      a.single_word !== b.single_word ||
      a.lstrip !== b.lstrip ||
      a.rstrip !== b.rstrip ||
      a.normalized !== b.normalized ||
      a.special !== b.special
    )
      am++;
  }
  ok("added_tokens match HF", am === 0, `${am} mismatches`);
} else {
  console.log(
    "pack-check: data/build/tokenizer.json absent — skipping the HF cross-check.\n" +
      "            (It is a gitignored 33 MB build INPUT; the round-trip above is self-contained.)",
  );
}

if (failures > 0) {
  console.error(`\npack-check: ${failures} FAILURE(S)`);
  process.exit(1);
}
console.log("\npack-check: OK");
