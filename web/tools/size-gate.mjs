#!/usr/bin/env node
/**
 * BUILD GATE: the asset size budget (§5.2).
 *
 *   app JS bundle          <=    60,000 B brotli
 *   app CSS bundle         <=     8,000 B brotli
 *   tokenizer blob         <= 1,300,000 B
 *   total dist/**\/*.br     <= 1,400,000 B
 *
 * "CI gates (build fails, not a warning)." A budget that warns is a budget that
 * is already blown.
 *
 * The tokenizer blob and the sampling table live in `data/assets/` and are
 * served by the Python app, not emitted by vite, so they are measured from
 * there. Everything else is measured from `dist/`.
 */
import { readFileSync, readdirSync, statSync, existsSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { brotliCompressSync, constants as zc } from "node:zlib";

const WEB = fileURLToPath(new URL("../", import.meta.url));
const REPO = fileURLToPath(new URL("../../", import.meta.url));
const DIST = join(WEB, "dist");

const BUDGET = {
  js: 60_000,
  css: 8_000,
  tokenizerBlob: 1_300_000,
  totalBr: 1_400_000,
};

let failures = 0;
const gate = (label, actual, limit) => {
  const pass = actual <= limit;
  const pct = ((actual / limit) * 100).toFixed(1);
  console.log(
    `  ${pass ? "ok  " : "FAIL"}  ${label.padEnd(34)} ${String(actual).padStart(9)} / ${String(limit).padStart(9)} B  (${pct}%)`,
  );
  if (!pass) failures += 1;
};

function walk(dir, out = []) {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) walk(p, out);
    else out.push(p);
  }
  return out;
}

const brotli = (buf) =>
  brotliCompressSync(buf, {
    params: {
      [zc.BROTLI_PARAM_QUALITY]: 11,
      [zc.BROTLI_PARAM_LGWIN]: 24,
      [zc.BROTLI_PARAM_SIZE_HINT]: buf.length,
    },
  }).length;

console.log("size-gate: §5.2 asset budget");

const blob = join(REPO, "data/assets/gemma3-tok.v1.bin.br");
if (existsSync(blob)) gate("tokenizer blob (br, on the wire)", statSync(blob).size, BUDGET.tokenizerBlob);

if (!existsSync(DIST)) {
  console.error("\nsize-gate: dist/ does not exist. Run `npm run build` first.");
  process.exit(1);
}

const files = walk(DIST);
const all = new Set(files);
let jsBr = 0;
let cssBr = 0;
let totalBr = 0;

// Measure each ORIGINAL asset exactly once, at its brotli size. Where vite
// already emitted a `.br` sibling we take its size; where it did not (threshold,
// or a plugin misconfiguration) we compress here, so a missing sibling shows up
// as a size, never as a free pass.
for (const f of files) {
  if (f.endsWith(".gz") || f.endsWith(".br")) continue;
  const sibling = `${f}.br`;
  const size = all.has(sibling) ? statSync(sibling).size : brotli(readFileSync(f));
  totalBr += size;
  if (f.endsWith(".js")) jsBr += size;
  else if (f.endsWith(".css")) cssBr += size;
}

gate("app JS (br)", jsBr, BUDGET.js);
gate("app CSS (br)", cssBr, BUDGET.css);
gate("total dist/**/*.br", totalBr, BUDGET.totalBr);

if (failures > 0) {
  console.error(`\nsize-gate: ${failures} budget(s) blown`);
  process.exit(1);
}
console.log("\nsize-gate: OK");
