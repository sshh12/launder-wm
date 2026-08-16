#!/usr/bin/env node
/**
 * `npm run slop-check` — TECH_PLAN.md §10.6's quality floor, as a build gate.
 *
 * The plan calls this "a grep script, wired into CI", and that is exactly what
 * it is: a list of the specific visual habits that make an interface look
 * generated rather than designed, each one checked by pattern against the
 * shipped CSS, HTML and TS. It is not a linter for taste. Every rule below is
 * quoted from §10.6, and a rule with a documented exception carries the
 * exception in code rather than in someone's memory.
 *
 * Exit 1 on any violation, with the file and line.
 */
import { readFileSync, readdirSync, statSync } from "node:fs";
import { extname, join, relative } from "node:path";
import { fileURLToPath } from "node:url";

const WEB = fileURLToPath(new URL("..", import.meta.url));
const ROOTS = ["src", "index.html"];
const EXTS = new Set([".css", ".html", ".ts", ".mjs", ".js"]);

function walk(p, out = []) {
  const s = statSync(p);
  if (s.isDirectory()) {
    for (const name of readdirSync(p).sort()) walk(join(p, name), out);
  } else if (EXTS.has(extname(p))) {
    out.push(p);
  }
  return out;
}

/**
 * Blank out every `/* ... *\/` and `<!-- ... -->` region, preserving newlines
 * so line numbers stay honest.
 *
 * Doing this by line-prefix instead (the obvious shortcut) reads the middle of
 * a multi-line comment as code, which is how the first version of this script
 * reported `border-radius: because`, `border-radius: a`, `border-radius: span`
 * from one sentence of prose. A gate that cries wolf gets switched off.
 */
function stripComments(src) {
  return src
    .replace(/\/\*[\s\S]*?\*\//g, (m) => m.replace(/[^\n]/g, " "))
    .replace(/<!--[\s\S]*?-->/g, (m) => m.replace(/[^\n]/g, " "))
    .replace(/^([^\n]*?)\/\/[^\n]*$/gm, (m, keep) => keep + " ".repeat(m.length - keep.length));
}

/** `:root` custom properties, so `border-radius: var(--r)` can be resolved. */
function customProperties(fileList) {
  const map = new Map();
  for (const abs of fileList) {
    if (extname(abs) !== ".css") continue;
    for (const m of stripComments(readFileSync(abs, "utf8")).matchAll(
      /(--[\w-]+)\s*:\s*([^;{}]+)/g,
    )) {
      if (!map.has(m[1])) map.set(m[1], m[2].trim());
    }
  }
  return map;
}

const files = ROOTS.flatMap((r) => walk(join(WEB, r)));
const VARS = customProperties(files);

/** Resolve `var(--x, fallback)` one level deep. */
function resolveVars(value) {
  return value.replace(/var\(\s*(--[\w-]+)\s*(?:,\s*([^)]*))?\)/g, (_, name, fallback) =>
    (VARS.get(name) ?? fallback ?? name).trim(),
  );
}
const violations = [];
const counts = new Map();

function bump(rule, file, line, text) {
  const list = counts.get(rule) ?? [];
  list.push({ file, line, text: text.trim().slice(0, 100) });
  counts.set(rule, list);
}

function fail(rule, file, line, text, why) {
  violations.push({ rule, file, line, text: text.trim().slice(0, 100), why });
}

// U+1F9FC SOAP is allowed: it lives inside the copied SHARE STRING, which is a
// portable artifact posted elsewhere, not interface. Everything else is out.
const SHARE_EMOJI = "\u{1F9FC}";
const EMOJI =
  /[\u{1F300}-\u{1FAFF}\u{1F000}-\u{1F0FF}\u{2600}-\u{27BF}\u{FE0F}\u{1F1E6}-\u{1F1FF}]/gu;

for (const abs of files) {
  const file = relative(WEB, abs).replaceAll("\\", "/");
  const raw_src = readFileSync(abs, "utf8");
  const code = stripComments(raw_src);
  const raw_lines = raw_src.split("\n");
  const lines = code.split("\n");

  lines.forEach((line, i) => {
    const n = i + 1;
    const raw = raw_lines[i] ?? line;
    // Comments are already blanked; anything left is code.
    const isComment = line.trim() === "";

    // --- colour ----------------------------------------------------------
    if (/\b(indigo|violet|rebeccapurple|blueviolet)\b/i.test(line) && !isComment) {
      fail("indigo/violet accent", file, n, raw, "§10.6 bans the AI-slop accent hues");
    }
    const okl = /oklch\(\s*[\d.]+%?\s+([\d.]+)\s+([\d.]+)/i.exec(line);
    if (okl && Number(okl[1]) > 0.08 && Number(okl[2]) >= 255 && Number(okl[2]) <= 320) {
      fail("oklch hue 255-320 above chroma 0.08", file, n, raw, "§10.6");
    }
    if (/background(-color)?\s*:\s*#fff(fff)?\b/i.test(line)) {
      fail("#FFFFFF background", file, n, raw, "§10.6: the palette is Dead Air, not paper white");
    }
    if (/(^|[^-\w])color\s*:\s*#000(000)?\b/i.test(line)) {
      fail("#000000 text", file, n, raw, "§10.6");
    }

    // --- effects ---------------------------------------------------------
    if (/backdrop-filter\s*:/i.test(line) && !isComment) {
      fail("backdrop-filter", file, n, raw, "§10.6: zero, anywhere");
    }
    if (/background-clip\s*:\s*text/i.test(line) && !isComment) {
      fail("background-clip: text", file, n, raw, "§10.6");
    }
    if (/transition\s*:\s*all\b/i.test(line) && !isComment) {
      fail("transition: all", file, n, raw, "§10.6");
    }
    if (/animation-iteration-count\s*:\s*infinite/i.test(line) && !isComment) {
      fail("infinite animation", file, n, raw, "§10.6: no shimmer, no pulse");
    }
    if (/:hover[^{]*\{[^}]*translateY\(\s*-/i.test(line) && !isComment) {
      fail("translateY(- hover lift", file, n, raw, "§10.6");
    }
    if (/!important/.test(line) && !isComment) {
      fail("!important", file, n, raw, "§10.6: cancelling rules are the classic failure");
    }

    // --- counted rules ---------------------------------------------------
    if (!isComment) {
      if (/-gradient\(/i.test(line)) bump("gradient", file, n, raw);
      // §10.4 REQUIRES `box-shadow: inset` on hot tokens as the redundant,
      // colour-blind-safe encoding of heat; a border or outline there would
      // change line breaking and desynchronise the mirror, which the same
      // paragraph forbids. So the "at most one box-shadow" rule counts DROP
      // shadows only.
      const shadow = /box-shadow\s*:\s*([^;}]+)/i.exec(line);
      if (shadow) {
        const v = resolveVars(shadow[1]).trim().toLowerCase();
        // `none` is a RESET, not a shadow, and `inset` is §10.4's required
        // colour-blind-safe heat encoding on `.tk`.
        if (v !== "none" && !v.startsWith("inset")) bump("drop box-shadow", file, n, raw);
      }
      for (const m of line.matchAll(/border-radius\s*:\s*([^;}]+)/gi)) {
        for (const v of resolveVars(m[1]).split(/[\s/]+/)) {
          const t = v.trim().toLowerCase();
          if (!/^-?[\d.]+(px|rem|em|%|)$/.test(t)) continue; // not a length
          if (t === "0" || t === "0px" || t === "0%") continue;
          bump(`border-radius:${t}`, file, n, raw);
        }
      }
    }

    // --- emoji -----------------------------------------------------------
    const found = [...raw.matchAll(EMOJI)].map((m) => m[0]).filter((c) => c !== SHARE_EMOJI);
    if (found.length && !isComment) {
      fail(`emoji ${found.join("")}`, file, n, raw, "§10.6: none in the UI, only in the share string");
    }
  });
}

// --- the counted limits ----------------------------------------------------
const LIMITS = { gradient: 2, "drop box-shadow": 1 };
for (const [rule, max] of Object.entries(LIMITS)) {
  const hits = counts.get(rule) ?? [];
  if (hits.length > max) {
    for (const h of hits) {
      fail(`${rule} (${hits.length} > ${max} allowed)`, h.file, h.line, h.text, "§10.6");
    }
  }
}

// At most TWO border-radius values in the whole app: 2px and 999px.
const radii = [...counts.keys()].filter((k) => k.startsWith("border-radius:"));
const allowed = new Set(["border-radius:2px", "border-radius:999px"]);
for (const r of radii) {
  if (!allowed.has(r)) {
    const h = counts.get(r)[0];
    fail(`${r}`, h.file, h.line, h.text, "§10.6 allows 2px and 999px only");
  }
}

for (const v of violations) {
  console.error(`slop-check: ${v.file}:${v.line}  ${v.rule}\n    ${v.text}\n    ${v.why}`);
}
if (violations.length) {
  console.error(`\nslop-check: ${violations.length} violation(s)`);
  process.exit(1);
}
console.log(
  `slop-check: OK — ${files.length} files, ` +
    `${(counts.get("gradient") ?? []).length} gradient(s), ` +
    `${(counts.get("drop box-shadow") ?? []).length} drop shadow(s), ` +
    `radii ${radii.map((r) => r.split(":")[1]).join(" ") || "none"}`,
);
