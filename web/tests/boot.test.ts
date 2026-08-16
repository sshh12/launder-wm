/**
 * The inlined boot payload is the page's contract with the server (§5.4 step 1,
 * §9.1): the passage is in the HTML, the strings are in the HTML, and the game
 * is correct before any network call and before any JS.
 *
 * The checked-in payload is a fixture, and a fixture that drifts from
 * data/config/copy.toml is worse than no fixture: it is a second source of
 * player-facing truth, which §10.7 exists to forbid. So this file re-reads the
 * real TOML and compares.
 *
 * It also mirrors, client-side, the two `forge lint-copy` rules (§10.7):
 *   1. every check named by the level has label + blurb + reject;
 *   2. every {placeholder} in a label resolves against that check's params.
 */

import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

import type { Boot, CopyTree } from "../src/state";

const REPO = new URL("../../", import.meta.url);
const HTML = readFileSync(new URL("../index.html", import.meta.url), "utf8");
const COPY_TOML = readFileSync(new URL("data/config/copy.toml", REPO), "utf8");
const PASSAGE = readFileSync(new URL("data/dev/passage.txt", REPO), "utf8").trim();

/**
 * A TOML reader for exactly the subset copy.toml uses: table headers and
 * single-line basic strings. Deliberately not a general parser — it exists so
 * this test needs no dependency, and it throws on anything it does not
 * understand rather than skipping it.
 */
function parseCopyToml(src: string): CopyTree {
  const root: CopyTree = {};
  let table = root;
  for (const rawLine of src.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (line === "" || line.startsWith("#")) continue;
    const header = /^\[([A-Za-z0-9_.]+)\]$/.exec(line);
    if (header) {
      table = root;
      for (const part of (header[1] ?? "").split(".")) {
        const next = table[part];
        if (typeof next === "object" && next !== null) table = next;
        else {
          const fresh: CopyTree = {};
          table[part] = fresh;
          table = fresh;
        }
      }
      continue;
    }
    const kv = /^([A-Za-z0-9_]+)\s*=\s*"/.exec(line);
    if (!kv) throw new Error(`copy.toml line this test cannot read: ${line}`);
    // Scan to the closing quote honouring \" and \\, then allow a trailing
    // comment. A regex cannot do this without either eating an escaped quote or
    // stopping at a `#` that lives inside the string.
    let out = "";
    let i = kv[0].length;
    for (; i < line.length; i++) {
      const ch = line[i];
      if (ch === "\\") {
        const next = line[i + 1];
        out += next === "n" ? "\n" : (next ?? "");
        i += 1;
      } else if (ch === '"') break;
      else out += ch;
    }
    const rest = line.slice(i + 1).trim();
    if (rest !== "" && !rest.startsWith("#")) {
      throw new Error(`copy.toml line this test cannot read: ${line}`);
    }
    table[kv[1] ?? ""] = out;
  }
  return root;
}

function readInlinedBoot(): Boot {
  const m = /<script type="application\/json" id="launder-boot">([\s\S]*?)<\/script>/.exec(HTML);
  expect(m, "index.html must inline #launder-boot").not.toBeNull();
  return JSON.parse((m?.[1] ?? "").replace(/\\u003c/g, "<")) as Boot;
}

describe("index.html boot payload", () => {
  const boot = readInlinedBoot();
  const toml = parseCopyToml(COPY_TOML);
  delete toml.schema;

  it("inlines data/config/copy.toml verbatim — one source of truth", () => {
    expect(boot.copy).toEqual(toml);
  });

  it("inlines the passage into the textarea, with no leading newline", () => {
    // An HTML parser eats one newline directly after <textarea ...>. If the
    // server emitted one the player would start from a different string than
    // the one the server scored, silently.
    const m = /<textarea\b[^>]*>([\s\S]*?)<\/textarea/.exec(HTML);
    const value = (m?.[1] ?? "").replace(/&lt;/g, "<").replace(/&amp;/g, "&");
    expect(value.startsWith("\n")).toBe(false);
    expect(value).toBe(PASSAGE);
  });

  it("ships no reading and no expected_z it did not measure", () => {
    // A heat value is a detector output. The dev fixture has none, so the
    // mirror renders unstained rather than inventing a plausible-looking wash.
    expect(boot.dev).toBe(true);
    expect(boot.reading).toBeNull();
    expect(boot.primer_demo).toBeNull();
  });

  it("carries a threshold and a scale the needle can be drawn against", () => {
    expect(boot.detector.z_star).toBeGreaterThan(0);
    expect(boot.detector.scale.max).toBeGreaterThan(boot.detector.scale.min);
    expect(boot.detector.scale.min).toBeLessThanOrEqual(boot.detector.expected_z);
  });

  it("names a level whose checks are all in the copy file (lint rule 1)", () => {
    const checks = toml.check as CopyTree;
    expect(boot.level.checks.length).toBeGreaterThan(0);
    for (const spec of boot.level.checks) {
      const block = checks[spec.check] as CopyTree | undefined;
      expect(block, `copy.toml has no [check.${spec.check}] block`).toBeDefined();
      for (const key of ["label", "blurb", "reject"]) {
        expect(typeof block?.[key], `[check.${spec.check}].${key}`).toBe("string");
      }
    }
  });

  it("resolves every {placeholder} in a check label against that check's params (lint rule 2)", () => {
    const checks = toml.check as CopyTree;
    for (const spec of boot.level.checks) {
      const block = checks[spec.check] as CopyTree | undefined;
      const label = typeof block?.label === "string" ? block.label : "";
      for (const m of label.matchAll(/\{([a-z0-9_]+)\}/gi)) {
        const key = m[1] ?? "";
        expect(
          Object.prototype.hasOwnProperty.call(spec.params ?? {}, key),
          `[check.${spec.check}].label needs {${key}}, which the level does not supply`,
        ).toBe(true);
      }
    }
  });

  it("keeps the readout honest: never 'human', never a percentage", () => {
    const readout = toml.readout as CopyTree;
    expect(String(readout.below).toLowerCase()).not.toContain("human");
    expect(String(readout.above)).not.toMatch(/%/);
    expect(String(readout.below)).not.toMatch(/%/);
  });

  it("uses no emoji in the interface — only inside the copied share string", () => {
    // §10.6 slop rule. The share template is a portable artifact, not interface.
    const emoji = /\p{Extended_Pictographic}/u;
    const walk = (node: CopyTree, path: string): void => {
      for (const [k, v] of Object.entries(node)) {
        const here = path === "" ? k : `${path}.${k}`;
        if (typeof v === "string") {
          if (here === "readout.share_template") continue;
          expect(emoji.test(v), `${here} contains an emoji`).toBe(false);
        } else walk(v, here);
      }
    };
    walk(toml, "");
    expect(emoji.test(String((toml.readout as CopyTree).share_template))).toBe(true);
  });
});

describe("index.html structure", () => {
  it("has no transform on any ancestor of the passage box", () => {
    // A transformed ancestor changes the RENDERED font size, which is what iOS
    // checks before deciding to zoom on focus — and it never resets (§10.3).
    expect(HTML).not.toMatch(/style="[^"]*transform/);
  });

  it("never sets user-scalable=no", () => {
    expect(HTML).not.toContain("user-scalable");
    expect(HTML).toContain("viewport-fit=cover");
  });

  it("turns off every input transform that would rewrite the player's text", () => {
    for (const attr of [
      'autocorrect="off"',
      'autocapitalize="off"',
      'autocomplete="off"',
      'spellcheck="false"',
    ]) {
      expect(HTML).toContain(attr);
    }
  });

  it("exposes the needle as a meter with an initial value", () => {
    expect(HTML).toMatch(/role="meter"/);
    expect(HTML).toMatch(/--init-x:/);
    expect(HTML).toMatch(/--init-n:/);
  });
});
