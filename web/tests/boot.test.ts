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

import { points } from "../src/game/needle";
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

  it("speaks launder.boot/2: a campaign position, never a day or a puzzle number", () => {
    // The game is a linear 15-level campaign, not a daily. A payload still
    // carrying `day` or `puzzle_number` is a server that never got the memo,
    // and the rail would render "#212" for a level number nobody assigned.
    expect(boot.schema).toBe("launder.boot/2");
    expect(Number.isInteger(boot.level_n)).toBe(true);
    expect(boot.level_n).toBeGreaterThanOrEqual(1);
    expect(boot.level_n).toBeLessThanOrEqual(boot.level_count);
    // Not "> 1": this fixture is dev:true, and dev:true means data/passages/
    // was empty and data/dev/passage.txt is standing in as the entire
    // campaign — level 1 of 1. The pair is asserted below.
    expect(boot.level_count).toBeGreaterThanOrEqual(1);
    const loose = boot as unknown as Record<string, unknown>;
    expect(loose.day).toBeUndefined();
    expect(loose.puzzle_number).toBeUndefined();
  });

  it("ships no reading and no expected_z it did not measure", () => {
    // A heat value is a detector output. The dev fixture has none, so the
    // mirror renders unstained rather than inventing a plausible-looking wash.
    expect(boot.dev).toBe(true);
    expect(boot.reading).toBeNull();
    expect(boot.primer_demo).toBeNull();
  });

  it("is a state the server could actually emit", () => {
    // A fixture is only worth checking in if it is REPRODUCIBLE. `dev` is set
    // exactly when data/passages/ is empty and data/dev/passage.txt stands in
    // as the whole campaign, which the loader builds as level 1 of 1 — so
    // dev:true beside level_count:15 described a page no code path can
    // produce, and anyone debugging against it was debugging a fiction.
    expect(boot.dev).toBe(true);
    expect(boot.passage_id).toBe("p_dev");
    expect(boot.level_count).toBe(1);
    expect(boot.level_n).toBe(1);
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
    // The displayed number is now 0-100, which is one `%` away from looking
    // like a "% AI" figure — the exact claim the detector cannot support and
    // the one thing [readout] has always forbidden. No template that renders
    // the reading may carry a percent sign.
    const readout = toml.readout as CopyTree;
    expect(String(readout.below).toLowerCase()).not.toContain("human");
    for (const key of [
      "above",
      "below",
      "threshold_label",
      "meter_name",
      "valuetext_above",
      "valuetext_below",
    ]) {
      expect(String(readout[key]), `readout.${key}`).not.toMatch(/%/);
    }
  });

  it("shares a level and a campaign, and never a day", () => {
    const readout = toml.readout as CopyTree;
    expect(String(readout.share_template)).toContain("{level_n}");
    expect(String(readout.share_template)).toContain("{distance}");
    const all = String(readout.share_all_template);
    for (const ph of ["{level_count}", "{total}", "{per_level}", "{url}"]) {
      expect(all, `share_all_template needs ${ph}`).toContain(ph);
    }
    // `{url}` is location.origin at runtime; a domain baked into the copy would
    // make every preview deployment advertise production.
    expect(all).not.toMatch(/https?:\/\//);
  });

  it("uses no emoji in the interface — only inside the copied share strings", () => {
    // §10.6 slop rule. A share template is a portable artifact, not interface.
    const emoji = /\p{Extended_Pictographic}/u;
    const shares = ["readout.share_template", "readout.share_all_template"];
    const walk = (node: CopyTree, path: string): void => {
      for (const [k, v] of Object.entries(node)) {
        const here = path === "" ? k : `${path}.${k}`;
        if (typeof v === "string") {
          if (shares.includes(here)) continue;
          expect(emoji.test(v), `${here} contains an emoji`).toBe(false);
        } else walk(v, here);
      }
    };
    walk(toml, "");
    for (const key of shares) {
      const value = String((toml.readout as CopyTree)[key.split(".")[1] ?? ""]);
      expect(emoji.test(value), `${key} should carry the soap`).toBe(true);
    }
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

  it("paints the fixture's needle where the fixture's number says it is", () => {
    // --init-x is a PERCENTAGE of the face (rail.css, and what boot.py writes),
    // so a checked-in `0px` is not a unit slip — it is a needle pinned to the
    // left edge of an instrument whose own readout says 17. Nobody looking at
    // the un-served page can tell whether the geometry is broken or the
    // fixture is. Both numbers come from expected_z; they are derived here so
    // they cannot drift apart again.
    const boot = readInlinedBoot();
    const { min, max } = boot.detector.scale;
    const pct = ((boot.detector.expected_z - min) / (max - min)) * 100;
    const style = /<div class="face" id="face" style="([^"]*)"/.exec(HTML)?.[1] ?? "";
    expect(style).toBe(`--init-x: ${pct.toFixed(4)}%; --init-n: ${(pct / 100).toFixed(4)}`);

    const shown = String(
      points(boot.detector.expected_z, {
        zStar: boot.detector.z_star,
        scale: boot.detector.scale,
      }),
    );
    expect(HTML).toContain(`data-pending="1">${shown}</span>`);
    expect(HTML).toContain(`aria-valuenow="${shown}"`);
  });

  it("bounds the meter 0..100, because the displayed number is points", () => {
    // The first paint is server-rendered and the first repaint is main.ts's.
    // If the checked-in bounds still said -2..10 a screen reader would announce
    // a value outside the range it was just given.
    expect(HTML).toContain('aria-valuemin="0"');
    expect(HTML).toContain('aria-valuemax="100"');
  });

  it("names the campaign position and never the ruleset id", () => {
    // "L1" is wire and DB vocabulary for a rule LIST. The player sees a level
    // NUMBER and the ruleset's name, and never both numbering systems at once.
    expect(HTML).toContain('id="levelno"');
    expect(HTML).toContain('id="lvlname"');
    expect(HTML).not.toContain('id="lvlid"');
    expect(HTML).not.toContain('id="daily"');
  });

  it("carries the campaign furniture the result sheet needs", () => {
    expect(HTML).toContain('id="nextlvl"');
    expect(HTML).toContain('id="rallclear"');
    expect(HTML).toContain('id="ractions"');
  });

  it("keeps the share's copy button alive through applyCopySlots", () => {
    // `data-copy-opt` REMOVES its element when the key is absent from
    // copy.toml. #copyres asked for `screen.copy_label`, which did not exist,
    // so the button was deleted at boot, gate.ts bound its click handler to a
    // detached element, and the end-of-campaign share — the only thing the
    // game has left to give a player who finished it — could not be copied.
    // The key must exist for the element to survive; assert both ends.
    const toml = parseCopyToml(COPY_TOML);
    expect(HTML).toMatch(/id="copyres"[^>]*data-copy-opt="screen\.copy_label"/);
    expect(typeof (toml.screen as CopyTree).copy_label).toBe("string");
  });

  it("credits the author with two real links, opened safely", () => {
    expect(HTML).toContain('href="https://x.com/ShrivuShankar"');
    expect(HTML).toContain('href="https://github.com/sshh12/launder-wm"');
    // target=_blank without rel=noopener hands the new tab a window.opener
    // handle back into this document.
    const links = HTML.match(/<a\b[\s\S]*?>/g) ?? [];
    expect(links.length).toBeGreaterThan(0);
    for (const tag of links) {
      if (tag.includes('target="_blank"')) expect(tag).toContain('rel="noopener"');
    }
  });

  it("is Launder WM, never Launder LM", () => {
    expect(HTML).toContain("<title>Launder WM</title>");
    expect(HTML).not.toContain("Launder LM");
  });
});
