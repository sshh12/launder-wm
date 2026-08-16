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
const TOKENS_CSS = readFileSync(new URL("../src/styles/tokens.css", import.meta.url), "utf8");

/** index.html with every `<!-- ... -->` removed.
 *
 *  Needed because the head's comments NAME the tags that are deliberately
 *  absent — og:image, apple-touch-icon — and explain why. A "this page must not
 *  contain X" assertion run against the raw file would fail on the paragraph
 *  explaining that the page must not contain X, and the obvious fix (delete the
 *  explanation) is the wrong one. */
const MARKUP = HTML.replace(/<!--[\s\S]*?-->/g, "");

/** Every value a custom property is given anywhere in tokens.css, lowercased.
 *  `--bg` and `--ink` each have three declarations — light, the
 *  prefers-color-scheme block, and the explicit [data-theme] block — so this is
 *  a set, and "is this colour in the palette" is a membership test against it. */
function tokenValues(name: string): Set<string> {
  const out = new Set<string>();
  for (const m of TOKENS_CSS.matchAll(new RegExp(`${name}:\\s*([^;]+);`, "g"))) {
    out.add((m[1] ?? "").trim().toLowerCase());
  }
  return out;
}

/** The head is not markup this file may parse loosely: a second `og:url` or a
 *  missing `description` is a silently wrong share card, so every lookup here
 *  asserts the tag appears exactly once. */
function headAttr(pattern: RegExp): string {
  const all = [...HTML.matchAll(new RegExp(pattern.source, `${pattern.flags}g`))];
  expect(all.length, `expected exactly one match for ${pattern}`).toBe(1);
  return all[0]?.[1] ?? "";
}

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

/**
 * The head — the one player-facing surface that is NOT copy.toml.
 *
 * The reasoning is written out at length in web/index.html and is not repeated
 * here; the short version is that [data-copy] fills elements after main.ts
 * runs, and a crawler reads the head and leaves. The consequence is what this
 * block exists for: head metadata sits outside `forge lint-copy`, so the COPY
 * STANDARD has to be enforced somewhere, and this is that somewhere.
 *
 * The palette assertions read src/styles/tokens.css rather than restating a
 * hex. A `theme-color` that drifts from `--bg` paints a seam between the
 * address bar and the page, and it drifts by somebody editing one file.
 */
describe("index.html head metadata", () => {
  const DESCRIPTION = headAttr(/<meta\s+name="description"\s+content="([^"]*)"/);

  it("describes the game, in one string used by both search and the share card", () => {
    // ONE sentence pair, two tags. Two different descriptions is two things to
    // keep honest, and the one that is wrong is always the one nobody reads.
    const og = headAttr(/<meta\s+property="og:description"\s+content="([^"]*)"/);
    expect(DESCRIPTION).not.toBe("");
    expect(og).toBe(DESCRIPTION);
    // Google truncates a snippet near 160 characters, and the tail of this one
    // is the part that says what the player does.
    expect(DESCRIPTION.length).toBeLessThanOrEqual(160);
  });

  it("never claims more than the detector can support", () => {
    // copy.toml's standing rule 2, applied to the one string that file does not
    // own. The detector finds a watermark; it does not find an author. A
    // description that overclaims is read by more people than any string in the
    // product, because it is what gets posted.
    const lower = DESCRIPTION.toLowerCase();
    expect(lower).not.toContain("human");
    expect(lower).not.toContain("secure");
    expect(lower).not.toContain("written by");
    expect(DESCRIPTION).not.toMatch(/%/);
    // ...and no "N% AI" figure by any other spelling.
    expect(DESCRIPTION).not.toMatch(/\d/);
  });

  it("restates the document's own identity, rather than inventing a second name", () => {
    // <title> has been static here since M0 and copy.toml has never owned it.
    // og:title and og:site_name are the same identity for a different reader,
    // so they follow the string that is already static.
    expect(headAttr(/<meta\s+property="og:title"\s+content="([^"]*)"/)).toBe("Launder WM");
    expect(headAttr(/<meta\s+property="og:site_name"\s+content="([^"]*)"/)).toBe("Launder WM");
    expect(headAttr(/<meta\s+property="og:type"\s+content="([^"]*)"/)).toBe("website");
  });

  it("asks for the card it can actually fill", () => {
    // `summary_large_image` reserves a picture-sized area and renders the hole
    // when no og:image arrives. There is no og:image because every unfurler
    // fetches it over the network as PNG/JPEG/GIF/WEBP — an SVG is not
    // accepted and a data: URI is not fetchable — and this repo ships no
    // binary assets for the page.
    expect(headAttr(/<meta\s+name="twitter:card"\s+content="([^"]*)"/)).toBe("summary");
    expect(MARKUP).not.toContain("og:image");
  });

  it("advertises no host, exactly like the share string", () => {
    // The mistake copy.toml names around `share_all_template`: a domain baked
    // into the checked-in file makes every preview deploy and every localhost
    // run post a production link. launder-serve rewrites both of these to the
    // origin the request arrived on (boot.py's `render_origin`); the value
    // here is what an un-served copy of this file says, and "/" resolves
    // against whatever document it is in.
    expect(headAttr(/<link\s+rel="canonical"\s+href="([^"]*)"/)).toBe("/");
    expect(headAttr(/<meta\s+property="og:url"\s+content="([^"]*)"/)).toBe("/");
  });

  it("paints the browser chrome with --bg, in both modes", () => {
    const themes = [...HTML.matchAll(/<meta\s+name="theme-color"\s+content="([^"]*)"\s+media="([^"]*)"/g)];
    expect(themes.length, "one theme-color per scheme").toBe(2);
    const bg = tokenValues("--bg");
    for (const [, colour, media] of themes) {
      expect(bg, `theme-color ${colour} is not a --bg in tokens.css`).toContain(
        (colour ?? "").toLowerCase(),
      );
      expect(media).toMatch(/^\(prefers-color-scheme: (light|dark)\)$/);
    }
    expect(new Set(themes.map((m) => m[2])).size).toBe(2);
  });
});

describe("the icon", () => {
  const HREF = headAttr(/<link\s+rel="icon"\s+href="([^"]*)"/);
  const SVG = decodeURIComponent(HREF);
  /** The SVG namespace is an IDENTIFIER, not a URL anything fetches. It is the
   *  one `http://` allowed in here, and dropping it is what lets the rule below
   *  be "no network, at all". */
  const OFFLINE = SVG.replace("xmlns='http://www.w3.org/2000/svg'", "");

  it("is an inline SVG and costs no request", () => {
    // vite.config.ts sets `publicDir: false`, so a file in web/public would not
    // even reach dist. And with no icon declared at all the browser asks for
    // /favicon.ico on every cold visit, which falls through to the static
    // mount and 404s — a declared data: URI removes the round trip rather
    // than answering it.
    expect(HREF.startsWith("data:image/svg+xml,")).toBe(true);
    expect(OFFLINE).not.toMatch(/https?:\/\//);
    // No CDN, no <image href>, no @import, no url() — all of which an SVG
    // document is perfectly willing to go and fetch.
    expect(OFFLINE).not.toMatch(/url\(|<image\b|<use\b|@import/);
    expect(MARKUP).not.toMatch(/rel="icon"[^>]*\.(png|ico|gif|jpe?g)/);
  });

  it("is thick enough to survive 16px", () => {
    // The in-app care symbols are 1.35 units on a 24 grid (gate.ts). Rendered
    // into a 16px tab icon that is 1.35 * 16/24 = 0.9 DEVICE PIXELS, which
    // antialiases to grey. Anything under ~2 units here is the same bug.
    expect(Number(/stroke-width='([\d.]+)'/.exec(SVG)?.[1])).toBeGreaterThanOrEqual(2);
    // ...and it is still the same drawing system as the in-app glyphs.
    expect(SVG).toContain("viewBox='0 0 24 24'");
    expect(SVG).toContain("fill='none'");
    expect(SVG).toContain("stroke-linecap='round'");
    expect(SVG).toContain("stroke-linejoin='round'");
  });

  it("carries only two strokes, because a third one merges at 16px", () => {
    // `wash` in gate.ts is three parts — basin, a three-bump wave, and the arc
    // over the rim. At 16px the arc sits under a device pixel from the rim and
    // the bumps are 1.6px wide. Dropping to basin + one wave is the whole
    // adaptation, and a future path added back here would undo it.
    expect([...SVG.matchAll(/<path\b/g)].length).toBe(2);
  });

  it("is legible on light and dark browser chrome, in palette colours", () => {
    // `stroke="currentColor"` — what symbolSvg() uses — is wrong in a favicon:
    // it is its own document with no cascade from the page, so currentColor
    // resolves to the initial black and vanishes against dark chrome.
    expect(SVG).not.toContain("currentColor");
    expect(SVG).toContain("@media(prefers-color-scheme:dark)");
    const ink = tokenValues("--ink");
    const used = [...SVG.matchAll(/#[0-9a-f]{3,8}\b/gi)].map((m) => m[0].toLowerCase());
    expect(used.length).toBe(2);
    for (const colour of used) {
      expect(ink, `${colour} is not an --ink in tokens.css`).toContain(colour);
    }
    // The presentation attribute is the LIGHT value, so a renderer that draws
    // SVG icons but ignores the media query still gets readable-on-white ink.
    expect(/stroke='(#[0-9a-f]{6})'/i.exec(SVG)?.[1]?.toLowerCase()).toBe("#1b1a17");
  });

  it("declares no apple-touch-icon, because it cannot without a PNG", () => {
    // iOS does not accept SVG for apple-touch-icon; Safari wants a PNG and
    // falls back to a screenshot of the page when it cannot decode one, which
    // is a worse home-screen tile than the default. Asserted rather than
    // merely omitted so nobody adds `rel="apple-touch-icon"` pointing at the
    // data: URI above and assumes it works.
    expect(MARKUP).not.toContain("apple-touch-icon");
  });
});
