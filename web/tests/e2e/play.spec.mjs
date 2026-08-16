// @ts-check
/**
 * The playing screen, end to end in a real browser.
 *
 * These are the behaviours §10.5 and §10.7 call requirements rather than
 * suggestions: the primer's once-only contract, the four things that never
 * collapse, the keyboard-open compaction, the tappable constraint disclosures,
 * and the fact that no player-facing string is ever written in TypeScript.
 *
 * JavaScript rather than TypeScript for the reason given in
 * mirror-alignment.spec.mjs: `@playwright/test` is not yet a dependency and
 * `tests/**` is type-checked by `npm run build`.
 */

import { readFileSync } from "node:fs";

import { expect, test } from "@playwright/test";

/**
 * Every ruleset name the rail can be asked to print, read from the file that
 * defines them rather than copied here — a name added to levels.toml is a
 * string the railhead has to fit, and this list is how the layout finds out.
 */
const RULESET_NAMES = [
  ...readFileSync(new URL("../../../data/config/levels.toml", import.meta.url), "utf8").matchAll(
    /^\s*name\s*=\s*"([^"]+)"/gm,
  ),
].map((m) => m[1]);

async function fresh(page) {
  await page.goto("/?level=1");
}

/**
 * A returning player: primer already seen, and pinned to level 1 with
 * `?level=`. The pin matters — without it, a spec that clears a level leaves a
 * `launder_level` cookie behind and the NEXT spec in the same browser context
 * lands on a different passage than the one it was written against.
 */
async function returning(page) {
  await page.addInitScript(() => {
    try {
      localStorage.setItem("launderwm.primer.v1", "1");
    } catch {
      /* private mode */
    }
  });
  await page.goto("/?level=1");
}

test.describe("first land", () => {
  test("opens the primer once, and never again", async ({ page }) => {
    await fresh(page);
    const sheet = page.locator("#primersheet");
    await expect(sheet).toHaveAttribute("data-open", "1");
    // Wordle's contract: the strings come from copy.toml, so assert the shape
    // rather than the words — the words are allowed to change without a code
    // change, which is the entire point of §10.7.
    await expect(page.locator("#primersheet .pr-lede").first()).not.toBeEmpty();
    await expect(page.locator("#primersheet .pr-body").first()).not.toBeEmpty();
    await page.locator("#prplay").click();
    await expect(sheet).toHaveAttribute("data-open", "0");

    await page.reload();
    await expect(page.locator("#primersheet")).toHaveAttribute("data-open", "0");
  });

  test("the ? in the rail is the one way back in", async ({ page }) => {
    await returning(page);
    await expect(page.locator("#primersheet")).toHaveAttribute("data-open", "0");
    await page.locator("#whatbtn").click();
    await expect(page.locator("#primersheet")).toHaveAttribute("data-open", "1");
    await page.keyboard.press("Escape");
    await expect(page.locator("#primersheet")).toHaveAttribute("data-open", "0");
  });
});

test.describe("the instrument", () => {
  test("is a meter with a spoken value and a threshold", async ({ page }) => {
    await returning(page);
    const meter = page.locator("#meter");
    await expect(meter).toHaveAttribute("role", "meter");
    await expect(meter).toHaveAttribute("aria-valuenow", /-?\d+(\.\d+)?/);
    await expect(meter).toHaveAttribute("aria-valuetext", /.+/);
    await expect(meter).toHaveAttribute("aria-label", /.+/);
    // The threshold label is "Under N to clear" — never the word "floor",
    // which already means the word minimum (§10.7 rule 1).
    const label = await page.locator("#floorlbl").textContent();
    expect(label?.toLowerCase()).not.toContain("floor");
    expect(label?.length ?? 0).toBeGreaterThan(0);
  });

  test("prints 21 ticks and a scale that runs 0 to 100", async ({ page }) => {
    await returning(page);
    await expect(page.locator("#face .tick")).toHaveCount(21);
    const labels = await page.locator("#scalerow span").allTextContents();
    // z is the statistic and stays internal; a face printed -2..10 reads as a
    // broken instrument.
    expect(labels).toEqual(["0", "25", "50", "75", "100"]);
  });

  test("reads as an instrument and offers no control", async ({ page }) => {
    await returning(page);
    // role="meter" is strictly read-only. Nothing inside it takes a drag, so
    // nothing inside it may LOOK like it does: no control, no focus stop, and
    // a cursor that does not promise one.
    await expect(
      page.locator("#meter input, #meter button, #meter [tabindex], #meter [role='slider']"),
    ).toHaveCount(0);
    const meter = await page.locator("#meter").evaluate((el) => ({
      cursor: getComputedStyle(el).cursor,
      role: el.getAttribute("role"),
    }));
    expect(meter.role).toBe("meter");
    expect(meter.cursor).toBe("default");
  });

  test("the threshold mark stays legible against the fill that covers it", async ({ page }) => {
    await returning(page);
    const paint = await page.evaluate(() => {
      const el = (id) => document.getElementById(id);
      const box = (node) => node.getBoundingClientRect();
      // Resolve a token to the same rgb() spelling getComputedStyle returns,
      // so the comparison below is about COLOUR and not about notation.
      const rgb = (value) => {
        const probe = document.createElement("div");
        probe.style.backgroundColor = value;
        document.body.appendChild(probe);
        const out = getComputedStyle(probe).backgroundColor;
        probe.remove();
        return out;
      };
      const root = getComputedStyle(document.documentElement);
      const s = getComputedStyle(el("notch"));
      const notch = box(el("notch"));
      const tri = box(el("tri"));
      // Whatever the notch is actually drawn with — a background, a border, or
      // both — every colour it puts on the face.
      const notchInk = [];
      if (s.backgroundColor !== "rgba(0, 0, 0, 0)") notchInk.push(s.backgroundColor);
      if (s.backgroundImage !== "none") notchInk.push(s.backgroundImage);
      if (s.borderLeftStyle !== "none" && Number.parseFloat(s.borderLeftWidth) > 0) {
        notchInk.push(s.borderLeftColor);
      }
      return {
        // The fill's two states. Compared against the tokens rather than
        // against the element, whose background-color is mid-transition for
        // 380ms after any reading lands.
        fillStates: [rgb(root.getPropertyValue("--hot")), rgb(root.getPropertyValue("--cold"))],
        notchInk,
        notchCx: notch.left + notch.width / 2,
        triCx: tri.left + tri.width / 2,
        faceBottom: box(el("face")).bottom,
        gutTop: box(document.querySelector(".gutrow")).top,
        scaleTop: box(el("scalerow")).top,
      };
    });
    // It is painted with something...
    expect(paint.notchInk.length).toBeGreaterThan(0);
    // ...and with nothing the fill also uses. The fill runs from 0 up to the
    // needle and passes straight over the one mark that defines winning; in
    // either of the fill's own two colours that mark is camouflaged against
    // the bar that moves.
    for (const ink of paint.notchInk) expect(paint.fillStates).not.toContain(ink);
    // "Under N to clear" and its pointer belong to the notch: same centre line,
    // and directly under the face rather than below the printed numerals.
    expect(Math.abs(paint.notchCx - paint.triCx)).toBeLessThanOrEqual(1);
    expect(paint.gutTop).toBeLessThan(paint.scaleTop);
    expect(paint.gutTop - paint.faceBottom).toBeLessThanOrEqual(4);
  });

  test("reads a bare 0-100 number, never a decimal and never a percentage", async ({ page }) => {
    await returning(page);
    const num = (await page.locator("#num").textContent()) ?? "";
    expect(num).toMatch(/^\d{1,3}$/);
    expect(Number(num)).toBeGreaterThanOrEqual(0);
    expect(Number(num)).toBeLessThanOrEqual(100);
    const word = (await page.locator("#stateword").textContent()) ?? "";
    expect(word.toLowerCase()).not.toContain("human");
    expect(word).not.toContain("%");
  });
});

test.describe("the campaign", () => {
  test("names the level and its total, and never the ruleset id", async ({ page }) => {
    await returning(page);
    // "Level 1 of 15". The player never sees "L1" — that is the rule LIST's id,
    // and two numbering systems on one screen is how they came to be confused.
    const levelno = (await page.locator("#levelno").textContent()) ?? "";
    expect(levelno).toMatch(/\d+.*\d+/);
    await expect(page.locator("#lvlname")).not.toBeEmpty();
    await expect(page.locator("#lvlid")).toHaveCount(0);
  });

  /**
   * THE TRUNCATION REGRESSION. The railhead was one flex row in which every
   * item was `flex: none` except the ruleset name, which carried
   * `text-overflow: ellipsis` — so the name was the only thing that could
   * absorb overflow, and lengthening the readout from "AI detected" to "AI
   * watermark detected" collapsed "Clean it" to "C…" on a phone. A level whose
   * name the player cannot read is a level whose rules they cannot anticipate.
   */
  test.describe("at 320px, the narrowest supported width", () => {
    test.use({ viewport: { width: 320, height: 844 } });

    test("prints the level, its total AND the ruleset name in full", async ({ page }) => {
      await returning(page);
      const rows = await page.evaluate((names) => {
        const chip = document.getElementById("lvlchip");
        const name = document.getElementById("lvlname");
        const levelno = document.getElementById("levelno");
        // scrollWidth > clientWidth is the clip itself, whatever draws it —
        // an ellipsis, a hidden overflow, or a box squeezed to nothing.
        const clipped = (el) => el.scrollWidth > el.clientWidth + 1;
        const out = [];
        for (const text of names) {
          name.textContent = text;
          out.push({
            text,
            rendered: name.textContent,
            nameWidth: name.getBoundingClientRect().width,
            levelnoWidth: levelno.getBoundingClientRect().width,
            chipClipped: clipped(chip),
            levelnoClipped: clipped(levelno),
            hOverflow:
              document.documentElement.scrollWidth - document.documentElement.clientWidth,
          });
        }
        return out;
      }, RULESET_NAMES);

      expect(rows.length).toBeGreaterThan(0);
      for (const row of rows) {
        expect(row.chipClipped, `"${row.text}" is clipped in the railhead at 320px`).toBe(false);
        expect(row.levelnoClipped, `the level counter is clipped at 320px`).toBe(false);
        // Both registers are actually on screen, not merely un-ellipsised.
        expect(row.nameWidth, `"${row.text}" renders at zero width`).toBeGreaterThan(0);
        expect(row.levelnoWidth).toBeGreaterThan(0);
        // Nothing was solved by pushing the rail off the side of the phone.
        expect(row.hOverflow, `"${row.text}" makes the page scroll sideways`).toBeLessThanOrEqual(
          0,
        );
      }
    });
  });

  test("credits the author with two tappable links", async ({ page }) => {
    await returning(page);
    const links = page.locator(".foot .credit");
    await expect(links).toHaveCount(2);
    await expect(links.first()).toHaveAttribute("href", "https://x.com/ShrivuShankar");
    await expect(links.last()).toHaveAttribute("href", "https://github.com/sshh12/launder-wm");
    for (const link of await links.all()) {
      await expect(link).not.toBeEmpty();
      await expect(link).toHaveAttribute("rel", "noopener");
      // Not a disabled hint: it has to be underlined and at ink weight.
      const decoration = await link.evaluate((el) => getComputedStyle(el).textDecorationLine);
      expect(decoration).toContain("underline");
    }
  });
});

test.describe("mobile, 390x844 with the keyboard open", () => {
  test.use({ viewport: { width: 390, height: 844 } });

  test("collapses to the status strip while typing and keeps the four essentials", async ({
    page,
  }) => {
    await returning(page);
    await page.locator("#raw").click();
    await expect(page.locator("body")).toHaveAttribute("data-typing", "1");
    // There is no accidental submit path while typing (§10.5).
    await expect(page.locator("#actions")).toBeHidden();
    await expect(page.locator("#strip")).toBeVisible();
    // Never collapses at any width: needle, notch, passage box, changed count.
    await expect(page.locator("#needle")).toBeVisible();
    await expect(page.locator("#notch")).toBeVisible();
    await expect(page.locator("#passage")).toBeVisible();
    await expect(page.locator("#stripcount")).toBeVisible();
  });

  test("Enter inserts a newline instead of submitting", async ({ page }) => {
    await returning(page);
    const raw = page.locator("#raw");
    await raw.click();
    const before = await raw.inputValue();
    await page.keyboard.press("End");
    await page.keyboard.press("Enter");
    const after = await raw.inputValue();
    expect(after.length).toBeGreaterThan(before.length);
    await expect(page.locator("#resultsheet")).toHaveAttribute("data-open", "0");
    await expect(page.locator("#gatesheet")).toHaveAttribute("data-open", "0");
  });
});

test.describe("the gate is legible", () => {
  test("pips open the checklist and every rule explains itself in place", async ({ page }) => {
    await returning(page);
    await page.locator("#pips").click();
    await expect(page.locator("#checksheet")).toHaveAttribute("data-open", "1");
    const rows = page.locator("#listchecks .checkrow[aria-controls]");
    await expect(rows.first()).toBeVisible();
    const first = rows.first();
    await expect(first).toHaveAttribute("aria-expanded", "false");
    const bodyId = await first.getAttribute("aria-controls");
    await expect(page.locator(`#${bodyId}`)).toBeHidden();
    await first.click();
    await expect(first).toHaveAttribute("aria-expanded", "true");
    await expect(page.locator(`#${bodyId}`)).toBeVisible();
    // The blurb answers "what does that actually mean" and never restates the
    // label, so it is necessarily longer than the label.
    const label = (await first.locator("span").first().textContent()) ?? "";
    const blurb = (await page.locator(`#${bodyId}`).textContent()) ?? "";
    expect(blurb.length).toBeGreaterThan(label.length);
  });

  test("draws one care symbol per check in the level", async ({ page }) => {
    await returning(page);
    const boot = await page.evaluate(() =>
      JSON.parse(document.getElementById("launder-boot").textContent),
    );
    await expect(page.locator("#pips svg")).toHaveCount(boot.level.checks.length);
  });
});

test.describe("no invented copy", () => {
  test("no element renders a raw copy key or an unresolved placeholder", async ({ page }) => {
    await returning(page);
    const text = await page.evaluate(() => document.body.innerText);
    // A missing key renders as its dotted path and an unresolved placeholder
    // keeps its braces; both are deliberate, loud failures (§10.7 lint rules).
    expect(text).not.toMatch(/\{[a-z0-9_]+\}/i);
    expect(text).not.toMatch(/\b(readout|screen|gate|errors|primer|check)\.[a-z0-9_.]+\b/i);
  });

  test("no emoji anywhere in the interface", async ({ page }) => {
    await returning(page);
    const text = await page.evaluate(() => document.body.innerText);
    expect(/\p{Extended_Pictographic}/u.test(text)).toBe(false);
  });
});
