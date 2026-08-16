// @ts-check
/**
 * The window — `detector_floor`, in a real browser.
 *
 * Levels 7 and 8 require the reading to LAND between a floor and the notch:
 * scrub too far down and you fail for looking scrubbed. The checklist has said
 * "Not too clean" since the rule shipped, and the face said nothing at all —
 * the player had no way to see where the bottom was until they tripped it.
 *
 * The two assertions that matter are the pair: the mark is DRAWN on a level
 * that runs the check and ABSENT on one that does not. A mark that is always
 * there is a mark that lies on thirteen of the fifteen levels.
 *
 * JavaScript rather than TypeScript for the reason given in play.spec.mjs.
 */

import { expect, test } from "@playwright/test";

/** As it ships in data/config/levels.toml for L8 and L9. */
const FLOOR_CHECK = { check: "detector_floor", params: { min_z: -1.2 } };

/** The shipped geometry: z* = 2.3263 on a -2..10 face is 36; the floor at
 *  z* - 1.2 is 26. A ten-point window on a hundred-point face. */
const FLOOR_AT = 26;
const NOTCH_AT = 36;

/**
 * Serve the page with `detector_floor` spliced into this level's checks.
 *
 * The dev fixture is a level with a top and no bottom, and the vite server
 * hands back exactly that file, so the only honest way to see a floor level in
 * this harness is to hand the page a boot payload that runs the check — which
 * is precisely the input the real server sends on levels 7 and 8. Nothing else
 * about the page changes.
 */
async function serveWithFloor(page) {
  // The payload is pretty-printed and the file's line endings are governed by
  // .gitattributes, so this matches the array's opening bracket and nothing
  // about the whitespace around it.
  const opensChecks = /"checks"\s*:\s*\[/;
  await page.route(
    (url) => url.pathname === "/",
    async (route) => {
      const response = await route.fetch();
      const html = await response.text();
      if (!opensChecks.test(html)) {
        throw new Error(
          "the boot payload in web/index.html has no `checks` array; this harness splices " +
            "detector_floor into it to stand in for levels 7 and 8.",
        );
      }
      await route.fulfill({
        response,
        body: html.replace(opensChecks, (m) => `${m}${JSON.stringify(FLOOR_CHECK)},`),
      });
    },
  );
}

/** A returning player, pinned to level 1, with the primer already seen so the
 *  sheet is not sitting over the instrument in the screenshots. */
async function returning(page, { floor = false } = {}) {
  await page.addInitScript(() => {
    try {
      localStorage.setItem("launderwm.primer.v1", "1");
    } catch {
      /* private mode */
    }
  });
  if (floor) await serveWithFloor(page);
  await page.goto("/?level=1");
  await expect(page.locator("#num")).not.toBeEmpty();
}

/** Everything the window is: where its two elements sit on the face, and what
 *  the floor is actually painted with next to what the notch is painted with. */
async function windowGeometry(page) {
  return page.evaluate(() => {
    const box = (id) => {
      const el = document.getElementById(id);
      return el === null ? null : el.getBoundingClientRect();
    };
    const face = box("face");
    const at = (rect) => (rect === null ? null : ((rect.left - face.left) / face.width) * 100);
    const ink = (id) => {
      const el = document.getElementById(id);
      if (el === null) return null;
      const s = getComputedStyle(el);
      return {
        borderLeftStyle: s.borderLeftStyle,
        borderLeftWidth: s.borderLeftWidth,
        borderTopStyle: s.borderTopStyle,
        borderRightWidth: s.borderRightWidth,
        height: Math.round(el.getBoundingClientRect().height),
      };
    };
    const band = box("band");
    return {
      faceWidth: face.width,
      faceHeight: Math.round(face.height),
      floorAt: at(box("floormark")),
      bandFrom: at(band),
      bandTo: band === null ? null : ((band.right - face.left) / face.width) * 100,
      notchAt: at(box("notch")),
      floorInk: ink("floormark"),
      notchInk: ink("notch"),
      // Nothing was solved by pushing the rail off the side of the phone.
      hOverflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
    };
  });
}

/** How far a mark stands from the number it claims, in PIXELS. Percentages are
 *  what the marks are placed with, but a percentage of a 292px face rounds to
 *  the device pixel grid, and a tolerance expressed in percent tightens as the
 *  viewport narrows — which is the one place it must not. */
function driftPx(geo, atPct, targetPct) {
  return (Math.abs(atPct - targetPct) / 100) * geo.faceWidth;
}

test.describe("a level that runs the check", () => {
  test("marks the floor and opens the window between it and the notch", async ({ page }) => {
    await returning(page, { floor: true });

    // The payload really does run the check — otherwise everything below could
    // pass by drawing a mark unconditionally.
    const checks = await page.evaluate(() =>
      JSON.parse(document.getElementById("launder-boot").textContent).level.checks.map(
        (c) => c.check,
      ),
    );
    expect(checks).toContain("detector_floor");

    await expect(page.locator("#floormark")).toHaveCount(1);
    await expect(page.locator("#band")).toHaveCount(1);
    await expect(page.locator("#floormark")).toBeVisible();
    await expect(page.locator("#band")).toBeVisible();

    const geo = await windowGeometry(page);
    // The mark stands on the number, not near it: 26 on a notch of 36.
    expect(driftPx(geo, geo.floorAt, FLOOR_AT)).toBeLessThanOrEqual(1);
    expect(driftPx(geo, geo.bandFrom, FLOOR_AT)).toBeLessThanOrEqual(1);
    expect(driftPx(geo, geo.bandTo, NOTCH_AT)).toBeLessThanOrEqual(1);
    expect(driftPx(geo, geo.notchAt, NOTCH_AT)).toBeLessThanOrEqual(2);
    // The band is the TARGET, so it lies between the two bounds and nowhere
    // else — a band that started at the left edge would be a filled track.
    expect(geo.bandFrom).toBeGreaterThan(1);
    expect(geo.bandTo).toBeLessThanOrEqual(geo.notchAt + 1);
  });

  test("draws the floor in a different hand from the notch", async ({ page }) => {
    await returning(page, { floor: true });
    const { floorInk, notchInk, faceHeight } = await windowGeometry(page);
    // They bound the same window and they do not mean the same thing, so they
    // must not read as a pair of identical ticks. Solid against dashed...
    expect(notchInk.borderLeftStyle).toBe("dashed");
    expect(floorInk.borderLeftStyle).toBe("solid");
    expect(floorInk.borderLeftStyle).not.toBe(notchInk.borderLeftStyle);
    // ...a bracket against a bare line: the floor turns serifs into the window.
    expect(floorInk.borderTopStyle).toBe("solid");
    expect(notchInk.borderTopStyle).toBe("none");
    expect(Number.parseFloat(floorInk.borderRightWidth)).toBe(0);
    // ...and short where the notch spans the whole face, so it cannot be read
    // as a second full-height rule. (The notch fills the face's INNER height,
    // which is the face minus its 1px border top and bottom.)
    expect(floorInk.height).toBeLessThan(notchInk.height);
    expect(notchInk.height).toBeGreaterThanOrEqual(faceHeight - 2);
  });

  test("stays a meter, and adds no control", async ({ page }) => {
    await returning(page, { floor: true });
    const meter = page.locator("#meter");
    await expect(meter).toHaveAttribute("role", "meter");
    await expect(meter).toHaveAttribute("aria-valuemin", "0");
    await expect(meter).toHaveAttribute("aria-valuemax", "100");
    await expect(meter).toHaveAttribute("aria-valuenow", /^\d{1,3}$/);
    await expect(meter).toHaveAttribute("aria-label", /.+/);
    await expect(
      page.locator("#meter input, #meter button, #meter [tabindex], #meter [role='slider']"),
    ).toHaveCount(0);
    expect(await meter.evaluate((el) => getComputedStyle(el).cursor)).toBe("default");
    // The window is printing, not furniture the player can grab or select.
    for (const id of ["band", "floormark"]) {
      expect(await page.locator(`#${id}`).evaluate((el) => getComputedStyle(el).pointerEvents)).toBe(
        "none",
      );
    }
  });

  /**
   * A mark on the face is no use to anyone reading the instrument through a
   * screen reader, so the spoken value carries the floor instead — in the words
   * copy.toml already writes for this rule, with both numbers resolved. No new
   * string is invented in TypeScript (§10.7).
   */
  test("speaks the floor when the needle is under it", async ({ page }) => {
    await returning(page, { floor: true });
    const spoken = await page.evaluate(() => {
      const boot = JSON.parse(document.getElementById("launder-boot").textContent);
      return {
        expectedZ: boot.detector.expected_z,
        floorZ: boot.detector.z_star - 1.2,
        template: boot.copy.check.detector_floor.reject,
        lineTemplate: boot.copy.readout.valuetext_below,
        num: document.getElementById("num").textContent,
        valuetext: document.getElementById("meter").getAttribute("aria-valuetext"),
      };
    });
    // Stated out loud, because the assertion below rests on it: the fixture's
    // pristine reading already sits under the floor this level would impose.
    expect(spoken.expectedZ, "the fixture no longer boots under the floor").toBeLessThan(
      spoken.floorZ,
    );
    expect(spoken.valuetext).toBe(
      spoken.template.replace("{z_display}", spoken.num).replace("{z_floor_display}", "26"),
    );
    // The floor's sentence REPLACES the line's rather than joining it: both are
    // true down here and only one of them is what stands between the player and
    // the clear.
    expect(spoken.valuetext).not.toContain(
      spoken.lineTemplate.replace("{z_display}", spoken.num),
    );
    expect(spoken.valuetext).not.toMatch(/\{[a-z0-9_]+\}/i);
    expect(spoken.valuetext).not.toMatch(/\bcheck\.[a-z0-9_.]+\b/i);
  });

  test("keeps the window inside the face at every supported width", async ({ page }) => {
    await returning(page, { floor: true });
    for (const width of [320, 390, 1280]) {
      await page.setViewportSize({ width, height: 844 });
      await page.reload();
      await expect(page.locator("#floormark")).toBeVisible();
      const geo = await windowGeometry(page);
      expect(driftPx(geo, geo.floorAt, FLOOR_AT), `floor drifts at ${width}px`).toBeLessThanOrEqual(
        1,
      );
      expect(geo.bandTo, `window overshoots the notch at ${width}px`).toBeLessThanOrEqual(
        geo.notchAt + 1,
      );
      expect(
        geo.hOverflow,
        `the window makes the page scroll sideways at ${width}px`,
      ).toBeLessThanOrEqual(0);
      await page
        .locator("#rail")
        .screenshot({ path: test.info().outputPath(`window-${width}.png`) });
    }
  });
});

test.describe("a level that does not", () => {
  test("draws no floor and no window at all", async ({ page }) => {
    await returning(page);
    const checks = await page.evaluate(() =>
      JSON.parse(document.getElementById("launder-boot").textContent).level.checks.map(
        (c) => c.check,
      ),
    );
    expect(checks).not.toContain("detector_floor");

    // Not hidden, not zero-width: absent. There is no bottom on this level, so
    // there is nothing on the face that could be read as one.
    await expect(page.locator("#floormark")).toHaveCount(0);
    await expect(page.locator("#band")).toHaveCount(0);
    await expect(page.locator("#face .floormark, #face .band")).toHaveCount(0);

    // The top of the scale is untouched by any of this, and so is the sentence
    // the meter speaks: no floor, no floor talk.
    await expect(page.locator("#notch")).toBeVisible();
    await expect(page.locator("#floorlbl")).not.toBeEmpty();
    const spoken = await page.evaluate(() => {
      const boot = JSON.parse(document.getElementById("launder-boot").textContent);
      return {
        valuetext: document.getElementById("meter").getAttribute("aria-valuetext"),
        expected: boot.copy.readout.valuetext_below.replace(
          "{z_display}",
          document.getElementById("num").textContent,
        ),
      };
    });
    expect(spoken.valuetext).toBe(spoken.expected);
  });

  test("looks the same at every supported width", async ({ page }) => {
    await returning(page);
    for (const width of [320, 390, 1280]) {
      await page.setViewportSize({ width, height: 844 });
      await page.reload();
      await expect(page.locator("#notch")).toBeVisible();
      await expect(page.locator("#floormark")).toHaveCount(0);
      await page
        .locator("#rail")
        .screenshot({ path: test.info().outputPath(`no-window-${width}.png`) });
    }
  });
});
