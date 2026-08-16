// @ts-check
/**
 * THREE PIECES OF WIRING, EACH OF WHICH WAS CONNECTED TO THE WRONG THING.
 *
 * None of these is arithmetic — `live.test.ts`, `mirror.test.ts` and
 * `points.test.ts` already prove the arithmetic, and all three of these bugs
 * were invisible to them because every function involved returned the right
 * answer. What was wrong was WHICH answer reached the screen, WHEN it was
 * asked for, and WHICH layout it was measured against. That only shows up with
 * the whole page assembled, which is what this file is.
 *
 * JavaScript rather than TypeScript for the reason given in
 * mirror-alignment.spec.mjs: `tests/**` is type-checked by `npm run build` and
 * a `.spec.ts` would break a checkout that has not installed Playwright.
 *
 * The dev server serves the checked-in fixture — level "L2", five checks, no
 * API behind it. Where a test needs a server that the dev server does not
 * have, it routes the request rather than pretending: the response bodies below
 * are the wire shapes in `launder_core.schemas.api`, and nothing about the
 * client is stubbed.
 */

import { expect, test } from "@playwright/test";

/** A returning player: the primer sets `inert` on the app while it is open. */
async function returning(page) {
  await page.addInitScript(() => {
    try {
      localStorage.setItem("launderwm.primer.v1", "1");
    } catch {
      /* private mode */
    }
  });
}

/** The pip states, in level order: "pass" | "fail" | "error" | "pending". */
const states = (page) =>
  page.locator("#pips svg").evaluateAll((els) => els.map((e) => e.getAttribute("data-s")));

/** The fixture's check list, read from the page rather than hardcoded here —
 *  the ordering belongs to levels.toml. */
const checksInPage = (page) =>
  page.evaluate(
    () =>
      JSON.parse(document.getElementById("launder-boot").textContent).level.checks.map(
        (c) => c.check,
      ),
  );

/* ------------------------------------------------------------------ *
 * 1. The reading the server inlined
 * ------------------------------------------------------------------ */

/**
 * Serve the fixture with a pristine reading in its boot payload, which is what
 * `launder-serve` does in production and what the checked-in dev fixture
 * deliberately does not (a reading is a detector output and this repo will not
 * ship an invented one). `text_hash` is empty on purpose: the store treats ""
 * as "no digest offered" and the exact-text arm of the reconciliation rule
 * carries the guard, which is the same path the server's own first paint takes
 * before `crypto.subtle` has answered.
 */
async function withInlinedReading(page, reading) {
  await page.route(
    (url) => url.pathname === "/",
    async (route) => {
      const response = await route.fetch();
      const body = (await response.text()).replace(
        '"reading": null',
        `"reading": ${JSON.stringify(reading)}`,
      );
      await route.fulfill({ response, body });
    },
  );
}

const PRISTINE_READING = {
  seq: 1,
  text_hash: "",
  score: 0.5312,
  z: 7.33,
  z_star: 2.3263,
  n_scored: 168,
  n_tokens: 200,
  tokens: [],
  preview_distance: 0,
};

test.describe("the reading the server already inlined", () => {
  test("reaches the live detector rule, before the player types anything", async ({ page }) => {
    await returning(page);
    await withInlinedReading(page, PRISTINE_READING);
    await page.goto("/?level=1");
    const checks = await checksInPage(page);
    const threshold = checks.indexOf("detector_threshold");
    expect(threshold).toBeGreaterThanOrEqual(0);

    // z 7.33 against a notch at 2.3263 and max_z 0: the win condition is not
    // met, and the level's own rule has to SAY so. It used to read "Not checked
    // yet" — `store.applyReading` ran before `store.subscribe`, so the one
    // emission the pristine passage ever produces reached nobody and
    // `gate.setReading` was never called with it.
    expect((await states(page))[threshold]).toBe("fail");

    // The judge is the row that genuinely has not run, and it must not move.
    expect((await states(page))[checks.indexOf("llm_gate")]).toBe("pending");
  });

  test("puts the needle on the reading, not on the fixture's expectation", async ({ page }) => {
    await returning(page);
    await withInlinedReading(page, PRISTINE_READING);
    await page.goto("/?level=1");
    // 7.33 on a -2..10 scale is 78 points. The same emission carries it.
    await expect(page.locator("#num")).toHaveText("78");
    await expect(page.locator("#meter")).toHaveAttribute("aria-valuenow", "78");
  });
});

/* ------------------------------------------------------------------ *
 * 2. Where the ripple starts
 * ------------------------------------------------------------------ */

test.describe("the ripple starts where the player typed", () => {
  test("the impulse lands on the span the caret is in, not on the one before it", async ({
    page,
  }) => {
    await returning(page);
    await page.goto("/?level=1");
    await page.waitForSelector("#mirror .tk");

    await page.locator("#raw").focus();
    await page.evaluate(() => {
      const t = document.getElementById("raw");
      t.selectionStart = t.selectionEnd = t.value.length;
    });
    // ONE character: each keystroke reanchors against the frame before it, so
    // a three-character burst leaves three one-character dirty spans and the
    // last of them would be "d" rather than "Zed". That is the renderer working
    // as designed, and it is not what this test is about.
    await page.keyboard.type("Z");

    const seen = await page.evaluate(() => {
      const spans = [...document.querySelectorAll("#mirror .tk")];
      const at = spans.findIndex((s) => s.hasAttribute("data-impulse"));
      return { at, count: spans.length, text: spans[at]?.textContent ?? null };
    });
    // Typing at the very end makes the last span the dirty one. The origin used
    // to be resolved against the layout the mirror still held from BEFORE the
    // keystroke, so it named the second-to-last span — and a paste of N
    // characters moved it N characters' worth of tokens to the right.
    expect(seen.at).toBe(seen.count - 1);
    expect(seen.text).toBe("Z");
  });

  test("the propagation delay counts outward from that span", async ({ page }) => {
    await returning(page);
    await page.goto("/?level=1");
    await page.waitForSelector("#mirror .tk");

    // Put the caret in the middle of the passage and type there.
    const caret = await page.evaluate(() => {
      const t = document.getElementById("raw");
      const at = Math.floor(t.value.length / 2);
      t.focus();
      t.selectionStart = t.selectionEnd = at;
      return at;
    });
    // A PASTE, not a keystroke: it is the case that separates the two layouts
    // by more than one character. `insertText` delivers the whole run in one
    // `input` event, so the caret ends 24 characters right of where the old
    // layout's spans said it was — several tokens' worth.
    await page.keyboard.insertText("Q pasted run of new words");

    const seen = await page.evaluate(() => {
      const spans = [...document.querySelectorAll("#mirror .tk")];
      const at = spans.findIndex((s) => s.hasAttribute("data-impulse"));
      return {
        at,
        d: spans.map((s) => Number(s.style.getPropertyValue("--d"))),
        offsets: spans.map((s) => s.textContent ?? ""),
      };
    });
    expect(seen.at).toBeGreaterThan(0);
    // `--d` is 0 everywhere at or before the origin and climbs by one per token
    // after it, clamped at 16. The origin is therefore the LAST index whose
    // delay is still 0, which pins the two to each other.
    expect(seen.d[seen.at]).toBe(0);
    expect(seen.d[seen.at + 1]).toBe(1);
    expect(seen.d[seen.at - 1]).toBe(0);
    // and the span it landed on is the one holding the character just typed
    expect(seen.offsets[seen.at]).toContain("Q");
    expect(caret).toBeGreaterThan(0);
  });
});

/* ------------------------------------------------------------------ *
 * 3. The care label after a provisional clear
 * ------------------------------------------------------------------ */

/** A `SubmitResponse` in which one fail-open check ERRORED — the judge
 *  unreachable, or the daily spend cap hit. `run_gate` clears those runs
 *  provisionally; every rejection path sets `provisional=False`, so this is the
 *  only shape in which the flag can ever arrive. */
async function withSubmitOutcome(page, build) {
  const checks = await checksInPage(page);
  await page.route("**/api/submit", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(build(checks)),
    });
  });
  return checks;
}

const cleared = (trace, provisional) => ({
  cleared: true,
  provisional,
  score: { distance: 4, ops: [] },
  detector: { score: 0.5, z: 1.1, z_star: 2.3263, n_scored: 168 },
  failure: null,
  trace,
  par: 4,
  rank: null,
  share: "",
});

test.describe("the result sheet reports the checks that actually ran", () => {
  test("a provisional clear does not draw the errored check as a tick", async ({ page }) => {
    await returning(page);
    await page.goto("/?level=1");
    const checks = await withSubmitOutcome(page, (list) =>
      cleared(
        list.map((check) =>
          check === "llm_gate"
            ? { status: "error", check, code: "judge_unavailable" }
            : { status: "pass", check },
        ),
        true,
      ),
    );

    await page.locator("#check").click();
    await expect(page.locator("#resultsheet")).toHaveAttribute("data-open", "1");

    const drawn = await page
      .locator("#rsyms svg")
      .evaluateAll((els) => els.map((e) => e.getAttribute("data-s")));
    expect(drawn).toHaveLength(checks.length);
    // The care label used to hardcode "pass" for every row, which claimed the
    // judge had read the text on the one outcome where it demonstrably had not
    // — and that clear is the one the client deliberately does not record.
    expect(drawn[checks.indexOf("llm_gate")]).toBe("error");
    for (const check of checks) {
      if (check !== "llm_gate") expect(drawn[checks.indexOf(check)]).toBe("pass");
    }
  });

  test("an ordinary clear still draws every check as passed", async ({ page }) => {
    await returning(page);
    await page.goto("/?level=1");
    const checks = await withSubmitOutcome(page, (list) =>
      cleared(
        list.map((check) => ({ status: "pass", check })),
        false,
      ),
    );

    await page.locator("#check").click();
    await expect(page.locator("#resultsheet")).toHaveAttribute("data-open", "1");
    const drawn = await page
      .locator("#rsyms svg")
      .evaluateAll((els) => els.map((e) => e.getAttribute("data-s")));
    expect(drawn).toEqual(checks.map(() => "pass"));
  });
});
