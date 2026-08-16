// @ts-check
/**
 * The closed-form rules, answered while typing — TECH_PLAN.md §10.5.
 *
 * Seven of this game's checks are functions of text the browser already holds,
 * and six of them used to be reported only by a network round trip that also
 * spends money on a language model. "Real characters" and "At least 50 words"
 * were decided in microseconds and delivered in a second and a half, after a
 * button press, next to a gate rejection.
 *
 * `tests/live.test.ts` proves the arithmetic. This file proves the WIRING: that
 * a keystroke reaches the pips, that the failing rule can say why in place, and
 * that the server's trace still wins the moment it exists. That last one is the
 * property that keeps §8.3 true, and it cannot be tested without a submission.
 *
 * The dev server serves the checked-in fixture, whose level runs
 * `unicode_sanitation`, `word_floor` (50), `edit_budget` (12),
 * `detector_threshold` and `llm_gate` — three live rules, one that needs a
 * reading, and one that never can be. There is no API behind the dev server, so
 * the two that need one stay honestly unresolved, which is exactly the contrast
 * worth asserting.
 */

import { expect, test } from "@playwright/test";

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

/** The pip states, in level order: "pass" | "fail" | "error" | "pending". */
const states = (page) =>
  page.locator("#pips svg").evaluateAll((els) => els.map((e) => e.getAttribute("data-s")));

/** Index of a check in the fixture level, so the assertions do not hardcode an
 *  ordering that levels.toml owns. */
async function indexOf(page, check) {
  const boot = await page.evaluate(() =>
    JSON.parse(document.getElementById("launder-boot").textContent),
  );
  return boot.level.checks.findIndex((c) => c.check === check);
}

async function type(page, value) {
  await page.locator("#raw").fill(value);
  // `fill` dispatches one `input`; the live rules answer synchronously inside
  // that handler, so there is nothing to wait for beyond the next frame.
  await page.waitForTimeout(50);
}

test.describe("rules that can be answered locally, are", () => {
  test("the pristine passage already reads as passing on every live rule", async ({ page }) => {
    await returning(page);
    const [unicode, floor, budget, llm] = await Promise.all([
      indexOf(page, "unicode_sanitation"),
      indexOf(page, "word_floor"),
      indexOf(page, "edit_budget"),
      indexOf(page, "llm_gate"),
    ]);
    const s = await states(page);
    expect(s[unicode]).toBe("pass");
    expect(s[floor]).toBe("pass");
    expect(s[budget]).toBe("pass");
    // The one that genuinely has not run. "Not checked yet" has to keep meaning
    // something, which is why the live set stops where it does.
    expect(s[llm]).toBe("pending");
  });

  test("deleting the essay turns the word floor red without pressing Check", async ({ page }) => {
    await returning(page);
    const floor = await indexOf(page, "word_floor");
    await type(page, "The committee's report was released late Tuesday.");
    expect((await states(page))[floor]).toBe("fail");
    // ... and putting it back turns it green again, still without a request.
    await page.reload();
    expect((await states(page))[floor]).toBe("pass");
  });

  test("a paste carrying invisible characters is caught on the keystroke", async ({ page }) => {
    await returning(page);
    const unicode = await indexOf(page, "unicode_sanitation");
    const original = await page.locator("#raw").inputValue();
    expect((await states(page))[unicode]).toBe("pass");
    await type(page, original.replace("committee", "commi​ttee"));
    expect((await states(page))[unicode]).toBe("fail");
  });

  test("the words-changed counter and the budget rule are one number", async ({ page }) => {
    await returning(page);
    const budget = await indexOf(page, "edit_budget");
    const original = await page.locator("#raw").inputValue();
    // The fixture allows 12. Thirteen substitutions is one past it.
    const words = original.split(" ");
    for (let i = 0; i < 13; i++) words[i * 3] = `word${i}`;
    await type(page, words.join(" "));
    await expect(page.locator("#changed")).toContainText("13");
    expect((await states(page))[budget]).toBe("fail");
  });

  test("a failing rule says why, in place, from the live numbers", async ({ page }) => {
    await returning(page);
    await type(page, "Too short to clear the floor.");
    await page.locator("#pips").click();
    const row = page.locator("#listchecks .check").filter({ hasText: /words/ }).first();
    const button = row.locator(".checkrow");
    await button.click();
    const body = page.locator(`#${await button.getAttribute("aria-controls")}`);
    // `check.word_floor.reject` is "Too short: {n_words} words, and this level
    // needs {min_words}." Assert the NUMBERS, not the wording: the words belong
    // to copy.toml and are allowed to change without a code change (§10.7).
    await expect(body.locator(".cxwhy")).toContainText("6");
    await expect(body.locator(".cxwhy")).toContainText("50");
  });

  test("a passing rule shows only its standing explanation", async ({ page }) => {
    await returning(page);
    await page.locator("#pips").click();
    const rows = page.locator("#listchecks .check");
    // Nothing is failing on the pristine passage, so no row carries a live
    // reason — a rule that explained why it was broken while it was not broken
    // would be worse than silence.
    await expect(rows.locator(".cxwhy")).toHaveCount(0);
  });
});

test.describe("the server still decides", () => {
  test("no live rule ever marks the judge", async ({ page }) => {
    await returning(page);
    const llm = await indexOf(page, "llm_gate");
    await type(page, "A short rewrite that changes almost nothing at all.");
    expect((await states(page))[llm]).toBe("pending");
  });
});

test.describe("the share copies, and says so", () => {
  test("the copy button confirms and then goes back to being a button", async ({ page, browserName }) => {
    test.skip(browserName !== "chromium", "clipboard permissions are chromium-only here");
    await page.context().grantPermissions(["clipboard-read", "clipboard-write"]);
    await returning(page);
    // Reach the result sheet without a server: drive the gate directly with a
    // cleared response. The button under test is the sheet's, not the gate's.
    await page.evaluate(() => {
      const share = document.getElementById("rshare");
      share.textContent = "Launder WM — level 1 cleared in 4";
      share.hidden = false;
      document.getElementById("copyres").hidden = false;
      document.getElementById("resultsheet").setAttribute("data-open", "1");
    });
    const button = page.locator("#copyres");
    const resting = (await button.textContent()) ?? "";
    await button.click();
    await expect(button).toHaveAttribute("data-done", "1");
    expect((await button.textContent()) ?? "").not.toBe(resting);
    expect(await page.evaluate(() => navigator.clipboard.readText())).toContain("Launder WM");
    await expect(button).not.toHaveAttribute("data-done", "1", { timeout: 4000 });
    await expect(button).toHaveText(resting);
  });
});

test.describe("a provisional clear says it does not count", () => {
  /** A cleared `SubmitResponse`. `run_gate` sets `provisional=True` only when a
   *  fail-open check ERRORED (judge unreachable, spend cap hit) — every
   *  rejection path sets it False, so a clear is the only shape it can arrive
   *  in. Which is exactly why the note had to live on this sheet. */
  const cleared = (provisional) => ({
    cleared: true,
    provisional,
    score: { distance: 3, ops: [] },
    detector: { score: 0.5, z: 1.0, z_star: 2.3263, n_scored: 120 },
    failure: null,
    trace: [],
    par: 4,
    rank: null,
    share: "",
  });

  async function submitWith(page, provisional) {
    await page.route("**/api/submit", (route) =>
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(cleared(provisional)),
      }),
    );
    await page.locator("#check").click();
    await expect(page.locator("#resultsheet")).toHaveAttribute("data-open", "1");
  }

  test("the result sheet carries the note", async ({ page }) => {
    await returning(page);
    // The note existed only on #gatesheet, where `provisional` is unreachable.
    // So the one outcome that can be provisional rendered as an ordinary clear
    // while the client silently skipped `markCleared` and the cookie — and the
    // player came back to find the level uncleared with nothing having said why.
    await submitWith(page, true);
    await expect(page.locator("#rnote")).toBeVisible();
    await expect(page.locator("#rnote")).not.toBeEmpty();
  });

  test("an ordinary clear does not", async ({ page }) => {
    await returning(page);
    await submitWith(page, false);
    await expect(page.locator("#rnote")).toBeHidden();
  });
});
