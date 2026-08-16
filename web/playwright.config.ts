import { defineConfig, devices } from "@playwright/test";

/**
 * The browser half of the mirror-alignment harness — TECH_PLAN.md §14.1 risk 4.
 *
 * `tests/mirror.test.ts` proves the STRUCTURE (every character exactly once,
 * spans reused rather than replaced, the CSS pair rule). It cannot prove the
 * thing that actually bites, which is whether two different layout paths — a
 * form control's internal line-box layout and ordinary block layout — put the
 * same glyph at the same pixel. That needs a layout engine, so it needs this.
 *
 * WHAT THIS CANNOT DO, AND THE USER SHOULD KNOW IT: the three failure modes
 * §10.3 calls the killers are device failures, not engine failures. Android
 * Chrome's font boosting only happens on a real Android phone with a real
 * system font scale; iOS focus-zoom only happens on a real iPhone; and the
 * font stack resolves differently on every OS, which is the whole reason
 * kerning and ligatures are forced off. Emulated `Mobile Safari` here is
 * desktop WebKit with a small viewport and a touch flag. It catches CSS
 * regressions. It does not catch device bugs.
 *
 * To run it for real, on hardware, from this same file:
 *   npm i -D @playwright/test && npx playwright install
 *   npx playwright test                       # local engines
 *   npx playwright test --project="Mobile Safari"
 *   # then, on a phone on the same LAN, with the dev server bound wide:
 *   npm run dev -- --host    → open http://<lan-ip>:5173 on the device
 *   # and read the numbers the harness prints: tests/e2e/mirror-alignment.spec.ts
 *   # exposes window.__alignmentReport() for exactly that manual run.
 */
export default defineConfig({
  testDir: "./tests/e2e",
  testMatch: /.*\.spec\.(ts|mjs)/,
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? "github" : "list",
  timeout: 30_000,
  expect: { timeout: 5_000 },

  use: {
    baseURL: "http://localhost:4321",
    trace: "on-first-retry",
    // The passage box must never be inside a transformed ancestor and the page
    // must never need horizontal scroll; a non-integer DPR is where subpixel
    // rounding bugs surface, so one project runs at 2.
    deviceScaleFactor: 1,
  },

  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
    {
      name: "webkit",
      use: { ...devices["Desktop Safari"] },
    },
    {
      name: "firefox",
      use: { ...devices["Desktop Firefox"] },
    },
    {
      // The design target: iPhone 390x844 with the keyboard open -> ~390x390
      // usable (§10.5). Emulated, which is not the same as an iPhone.
      name: "Mobile Safari",
      use: { ...devices["iPhone 13"] },
    },
    {
      name: "Mobile Chrome",
      use: { ...devices["Pixel 7"] },
    },
    {
      name: "reduced-motion",
      use: { ...devices["Desktop Chrome"], reducedMotion: "reduce" },
    },
  ],

  webServer: {
    command: "npm run dev -- --port 4321 --strictPort",
    url: "http://localhost:4321",
    reuseExistingServer: !process.env.CI,
    timeout: 60_000,
  },
});
