// @ts-check
/**
 * MIRROR ALIGNMENT — the browser half. TECH_PLAN.md §10.3, §14.1 risk 4.
 *
 * Written in JavaScript on purpose: `web/tsconfig.json` type-checks everything
 * under `tests/`, and `@playwright/test` is not in this project's
 * devDependencies yet. A `.spec.ts` here would break `npm run build` on a
 * checkout that has not installed Playwright. `.mjs` keeps the harness present
 * and runnable without holding the build hostage. Convert it to `.ts` the day
 * `@playwright/test` lands in package.json.
 *
 * THE ASSERTION. For every token span in the mirror we take a point inside its
 * first client rect and ask the DOCUMENT which character of the TEXTAREA sits
 * under that point (`caretPositionFromPoint`, or `caretRangeFromPoint` on
 * WebKit). If the mirror and the textarea agree on where glyphs go, the answer
 * is the character index that span covers. If the mirror has drifted — one
 * ligature, one rounding difference in line-height, one font-boosted block —
 * the answer is a different character, and the drift accumulates towards the
 * end of the passage, which is exactly where this test looks hardest.
 *
 * The second assertion is cheaper and catches the same class earlier: every
 * property §10.3 says must match is compared as a COMPUTED style on both
 * elements, in the real engine, after the cascade.
 */

import { expect, test } from "@playwright/test";

/** Properties that must match exactly on the mirror and the textarea (§10.3). */
const MUST_MATCH = [
  "boxSizing",
  "width",
  "marginTop",
  "marginRight",
  "marginBottom",
  "marginLeft",
  "borderTopWidth",
  "borderRightWidth",
  "borderBottomWidth",
  "borderLeftWidth",
  "paddingTop",
  "paddingRight",
  "paddingBottom",
  "paddingLeft",
  "fontFamily",
  "fontSize",
  "fontWeight",
  "fontStyle",
  "lineHeight",
  "letterSpacing",
  "wordSpacing",
  "textIndent",
  "textTransform",
  "textRendering",
  "fontKerning",
  "fontVariantLigatures",
  "fontFeatureSettings",
  "webkitFontSmoothing",
  "whiteSpace",
  "overflowWrap",
  "wordBreak",
  "hyphens",
  "tabSize",
  "direction",
  "unicodeBidi",
];

/** Suppress the first-land primer: it sets `inert` on the app while open. */
async function open(page) {
  await page.addInitScript(() => {
    try {
      localStorage.setItem("launderwm.primer.v1", "1");
    } catch {
      /* private mode */
    }
  });
  await page.goto("/");
  await page.waitForSelector("#mirror .tk");
}

test.describe("the heat mirror and the textarea are the same box", () => {
  test("every property §10.3 requires to match, matches in the real engine", async ({ page }) => {
    await open(page);
    const diffs = await page.evaluate((props) => {
      const mirror = document.getElementById("mirror");
      const raw = document.getElementById("raw");
      const a = getComputedStyle(mirror);
      const b = getComputedStyle(raw);
      const out = [];
      for (const p of props) {
        if (a[p] !== b[p]) out.push(`${p}: mirror=${a[p]} textarea=${b[p]}`);
      }
      return out;
    }, MUST_MATCH);
    expect(diffs, `computed style drift:\n${diffs.join("\n")}`).toEqual([]);
  });

  test("line-height is a whole number of pixels", async ({ page }) => {
    await open(page);
    const lh = await page.evaluate(() => {
      const v = getComputedStyle(document.getElementById("mirror")).lineHeight;
      return Number.parseFloat(v);
    });
    expect(Number.isInteger(lh)).toBe(true);
  });

  test("the rendered font-size is at least 16px, so iOS does not zoom on focus", async ({
    page,
  }) => {
    await open(page);
    const size = await page.evaluate(() => {
      const raw = document.getElementById("raw");
      const declared = Number.parseFloat(getComputedStyle(raw).fontSize);
      // The check iOS performs is on the RENDERED size, so a transform on any
      // ancestor changes the answer. Walk up and multiply.
      let scale = 1;
      for (let el = raw; el; el = el.parentElement) {
        const t = getComputedStyle(el).transform;
        if (t && t !== "none") {
          const m = new DOMMatrixReadOnly(t);
          scale *= m.a;
        }
      }
      return declared * scale;
    });
    expect(size).toBeGreaterThanOrEqual(16);
  });

  test("the mirror defines the height and the textarea never scrolls internally", async ({
    page,
  }) => {
    await open(page);
    const box = await page.evaluate(() => {
      const mirror = document.getElementById("mirror");
      const raw = document.getElementById("raw");
      return {
        mirrorHeight: mirror.getBoundingClientRect().height,
        rawHeight: raw.getBoundingClientRect().height,
        scrollHeight: raw.scrollHeight,
        clientHeight: raw.clientHeight,
        overflow: getComputedStyle(raw).overflow,
      };
    });
    expect(Math.abs(box.mirrorHeight - box.rawHeight)).toBeLessThanOrEqual(1);
    expect(box.overflow).toBe("hidden");
    // The page scrolls; the box does not. One decision, and it removes every
    // scroll-sync bug listed in §10.3.
    expect(box.scrollHeight - box.clientHeight).toBeLessThanOrEqual(1);
  });

  test("the page never scrolls horizontally at 390px", async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await open(page);
    const overflow = await page.evaluate(
      () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
    );
    expect(overflow).toBeLessThanOrEqual(0);
  });

  /**
   * THE REAL ONE. Every token span, its own glyphs, the textarea's opinion.
   */
  test("every token span sits over the characters it describes", async ({ page }) => {
    await open(page);

    // Probe in viewport-sized passes. A point below the fold is not hit-tested
    // — caretPositionFromPoint answers about the visual viewport only — so on a
    // 390x844 phone the passage has to be walked, not sampled once.
    const totals = { spans: 0, probed: 0, unsupported: 0, failures: [] };
    const pageHeight = await page.evaluate(() => document.documentElement.scrollHeight);
    const viewport = page.viewportSize()?.height ?? 800;
    for (let top = 0; top < pageHeight; top += Math.floor(viewport * 0.55)) {
      await page.evaluate((y) => window.scrollTo(0, y), top);
      await page.waitForTimeout(30);
      const pass = await page.evaluate(() => {
        const mirror = document.getElementById("mirror");
        const raw = document.getElementById("raw");

        // Char offset of each token span, derived from the mirror's own
        // content — the mirror holds every character exactly once, in order,
        // so a running sum over childNodes IS the offset table.
        const spans = [];
        let offset = 0;
        for (const node of mirror.childNodes) {
          const text = node.nodeName === "BR" ? "" : (node.textContent ?? "");
          if (node.nodeType === 1 && node.classList?.contains("tk")) {
            spans.push({ el: node, s: offset, e: offset + text.length });
          }
          offset += text.length;
        }

        const value = raw.value;
        const charAtPoint = (x, y) => {
          if (document.caretPositionFromPoint) {
            const pos = document.caretPositionFromPoint(x, y);
            return pos ? pos.offset : null;
          }
          if (document.caretRangeFromPoint) {
            const range = document.caretRangeFromPoint(x, y);
            return range ? range.startOffset : null;
          }
          return null;
        };

        const failures = [];
        const seen = [];
        let unsupported = 0;
        const h = window.innerHeight;
        const w = window.innerWidth;
        for (const span of spans) {
          const rects = span.el.getClientRects();
          if (rects.length === 0) continue;
          const r = rects[0];
          // A point one third into the first glyph box, vertically centred.
          const x = r.left + Math.min(r.width * 0.33, r.width - 1);
          const y = r.top + r.height / 2;
          // Only points the compositor will actually hit-test...
          if (y < 4 || y > h - 4 || x < 4 || x > w - 4) continue;
          // ...and only points where the TEXTAREA is the topmost element. The
          // rail is position: sticky and covers the passage as the page
          // scrolls; a caret query there answers about the rail, not the text.
          if (document.elementFromPoint(x, y) !== raw) continue;
          const got = charAtPoint(x, y);
          if (got === null) {
            unsupported += 1;
            continue;
          }
          seen.push(span.s);
          // One character of slack: hit-testing rounds, and a caret sitting on
          // a boundary is legitimately either side of it.
          if (got < span.s - 1 || got > span.e + 1) {
            failures.push({
              expected: [span.s, span.e],
              got,
              text: value.slice(span.s, span.e),
              around: value.slice(Math.max(0, got - 12), got + 12),
            });
          }
        }
        return { spans: spans.length, seen, unsupported, failures };
      });
      totals.spans = pass.spans;
      totals.probed += pass.seen.length;
      totals.unsupported += pass.unsupported;
      totals.failures.push(...pass.failures);
    }
    await page.evaluate(() => window.scrollTo(0, 0));

    test.info().annotations.push({
      type: "alignment",
      description: `${totals.probed} probes over ${totals.spans} spans, ${totals.failures.length} misaligned, ${totals.unsupported} unsupported`,
    });
    test.skip(
      totals.probed === 0,
      "this engine exposes neither caretPositionFromPoint nor caretRangeFromPoint",
    );
    expect(totals.probed).toBeGreaterThan(20);
    expect(
      totals.failures,
      `misaligned spans:
${JSON.stringify(totals.failures.slice(0, 5), null, 2)}`,
    ).toEqual([]);
  });

  test("stays aligned after the player types, including at the end of a long line", async ({
    page,
  }) => {
    await open(page);
    const raw = page.locator("#raw");
    await raw.click();
    // Insert in the middle, where a reflow moves every following line.
    await page.evaluate(() => {
      const el = document.getElementById("raw");
      el.setSelectionRange(120, 120);
    });
    await raw.pressSequentially(" laundered ", { delay: 15 });
    await page.waitForTimeout(300);
    const same = await page.evaluate(() => {
      const mirror = document.getElementById("mirror");
      const raw = document.getElementById("raw");
      let rendered = "";
      for (const node of mirror.childNodes) {
        if (node.nodeName === "BR") continue;
        rendered += node.textContent ?? "";
      }
      return {
        equal: rendered === raw.value,
        mirrorLen: rendered.length,
        rawLen: raw.value.length,
        heights: [
          mirror.getBoundingClientRect().height,
          raw.getBoundingClientRect().height,
        ],
      };
    });
    expect(same.equal, `mirror ${same.mirrorLen} chars, textarea ${same.rawLen}`).toBe(true);
    expect(Math.abs(same.heights[0] - same.heights[1])).toBeLessThanOrEqual(1);
  });
});
