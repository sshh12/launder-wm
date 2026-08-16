/**
 * Campaign persistence (§7).
 *
 * Two things here can end the game rather than degrade it, and both get most
 * of the attention below:
 *
 *   1. THE RELOAD LOOP. `GET /` picks the level from the `launder_level`
 *      cookie. When the cookie cannot be written, the reload lands back on the
 *      level we tried to leave and asks for the same reload again. The
 *      sessionStorage guard is the only thing between that and a page that
 *      thrashes forever, so it is tested against a cookie jar that silently
 *      swallows writes — the realistic failure, not a thrown error.
 *   2. A CORRUPT RECORD. localStorage is a string the user can edit and a disk
 *      can truncate. Nothing in it may be trusted enough to throw on.
 */

import { describe, expect, it } from "vitest";

import {
  PROGRESS_KEY,
  SYNC_GUARD_KEY,
  emptyProgress,
  markCleared,
  mergeCleared,
  perLevelLine,
  readProgress,
  reconcile,
  totalChanged,
  unlockedLevel,
  writeLevelCookie,
  writeProgress,
  type CookieJar,
  type Nav,
} from "../src/game/progress";

/** The smallest thing that satisfies the Storage surface this module uses. */
function memStorage(seed: Record<string, string> = {}): Storage {
  const map = new Map(Object.entries(seed));
  return {
    get length() {
      return map.size;
    },
    clear: () => map.clear(),
    getItem: (k: string) => map.get(k) ?? null,
    key: (i: number) => Array.from(map.keys())[i] ?? null,
    removeItem: (k: string) => void map.delete(k),
    setItem: (k: string, v: string) => void map.set(k, v),
  };
}

/** Private mode: throws on ACCESS, not only on write. */
function hostileStorage(): Storage {
  const boom = (): never => {
    throw new Error("SecurityError");
  };
  return {
    get length(): number {
      return boom();
    },
    clear: boom,
    getItem: boom,
    key: boom,
    removeItem: boom,
    setItem: boom,
  };
}

/** A jar that keeps what it is given, like `document.cookie` in the happy case
 *  (close enough: this module only ever writes one cookie). */
function jar(): CookieJar {
  return { cookie: "" };
}

/** Third-party cookie blocking: the assignment succeeds and does nothing. */
function deafJar(): CookieJar {
  return {
    get cookie() {
      return "";
    },
    set cookie(_v: string) {
      /* silently dropped */
    },
  };
}

function navTo(search: string): Nav & { visited: string[] } {
  const visited: string[] = [];
  return { search, replace: (url: string) => visited.push(url), visited };
}

describe("the record", () => {
  it("starts empty and round-trips", () => {
    const s = memStorage();
    expect(readProgress(s)).toEqual(emptyProgress());
    writeProgress(s, { v: 1, cleared: [1, 2], best: { "1": 4, "2": 6 } });
    expect(readProgress(s)).toEqual({ v: 1, cleared: [1, 2], best: { "1": 4, "2": 6 } });
  });

  it("keeps the LOWER distance when a level is replayed", () => {
    const s = memStorage();
    markCleared(s, 3, 9);
    expect(readProgress(s).best["3"]).toBe(9);
    markCleared(s, 3, 5);
    expect(readProgress(s).best["3"]).toBe(5);
    // a worse replay must not overwrite a better score
    markCleared(s, 3, 11);
    expect(readProgress(s).best["3"]).toBe(5);
    expect(readProgress(s).cleared).toEqual([3]);
  });

  it("keeps `cleared` ascending and free of duplicates", () => {
    const s = memStorage();
    markCleared(s, 4, 1);
    markCleared(s, 2, 1);
    markCleared(s, 4, 1);
    expect(readProgress(s).cleared).toEqual([2, 4]);
  });

  it("reads a corrupt, truncated or hostile record as 'nothing cleared'", () => {
    for (const raw of [
      "",
      "{",
      "null",
      "[1,2,3]",
      '"nope"',
      '{"v":2,"cleared":[1]}',
      '{"v":1,"cleared":"1,2"}',
      '{"v":1,"cleared":[0,-3,1.5,"2",1],"best":{"x":1,"2":"nope","3":-4,"4":7}}',
    ]) {
      const p = readProgress(memStorage({ [PROGRESS_KEY]: raw }));
      expect(p.v, raw).toBe(1);
      expect(Array.isArray(p.cleared), raw).toBe(true);
      for (const n of p.cleared) expect(Number.isInteger(n) && n >= 1, raw).toBe(true);
      for (const v of Object.values(p.best)) expect(v >= 0, raw).toBe(true);
    }
    // the last case above keeps only what was actually well-formed
    const mixed = readProgress(
      memStorage({
        [PROGRESS_KEY]: '{"v":1,"cleared":[0,-3,1.5,"2",1],"best":{"x":1,"2":"nope","3":-4,"4":7}}',
      }),
    );
    expect(mixed.cleared).toEqual([1]);
    expect(mixed.best).toEqual({ "4": 7 });
  });

  it("survives storage that throws on access, and storage that is absent", () => {
    for (const s of [hostileStorage(), null]) {
      expect(readProgress(s)).toEqual(emptyProgress());
      expect(() => writeProgress(s, emptyProgress())).not.toThrow();
      expect(() => markCleared(s, 2, 3)).not.toThrow();
      expect(() => mergeCleared(s, [1, 2])).not.toThrow();
    }
  });

  it("unions the server's list without losing the local one", () => {
    const s = memStorage();
    markCleared(s, 1, 4);
    markCleared(s, 2, 6);
    // the server knows about a level cleared on another device, and does not
    // know about one cleared here before a session id existed
    const merged = mergeCleared(s, [2, 3, 4]);
    expect(merged.cleared).toEqual([1, 2, 3, 4]);
    // it carries no distances, so those levels stay `-` in the share
    expect(merged.best).toEqual({ "1": 4, "2": 6 });
  });
});

describe("unlockedLevel", () => {
  it("is min(highest cleared + 1, level_count)", () => {
    expect(unlockedLevel(emptyProgress(), 15)).toBe(1);
    expect(unlockedLevel({ v: 1, cleared: [1, 2, 3], best: {} }, 15)).toBe(4);
    // a gap does not hold the player back: the highest clear wins
    expect(unlockedLevel({ v: 1, cleared: [1, 7], best: {} }, 15)).toBe(8);
    // the campaign has an end
    expect(unlockedLevel({ v: 1, cleared: [15], best: {} }, 15)).toBe(15);
  });
});

describe("writeLevelCookie", () => {
  it("writes the whole attribute set, so it outlives the tab and the session", () => {
    const j = jar();
    writeLevelCookie(j, 7);
    expect(j.cookie).toBe("launder_level=7; path=/; max-age=31536000; samesite=lax");
  });

  it("does not throw when the jar does", () => {
    const angry: CookieJar = {
      get cookie() {
        return "";
      },
      set cookie(_v: string) {
        throw new Error("sandboxed frame");
      },
    };
    expect(() => writeLevelCookie(angry, 2)).not.toThrow();
  });
});

describe("reconcile", () => {
  const cleared = (n: number): Storage => {
    const s = memStorage();
    for (let i = 1; i <= n; i++) markCleared(s, i, 4);
    return s;
  };

  it("does nothing at all when ?level= is pinned", () => {
    // The hidden test escape hatch renders a level regardless of progress. A
    // reload here would throw the query parameter away and bounce the tester
    // back to their real position.
    const j = jar();
    const nav = navTo("?level=9");
    expect(
      reconcile({
        storage: cleared(3),
        session: memStorage(),
        jar: j,
        nav,
        levelN: 9,
        levelCount: 15,
      }),
    ).toBe("pinned");
    expect(j.cookie).toBe("");
    expect(nav.visited).toEqual([]);
  });

  it("stays put when the server already rendered the unlocked level", () => {
    const j = jar();
    const nav = navTo("");
    expect(
      reconcile({
        storage: cleared(3),
        session: memStorage(),
        jar: j,
        nav,
        levelN: 4,
        levelCount: 15,
      }),
    ).toBe("aligned");
    expect(j.cookie).toBe("");
    expect(nav.visited).toEqual([]);
  });

  it("writes the cookie and reloads once when the page is on the wrong level", () => {
    const j = jar();
    const nav = navTo("");
    const session = memStorage();
    expect(
      reconcile({ storage: cleared(3), session, jar: j, nav, levelN: 1, levelCount: 15 }),
    ).toBe("reloaded");
    expect(j.cookie).toContain("launder_level=4");
    expect(nav.visited).toEqual(["/"]);
    expect(session.getItem(SYNC_GUARD_KEY)).toBe("1");
  });

  it("NEVER reloads twice in one tab, even when the cookie is silently dropped", () => {
    // The loop: cookie blocked -> the server keeps rendering level 1 -> we keep
    // asking for the same reload. The guard is spent before the first reload is
    // issued, so the second call can only write the cookie and give up.
    const storage = cleared(3);
    const session = memStorage();
    const j = deafJar();
    const nav = navTo("");
    const first = reconcile({ storage, session, jar: j, nav, levelN: 1, levelCount: 15 });
    const second = reconcile({ storage, session, jar: j, nav, levelN: 1, levelCount: 15 });
    expect([first, second]).toEqual(["reloaded", "guarded"]);
    expect(nav.visited).toEqual(["/"]);
  });

  it("refuses to reload at all when sessionStorage is unavailable", () => {
    // No budget to spend means no reload: playing the wrong level beats
    // watching a page thrash, and the cookie still gets written for next time.
    //
    // BOTH shapes of "unavailable" are here on purpose. `safeSession()` returns
    // null when sessionStorage is simply ABSENT (an embedded WebView) and a
    // throwing object never reaches the caller at all — so the null case is the
    // one the try/catch cannot see. Guarding it with `o.session?.getItem(...)
    // === "1"` made `undefined === "1"` read as "budget unspent", the optional
    // setItem did nothing, nothing threw, and every reload landed on an
    // identical document that asked for the same reload: a page that reloads
    // forever.
    for (const session of [hostileStorage(), null]) {
      const j = jar();
      const nav = navTo("");
      expect(
        reconcile({
          storage: cleared(2),
          session,
          jar: j,
          nav,
          levelN: 1,
          levelCount: 15,
        }),
      ).toBe("guarded");
      expect(j.cookie).toContain("launder_level=3");
      expect(nav.visited).toEqual([]);
    }
  });

  it("sends a returning player forward, and a reset player back", () => {
    const nav = navTo("");
    const j = jar();
    // cleared 1..5 but the server rendered 1 (cookie lost): go forward to 6
    reconcile({ storage: cleared(5), session: memStorage(), jar: j, nav, levelN: 1, levelCount: 15 });
    expect(j.cookie).toContain("launder_level=6");
    // nothing cleared but the cookie says 9 (site data wiped): go back to 1
    const j2 = jar();
    reconcile({
      storage: memStorage(),
      session: memStorage(),
      jar: j2,
      nav: navTo(""),
      levelN: 9,
      levelCount: 15,
    });
    expect(j2.cookie).toContain("launder_level=1");
  });
});

describe("the end-of-campaign share", () => {
  const full = (): Storage => {
    const s = memStorage();
    const scores = [4, 6, 5, 4, 7, 5, 6, 4, 8, 6, 5, 7, 4, 6, 4];
    scores.forEach((d, i) => markCleared(s, i + 1, d));
    return s;
  };

  it("groups fifteen distances in fives", () => {
    expect(perLevelLine(readProgress(full()), 15)).toBe("4 6 5 4 7 · 5 6 4 8 6 · 5 7 4 6 4");
    expect(totalChanged(readProgress(full()), 15)).toBe(81);
  });

  it("renders a level with no recorded distance as `-`, never as 0", () => {
    // A zero would claim a perfect level. This is the level the server knew
    // about but the local record does not have a distance for.
    const s = memStorage();
    markCleared(s, 1, 4);
    mergeCleared(s, [2, 3]);
    const p = readProgress(s);
    expect(perLevelLine(p, 5)).toBe("4 - - - -");
    expect(totalChanged(p, 5)).toBe(4);
  });
});
