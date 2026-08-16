/**
 * Campaign progress — the client's half of the 15-level campaign.
 *
 * Three stores, and NOT ONE of them is allowed to break the game when it is
 * unavailable:
 *
 *   localStorage["launderwm.progress.v1"]  what this browser has cleared and
 *       its best (lowest) distance per level. It is the record that survives
 *       the server not knowing who you are.
 *   cookie `launder_level`                 written by the CLIENT only. It
 *       exists so `GET /` can render the right level with zero API calls and
 *       no flash — the server never sets it and never trusts it for anything
 *       but which page to render.
 *   sessionStorage["launderwm.synced"]     the one-reload-per-tab guard.
 *
 * The reconciliation is the whole point, and it is also the one thing here
 * that can loop. The page the server rendered was chosen FROM THE COOKIE, so
 * when the cookie cannot be written — a privacy mode, third-party cookie
 * blocking, an embedded WebView, a sandboxed frame — the reload lands right
 * back on the level we just tried to leave and asks for the same reload again,
 * forever. That is why the sessionStorage flag is set BEFORE the reload is
 * issued rather than after it: at most one reload happens per tab, ever. A
 * browser that also drops sessionStorage cannot reload at all, which is the
 * correct failure — the player plays level 1 with a working game instead of
 * watching a page thrash.
 *
 * Every read and every write is wrapped, in exactly the way primer.ts's
 * `safeStorage` is: `localStorage` throws on ACCESS in some privacy modes and
 * not merely on write, `document.cookie` throws in a sandboxed frame, and both
 * can accept a write and silently discard it. A storage failure costs
 * persistence. It never costs play.
 */

export const PROGRESS_KEY = "launderwm.progress.v1";
export const LEVEL_COOKIE = "launder_level";
export const SYNC_GUARD_KEY = "launderwm.synced";

/** One year. The campaign has no expiry; the cookie should outlast a break. */
const COOKIE_MAX_AGE_S = 31536000;

/** `{"v":1,"cleared":[1,2],"best":{"1":4}}` — `best` is keyed by level number
 *  as a string because it is JSON, and JSON has no integer keys. */
export interface Progress {
  v: 1;
  /** ascending, no duplicates */
  cleared: number[];
  /** level number -> lowest distance ever recorded for it */
  best: Record<string, number>;
}

export function emptyProgress(): Progress {
  return { v: 1, cleared: [], best: {} };
}

/** sessionStorage throws on access in the same places localStorage does. */
export function safeSession(): Storage | null {
  try {
    return globalThis.sessionStorage ?? null;
  } catch {
    return null;
  }
}

function isLevel(n: unknown): n is number {
  return typeof n === "number" && Number.isInteger(n) && n >= 1;
}

/**
 * Read the record, tolerating anything. A record written by a future version,
 * truncated by a full disk, or hand-edited into nonsense reads as "nothing
 * cleared yet" — a corrupt record must cost the player their history, never
 * their ability to play. Unknown fields are dropped rather than carried, so a
 * malformed `best` cannot reach the share string.
 */
export function readProgress(storage: Storage | null): Progress {
  let raw: string | null = null;
  try {
    raw = storage?.getItem(PROGRESS_KEY) ?? null;
  } catch {
    return emptyProgress();
  }
  if (raw === null || raw === "") return emptyProgress();
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return emptyProgress();
  }
  if (typeof parsed !== "object" || parsed === null) return emptyProgress();
  const rec = parsed as Partial<Progress>;
  if (rec.v !== 1) return emptyProgress();
  const out = emptyProgress();
  if (Array.isArray(rec.cleared)) {
    out.cleared = Array.from(new Set(rec.cleared.filter(isLevel))).sort((a, b) => a - b);
  }
  if (typeof rec.best === "object" && rec.best !== null) {
    for (const [k, v] of Object.entries(rec.best)) {
      const n = Number.parseInt(k, 10);
      if (isLevel(n) && typeof v === "number" && Number.isFinite(v) && v >= 0) {
        out.best[String(n)] = v;
      }
    }
  }
  return out;
}

export function writeProgress(storage: Storage | null, p: Progress): void {
  try {
    storage?.setItem(PROGRESS_KEY, JSON.stringify(p));
  } catch {
    /* private mode, or the quota is full: progress this session only */
  }
}

/**
 * Record a clear, KEEPING THE LOWER DISTANCE — the same rule the server's
 * `progress` upsert uses. Replaying a level you already beat must never make
 * your recorded score worse, and the end-of-campaign share is built from this
 * map.
 */
export function markCleared(
  storage: Storage | null,
  levelN: number,
  distance: number,
): Progress {
  const p = readProgress(storage);
  if (!isLevel(levelN)) return p;
  if (!p.cleared.includes(levelN)) {
    p.cleared.push(levelN);
    p.cleared.sort((a, b) => a - b);
  }
  const key = String(levelN);
  const previous = p.best[key];
  if (Number.isFinite(distance) && distance >= 0) {
    p.best[key] = previous === undefined ? distance : Math.min(previous, distance);
  }
  writeProgress(storage, p);
  return p;
}

/**
 * Union the server's `cleared` list into the local record. The server is the
 * authority on nothing here — it is a SECOND witness. A player who cleared
 * levels on their phone and opened the game on a laptop has a local record
 * that is missing them, and a player whose session id was never sent has a
 * server record that is missing everything; taking the union is the only
 * merge that loses neither.
 */
export function mergeCleared(storage: Storage | null, levels: readonly number[]): Progress {
  const p = readProgress(storage);
  let changed = false;
  for (const n of levels) {
    if (!isLevel(n) || p.cleared.includes(n)) continue;
    p.cleared.push(n);
    changed = true;
  }
  if (changed) {
    p.cleared.sort((a, b) => a - b);
    writeProgress(storage, p);
  }
  return p;
}

/** `min(max(cleared) + 1, level_count)`, the same arithmetic as the server's
 *  `GET /api/progress`. Nothing cleared unlocks level 1. */
export function unlockedLevel(p: Progress, levelCount: number): number {
  let highest = 0;
  for (const n of p.cleared) if (n > highest) highest = n;
  return Math.min(highest + 1, Math.max(1, levelCount));
}

/** `document` satisfies this structurally; a test can pass `{ cookie: "" }`. */
export interface CookieJar {
  cookie: string;
}

export function writeLevelCookie(jar: CookieJar, levelN: number): void {
  try {
    jar.cookie = `${LEVEL_COOKIE}=${String(levelN)}; path=/; max-age=${String(
      COOKIE_MAX_AGE_S,
    )}; samesite=lax`;
  } catch {
    // A sandboxed frame throws on the assignment itself. Nothing to fall back
    // to: the reconciliation's reload guard is what stops this becoming a loop.
  }
}

/** The navigation surface reconciliation needs, so it can be driven in a test
 *  without a real Location (whose `replace` cannot be called detached). */
export interface Nav {
  /** the current query string, including the leading `?` when non-empty */
  search: string;
  /** navigate WITHOUT adding a history entry — the level we are leaving must
   *  not be sitting behind the back button */
  replace: (url: string) => void;
}

export interface ReconcileOptions {
  storage: Storage | null;
  session: Storage | null;
  jar: CookieJar;
  nav: Nav;
  /** the level the server actually rendered */
  levelN: number;
  levelCount: number;
}

export type ReconcileResult =
  /** `?level=` is pinned: the hidden test escape hatch renders that level
   *  regardless of progress, so we neither write the cookie nor reload. */
  | "pinned"
  /** the server already rendered the unlocked level */
  | "aligned"
  /** cookie written, reload issued */
  | "reloaded"
  /** cookie written, but no reload: this tab has already spent its one, or it
   *  has nowhere to record spending one and so may not take it */
  | "guarded";

/**
 * Point the browser at the level the player has actually unlocked.
 *
 * Called twice: once at boot from the localStorage record alone, and again
 * after `GET /api/progress` has been unioned in. Both calls share the ONE
 * reload budget, which is why the guard lives in sessionStorage and not in a
 * module-level variable — a variable would be reset by the very reload it is
 * meant to prevent.
 */
export function reconcile(o: ReconcileOptions): ReconcileResult {
  // `?level=` wins over everything, including over us: it is how the hidden
  // tests reach a level without playing to it, and a reload here would throw
  // the query parameter away and bounce them back to their real position.
  if (new URLSearchParams(o.nav.search).has("level")) return "pinned";

  const target = unlockedLevel(readProgress(o.storage), o.levelCount);
  if (target === o.levelN) return "aligned";

  // Written even when the reload cannot happen: the next natural page load
  // then lands on the right level with no reload at all.
  writeLevelCookie(o.jar, target);

  // The budget may only be spent if it can be RECORDED, which is why this is
  // an explicit null check and not `o.session?.getItem(...) === "1"`. On a
  // browser with no sessionStorage — an embedded WebView, a sandboxed frame —
  // `safeSession()` hands us null, that expression is `undefined === "1"`, and
  // the whole guard inverts: the budget reads as unspent, the optional
  // `setItem` is a silent no-op, nothing throws so the catch never runs, and we
  // reload into a document that is in the identical state and asks for the
  // identical reload. That is the unbounded loop the guard exists to prevent.
  // So a guard that cannot be written counts as ALREADY SPENT: a missing
  // reload costs a returning player one extra click, an endless one costs them
  // the game.
  let spent = true;
  try {
    const session = o.session;
    if (session !== null) {
      spent = session.getItem(SYNC_GUARD_KEY) === "1";
      // Set FIRST. If the reload commits before the flag is stored, the fresh
      // document reads an unset flag and reloads again, which is the loop.
      if (!spent) session.setItem(SYNC_GUARD_KEY, "1");
    }
  } catch {
    // sessionStorage that throws on access is the same situation as one that
    // is absent: no budget to spend, so refuse the reload.
    spent = true;
  }
  if (spent) return "guarded";

  o.nav.replace("/");
  return "reloaded";
}

/**
 * The per-level line of the end-of-campaign share: every level's best distance
 * in level order, in groups of five, so fifteen numbers read as a shape rather
 * than as a run-on. A level with no recorded distance renders `-` — the record
 * can be missing a level the SERVER knows about (cleared before the session id
 * existed), and printing a zero there would claim a perfect score.
 */
export function perLevelLine(p: Progress, levelCount: number): string {
  const cells: string[] = [];
  for (let n = 1; n <= levelCount; n++) {
    const d = p.best[String(n)];
    cells.push(d === undefined ? "-" : String(d));
  }
  const groups: string[] = [];
  for (let i = 0; i < cells.length; i += 5) groups.push(cells.slice(i, i + 5).join(" "));
  return groups.join(" · ");
}

/** Sum of the recorded distances. A missing level contributes nothing rather
 *  than a guess. */
export function totalChanged(p: Progress, levelCount: number): number {
  let total = 0;
  for (let n = 1; n <= levelCount; n++) {
    const d = p.best[String(n)];
    if (d !== undefined) total += d;
  }
  return total;
}
