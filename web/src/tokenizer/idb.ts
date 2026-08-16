/**
 * Caching layer 3 (§5.3): persist the DERIVED merge pairs to IndexedDB.
 *
 * Measured cold-start cost on desktop V8 is dominated by deriving the 514,906
 * merges from the vocab (~350 ms desktop, ~1.4-2.1 s on a mid-range phone).
 * The derivation output is a pure function of the vocab strings, so it is
 * cacheable under the blob's content hash. Later cold loads read a flat
 * `Int32Array` back and skip derivation entirely: the difference between ~1.5 s
 * of jank on every cold load and 1.5 s once, ever.
 *
 * Stored form: the merge list as INDICES into the candidate list is not enough
 * (rebuilding the candidate list is the expensive part). We store the merge
 * pairs as two vocab IDs each — `514,906 x 2 x 4 = 4,119,248 B` — which is
 * exactly the §5.3 figure, and rehydration is an array lookup per entry.
 *
 * ---------------------------------------------------------------------------
 * EVERY function here degrades silently.
 * ---------------------------------------------------------------------------
 * §5.4 item 5: "If the worker fails to build (OOM on a low-end device, storage
 * blocked in private mode), stay in SERVER mode forever and never retry in a
 * loop." Storage being unavailable must cost latency and nothing else, so a
 * failed read returns `null` and a failed write returns `false`. Neither ever
 * throws, and neither is ever retried. Safari in private mode, Firefox with
 * `dom.indexedDB.enabled=false`, a full quota, and a corrupt object store all
 * land in the same quiet path.
 */

const DB_NAME = "launder-tokenizer";
const DB_VERSION = 1;
const STORE = "merges";

export interface CachedMerges {
  /** flat `[leftId, rightId, ...]`, merge order preserved */
  readonly pairs: Int32Array;
  /** number of merges (`pairs.length / 2`) */
  readonly count: number;
}

function openDB(): Promise<IDBDatabase | null> {
  return new Promise((resolve) => {
    let idb: IDBFactory | undefined;
    try {
      idb = globalThis.indexedDB;
    } catch {
      // Firefox throws on `indexedDB` access when storage is blocked entirely.
      resolve(null);
      return;
    }
    if (!idb) {
      resolve(null);
      return;
    }
    let req: IDBOpenDBRequest;
    try {
      req = idb.open(DB_NAME, DB_VERSION);
    } catch {
      resolve(null);
      return;
    }
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) db.createObjectStore(STORE);
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => resolve(null);
    req.onblocked = () => resolve(null);
  });
}

/**
 * Read the cached merge pairs for a content hash.
 * Returns `null` on any failure — including "storage works but this key is
 * absent", which is the common case on a first visit.
 */
export async function loadMerges(contentHash: string): Promise<CachedMerges | null> {
  const db = await openDB();
  if (db === null) return null;
  try {
    return await new Promise<CachedMerges | null>((resolve) => {
      let tx: IDBTransaction;
      try {
        tx = db.transaction(STORE, "readonly");
      } catch {
        resolve(null);
        return;
      }
      const req = tx.objectStore(STORE).get(contentHash);
      req.onsuccess = () => {
        const v: unknown = req.result;
        if (v instanceof ArrayBuffer && v.byteLength % 8 === 0) {
          const pairs = new Int32Array(v);
          resolve({ pairs, count: pairs.length / 2 });
        } else {
          resolve(null);
        }
      };
      req.onerror = () => resolve(null);
      tx.onabort = () => resolve(null);
    });
  } catch {
    return null;
  } finally {
    db.close();
  }
}

/** Persist the derived merge pairs. Returns `false` on any failure. */
export async function storeMerges(contentHash: string, pairs: Int32Array): Promise<boolean> {
  const db = await openDB();
  if (db === null) return false;
  try {
    return await new Promise<boolean>((resolve) => {
      let tx: IDBTransaction;
      try {
        tx = db.transaction(STORE, "readwrite");
      } catch {
        resolve(false);
        return;
      }
      // Structured-clone the buffer, not the view: a view would serialise its
      // backing buffer plus offsets and round-trip as a plain object in some
      // older engines.
      const buf = pairs.buffer.slice(
        pairs.byteOffset,
        pairs.byteOffset + pairs.byteLength,
      ) as ArrayBuffer;
      const req = tx.objectStore(STORE).put(buf, contentHash);
      req.onerror = () => resolve(false);
      tx.oncomplete = () => resolve(true);
      tx.onabort = () => resolve(false);
      tx.onerror = () => resolve(false);
    });
  } catch {
    return false;
  } finally {
    db.close();
  }
}

/** Drop every cached derivation. Used when the asset bundle changes. */
export async function clearMerges(): Promise<boolean> {
  const db = await openDB();
  if (db === null) return false;
  try {
    return await new Promise<boolean>((resolve) => {
      try {
        const tx = db.transaction(STORE, "readwrite");
        tx.objectStore(STORE).clear();
        tx.oncomplete = () => resolve(true);
        tx.onabort = () => resolve(false);
        tx.onerror = () => resolve(false);
      } catch {
        resolve(false);
      }
    });
  } catch {
    return false;
  } finally {
    db.close();
  }
}

/** Flatten `[left, right]` string pairs into vocab-id pairs for storage. */
export function mergesToPairs(
  merges: readonly (readonly [string, string])[],
  vocab: ReadonlyMap<string, number>,
): Int32Array | null {
  const out = new Int32Array(merges.length * 2);
  for (let i = 0; i < merges.length; i++) {
    const pair = merges[i] as readonly [string, string];
    const l = vocab.get(pair[0]);
    const r = vocab.get(pair[1]);
    // A merge whose halves are not both in the vocab cannot be represented and
    // means the derivation disagreed with the blob. Do not cache a half-truth.
    if (l === undefined || r === undefined) return null;
    out[i * 2] = l;
    out[i * 2 + 1] = r;
  }
  return out;
}

/** Inverse of `mergesToPairs`. */
export function pairsToMerges(pairs: Int32Array, pieces: readonly string[]): [string, string][] {
  const out = new Array<[string, string]>(pairs.length / 2);
  for (let i = 0; i < out.length; i++) {
    const l = pieces[pairs[i * 2] as number];
    const r = pieces[pairs[i * 2 + 1] as number];
    if (l === undefined || r === undefined) {
      throw new Error(`cached merges: entry ${i} references an id outside the vocab`);
    }
    out[i] = [l, r];
  }
  return out;
}

/** sha256 hex of the raw asset bytes — the cache key. */
export async function contentHash(bytes: Uint8Array): Promise<string> {
  const view = new Uint8Array(bytes); // copy: subtle.digest wants a plain buffer
  const digest = await crypto.subtle.digest("SHA-256", view);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join("");
}
