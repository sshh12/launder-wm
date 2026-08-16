/**
 * Unrestricted Damerau-Levenshtein (Lowrance-Wagner) over the WORD sequence.
 * TS port of `launder_core.scoring.damerau`.
 *
 * **THE SERVER IS AUTHORITY** (§8.3). This is the live preview; the server
 * recomputes and its number is the one persisted, ranked and shared.
 *
 * Unrestricted, NOT Optimal String Alignment. OSA forbids editing between
 * transposed elements and would price a legitimate "swap two words then change
 * one of them" at 3 instead of 2. CONCEPT.md's requirement — "a reorder counts
 * as 1, so reordering is cheap, not free" — is the unrestricted variant's
 * property (§8.2).
 *
 * Backtrace determinism matters (two clients must render the same diff), so
 * tie-breaking is fixed at **substitute/match > delete > insert > transpose**
 * and pinned by golden case #8.
 */

export type EditOpKind = "sub" | "ins" | "del" | "transpose";

export interface EditOp {
  readonly op: EditOpKind;
  /** index into the original word sequence */
  readonly i: number;
  /** index into the submission word sequence */
  readonly j: number;
  readonly from: string;
  readonly to: string;
}

export interface DistanceResult {
  readonly distance: number;
  readonly ops: EditOp[];
  readonly aWords: readonly string[];
  readonly bWords: readonly string[];
}

/**
 * The DP table, with the Lowrance-Wagner sentinel row/column.
 *
 * Indices are shifted by one: `D(i, j)` is the cost of aligning `a[0..i)` with
 * `b[0..j)`, and `D(-1, *) = D(*, -1) = m + n` is the sentinel that makes the
 * transpose branch unreachable when there is no earlier occurrence.
 */
function buildTable(
  a: readonly string[],
  b: readonly string[],
): { d: Int32Array; kMat: Int32Array; lMat: Int32Array; w: number } {
  const m = a.length;
  const n = b.length;
  const big = m + n;
  const w = n + 2; // row stride, with the sentinel column at index 0
  const d = new Int32Array((m + 2) * w);
  const kMat = new Int32Array((m + 2) * w);
  const lMat = new Int32Array((m + 2) * w);

  const at = (i: number, j: number): number => (i + 1) * w + (j + 1);

  for (let i = -1; i <= m; i++) d[at(i, -1)] = big;
  for (let j = -1; j <= n; j++) d[at(-1, j)] = big;
  for (let i = 0; i <= m; i++) d[at(i, 0)] = i;
  for (let j = 0; j <= n; j++) d[at(0, j)] = j;

  const da = new Map<string, number>(); // last row index at which each word occurred in a

  for (let i = 1; i <= m; i++) {
    let db = 0; // last col index in b matching a[i]
    for (let j = 1; j <= n; j++) {
      const ai = a[i - 1] as string;
      const bj = b[j - 1] as string;
      const k = da.get(bj) ?? 0; // last occurrence of b[j] in a[:i]
      const l = db;
      const cost = ai === bj ? 0 : 1;
      if (cost === 0) db = j;

      const sub = (d[at(i - 1, j - 1)] as number) + cost;
      const ins = (d[at(i, j - 1)] as number) + 1;
      const del = (d[at(i - 1, j)] as number) + 1;
      const tra = (d[at(k - 1, l - 1)] as number) + (i - k - 1) + 1 + (j - l - 1);

      d[at(i, j)] = Math.min(sub, ins, del, tra);
      kMat[at(i, j)] = k;
      lMat[at(i, j)] = l;
    }
    da.set(a[i - 1] as string, i);
  }
  return { d, kMat, lMat, w };
}

/** Distance only. */
export function damerauDistance(a: readonly string[], b: readonly string[]): number {
  const { d, w } = buildTable(a, b);
  return d[(a.length + 1) * w + (b.length + 1)] as number;
}

/**
 * Distance plus the backtrace, which IS the shareable diff.
 *
 * Tie-break order, fixed: substitute/match, then delete, then insert, then
 * transpose. Reordering these changes the rendered diff without changing the
 * distance, which is why the order is config in `scoring.toml` and pinned by a
 * golden case rather than left to whoever last touched the file.
 *
 * A transpose step consumes the whole `a[k..i] / b[l..j]` block: the
 * `(i-k-1)` intervening words of `a` are deletions, the `(j-l-1)` intervening
 * words of `b` are insertions, and the swap itself is one op. They are emitted
 * deletions-first, then insertions, then the `transpose` op, so the op list is
 * a deterministic function of the inputs alone.
 *
 * The `transpose` op's fields: `i`/`j` are the 0-based positions of the FIRST
 * word of the transposed pair in `a` and `b`; `from` is that word and `to` is
 * the word it swapped with (`a[i-1]`, 0-based `i-1`). So `{from, to}` in `a`
 * became `{to, from}` in `b`.
 */
export function damerau(a: readonly string[], b: readonly string[]): DistanceResult {
  const m = a.length;
  const n = b.length;
  const { d, kMat, lMat, w } = buildTable(a, b);
  const at = (i: number, j: number): number => (i + 1) * w + (j + 1);
  const distance = d[at(m, n)] as number;

  const rev: EditOp[] = [];
  let i = m;
  let j = n;
  let guard = (m + n) * 2 + 8;

  while ((i > 0 || j > 0) && guard-- > 0) {
    const here = d[at(i, j)] as number;

    // 1. substitute / match
    if (i > 0 && j > 0) {
      const ai = a[i - 1] as string;
      const bj = b[j - 1] as string;
      const cost = ai === bj ? 0 : 1;
      if (here === (d[at(i - 1, j - 1)] as number) + cost) {
        if (cost === 1) rev.push({ op: "sub", i: i - 1, j: j - 1, from: ai, to: bj });
        i -= 1;
        j -= 1;
        continue;
      }
    }
    // 2. delete
    if (i > 0 && here === (d[at(i - 1, j)] as number) + 1) {
      rev.push({ op: "del", i: i - 1, j, from: a[i - 1] as string, to: "" });
      i -= 1;
      continue;
    }
    // 3. insert
    if (j > 0 && here === (d[at(i, j - 1)] as number) + 1) {
      rev.push({ op: "ins", i, j: j - 1, from: "", to: b[j - 1] as string });
      j -= 1;
      continue;
    }
    // 4. transpose
    if (i > 0 && j > 0) {
      const k = kMat[at(i, j)] as number;
      const l = lMat[at(i, j)] as number;
      const tra = (d[at(k - 1, l - 1)] as number) + (i - k - 1) + 1 + (j - l - 1);
      if (k > 0 && l > 0 && here === tra) {
        // `rev` is flipped at the end, so the LAST push lands FIRST. Final
        // order is therefore: transpose, then the (i-k-1) deletions in
        // ascending a-order, then the (j-l-1) insertions in ascending b-order.
        // Every op in the block is anchored at the pair start (k-1, l-1).
        for (let p = j - 1; p >= l + 1; p--) {
          rev.push({ op: "ins", i: k - 1, j: p - 1, from: "", to: b[p - 1] as string });
        }
        for (let p = i - 1; p >= k + 1; p--) {
          rev.push({ op: "del", i: p - 1, j: l - 1, from: a[p - 1] as string, to: "" });
        }
        rev.push({
          op: "transpose",
          i: k - 1,
          j: l - 1,
          from: a[k - 1] as string,
          to: a[i - 1] as string,
        });
        i = k - 1;
        j = l - 1;
        continue;
      }
    }
    throw new Error(
      `damerau backtrace stuck at (${i}, ${j}) with cost ${here}. The DP table and the ` +
        `backtrace disagree, which means one of them was edited without the other.`,
    );
  }
  if (guard <= 0) throw new Error("damerau backtrace did not terminate");

  rev.reverse();
  return { distance, ops: rev, aWords: a, bWords: b };
}
