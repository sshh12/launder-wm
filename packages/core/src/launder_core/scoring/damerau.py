"""The distance — TECH_PLAN.md §8.2.

**Unrestricted Damerau-Levenshtein (Lowrance-Wagner), over the WORD sequence,
unit costs, adjacent transposition = 1.**

Not Optimal String Alignment. OSA forbids editing between transposed elements
and would price a legitimate "swap two words then change one of them" at 3
instead of 2. CONCEPT.md's requirement — *a reorder counts as 1, so reordering
is cheap, not free* — is the unrestricted variant's property, and it is why the
DP below carries the `da`/`db` side tables and the ``m + n`` sentinel row and
column rather than the three-line OSA recurrence.

THIS MODULE IS THE AUTHORITY. The TypeScript port must match it exactly — same
distance AND same op list, because the op list *is* the shareable diff and the
leaderboard renders it. Two clients rendering different diffs for one clear is
the parity bug golden case #8 exists to catch. Everything the port needs is
pinned here:

* the recurrence, verbatim from §8.2;
* the backtrace tie-break order — **substitute/match > delete > insert >
  transpose** — applied at every cell;
* the op-emission order inside a transposition span: the ``transpose`` op
  first, then the span's deletions ascending, then its insertions ascending.
  `apply_ops` depends on that order and documents why.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from launder_core.schemas import EditOp, ScoreResult
from launder_core.scoring.normalize import DEFAULT_NORMALIZE, NormalizeConfig, normalize, words

__all__ = [
    "TIE_BREAK",
    "apply_ops",
    "damerau_levenshtein",
    "damerau_levenshtein_ops",
    "score",
]

#: Fixed, and pinned by golden case #8. Backtrace determinism is what makes two
#: independent clients render the same diff for the same clear.
TIE_BREAK: Final[tuple[str, ...]] = ("substitute", "delete", "insert", "transpose")


def _table(a: Sequence[str], b: Sequence[str]) -> tuple[list[int], list[int], list[int], int]:
    """The Lowrance-Wagner DP table plus the per-cell transposition witnesses.

    Returns ``(d, ka, lb, width)``. Cell ``(i, j)`` for ``i in -1..m`` and
    ``j in -1..n`` lives at ``d[(i + 1) * width + (j + 1)]`` — the ``+1`` offset
    is what makes the ``-1`` sentinel row and column expressible in a flat list.

    ``ka``/``lb`` record the ``k``/``l`` the transposition term used at each
    cell so the backtrace can replay it without a second search.
    """
    m, n = len(a), len(b)
    inf = m + n
    width = n + 2
    size = (m + 2) * width
    d = [0] * size
    ka = [0] * size
    lb = [0] * size

    d[0] = inf  # cell (-1, -1)
    for i in range(m + 1):
        d[(i + 1) * width] = inf  # (i, -1)
        d[(i + 1) * width + 1] = i  # (i, 0)
    for j in range(n + 1):
        d[j + 1] = inf  # (-1, j)
        d[width + j + 1] = j  # (0, j)

    da: dict[str, int] = {}
    for i in range(1, m + 1):
        ai = a[i - 1]
        db = 0
        row = (i + 1) * width
        prev = i * width
        for j in range(1, n + 1):
            bj = b[j - 1]
            k = da.get(bj, 0)
            ell = db
            cost = 0 if ai == bj else 1
            if cost == 0:
                db = j

            best = d[prev + j] + cost  # substitute / match: (i-1, j-1)
            candidate = d[row + j] + 1  # insert: (i, j-1)
            if candidate < best:
                best = candidate
            candidate = d[prev + j + 1] + 1  # delete: (i-1, j)
            if candidate < best:
                best = candidate
            if k > 0 and ell > 0:
                # transpose: (k-1, l-1) + the span's interior edits.
                # The guard is equivalent to the plan's unguarded form: with
                # k == 0 or l == 0 the term reads the `m + n` sentinel and is
                # then >= m + n, while the true optimum at any cell with
                # i, j >= 1 is <= max(m, n) <= m + n - 1. It can never win.
                candidate = d[k * width + ell] + (i - k - 1) + 1 + (j - ell - 1)
                if candidate < best:
                    best = candidate
            d[row + j + 1] = best
            ka[row + j + 1] = k
            lb[row + j + 1] = ell
        da[ai] = i

    return d, ka, lb, width


def damerau_levenshtein(a: Sequence[str], b: Sequence[str]) -> int:
    """Distance only. O(mn) time, O(mn) space; ~200 words is sub-millisecond."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    d, _ka, _lb, width = _table(a, b)
    return d[(len(a) + 1) * width + len(b) + 1]


def damerau_levenshtein_ops(a: Sequence[str], b: Sequence[str]) -> tuple[int, tuple[EditOp, ...]]:
    """Distance AND the edit script, backtraced over the same table.

    Op indices are 0-based: ``i`` indexes ``a``, ``j`` indexes ``b``. For a
    ``transpose``, ``i``/``j`` address the FIRST element of the swapped pair on
    each side, ``from_`` is the word that starts the pair in ``a`` and ``to`` is
    the word that starts it in ``b`` — so a renderer reads it as
    ``"{from_} {to}" -> "{to} {from_}"``.
    """
    m, n = len(a), len(b)
    if m == 0:
        return n, tuple(EditOp(op="ins", i=0, j=j, to=b[j]) for j in range(n))
    if n == 0:
        return m, tuple(EditOp(op="del", i=i, j=0, from_=a[i]) for i in range(m))

    d, ka, lb, width = _table(a, b)

    def cell(i: int, j: int) -> int:
        return d[(i + 1) * width + j + 1]

    ops: list[EditOp] = []
    i, j = m, n
    while i > 0 or j > 0:
        here = cell(i, j)

        # 1. substitute / match — first in the tie-break order, so a cell that
        #    can be reached both by substituting and by transposing renders as
        #    a substitution on every client.
        if i > 0 and j > 0:
            cost = 0 if a[i - 1] == b[j - 1] else 1
            if here == cell(i - 1, j - 1) + cost:
                if cost:
                    ops.append(EditOp(op="sub", i=i - 1, j=j - 1, from_=a[i - 1], to=b[j - 1]))
                i -= 1
                j -= 1
                continue

        # 2. delete
        if i > 0 and here == cell(i - 1, j) + 1:
            ops.append(EditOp(op="del", i=i - 1, j=j, from_=a[i - 1]))
            i -= 1
            continue

        # 3. insert
        if j > 0 and here == cell(i, j - 1) + 1:
            ops.append(EditOp(op="ins", i=i, j=j - 1, to=b[j - 1]))
            j -= 1
            continue

        # 4. transpose. `a[k-1]` and `a[i-1]` swap; the span's interior is
        #    wholly deleted from `a` and wholly inserted from `b` (that is what
        #    the (i-k-1) + (j-l-1) terms in the recurrence pay for), so there
        #    are never any matches inside it — which is what makes `apply_ops`
        #    able to replay the span from the op stream alone.
        idx = (i + 1) * width + j + 1
        k, ell = ka[idx], lb[idx]
        if (
            i > 0
            and j > 0
            and k > 0
            and ell > 0
            and here == cell(k - 1, ell - 1) + (i - k - 1) + 1 + (j - ell - 1)
        ):
            # Appended in reverse; after the final reverse() the span reads
            # transpose, then deletions ascending, then insertions ascending.
            for y in range(j - 2, ell - 1, -1):
                ops.append(EditOp(op="ins", i=k, j=y, to=b[y]))
            for x in range(i - 2, k - 1, -1):
                ops.append(EditOp(op="del", i=x, j=ell, from_=a[x]))
            ops.append(EditOp(op="transpose", i=k - 1, j=ell - 1, from_=a[k - 1], to=a[i - 1]))
            i, j = k - 1, ell - 1
            continue

        raise AssertionError(  # pragma: no cover - a broken DP table, not an input
            f"damerau backtrace stuck at ({i}, {j}) with value {here}; the DP table and the "
            "backtrace disagree, which means one of them was edited without the other."
        )

    ops.reverse()
    return cell(m, n), tuple(ops)


def apply_ops(a: Sequence[str], ops: Sequence[EditOp]) -> tuple[str, ...]:
    """Replay an edit script onto `a`. ``apply_ops(a, score(...).ops) == b``.

    This is the executable definition of what an op MEANS, and the reason the
    diff exhibit can be trusted: a renderer that draws these ops draws exactly
    the transformation the distance was charged for.

    The transposition span is the only non-obvious case. Its ops arrive as
    ``transpose``, then the span's deletions, then its insertions; the span's
    interior contains no matches, so the walk can consume the deletions while
    ``i`` still tracks the cursor and the insertions while ``j`` still tracks
    the output length, then place the moved word. Any op belonging to the span
    is contiguous with it and any op that does not belong fails both guards.
    """
    out: list[str] = []
    p = 0
    q = 0
    total = len(ops)
    while q < total:
        op = ops[q]
        while p < op.i:  # untouched words between ops
            out.append(a[p])
            p += 1
        kind = op.op
        if kind == "sub":
            out.append(op.to)
            p += 1
            q += 1
        elif kind == "del":
            p += 1
            q += 1
        elif kind == "ins":
            out.append(op.to)
            q += 1
        else:  # transpose
            out.append(op.to)
            p += 1
            q += 1
            while q < total and ops[q].op == "del" and ops[q].i == p:
                p += 1
                q += 1
            while q < total and ops[q].op == "ins" and ops[q].j == len(out):
                out.append(ops[q].to)
                q += 1
            out.append(op.from_)
            p += 1
    while p < len(a):
        out.append(a[p])
        p += 1
    return tuple(out)


def score(original: str, submission: str, cfg: NormalizeConfig = DEFAULT_NORMALIZE) -> ScoreResult:
    """The one scoring entry point (TECH_PLAN.md §8.2, §8.3).

    `edit_budget` calls this, `close_paraphrase`'s `word_distance_ratio` calls
    this, the client preview calls its port of this, and the server recomputes
    it on submit. The budget and the scoreboard therefore cannot disagree.
    """
    a = words(normalize(original, cfg))
    b = words(normalize(submission, cfg))
    distance, ops = damerau_levenshtein_ops(a, b)
    return ScoreResult(distance=distance, ops=ops, a_words=a, b_words=b)
