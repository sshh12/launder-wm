"""Vectorised SynthID arithmetic for bulk authoring work.

WHY THIS EXISTS, AND WHAT IT IS NOT
-----------------------------------
`launder_core.watermark` is the reference implementation and the authority. It
is pure Python over one sequence at a time, which is exactly right for the
server and for the golden vectors, and exactly wrong for the forge, which
scores 20,000 sequences during `calibrate` and tens of thousands of one-edit
neighbours during `solve`.

So this module is a **numpy port of the same integer arithmetic**, used only
inside forge, and held to the reference by an executable parity test
(`packages/forge/tests/test_numerics_parity.py`) that runs every function here
against `launder_core` on random sequences and asserts bit equality. If
`launder_core.watermark` is not importable the parity test skips and says so;
the numbers below are still the ones §4.2 specifies, checked against the
TECH_PLAN §4.5 golden constants in `test_numerics_golden.py`.

Do not add a *semantic* here that core does not have. If you need a new
watermark primitive, it belongs in core and gets ported here, never the
reverse.

Arithmetic notes
----------------
* All hashing is done in ``uint64``, which numpy defines as modular. That is
  bit-identical to torch's ``int64`` wraparound; the two differ only in how the
  same 64 bits are *printed*. `.view(np.int64)` recovers the signed value.
* The g-value index is ``((h % N) + N) % N`` with Python remainder semantics.
  For a power-of-two N that is exactly the low ``log2(N)`` bits of the uint64
  representation, so the implementation masks — and `test_numerics_golden.py`
  asserts the masked path equals the explicit double-remainder path on the
  worst case (a negative hash).
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

__all__ = [
    "HASH_IV",
    "LCG_INCREMENT",
    "LCG_MULTIPLIER",
    "ScoredPassage",
    "accumulate_hash",
    "compute_context_hashes",
    "compute_context_repetition_mask",
    "compute_eos_mask",
    "compute_g_values",
    "contributions",
    "depth_weights",
    "g_digest",
    "load_sampling_table",
    "ripple_span",
    "sample_index",
    "score_ids",
    "unpack_sampling_table",
    "weighted_mean_score",
    "z_from_score",
]

LCG_MULTIPLIER: Final[int] = 6364136223846793005
LCG_INCREMENT: Final[int] = 1
HASH_IV: Final[int] = 1

_U64_M: Final[np.uint64] = np.uint64(LCG_MULTIPLIER)
_U64_INC: Final[np.uint64] = np.uint64(LCG_INCREMENT)

U64Array = npt.NDArray[np.uint64]
I64Array = npt.NDArray[np.int64]
F64Array = npt.NDArray[np.float64]
U8Array = npt.NDArray[np.uint8]


# ---------------------------------------------------------------------------
# The hash
# ---------------------------------------------------------------------------


def accumulate_hash(current: int, data: list[int] | tuple[int, ...] | I64Array) -> int:
    """Scalar reference spelling, kept for goldens and error messages.

    ``h = wrap64(wrap64(h + tok) * 6364136223846793005 + 1)`` per token.
    Returns the *signed* int64 value, which is what the golden vectors record.
    """
    h = int(current) & 0xFFFF_FFFF_FFFF_FFFF
    for tok in data:
        h = (h + int(tok)) & 0xFFFF_FFFF_FFFF_FFFF
        h = (h * LCG_MULTIPLIER + LCG_INCREMENT) & 0xFFFF_FFFF_FFFF_FFFF
    return h - (1 << 64) if h >= (1 << 63) else h


def _step(h: U64Array, data: U64Array) -> U64Array:
    """One accumulate step, vectorised. Modular by definition of uint64."""
    with np.errstate(over="ignore"):
        return (h + data) * _U64_M + _U64_INC


def sample_index(h: int, table_size: int) -> int:
    """``((h % N) + N) % N`` — golden case #2, the negative-modulo sign trap.

    ``torch.remainder(-12345678901234567, 65536) == 46201``; JS ``BigInt %``
    gives ``-19335``. Getting this wrong yields a detector that reads ~0.5 on
    everything and never fails loudly.
    """
    return ((h % table_size) + table_size) % table_size


# ---------------------------------------------------------------------------
# The sampling table
# ---------------------------------------------------------------------------


def unpack_sampling_table(packed: bytes, size: int = 65536) -> U8Array:
    """``np.packbits`` MSB-first, the numpy default, is the on-disk format."""
    if len(packed) * 8 != size:
        raise ValueError(
            f"sampling table is {len(packed)} bytes = {len(packed) * 8} values, "
            f"expected {size // 8} bytes for {size} values"
        )
    return np.unpackbits(np.frombuffer(packed, dtype=np.uint8), bitorder="big").astype(np.uint8)


def load_sampling_table(path: str | bytes | object, size: int = 65536) -> U8Array:
    """Load `data/assets/sampling_table.v1.bin`. KEY MATERIAL — never rebuilt here.

    §4.3 #3: the table is built ONCE on CPU and committed. Every consumer loads
    this file, including the generator, which overwrites the HF processor's own
    device-built table with it.
    """
    from pathlib import Path

    p = Path(str(path))
    if not p.exists():
        raise FileNotFoundError(
            f"sampling table not found at {p}. It is key material and is committed to the "
            "repo; it is never regenerated by library code. Restore it from git."
        )
    return unpack_sampling_table(p.read_bytes(), size)


# ---------------------------------------------------------------------------
# g-values, contexts, masks
# ---------------------------------------------------------------------------


def compute_g_values(
    ids: I64Array | list[int],
    *,
    keys: tuple[int, ...],
    ngram_len: int,
    table: U8Array,
) -> U8Array:
    """(rows, depth) uint8 matrix, rows = ``len(ids) - ngram_len + 1``.

    Row `i` covers ``ids[i .. i+ngram_len-1]``; the FULL n-gram is hashed first
    and each watermark key is then folded in as one more accumulate step.
    """
    arr = np.ascontiguousarray(np.asarray(ids, dtype=np.int64)).view(np.uint64)
    rows = arr.shape[0] - ngram_len + 1
    if rows <= 0:
        return np.zeros((0, len(keys)), dtype=np.uint8)
    h = np.ones(rows, dtype=np.uint64)
    for k in range(ngram_len):
        h = _step(h, arr[k : k + rows])
    keys_u = np.asarray(keys, dtype=np.int64).view(np.uint64)
    hl = _step(np.broadcast_to(h[:, None], (rows, keys_u.shape[0])).copy(), keys_u[None, :])
    size = table.shape[0]
    if size & (size - 1):
        raise ValueError(f"sampling table size {size} is not a power of two")
    idx = (hl & np.uint64(size - 1)).astype(np.intp)
    return table[idx]


def compute_context_hashes(ids: I64Array | list[int], *, ngram_len: int) -> I64Array:
    """``ctx[i] = accumulate_hash(1, ids[i .. i+ngram_len-2])`` — the n-1 leading
    tokens. One per g-value row; signed int64, matching the golden vectors."""
    arr = np.ascontiguousarray(np.asarray(ids, dtype=np.int64)).view(np.uint64)
    rows = arr.shape[0] - ngram_len + 1
    if rows <= 0:
        return np.zeros(0, dtype=np.int64)
    c = np.ones(rows, dtype=np.uint64)
    for k in range(ngram_len - 1):
        c = _step(c, arr[k : k + rows])
    return c.view(np.int64)


def compute_context_repetition_mask(
    context_hashes: I64Array, *, context_history_size: int = 1024
) -> U8Array:
    """1 = not repeated (scored), 0 = repeated (dropped), causal.

    Replicates `SynthIDTextWatermarkLogitsProcessor.compute_context_repetition_mask`
    exactly, including the two details that are easy to get wrong:

    * the history is a **fixed-size window pre-filled with zeros**, so a context
      hash of exactly 0 is always "repeated";
    * the hash is pushed **even when it was a repeat**, and the oldest entry is
      evicted on every step (not only on a miss).
    """
    n = int(context_hashes.shape[0])
    mask = np.ones(n, dtype=np.uint8)
    if n == 0:
        return mask
    # Fast path, exact rather than approximate: if every context hash is
    # distinct and none is 0, no row can ever match a history entry — the
    # pre-filled zeros included. Saves the Python loop on the overwhelming
    # majority of passages, which matters at 20,000 of them in `calibrate`.
    if np.unique(context_hashes).shape[0] == n and not np.any(context_hashes == 0):
        return mask
    window: deque[int] = deque([0] * context_history_size, maxlen=context_history_size)
    counts: Counter[int] = Counter({0: context_history_size})
    for i in range(n):
        h = int(context_hashes[i])
        if counts.get(h, 0):
            mask[i] = 0
        evicted = window[0]
        window.append(h)
        counts[h] += 1
        counts[evicted] -= 1
        if counts[evicted] == 0:
            del counts[evicted]
    return mask


def compute_eos_mask(
    ids: I64Array | list[int], *, eos_token_id: int | None, ngram_len: int
) -> U8Array:
    """First eos index and everything after it -> 0, then sliced ``[ngram_len-1:]``.

    Length matches the g-value row count. `eos_token_id=None` means "the
    passage text alone was tokenized with add_special_tokens=False", which is
    the canonical scoring unit (§6.2) and cannot contain an eos.
    """
    arr = np.asarray(ids, dtype=np.int64)
    full = np.ones(arr.shape[0], dtype=np.uint8)
    if eos_token_id is not None:
        hits = np.nonzero(arr == eos_token_id)[0]
        if hits.size:
            full[hits[0] :] = 0
    return full[ngram_len - 1 :]


def ripple_span(edit_index: int, *, ngram_len: int, n_rows: int) -> tuple[int, int]:
    """Rows invalidated by changing token `edit_index`: ``[j-n+1, j]`` clipped.

    Returned half-open. §4.4: exactly `ngram_len` rows, never more, and rows
    beyond ``j`` are bit-identical because g-values are a pure function of token
    *content*, not position.
    """
    lo = max(0, edit_index - ngram_len + 1)
    hi = min(n_rows, edit_index + 1)
    return lo, max(lo, hi)


# ---------------------------------------------------------------------------
# The detector
# ---------------------------------------------------------------------------


def depth_weights(depth: int) -> F64Array:
    """``w = linspace(10, 1, m); w *= m / sum(w)`` so ``sum(w) == m``."""
    w = np.linspace(10.0, 1.0, depth, dtype=np.float64)
    return w * (depth / w.sum())


def weighted_mean_score(g: U8Array, mask: U8Array, weights: F64Array) -> tuple[float, int]:
    """``score = sum(mask_i w_L g_iL) / (m * sum(mask_i))`` and ``n_scored``."""
    n_scored = int(mask.sum())
    depth = int(weights.shape[0])
    if n_scored == 0:
        return 0.5, 0
    num = float((g.astype(np.float64) * weights[None, :]).sum(axis=1) @ mask.astype(np.float64))
    return num / (depth * n_scored), n_scored


def contributions(g: U8Array, weights: F64Array) -> F64Array:
    """``heat[i] = sum_L w_L g_iL / m`` in [0,1]; 0.5 is neutral. THE mirror value."""
    depth = int(weights.shape[0])
    if g.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    return (g.astype(np.float64) @ weights) / depth


def sigma_null(n_scored: int, depth: int, inflation: float = 1.0) -> float:
    """``1 / (2*sqrt(m*n))`` times an empirical inflation `kappa`.

    THE SAME CLOSED FORM `launder_core.detect.calibration.sigma_closed_form`
    uses, deliberately. Carrying the weights through the variance exactly would
    give ``sqrt(sum(w^2)/(4 m^2 n))``, which is a *constant multiple* of this
    (about 1.05 for `linspace(10,1,30)`) — and that constant is exactly the sort
    of thing `kappa` absorbs. Two different "closed forms" in one repo would
    make the forge's z and the server's z differ by 5% forever, which is worse
    than either choice. §6.5: the formula is the SHAPE, the measured percentile
    is the LEVEL.
    """
    if n_scored <= 0:
        raise ValueError("sigma_null needs at least one scored row")
    return inflation / (2.0 * float(np.sqrt(depth * n_scored)))


def z_from_score(score: float, n_scored: int, depth: int, inflation: float = 1.0) -> float:
    """``z = (score - 0.5) / sigma_null(n_scored)`` — the needle's quantity."""
    return (score - 0.5) / sigma_null(n_scored, depth, inflation)


def g_digest(g: U8Array, mask: U8Array) -> str:
    """`blake3:<hex>` over the packed g-matrix and mask.

    The highest-value field in the passage schema: it pins the exact 0/1 matrix
    the browser must reproduce, so a detector that diverges by one bit fails
    loudly on load instead of showing plausible numbers.
    """
    import blake3

    h = blake3.blake3()
    h.update(np.ascontiguousarray(g, dtype=np.uint8).tobytes())
    h.update(b"|")
    h.update(np.ascontiguousarray(mask, dtype=np.uint8).tobytes())
    return "blake3:" + h.hexdigest()


# ---------------------------------------------------------------------------
# One-call scoring
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ScoredPassage:
    """Everything the pipeline needs about one token sequence, computed once."""

    ids: I64Array
    g: U8Array
    context_hashes: I64Array
    rep_mask: U8Array
    mask: U8Array
    heat: F64Array
    score: float
    n_scored: int
    z: float

    @property
    def masked_fraction(self) -> float:
        rows = int(self.mask.shape[0])
        return 0.0 if rows == 0 else 1.0 - (self.n_scored / rows)

    @property
    def digest(self) -> str:
        return g_digest(self.g, self.mask)


def score_ids(
    ids: I64Array | list[int],
    *,
    keys: tuple[int, ...],
    ngram_len: int,
    table: U8Array,
    context_history_size: int = 1024,
    eos_token_id: int | None = None,
    inflation: float = 1.0,
) -> ScoredPassage:
    """The whole §4.2 pipeline for one sequence."""
    arr = np.asarray(ids, dtype=np.int64)
    g = compute_g_values(arr, keys=keys, ngram_len=ngram_len, table=table)
    ctx = compute_context_hashes(arr, ngram_len=ngram_len)
    rep = compute_context_repetition_mask(ctx, context_history_size=context_history_size)
    eos = compute_eos_mask(arr, eos_token_id=eos_token_id, ngram_len=ngram_len)
    mask = (rep * eos).astype(np.uint8)
    w = depth_weights(len(keys))
    score, n_scored = weighted_mean_score(g, mask, w)
    heat = contributions(g, w)
    z = z_from_score(score, n_scored, len(keys), inflation) if n_scored else 0.0
    return ScoredPassage(
        ids=arr,
        g=g,
        context_hashes=ctx,
        rep_mask=rep,
        mask=mask,
        heat=heat,
        score=score,
        n_scored=n_scored,
        z=z,
    )
