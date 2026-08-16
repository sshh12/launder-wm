"""G-values, context hashes, the repetition mask, the eos mask, and the ripple.

This is the part of the algorithm the browser runs on every keystroke, so the
shapes matter as much as the numbers. For `T` token ids and `n = ngram_len`:

    rows R  = max(0, T - n + 1)
    g       : (R, m) uint8      row i covers ids[i : i+n]
    ctx     : (R,)   int64      ctx[i] = accumulate_hash(1, ids[i : i+n-1])
    rep_mask: (R,)   bool       causal, checked BEFORE insertion into the ring
    eos_mask: (R,)   bool       full-length eos mask sliced [n-1:]
    mask    : (R,)   bool       rep_mask & eos_mask

Row `i`'s **current token** is `ids[i + n - 1]` — the last token of the window.
That mapping is what the heat mirror colours, and it is the reason the ripple
looks asymmetric when you write it in token coordinates (§4.4).

Everything here is a pure function of token *ids*. Nothing depends on position,
which is exactly why insertions and deletions do not shift downstream g-values
and why the ripple genuinely stops after `n` tokens.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from launder_core.watermark.config import (
    CANONICAL_CONFIG,
    HASH_IV,
    SCORING_EOS_TOKEN_ID,
    SynthIDConfig,
)
from launder_core.watermark.hash import accumulate_hash_batch, sliding_windows
from launder_core.watermark.table import load_sampling_table

__all__ = [
    "RippleSpan",
    "WatermarkFrame",
    "changed_span",
    "compute_context_hashes",
    "compute_context_repetition_mask",
    "compute_depth_hashes",
    "compute_eos_mask",
    "compute_eos_token_mask_full",
    "compute_frame",
    "compute_g_values",
    "compute_ngram_hashes",
    "ripple_span",
    "ripple_span_for_retokenization",
    "sample_index",
    "sample_indices",
]

BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.int64]
ByteArray = npt.NDArray[np.uint8]


def as_ids(ids: npt.ArrayLike) -> IntArray:
    """Coerce to a contiguous 1-D int64 array, raising on anything else."""
    a = np.ascontiguousarray(ids, dtype=np.int64)
    if a.ndim != 1:
        raise ValueError(f"expected a 1-D token id array, got shape {a.shape}")
    return a


def n_rows_for(n_tokens: int, ngram_len: int) -> int:
    """`T - n + 1`, floored at 0. A passage shorter than one n-gram scores nothing."""
    return max(0, n_tokens - ngram_len + 1)


# --------------------------------------------------------------------------
# hashes
# --------------------------------------------------------------------------


def compute_ngram_hashes(ids: npt.ArrayLike, ngram_len: int = 5) -> IntArray:
    """`h[i] = accumulate_hash(1, ids[i : i+n])` — the FULL n-gram, keys not yet folded."""
    windows = sliding_windows(as_ids(ids), ngram_len)
    return accumulate_hash_batch(HASH_IV, windows)


def compute_context_hashes(ids: npt.ArrayLike, ngram_len: int = 5) -> IntArray:
    """`ctx[i] = accumulate_hash(1, ids[i : i+n-1])` — the `n-1` leading tokens.

    Mirrors transformers' `compute_context_repetition_mask`, which unfolds
    `input_ids[:, :-1]` with size `ngram_len - 1`. Dropping the final token
    before unfolding is what makes the context count equal the g-value row
    count (`T - n + 1`) rather than `T - n + 2`.
    """
    a = as_ids(ids)
    if a.shape[0] < ngram_len:
        return np.empty(0, dtype=np.int64)
    windows = sliding_windows(a[:-1], ngram_len - 1)
    return accumulate_hash_batch(HASH_IV, windows)


def compute_depth_hashes(
    ngram_hashes: IntArray,
    keys: tuple[int, ...] | list[int],
) -> IntArray:
    """Fold each watermark key into each n-gram hash: `(R,) -> (R, m)`.

    The key is folded in as **one more `accumulate_hash` step**, not as part of
    the hashed data::

        hL = accumulate_hash(accumulate_hash(1, ngram), keys[L])

    This is the single most confusable line in the algorithm and the one that
    golden case #1 pins down.
    """
    h = np.ascontiguousarray(ngram_hashes, dtype=np.int64)
    if h.ndim != 1:
        raise ValueError(f"expected 1-D n-gram hashes, got shape {h.shape}")
    key_arr = np.asarray(list(keys), dtype=np.int64).reshape(1, len(keys), 1)
    data = np.broadcast_to(key_arr, (h.shape[0], len(keys), 1))
    return accumulate_hash_batch(h[:, None], data)


def sample_index(h: int, size: int = 65536) -> int:
    """`((h % size) + size) % size` — the negative-modulo trap, in isolation (§4.3 #1).

    Python's `%` already returns a non-negative result for a positive modulus,
    so the `+ size) % size` is a no-op *here*. It is written out anyway because
    this function is the spec the TypeScript port copies, and there `BigInt %`
    is truncating: `-12345678901234567n % 65536n === -19335n` where torch and
    Python both give `46201`. That divergence produces a detector that reads
    ~0.5 on everything, and it never raises.
    """
    return ((h % size) + size) % size


def sample_indices(hashes: IntArray, size: int = 65536) -> IntArray:
    """Vectorized `sample_index`. numpy's `remainder` follows Python, not C."""
    idx = np.remainder(np.ascontiguousarray(hashes, dtype=np.int64), np.int64(size))
    return np.remainder(idx + np.int64(size), np.int64(size))


# --------------------------------------------------------------------------
# g-values
# --------------------------------------------------------------------------


def compute_g_values(
    ids: npt.ArrayLike,
    config: SynthIDConfig = CANONICAL_CONFIG,
    table: ByteArray | None = None,
) -> ByteArray:
    """The `(R, m)` uint8 g-value matrix. `R = T - n + 1`, `m = len(config.keys)`.

    Equivalent to `SynthIDTextWatermarkLogitsProcessor.compute_g_values` on a
    batch of one, and asserted bit-identical against it in `test_watermark.py`
    whenever transformers is importable.
    """
    tbl = load_sampling_table(size=config.sampling_table_size) if table is None else table
    if tbl.shape != (config.sampling_table_size,):
        raise ValueError(
            f"sampling table has shape {tbl.shape}, expected ({config.sampling_table_size},)"
        )
    ngram_hashes = compute_ngram_hashes(ids, config.ngram_len)
    if ngram_hashes.shape[0] == 0:
        return np.empty((0, config.depth), dtype=np.uint8)
    depth_hashes = compute_depth_hashes(ngram_hashes, config.keys)
    idx = sample_indices(depth_hashes, config.sampling_table_size)
    out: ByteArray = np.ascontiguousarray(tbl[idx], dtype=np.uint8)
    return out


# --------------------------------------------------------------------------
# masks
# --------------------------------------------------------------------------


def compute_context_repetition_mask(
    ids: npt.ArrayLike,
    config: SynthIDConfig = CANONICAL_CONFIG,
    *,
    context_hashes: IntArray | None = None,
) -> BoolArray:
    """`True` where row `i`'s context is NOT a repeat of one of the previous 1024.

    Faithful to transformers, including two details that are easy to "clean up"
    into a different algorithm:

    * The history starts as `context_history_size` **zeros**, and membership is
      tested against those zeros. A context hash that happens to equal 0 is
      therefore treated as repeated. That is what the reference does; matching
      it matters more than the aesthetics of it.
    * The check happens **before** insertion, and the context is pushed into the
      ring **even when it was a repeat**. Skipping the push on a repeat would
      change the eviction schedule and quietly desynchronise us from the
      generator.

    O(T) with a hash multiset, which is why §4.4(b) can say "recompute the whole
    mask on every keystroke" — the mask has unbounded rightward reach, so
    incremental caching of *it* is a bug, while incremental caching of g-values
    is fine.
    """
    ctx = (
        compute_context_hashes(ids, config.ngram_len) if context_hashes is None else context_hashes
    )
    n = ctx.shape[0]
    out = np.ones(n, dtype=np.bool_)
    if n == 0:
        return out

    ring: deque[int] = deque([0] * config.context_history_size, maxlen=config.context_history_size)
    counts: Counter[int] = Counter({0: config.context_history_size})

    for i in range(n):
        h = int(ctx[i])
        out[i] = counts[h] == 0
        evicted = ring[0]  # deque is full from the start, so append always evicts
        ring.append(h)
        counts[evicted] -= 1
        if counts[evicted] == 0:
            del counts[evicted]
        counts[h] += 1
    return out


def compute_eos_token_mask_full(ids: npt.ArrayLike, eos_token_id: int | None) -> BoolArray:
    """Length-`T` mask: `False` at the first `eos_token_id` and everything after.

    `SynthIDTextWatermarkLogitsProcessor.compute_eos_token_mask`. `None` means
    "no eos in this text" and yields an all-`True` mask — which is the normal
    case for us, because the canonical scoring unit is the passage text alone,
    tokenized with `add_special_tokens=False` (§6.2).
    """
    a = as_ids(ids)
    out = np.ones(a.shape[0], dtype=np.bool_)
    if eos_token_id is None:
        return out
    hits = np.flatnonzero(a == np.int64(eos_token_id))
    if hits.size:
        out[int(hits[0]) :] = False
    return out


def compute_eos_mask(
    ids: npt.ArrayLike,
    eos_token_id: int | None = None,
    config: SynthIDConfig = CANONICAL_CONFIG,
) -> BoolArray:
    """The eos mask sliced to g-value rows: `full_mask[n-1:]`, length `R`.

    The slice is `[n-1:]`, i.e. it is indexed by the row's **current token**,
    not by the row's first token. §4.2 states it that way and transformers'
    detector does `eos_token_mask[:, self.ngram_len - 1 :]`.
    """
    a = as_ids(ids)
    rows = n_rows_for(a.shape[0], config.ngram_len)
    if rows == 0:
        return np.empty(0, dtype=np.bool_)
    full = compute_eos_token_mask_full(a, eos_token_id)
    return np.ascontiguousarray(full[config.ngram_len - 1 :][:rows])


# --------------------------------------------------------------------------
# the frame
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class WatermarkFrame:
    """Everything the detector needs for one exact token sequence.

    Frozen and self-consistent: the arrays are all length `R` (or `(R, m)`) and
    were computed from the same `ids` under the same `config`. Passing a frame
    around instead of five loose arrays is what stops a caller from scoring
    g-values for one text against a mask for another — which would read as a
    plausible number and be meaningless.
    """

    ids: IntArray
    g: ByteArray
    context_hashes: IntArray
    repetition_mask: BoolArray
    eos_mask: BoolArray
    mask: BoolArray
    ngram_len: int
    depth: int

    @property
    def n_tokens(self) -> int:
        return int(self.ids.shape[0])

    @property
    def n_rows(self) -> int:
        return int(self.g.shape[0])

    @property
    def n_scored(self) -> int:
        return int(self.mask.sum())

    @property
    def masked_fraction(self) -> float:
        """Fraction of rows removed by the mask. Logged on every clear (§4.4c).

        If this exceeds ~35% on more than a few percent of clears, the
        `max_masked_fraction` check gets added — to config, not to code.
        """
        if self.n_rows == 0:
            return 0.0
        return 1.0 - self.n_scored / self.n_rows

    def token_index(self, row: int) -> int:
        """Row -> index of that row's current token: `row + n - 1`."""
        return row + self.ngram_len - 1

    def row_index(self, token: int) -> int:
        """Current-token index -> row. Negative for the first `n-1` tokens, which
        are inside no window's final position and therefore carry no heat."""
        return token - self.ngram_len + 1


def compute_frame(
    ids: npt.ArrayLike,
    config: SynthIDConfig = CANONICAL_CONFIG,
    *,
    table: ByteArray | None = None,
    eos_token_id: int | None = SCORING_EOS_TOKEN_ID,
) -> WatermarkFrame:
    """One pass: g-values, context hashes, both masks, and their conjunction.

    `eos_token_id` defaults to `SCORING_EOS_TOKEN_ID` — the ONE shipped policy,
    which `web/src/detector/index.ts` reads from its own mirror of the same
    constant and which `data/config/watermark.toml [scoring]` records. It used to
    default to a bare `None` here while the browser defaulted to `1`, and the two
    runtimes reported z 0.109 and z 1.971 for the same sentence.
    """
    a = as_ids(ids)
    g = compute_g_values(a, config, table)
    ctx = compute_context_hashes(a, config.ngram_len)
    rep = compute_context_repetition_mask(a, config, context_hashes=ctx)
    eos = compute_eos_mask(a, eos_token_id, config)
    return WatermarkFrame(
        ids=a,
        g=g,
        context_hashes=ctx,
        repetition_mask=rep,
        eos_mask=eos,
        mask=rep & eos,
        ngram_len=config.ngram_len,
        depth=config.depth,
    )


# --------------------------------------------------------------------------
# the ripple (§4.4)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RippleSpan:
    """Half-open `[start, stop)` ranges of what an edit at `edit_index` invalidates.

    `rows` are g-value row indices; `tokens` are current-token indices, which is
    what the mirror colours. The invariant §4.4 states — *exactly `ngram_len`
    wide* — holds on `tokens` away from the array edges, and both ranges are
    clipped at the edges rather than running off the end.
    """

    edit_index: int
    ngram_len: int
    n_tokens: int
    rows: tuple[int, int]
    tokens: tuple[int, int]

    @property
    def row_indices(self) -> range:
        return range(*self.rows)

    @property
    def token_indices(self) -> range:
        return range(*self.tokens)

    @property
    def width(self) -> int:
        """Number of affected current-token positions. `ngram_len` in the interior."""
        return max(0, self.tokens[1] - self.tokens[0])


def ripple_span(
    edit_index: int,
    *,
    n_tokens: int,
    ngram_len: int = 5,
) -> RippleSpan:
    """Blast radius of changing the token at `edit_index`. §4.4, executable.

    Row `i` covers `ids[i : i+n]`, so changing `ids[j]` invalidates every row
    `i` with `j - n + 1 <= i <= j`. In current-token coordinates
    (`token = row + n - 1`) that is `j .. j + n - 1`: **the edited token itself
    plus the next `n-1`**.

    This corrects CONCEPT.md, which describes the ripple as `j+1 .. j+H` and so
    omits the edited token's own g-value — the largest single contribution the
    player is deleting. The correction makes the mechanic stronger: the edit
    site itself visibly cools.

    Row `j + n` and beyond are bit-identical, because g-values are a pure
    function of token content and not of position.
    """
    if ngram_len < 1:
        raise ValueError(f"ngram_len must be >= 1, got {ngram_len}")
    if n_tokens < 0:
        raise ValueError(f"n_tokens must be >= 0, got {n_tokens}")
    if n_tokens > 0 and not 0 <= edit_index < n_tokens:
        raise IndexError(f"edit_index {edit_index} out of range for {n_tokens} tokens")

    rows_total = n_rows_for(n_tokens, ngram_len)
    row_start = max(0, edit_index - ngram_len + 1)
    row_stop = min(edit_index + 1, rows_total)
    if row_stop < row_start:
        row_stop = row_start

    tok_start = min(edit_index, n_tokens)
    tok_stop = min(edit_index + ngram_len, n_tokens)
    return RippleSpan(
        edit_index=edit_index,
        ngram_len=ngram_len,
        n_tokens=n_tokens,
        rows=(row_start, row_stop),
        tokens=(tok_start, tok_stop),
    )


def changed_span(old_ids: npt.ArrayLike, new_ids: npt.ArrayLike) -> tuple[int, int, int]:
    """Longest-common-prefix / longest-common-suffix diff of two id arrays.

    Returns `(prefix, old_stop, new_stop)`: `old[:prefix] == new[:prefix]`,
    `old[old_stop:] == new[new_stop:]`, and the two middles are what changed.
    When the arrays are equal, `prefix == old_stop == new_stop == len`.

    This is §4.4(a)'s machinery. A one-*word* edit in a BPE tokenizer with
    `byte_fallback` is one to three *token* edits and can retokenize the
    neighbours (leading-space merges), so the browser re-tokenizes the whole
    text and diffs the id arrays rather than trying to predict which tokens
    changed.
    """
    a = as_ids(old_ids)
    b = as_ids(new_ids)
    na, nb = a.shape[0], b.shape[0]
    limit = min(na, nb)

    prefix = 0
    while prefix < limit and a[prefix] == b[prefix]:
        prefix += 1

    suffix = 0
    while suffix < (limit - prefix) and a[na - 1 - suffix] == b[nb - 1 - suffix]:
        suffix += 1

    return prefix, na - suffix, nb - suffix


def ripple_span_for_retokenization(
    old_ids: npt.ArrayLike,
    new_ids: npt.ArrayLike,
    ngram_len: int = 5,
) -> RippleSpan:
    """Rows/tokens of the NEW sequence that an edit invalidated.

    Diffs the arrays, then extends the changed span rightward by `n-1` — the
    same blast radius as `ripple_span`, generalized from one token to a range.
    Returns an empty span (`width == 0`) when the arrays are identical.

    Note what this does NOT cover: the repetition mask's rightward reach is
    unbounded (§4.4b), so only *g-values* may be cached against this span. The
    mask is always recomputed in full.
    """
    b = as_ids(new_ids)
    prefix, _old_stop, new_stop = changed_span(old_ids, b)
    n_tokens = b.shape[0]
    if prefix >= new_stop:
        # No substitution: either identical, or a pure deletion. A deletion
        # still invalidates the rows whose window straddles the seam.
        if np.array_equal(as_ids(old_ids), b):
            return RippleSpan(
                edit_index=prefix,
                ngram_len=ngram_len,
                n_tokens=n_tokens,
                rows=(0, 0),
                tokens=(0, 0),
            )
        new_stop = prefix + 1

    rows_total = n_rows_for(n_tokens, ngram_len)
    row_start = max(0, prefix - ngram_len + 1)
    row_stop = min(new_stop, rows_total)
    if row_stop < row_start:
        row_stop = row_start
    tok_start = min(prefix, n_tokens)
    tok_stop = min(new_stop - 1 + ngram_len, n_tokens)
    return RippleSpan(
        edit_index=prefix,
        ngram_len=ngram_len,
        n_tokens=n_tokens,
        rows=(row_start, row_stop),
        tokens=(tok_start, tok_stop),
    )
