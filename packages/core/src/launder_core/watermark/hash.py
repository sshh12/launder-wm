"""`accumulate_hash` — THE reference implementation (TECH_PLAN.md §3, §4.2).

Every other runner in this project is a port of this file: the TypeScript
detector in `web/src/detector/hash.ts`, and (transitively, via the sampling
table) the HF generator. If they disagree with this file, they are wrong.

The recurrence, per token, in int64 two's-complement arithmetic::

    h = wrap64(wrap64(h + tok) * 6364136223846793005 + 1)

with IV `h = 1`. It is an adapted linear congruential generator with
newlib/musl parameters, and it has the property

    f(x, data[:T]) == f(f(x, data[:T-1]), data[T])

which is exactly why a watermark key can be folded in as "one more token" after
the n-gram (see `launder_core.watermark.gvalues.compute_depth_hashes`).

Two traps, both from §4.3, both of which produce plausible numbers that mean
nothing rather than an exception:

1. **int64 wraparound.** torch does this for free because its `int64` wraps
   mod 2^64. Python `int` does not — it grows without bound — so every add and
   every multiply here is followed by an explicit mask-and-sign-fold. In TS the
   equivalent is `BigInt.asIntN(64, x)` after every operation.
2. **Unsigned intermediates.** The vectorized path works in `uint64` and
   `.view(np.int64)` at the end. `view` is a pure bit reinterpretation, which
   is the definition of two's complement; `astype` between signed and unsigned
   is a value cast whose out-of-range behaviour is not something to rely on.

Two implementations are provided on purpose. `accumulate_hash` is the scalar
one, written to be read next to §4.2's pseudocode. `accumulate_hash_batch` is
the numpy one that actually runs over a passage. `test_watermark.py` asserts
they agree over random inputs, so the fast path can never drift from the
readable one.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Final

import numpy as np
import numpy.typing as npt

from launder_core.watermark.config import HASH_IV, LCG_INCREMENT, LCG_MULTIPLIER

__all__ = [
    "HASH_IV",
    "LCG_INCREMENT",
    "LCG_MULTIPLIER",
    "MASK64",
    "MIN_INT64",
    "accumulate_hash",
    "accumulate_hash_batch",
    "sliding_windows",
    "wrap_int64",
]

MASK64: Final[int] = (1 << 64) - 1
_SIGN_BIT: Final[int] = 1 << 63
MIN_INT64: Final[int] = -(1 << 63)

_MULT_U64: Final[np.uint64] = np.uint64(LCG_MULTIPLIER)
_INCR_U64: Final[np.uint64] = np.uint64(LCG_INCREMENT)


def wrap_int64(x: int) -> int:
    """Reduce a Python int to its int64 two's-complement value.

    `wrap_int64(2**63) == -2**63`, `wrap_int64(-2**63 - 1) == 2**63 - 1`.
    """
    x &= MASK64
    return x - (1 << 64) if x >= _SIGN_BIT else x


def accumulate_hash(
    current_hash: int,
    data: Iterable[int],
    multiplier: int = LCG_MULTIPLIER,
    increment: int = LCG_INCREMENT,
) -> int:
    """Fold `data` into `current_hash`, one element at a time. Scalar reference.

    Mirrors `SynthIDTextWatermarkLogitsProcessor.accumulate_hash` exactly
    (transformers `generation/logits_process.py`), including the argument
    defaults, with torch's int64 wraparound made explicit.
    """
    h = wrap_int64(current_hash)
    for tok in data:
        h = wrap_int64(wrap_int64(h + int(tok)) * multiplier + increment)
    return h


def accumulate_hash_batch(
    current_hash: npt.NDArray[np.int64] | int,
    data: npt.NDArray[np.int64],
) -> npt.NDArray[np.int64]:
    """Vectorized `accumulate_hash` over the last axis of `data`.

    `data` has shape `(..., L)`; `current_hash` is a scalar or broadcastable to
    `data.shape[:-1]`. Returns an int64 array of shape `data.shape[:-1]`.

    Arithmetic is done in `uint64` (which wraps mod 2^64 silently and without
    the overflow warnings numpy raises on signed types) and reinterpreted as
    int64 at the end.
    """
    d = np.ascontiguousarray(data, dtype=np.int64).view(np.uint64)
    if d.ndim == 0:
        raise ValueError("data must have at least one axis (the accumulation axis)")

    if isinstance(current_hash, int):
        h = np.full(d.shape[:-1], np.uint64(wrap_int64(current_hash) & MASK64), dtype=np.uint64)
    else:
        seed = np.ascontiguousarray(current_hash, dtype=np.int64).view(np.uint64)
        h = np.broadcast_to(seed, d.shape[:-1]).astype(np.uint64, copy=True)

    with np.errstate(over="ignore"):
        for i in range(d.shape[-1]):
            h += d[..., i]
            h *= _MULT_U64
            h += _INCR_U64

    return h.view(np.int64)


def sliding_windows(ids: npt.NDArray[np.int64], width: int) -> npt.NDArray[np.int64]:
    """`(T - width + 1, width)` view of consecutive windows. Empty when `T < width`.

    Equivalent to torch's `Tensor.unfold(dimension=0, size=width, step=1)`, but
    it returns an empty `(0, width)` array instead of raising when the sequence
    is shorter than the window — a two-token passage is a legitimate mid-edit
    state in the browser, not an error.
    """
    if width < 1:
        raise ValueError(f"window width must be >= 1, got {width}")
    a = np.ascontiguousarray(ids, dtype=np.int64)
    if a.ndim != 1:
        raise ValueError(f"expected a 1-D token id array, got shape {a.shape}")
    if a.shape[0] < width:
        return np.empty((0, width), dtype=np.int64)
    view: npt.NDArray[np.int64] = np.lib.stride_tricks.sliding_window_view(a, width)
    return view
