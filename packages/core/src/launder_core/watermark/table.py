"""The sampling table: loader, digest assertion, and the CPU rebuild path.

`data/assets/sampling_table.v1.bin` is **key material** (TECH_PLAN.md §4.3).
It is 8,192 bytes: `np.packbits` of 65,536 zero/one values, MSB-first (numpy's
default bit order). Every consumer loads *this file* — the browser, the server,
and the generator, which explicitly overwrites the HF processor's own table
with it.

Why a committed file rather than `torch.randint(..., generator=Generator(device)
.manual_seed(0))` at each site: HF's construction is **device dependent**. CPU
generators are MT19937, CUDA generators are Philox. A table built on the 5090 at
generation time and rebuilt on CPU at detection time would give g-values
*uncorrelated with the watermark* — a needle that moves, looks fine, and
measures nothing. That is §4.3 failure #3, and it does not raise.

So: build once, on CPU, commit the bytes, and assert the digest on every load.
`build_table_cpu` exists so `forge` can regenerate and *compare* — it is a
verification tool, not the source of truth.

torch is imported INSIDE `build_table_cpu`. `import launder_core.watermark.table`
must succeed in a venv holding only pydantic, blake3 and numpy (§2.1), because
that invariant is what keeps torch out of the 250 MB Railway image.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from launder_core.watermark.config import CANONICAL_CONFIG, data_dir

__all__ = [
    "SAMPLING_TABLE_BLAKE3_PACKED",
    "SAMPLING_TABLE_BLAKE3_UNPACKED",
    "SAMPLING_TABLE_ONES",
    "SAMPLING_TABLE_PATH",
    "SAMPLING_TABLE_SHA256_PACKED",
    "SAMPLING_TABLE_SHA256_UNPACKED",
    "build_table_cpu",
    "digest_algorithm",
    "load_sampling_table",
    "pack_table",
    "sampling_table_digest",
    "unpack_table",
]

#: Path relative to `data/`.
SAMPLING_TABLE_PATH: Final[str] = "assets/sampling_table.v1.bin"

# The four digests, measured on the committed bytes and mirrored in
# data/config/watermark.toml's [table] section (test_watermark.py asserts the
# two copies agree, so neither can drift).
SAMPLING_TABLE_SHA256_PACKED: Final[str] = (
    "5151fe795a218d1adf0b7fd707204de23867ef2653fdebb8c9ff28d93f6aa55b"
)
SAMPLING_TABLE_SHA256_UNPACKED: Final[str] = (
    "a3e9e18ea34546849c5136cf9b06c7bfd03e50e601da14af47cd926f9e623a85"
)
SAMPLING_TABLE_BLAKE3_PACKED: Final[str] = (
    "d89a6eddd5e5e130284f57f1734712d29dd2b3543f126ab4f7cef9237042f1cd"
)
#: §4.3's `SAMPLING_TABLE_DIGEST`: blake3 over the 65,536 UNPACKED uint8 bytes.
SAMPLING_TABLE_BLAKE3_UNPACKED: Final[str] = (
    "db6fac16981fbdd4951314e0ffbbaea52d217e6d1633bc1842ad21514afdad84"
)
#: Number of ones among the 65,536 values. A weak check, but it is the one a
#: human can eyeball against a fresh build without a hex dump.
SAMPLING_TABLE_ONES: Final[int] = 32743

try:  # pragma: no cover - exercised by whichever branch the env provides
    from blake3 import blake3 as _blake3

    _HAVE_BLAKE3 = True
except ImportError:  # pragma: no cover
    _HAVE_BLAKE3 = False


def digest_algorithm() -> str:
    """`"blake3"` when the (declared, core) dependency is importable, else `"sha256"`.

    blake3 is a hard dependency of `launder-core`, so the sha256 branch should
    never be taken in a correctly installed environment. It exists because a
    digest assertion that gets *skipped* when a hashing library is missing is
    worse than useless, and this way the assertion always runs against
    something.
    """
    return "blake3" if _HAVE_BLAKE3 else "sha256"


def _hexdigest(payload: bytes) -> str:
    if _HAVE_BLAKE3:
        return str(_blake3(payload).hexdigest())
    return hashlib.sha256(payload).hexdigest()


def _expected(packed: bool) -> str:
    if _HAVE_BLAKE3:
        return SAMPLING_TABLE_BLAKE3_PACKED if packed else SAMPLING_TABLE_BLAKE3_UNPACKED
    return SAMPLING_TABLE_SHA256_PACKED if packed else SAMPLING_TABLE_SHA256_UNPACKED


def sampling_table_digest(table: npt.NDArray[np.uint8] | bytes) -> str:
    """Digest of an UNPACKED table: one byte per value, 65,536 bytes.

    This is the quantity §4.3's `assert` compares. Returns a bare hex string;
    the algorithm is `digest_algorithm()`.
    """
    if isinstance(table, bytes | bytearray | memoryview):
        payload = bytes(table)
    else:
        arr = np.ascontiguousarray(table, dtype=np.uint8)
        payload = arr.tobytes()
    return _hexdigest(payload)


def unpack_table(packed: bytes, size: int) -> npt.NDArray[np.uint8]:
    """`np.unpackbits` MSB-first, truncated to `size` values.

    MSB-first is numpy's default (`bitorder="big"`) and is stated explicitly
    here because it is the one parameter a port can get backwards while still
    producing a table with the right shape and the right number of ones.
    """
    expected_bytes = size // 8
    if len(packed) != expected_bytes:
        raise ValueError(
            f"packed sampling table is {len(packed)} bytes, expected {expected_bytes} "
            f"for {size} values (8 values per byte)"
        )
    bits: npt.NDArray[np.uint8] = np.unpackbits(
        np.frombuffer(packed, dtype=np.uint8), bitorder="big"
    )
    return np.ascontiguousarray(bits[:size], dtype=np.uint8)


def pack_table(table: npt.NDArray[np.uint8]) -> bytes:
    """Inverse of `unpack_table`. Used by `forge tok pack` / regeneration checks."""
    arr = np.ascontiguousarray(table, dtype=np.uint8)
    if arr.ndim != 1:
        raise ValueError(f"expected a 1-D table, got shape {arr.shape}")
    if not np.isin(arr, (0, 1)).all():
        raise ValueError("sampling table must contain only 0 and 1")
    return bytes(np.packbits(arr, bitorder="big").tobytes())


@lru_cache(maxsize=4)
def _load(path_str: str, size: int, verify: bool) -> npt.NDArray[np.uint8]:
    path = Path(path_str)
    if not path.is_file():
        raise FileNotFoundError(
            f"sampling table not found at {path}. It is committed key material "
            f"(data/{SAMPLING_TABLE_PATH}, 8,192 bytes); it is never generated at runtime, "
            "because a device-built table gives g-values uncorrelated with the watermark "
            "(TECH_PLAN.md sec 4.3 #3)."
        )
    packed = path.read_bytes()

    if verify:
        got_packed = _hexdigest(packed)
        want_packed = _expected(packed=True)
        if got_packed != want_packed:
            raise ValueError(
                f"sampling table digest mismatch at {path}: {digest_algorithm()}(packed) = "
                f"{got_packed}, expected {want_packed}. The table is key material; a table "
                "that differs by one bit produces a detector that reads plausible numbers "
                "meaning nothing. Restore the committed file; do NOT regenerate it."
            )

    table = unpack_table(packed, size)

    if verify:
        got_unpacked = sampling_table_digest(table)
        want_unpacked = _expected(packed=False)
        if got_unpacked != want_unpacked:  # pragma: no cover - implies a broken np.unpackbits
            raise ValueError(
                f"unpacked sampling table digest mismatch: {digest_algorithm()} = "
                f"{got_unpacked}, expected {want_unpacked}. The packed bytes verified, so "
                "this means the bit order is wrong (must be MSB-first / bitorder='big')."
            )
        ones = int(table.sum())
        if ones != SAMPLING_TABLE_ONES:  # pragma: no cover - unreachable if digests match
            raise ValueError(f"sampling table has {ones} ones, expected {SAMPLING_TABLE_ONES}")

    table.setflags(write=False)
    return table


def load_sampling_table(
    path: Path | str | None = None,
    *,
    size: int | None = None,
    verify: bool = True,
) -> npt.NDArray[np.uint8]:
    """Load, verify and cache the 65,536-entry sampling table.

    The returned array is read-only and shared between callers — the table is a
    constant, and handing out a mutable copy per call would let one caller
    silently corrupt the detector for the rest of the process. Copy it if you
    need to mutate.

    `verify=False` exists only for tests that deliberately load a *different*
    table (e.g. the `wm_other_key` control in L6, or a fuzzed table used to
    prove the digest assertion actually fires).
    """
    size = CANONICAL_CONFIG.sampling_table_size if size is None else size
    p = Path(path) if path is not None else data_dir() / SAMPLING_TABLE_PATH
    return _load(str(p.resolve()), size, verify)


def build_table_cpu(size: int = 65536, seed: int = 0) -> npt.NDArray[np.uint8]:
    """Rebuild the table the way HF transformers does, pinned to the CPU generator.

    Exactly `SynthIDTextWatermarkLogitsProcessor.__init__`'s construction::

        torch.randint(low=0, high=2, size=(size,),
                      generator=torch.Generator(device="cpu").manual_seed(seed))

    Returns numpy `uint8` so the result can be compared with `load_sampling_table()`
    directly. `forge` calls this to verify the committed bytes; nothing at
    runtime does, and nothing should.

    torch is imported here rather than at module scope on purpose: this file
    must import in the bare CI venv (pydantic + blake3 + numpy only).
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on the venv
        raise RuntimeError(
            "build_table_cpu() needs torch, which launder-core deliberately does not "
            "depend on (TECH_PLAN.md sec 2.1: core must import with only pydantic, blake3 "
            "and numpy, which is what keeps torch out of the Railway image). Run this "
            "from launder-forge: `uv sync --extra cpu`, or use load_sampling_table() "
            "which reads the committed bytes and needs no torch at all."
        ) from exc

    generator = torch.Generator(device="cpu").manual_seed(seed)
    values: Any = torch.randint(low=0, high=2, size=(size,), generator=generator, device="cpu")
    out: npt.NDArray[np.uint8] = values.numpy().astype(np.uint8)
    return out
