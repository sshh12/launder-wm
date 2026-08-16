"""`forge table build` and `forge check-table`.

`check-table` settles TECH_PLAN.md §14.2 item 1 — the *device-dependent
sampling table*, §4.3's silent failure #3. HF builds the table with
``torch.Generator(device=device).manual_seed(0)``; CPU is MT19937 and CUDA is
Philox. If the two disagree, generating on the 5090 and detecting in the
browser produces g-values **uncorrelated with the watermark** — a needle that
moves, looks plausible, and measures nothing.

The mitigation ships regardless of the answer: the table is built once on CPU,
committed as `data/assets/sampling_table.v1.bin`, and every consumer loads that
file — including the generator, which overwrites the processor's own table
(see `launder_forge.generate`). `check-table` exists so we *know*, and so a
future torch upgrade that changes MT19937 seeding is caught by CI rather than
by a broken daily.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from launder_forge.numerics import U8Array, unpack_sampling_table

__all__ = ["TableCheck", "build_table", "check_table", "digests_of", "pack_table"]


def build_table(*, size: int = 65536, seed: int = 0, device: str = "cpu") -> U8Array:
    """Exactly HF's construction: ``torch.randint(0, 2, (size,), generator=g, device=device)``.

    Reproduced rather than approximated — the point of the check is that the
    bytes match what `SynthIDTextWatermarkLogitsProcessor.__init__` would build.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "building a sampling table needs torch, because the table IS "
            "`torch.randint(0, 2, (65536,), generator=torch.Generator(device).manual_seed(0))` "
            "and reimplementing MT19937/Philox by hand would defeat the purpose of the check. "
            "Run `uv sync --extra cpu` (or --extra cuda)."
        ) from exc
    generator = torch.Generator(device=device).manual_seed(seed)
    t = torch.randint(low=0, high=2, size=(size,), generator=generator, device=device)
    return t.to("cpu").numpy().astype(np.uint8)


def pack_table(values: U8Array) -> bytes:
    """``np.packbits`` MSB-first — the numpy default and the on-disk format."""
    return np.packbits(values, bitorder="big").tobytes()


def digests_of(values: U8Array) -> dict[str, Any]:
    """The four digests recorded in `[table]` of `data/config/watermark.toml`."""
    import blake3

    packed = pack_table(values)
    unpacked = values.astype(np.uint8).tobytes()
    return {
        "values": int(values.shape[0]),
        "packed_bytes": len(packed),
        "ones": int(values.sum()),
        "sha256_packed": hashlib.sha256(packed).hexdigest(),
        "sha256_unpacked": hashlib.sha256(unpacked).hexdigest(),
        "blake3_packed": blake3.blake3(packed).hexdigest(),
        "blake3_unpacked": blake3.blake3(unpacked).hexdigest(),
    }


@dataclass(slots=True)
class TableCheck:
    """The result of `forge check-table`. `cuda_ran` is the honest part."""

    committed_path: Path
    size: int
    seed: int
    cpu_matches: bool
    cpu_first_mismatch: int | None
    cuda_ran: bool
    cuda_matches: bool | None = None
    cuda_first_mismatch: int | None = None
    cuda_vs_cpu_agree: bool | None = None
    committed_digests: dict[str, Any] = field(default_factory=dict)
    cpu_digests: dict[str, Any] = field(default_factory=dict)
    cuda_digests: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """CPU parity is the shipping requirement. CUDA disagreement is a
        *finding*, not a failure: the committed CPU table is authoritative and
        the generator overwrites the processor's table with it either way."""
        return self.cpu_matches


def _first_mismatch(a: U8Array, b: U8Array) -> int | None:
    diff = np.nonzero(a != b)[0]
    return None if diff.size == 0 else int(diff[0])


def check_table(committed: Path, *, size: int = 65536, seed: int = 0) -> TableCheck:
    values = unpack_sampling_table(committed.read_bytes(), size)
    cpu = build_table(size=size, seed=seed, device="cpu")
    result = TableCheck(
        committed_path=committed,
        size=size,
        seed=seed,
        cpu_matches=bool(np.array_equal(cpu, values)),
        cpu_first_mismatch=_first_mismatch(cpu, values),
        cuda_ran=False,
        committed_digests=digests_of(values),
        cpu_digests=digests_of(cpu),
    )
    if not result.cpu_matches:
        result.notes.append(
            "the committed table is NOT what this torch build produces on CPU. Either the "
            "table was built with a different seed/size, or torch changed MT19937 seeding. "
            "The committed file wins — it is key material and every shipped passage was "
            "scored with it — but this must be understood before the next regeneration."
        )

    try:
        import torch

        cuda_ok = torch.cuda.is_available()
    except ImportError:
        cuda_ok = False
        result.notes.append("torch is not installed; the CUDA half of §14.2 item 1 is unrun")
    if not cuda_ok:
        result.notes.append(
            "no CUDA device here, so the CUDA-vs-CPU half of the verification is UNRUN. "
            "It does not block anything: the generator overwrites the processor's table with "
            "the committed CPU bytes (§4.3), so a divergence would be neutralised, not hidden."
        )
        return result

    cuda = build_table(size=size, seed=seed, device="cuda")
    result.cuda_ran = True
    result.cuda_matches = bool(np.array_equal(cuda, values))
    result.cuda_first_mismatch = _first_mismatch(cuda, values)
    result.cuda_vs_cpu_agree = bool(np.array_equal(cuda, cpu))
    result.cuda_digests = digests_of(cuda)
    if not result.cuda_vs_cpu_agree:
        result.notes.append(
            "CONFIRMED: the CUDA generator (Philox) and the CPU generator (MT19937) produce "
            "DIFFERENT sampling tables for the same seed. This is §4.3 failure #3. The "
            "committed CPU table plus the generator-side override is what makes it harmless; "
            "never let a code path build the table on device."
        )
    else:
        result.notes.append(
            "CUDA and CPU produce the same table for this seed on this torch build. The "
            "override still ships — the guarantee is cheap and the failure mode is silent."
        )
    return result
