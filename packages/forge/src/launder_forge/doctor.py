"""`forge doctor` — environment assertions that ASSERT rather than trust.

TECH_PLAN.md §6.1. `torch.cuda.is_available()` returns True on a PTX-JIT-only
build that will then either die with `no kernel image is available` or crawl,
so the real smoke test is a bf16 matmul that is actually executed and timed.

The report is a plain dict so it can be embedded verbatim in
`data/runs/<run_id>.manifest.json` and in every passage's
`provenance.env` block.

Degradation contract: on a machine with no CUDA (CI, the Docker image, a
laptop) every check still runs and reports; nothing raises. `ok` is False and
`blocking` names what is missing in a sentence someone can act on.
"""

from __future__ import annotations

import platform
import sys
import time
from datetime import UTC, datetime
from typing import Any

__all__ = ["EXPECTED_CAPABILITY", "doctor_report", "run_id_now"]

#: RTX 5090 / Blackwell. A wheel that reports anything else is the wrong wheel;
#: cu126 installs cleanly, says `is_available() == True`, and is a trap.
EXPECTED_CAPABILITY = (12, 0)


def run_id_now(prefix: str = "run") -> str:
    return f"{prefix}_{datetime.now(UTC).strftime('%Y-%m-%dT%H-%M%SZ')}"


def _torch_module() -> Any | None:
    try:
        import torch
    except ImportError:
        return None
    return torch


def doctor_report(*, matmul_size: int = 4096) -> dict[str, Any]:
    """Run every environment check. Never raises; the caller decides severity."""
    report: dict[str, Any] = {
        "checked_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "torch": None,
        "torch_cuda_build": None,
        "cuda_available": False,
        "device_name": None,
        "capability": None,
        "capability_ok": False,
        "bf16_supported": False,
        "matmul_ok": False,
        "matmul_ms": None,
        "matmul_tflops": None,
        "vram_total_mib": None,
        "driver": None,
        "ok": False,
        "blocking": [],
        "notes": [],
    }
    blocking: list[str] = report["blocking"]
    notes: list[str] = report["notes"]

    torch = _torch_module()
    if torch is None:
        blocking.append(
            "torch is not installed in this environment. The forge pins it behind an extra so "
            "the workspace resolves on CPU-only machines: run `uv sync --extra cuda` on the "
            "5090 authoring box, or `uv sync --extra cpu` for a CPU checkout."
        )
        return report

    report["torch"] = torch.__version__
    report["torch_cuda_build"] = getattr(torch.version, "cuda", None)

    try:
        report["cuda_available"] = bool(torch.cuda.is_available())
    except Exception as exc:  # pragma: no cover - driver-level failure
        report["cuda_available"] = False
        notes.append(f"torch.cuda.is_available() raised {type(exc).__name__}: {exc}")

    if not report["cuda_available"]:
        build = report["torch_cuda_build"]
        blocking.append(
            f"no CUDA device visible to torch {report['torch']}"
            + (
                " — this is a CPU-only wheel (torch.version.cuda is None). "
                "Install the cu130 build: `uv sync --extra cuda`."
                if not build
                else f" — the wheel is built for CUDA {build} but no device is available; "
                "check the driver and that the GPU is not claimed by another process."
            )
        )
        notes.append(
            "generation (`forge gen`, `forge regen`, `forge analyze`) requires CUDA. Everything "
            "that operates on cached token ids — score, solve, triage, calibrate, vectors, "
            "pack, verify, tok pack, lint-copy — runs fine here."
        )
        return report

    try:
        report["device_name"] = torch.cuda.get_device_name(0)
        cap = tuple(torch.cuda.get_device_capability(0))
        report["capability"] = list(cap)
        report["capability_ok"] = cap == EXPECTED_CAPABILITY
        props = torch.cuda.get_device_properties(0)
        report["vram_total_mib"] = int(props.total_memory // (1024 * 1024))
        report["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
        if not report["capability_ok"]:
            blocking.append(
                f"expected Blackwell sm_{EXPECTED_CAPABILITY[0]}{EXPECTED_CAPABILITY[1]}, got "
                f"sm_{cap[0]}{cap[1]}. Wrong wheel — cu128/cu129 were removed in torch 2.13 and "
                "cu126 silently PTX-JITs. The correct pin is cu130."
            )
    except Exception as exc:  # pragma: no cover - driver-level failure
        blocking.append(f"could not query the CUDA device: {type(exc).__name__}: {exc}")
        return report

    # The actual smoke test. is_available() passing means nothing on its own.
    try:
        x = torch.randn(matmul_size, matmul_size, device="cuda", dtype=torch.bfloat16)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        total = (x @ x).sum().item()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        report["matmul_ok"] = total == total  # NaN would mean a broken kernel
        report["matmul_ms"] = round(dt * 1000, 3)
        report["matmul_tflops"] = round(2 * matmul_size**3 / dt / 1e12, 2)
        del x
        torch.cuda.empty_cache()
    except Exception as exc:
        blocking.append(
            f"the bf16 {matmul_size}x{matmul_size} matmul failed: {type(exc).__name__}: {exc}. "
            "This is the failure `torch.cuda.is_available()` hides — the wheel has no kernel "
            "image for this architecture."
        )
        return report

    try:  # nvidia-smi is optional; its absence is not blocking
        import subprocess

        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0:
            report["driver"] = out.stdout.strip().splitlines()[0].strip()
    except Exception:
        notes.append("nvidia-smi not available; driver version not recorded")

    report["ok"] = bool(report["capability_ok"] and report["matmul_ok"] and not blocking)
    return report
