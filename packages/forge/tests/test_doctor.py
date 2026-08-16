"""`forge doctor` must DEGRADE, not crash, and it must never call a machine
ready when it is not.

The failure this guards is specific: `torch.cuda.is_available()` returns True on
a PTX-JIT-only build that then dies with `no kernel image is available`, so
`ok` is allowed to be True only after a bf16 matmul has actually executed.
"""

from __future__ import annotations

import builtins

import pytest
from typer.testing import CliRunner

from launder_forge.cli import app
from launder_forge.doctor import EXPECTED_CAPABILITY, doctor_report

runner = CliRunner()


def test_report_never_raises_and_always_answers() -> None:
    report = doctor_report(matmul_size=64)
    for key in ("python", "platform", "torch", "cuda_available", "ok", "blocking", "notes"):
        assert key in report
    assert isinstance(report["ok"], bool)


def test_ok_requires_a_matmul_that_actually_ran() -> None:
    report = doctor_report(matmul_size=64)
    if report["ok"]:
        assert report["matmul_ok"] is True
        assert report["capability"] == list(EXPECTED_CAPABILITY)
        assert report["matmul_ms"] is not None
    else:
        assert report["blocking"], "a not-ready environment must say what is missing"


def test_no_cuda_names_the_fix() -> None:
    """ "No CUDA" has TWO causes, and the actionable fix differs between them.

    This test used to assert `"uv sync" in joined` for both, which is only true
    of the CPU-only-wheel branch: with a cu130 wheel and no visible device,
    `uv sync` is exactly the wrong advice — the packages are already right and
    the driver or the GPU is not. `doctor.py` has always said so; the test did
    not, and it asserted the wrong branch on any machine that installs the CUDA
    extra without a card. **CI is such a machine**, which is where this surfaced:
    the python job had been failing at Postgres service startup, so this
    assertion had not actually run in CI for as long as that was broken.
    """
    report = doctor_report(matmul_size=64)
    if report["cuda_available"]:
        pytest.skip("this machine has CUDA; the degradation path is covered elsewhere")
    assert report["ok"] is False
    joined = " ".join(report["blocking"] + report["notes"])

    if report["torch"] is None:
        # No torch at all — `test_missing_torch_is_reported_not_raised` owns the
        # detail; here it is enough that the sentence exists.
        assert "uv sync" in joined
    elif report["torch_cuda_build"] is None:
        # A CPU-only wheel. The fix IS a resync, and naming it is the point.
        assert "uv sync" in joined, joined
        assert "--extra cuda" in joined, joined
    else:
        # A CUDA wheel with no device. Reinstalling changes nothing, so the
        # message must send the reader at the driver instead.
        assert "uv sync" not in joined, "a resync does not fix a missing GPU"
        assert "driver" in joined, joined

    # Either way it must say what still works: the whole point of the report is
    # that a CPU checkout can do everything except generation.
    assert "forge gen" in joined or "generation" in joined


def test_missing_torch_is_reported_not_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    """The forge pins torch behind an extra, so a bare `uv sync` has none.
    Doctor must say that in a sentence, not raise ImportError."""
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "torch" or name.startswith("torch."):
            raise ImportError("No module named 'torch'")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)
    report = doctor_report(matmul_size=64)
    assert report["torch"] is None
    assert report["ok"] is False
    assert any("torch is not installed" in b for b in report["blocking"])


def test_cli_exits_2_when_the_environment_cannot_generate() -> None:
    """Exit 2 means 'not attempted here', 1 means 'broken'. CI depends on the
    distinction."""
    result = runner.invoke(app, ["doctor", "--matmul-size", "64"])
    report = doctor_report(matmul_size=64)
    assert result.exit_code == (0 if report["ok"] else 2)
    assert "torch" in result.output


def test_cli_json_mode_is_machine_readable() -> None:
    import json

    result = runner.invoke(app, ["doctor", "--json", "--matmul-size", "64"])
    payload = json.loads(result.output)
    assert "cuda_available" in payload
