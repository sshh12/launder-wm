"""THE PARITY GATE — Python against TypeScript, on the same inputs.

TECH_PLAN.md §4.5 says the golden file has *three* runners: pytest, vitest and
`forge verify`. Two of those check the same numbers against the same committed
file, which catches a golden that drifted from the assets but NOT the failure
this project actually fears: Python and TypeScript quietly computing different
things and each agreeing with its own half of the file.

So this test runs the real browser detector (`node web/tools/parity.mjs`,
esbuild-bundling the shipped TS modules) and compares its output to
`launder_core` — the shipped Python detector, not forge's numpy port — element
for element:

* **g-values BIT-IDENTICAL.** Not "close": the same 0/1 in every one of the
  ~4,700 cells of the dev passage's matrix. A score can agree by luck; a bit
  matrix cannot. This is where §4.3's four traps surface — int64 wraparound,
  the negative-modulo sign, the hash-then-fold key order, and the table's
  device-dependent generation.
* **|z_py - z_ts| < 1e-9**, which additionally pins the calibration curve and
  its interpolation axis, since z is the only number that reads them.
* **Per-token heat and masks**, which is what the mirror paints.
* **Token ids**, so a divergence in the tokenizer cannot hide behind a
  divergence in the detector (or vice versa).

Skips are environmental only, and each one names what is missing.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import Any

import numpy as np
import pytest

from launder_core.detect import detect_ids
from launder_core.detect.calibration import load_thresholds
from launder_core.watermark import compute_frame
from launder_core.watermark.config import CANONICAL_CONFIG, SCORING_EOS_TOKEN_ID
from launder_forge.paths import repo_paths

pytestmark = pytest.mark.golden

_NODE = shutil.which("node")


def _rows_to_bits(g: np.ndarray) -> list[str]:
    return ["".join(str(int(v)) for v in row) for row in g]


def _per_token(reading: object, n_tokens: int) -> tuple[list[float], list[int]]:
    """Core's per-ROW heat, laid out as the per-TOKEN wire array (§9.2).

    This mapping is a cross-package contract, not a test convenience, and it is
    written out here because it is the one place both sides of it are visible:
    row `i` covers `ids[i .. i+n-1]` and its heat belongs to the CURRENT token
    `i + n - 1`, so the leading `ngram_len - 1` tokens are the final token of no
    window. They read neutral 0.5 and `masked: true` — absence of evidence, not
    evidence of innocence. `launder_serve.api.detect` must build `tokens[]` the
    same way or the mirror's first four spans are wrong on every passage.
    """
    heat = [0.5] * n_tokens
    masked = [1] * n_tokens
    for token_index, h, is_masked in reading.token_heat():  # type: ignore[attr-defined]
        heat[token_index] = h
        masked[token_index] = 1 if is_masked else 0
    return heat, masked


@pytest.fixture(scope="module")
def ts_report(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Run the browser detector over every shipped input and read its report."""
    paths = repo_paths()
    runner = paths.root / "web" / "tools" / "parity.mjs"
    if _NODE is None:
        pytest.skip("node is not installed; the TS half of the parity gate cannot run here")
    if not runner.is_file():
        pytest.skip(f"{runner} is missing")
    if not (paths.root / "web" / "node_modules" / "esbuild").exists():
        pytest.skip("web/node_modules is not installed (run `npm install` in web/)")

    out = tmp_path_factory.mktemp("parity") / "ts.json"
    proc = subprocess.run(
        [_NODE, str(runner), "--emit", str(out), "--quiet"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(paths.root / "web"),
        timeout=600,
        check=False,
    )
    assert proc.returncode == 0, (
        "web/tools/parity.mjs failed — the TS detector disagrees with the golden file "
        f"before Python is even consulted:\n{proc.stdout}\n{proc.stderr}"
    )
    report: dict[str, Any] = json.loads(out.read_text(encoding="utf-8"))
    return report


def test_wm_config_id_agrees(ts_report: dict[str, Any]) -> None:
    assert ts_report["wm_config_id"] == CANONICAL_CONFIG.wm_config_id


def test_calibration_is_one_curve(ts_report: dict[str, Any]) -> None:
    """The browser and the server must be reading the same sigma(T)."""
    cal = load_thresholds()
    ts_cal = ts_report["calibration"]
    assert ts_cal["depth"] == cal.depth
    assert ts_cal["z_star"] == pytest.approx(cal.z_star, abs=0.0)
    assert [(b["n_scored"], b["sigma"]) for b in ts_cal["buckets"]] == [
        (b.n_scored, b.sigma) for b in cal.buckets
    ]


def test_g_values_bit_identical_on_the_golden_sequence(ts_report: dict[str, Any]) -> None:
    case = ts_report["cases"]["g_values"]
    frame = compute_frame(np.asarray(case["ids"], dtype=np.int64))
    assert _rows_to_bits(frame.g) == case["g_rows"]


def test_context_hashes_and_repetition_mask_agree(ts_report: dict[str, Any]) -> None:
    case = ts_report["cases"]["repetition_mask"]
    frame = compute_frame(np.asarray(case["ids"], dtype=np.int64))
    assert [str(int(h)) for h in frame.context_hashes] == case["context_hashes"]
    assert [int(v) for v in frame.repetition_mask] == case["repetition_mask"]
    assert [int(v) for v in frame.mask] == case["mask"]


def test_the_two_runtimes_share_one_eos_policy(ts_report: dict[str, Any]) -> None:
    """The gate's blind spot, closed.

    The TS runner used to be handed the golden file's `eos_token_id` explicitly,
    so the gate compared two runtimes in a configuration NEITHER of them shipped:
    the worker omitted the option and masked at id 1, `compute_frame` omitted it
    and masked nothing, and one sentence read z 1.971 against z 0.109. This
    asserts the browser's default IS core's default, and every score case below
    is now scored through that default on both sides.
    """
    assert ts_report["cases"]["score"]["eos_from_default"] is True
    assert ts_report["cases"]["score"]["eos_token_id"] == SCORING_EOS_TOKEN_ID


def test_score_cases_agree_to_1e_9(ts_report: dict[str, Any]) -> None:
    for cc in ts_report["cases"]["score"]["cases"]:
        ids = np.asarray(cc["ids"], dtype=np.int64)
        # No `eos_token_id=`: the DEFAULT is the thing under test.
        reading = detect_ids(ids)
        frame = compute_frame(ids)
        label = cc["label"]
        assert _rows_to_bits(frame.g) == cc["g_rows"], f"g-values differ on {label}"
        assert reading.n_scored == cc["n_scored"], label
        assert reading.n_tokens == cc["n_tokens"], label
        assert abs(reading.score - cc["score"]) < 1e-12, label
        assert abs(reading.z - cc["z"]) < 1e-9, f"z differs on {label}"
        heat, masked = _per_token(reading, cc["n_tokens"])
        assert len(heat) == len(cc["heat"]), label
        for i in range(len(heat)):
            assert abs(heat[i] - cc["heat"][i]) < 1e-12, f"{label} heat[{i}]"
            assert masked[i] == cc["masked"][i], f"{label} masked[{i}]"


def test_real_passages_agree_bit_for_bit(ts_report: dict[str, Any]) -> None:
    """The end-to-end assertion: real prose, real tokenizer, both detectors."""
    passages = ts_report["passages"]
    assert passages, (
        "the TS runner scored no passage — data/assets/gemma3-tok.v1.bin.br is missing, "
        "so the tokenizer half of the gate did not run"
    )
    for p in passages:
        ids = np.asarray(p["ids"], dtype=np.int64)
        frame = compute_frame(ids)
        reading = detect_ids(ids)
        assert _rows_to_bits(frame.g) == p["g_rows"], f"g-values differ on {p['id']}"
        assert [int(v) for v in frame.mask] == [
            1 - m for m in p["masked"][CANONICAL_CONFIG.ngram_len - 1 :]
        ], f"mask differs on {p['id']}"
        assert reading.n_scored == p["n_scored"], p["id"]
        assert abs(reading.score - p["score"]) < 1e-12, p["id"]
        assert abs(reading.z - p["z"]) < 1e-9, f"z differs on {p['id']}"
        heat, masked = _per_token(reading, p["n_tokens"])
        assert masked == p["masked"], f"masked differs on {p['id']}"
        for i in range(len(heat)):
            assert abs(heat[i] - p["heat"][i]) < 1e-12, f"{p['id']} heat[{i}]"


def test_python_tokenizes_the_dev_passage_to_the_same_ids(ts_report: dict[str, Any]) -> None:
    """A detector divergence must not be able to hide behind a tokenizer one."""
    from launder_forge.tokenizer import load_tokenizer

    paths = repo_paths()
    try:
        tok = load_tokenizer(paths)
    except RuntimeError as exc:  # no `tokenizers` wheel / no blob
        pytest.skip(f"python tokenizer unavailable: {exc}")

    dev = paths.root / "data" / "dev" / "passage.txt"
    if not dev.is_file():
        pytest.skip("data/dev/passage.txt is missing")
    text = dev.read_text(encoding="utf-8").removesuffix("\n")
    entry = next((p for p in ts_report["passages"] if p["id"] == "data/dev/passage.txt"), None)
    assert entry is not None
    assert tok.encode(text) == entry["ids"]


def test_normalize_agrees(ts_report: dict[str, Any]) -> None:
    from launder_core.scoring.normalize import normalize, words

    for e in ts_report["cases"]["normalize"]:
        assert normalize(e["input"]) == e["output"], repr(e["input"])
        assert list(words(e["input"])) == e["words"], repr(e["input"])


def test_damerau_agrees(ts_report: dict[str, Any]) -> None:
    from launder_core.scoring.damerau import damerau_levenshtein

    cases = ts_report["cases"].get("distance", [])
    for e in cases:
        assert damerau_levenshtein(e["a"], e["b"]) == e["distance"], (e["a"], e["b"])


def test_the_gate_covered_something(ts_report: dict[str, Any]) -> None:
    """A parity gate that silently scored nothing is worse than no gate."""
    cells = sum(len(r) for p in ts_report["passages"] for r in p["g_rows"])
    cells += sum(len(r) for r in ts_report["cases"]["g_values"]["g_rows"])
    assert cells > 4000, f"only {cells} g-value cells compared"
