"""`forge verify` must catch a corrupted MANIFEST digest, a corrupted asset,
and a passage whose text does not retokenize to its own ids.

Everything runs against `tmp_repo`, a throwaway copy — the real
`data/assets/sampling_table.v1.bin` is key material and no test gets to touch it.
"""

from __future__ import annotations

import json

import pytest

from launder_core.schemas import SynthIDConfig
from launder_forge.config import load_watermark_config
from launder_forge.manifest import build_manifest, write_manifest
from launder_forge.paths import Paths
from launder_forge.verify import verify_all


@pytest.fixture
def cfg(tmp_repo: Paths) -> SynthIDConfig:
    return load_watermark_config(tmp_repo)


def test_a_clean_tree_verifies(tmp_repo: Paths, cfg: SynthIDConfig) -> None:
    write_manifest(tmp_repo, build_manifest(tmp_repo))
    report = verify_all(tmp_repo, cfg, run_node=False)
    named = {name: (ok, detail) for name, ok, detail in report.checks}
    assert named["manifest.blake3"][0], named["manifest.blake3"]
    assert named["assets.sampling_table"][0], named["assets.sampling_table"]
    assert named["config.wm_config_id"][0], named["config.wm_config_id"]


def test_verify_catches_a_corrupted_manifest_digest(tmp_repo: Paths, cfg: SynthIDConfig) -> None:
    manifest = build_manifest(tmp_repo)
    target = "data/assets/sampling_table.v1.bin"
    assert target in manifest["files"]
    manifest["files"][target]["blake3"] = "blake3:" + "0" * 64
    write_manifest(tmp_repo, manifest)

    report = verify_all(tmp_repo, cfg, run_node=False)
    failed = dict((name, detail) for name, ok, detail in report.checks if not ok)
    assert "manifest.blake3" in failed
    assert target in failed["manifest.blake3"]
    assert not report.ok


def test_verify_catches_a_file_edited_after_the_manifest(
    tmp_repo: Paths, cfg: SynthIDConfig
) -> None:
    write_manifest(tmp_repo, build_manifest(tmp_repo))
    victim = tmp_repo.config / "copy.toml"
    victim.write_text(victim.read_text(encoding="utf-8") + "\n# tampered\n", encoding="utf-8")

    report = verify_all(tmp_repo, cfg, run_node=False)
    failed = dict((name, detail) for name, ok, detail in report.checks if not ok)
    assert "manifest.blake3" in failed
    assert "copy.toml" in failed["manifest.blake3"]


def test_verify_catches_a_corrupted_sampling_table(tmp_repo: Paths, cfg: SynthIDConfig) -> None:
    """The table is key material; a flipped bit changes every g-value and the
    detector keeps producing plausible numbers. It must fail loudly."""
    data = bytearray(tmp_repo.sampling_table.read_bytes())
    data[0] ^= 0x01
    tmp_repo.sampling_table.write_bytes(bytes(data))

    report = verify_all(tmp_repo, cfg, run_node=False)
    failed = dict((name, detail) for name, ok, detail in report.checks if not ok)
    assert "assets.sampling_table" in failed
    assert "sha256_packed" in failed["assets.sampling_table"]


def test_verify_catches_a_stale_wm_config_id(tmp_repo: Paths) -> None:
    """`load_watermark_config` asserts the recorded id, so a hand-edited key
    list with a stale id cannot even be loaded."""
    text = tmp_repo.watermark_toml.read_text(encoding="utf-8")
    tmp_repo.watermark_toml.write_text(
        text.replace("ngram_len              = 5", "ngram_len              = 6"), "utf-8"
    )
    with pytest.raises(ValueError, match="wm_config_id mismatch"):
        load_watermark_config(tmp_repo)


def test_verify_rejects_a_passage_that_does_not_retokenize(
    tmp_repo: Paths, cfg: SynthIDConfig
) -> None:
    """A passage whose text does not encode back to its own ids desyncs the
    browser on keystroke zero (§6.2)."""
    from launder_forge.numerics import g_digest, load_sampling_table, score_ids
    from launder_forge.tokenizer import load_tokenizer

    tokenizer = load_tokenizer(tmp_repo, "blob")
    text = "the cutting holds its own weather and keeps it through the afternoon"
    ids = tokenizer.encode(text)
    table = load_sampling_table(tmp_repo.sampling_table)
    scored = score_ids(
        ids,
        keys=cfg.keys,
        ngram_len=cfg.ngram_len,
        table=table,
        context_history_size=cfg.context_history_size,
    )

    tmp_repo.ensure(tmp_repo.passages)
    payload = {
        "schema": "launder.passage.public/1",
        "id": "p_test",
        "level_id": "L2",
        "wm_config_id": cfg.wm_config_id,
        "asset_bundle_id": "ab1:" + "0" * 64,
        "scoring_version": "sc1",
        "text": text + " tampered",  # <- the text no longer matches the numbers
        "n_words": len(text.split()) + 1,
        "detector": {
            "expected_n_scored": scored.n_scored,
            "expected_score": scored.score,
            "expected_z": scored.z,
            "g_digest": g_digest(scored.g, scored.mask),
        },
        "par": 3,
        "par_source": "solver_upper",
        "judge_prompt_id": "judge.observe.v3",
    }
    (tmp_repo.passages / "p_test.public.json").write_text(json.dumps(payload), encoding="utf-8")

    report = verify_all(tmp_repo, cfg, run_node=False)
    failed = dict((name, detail) for name, ok, detail in report.checks if not ok)
    assert "passages.rederived" in failed
    assert "p_test" in failed["passages.rederived"]


def test_node_check_reports_rather_than_crashes(tmp_repo: Paths, cfg: SynthIDConfig) -> None:
    """There is no TS parity runner in the temp tree; verify must say so."""
    report = verify_all(tmp_repo, cfg, run_node=True)
    detail = dict((n, d) for n, ok, d in report.checks if n == "ts.parity")["ts.parity"]
    assert "parity" in detail or "node" in detail
