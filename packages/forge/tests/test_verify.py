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


def test_ts_parity_refuses_to_settle_for_a_script_that_is_not_the_parity_runner(
    tmp_repo: Paths, cfg: SynthIDConfig
) -> None:
    """The candidate list used to fall back to `web/tools/pack-check.mjs`.

    pack-check round-trips the packed tokenizer blob. It never loads the
    detector, never reads data/golden/vectors.json and never scores a passage —
    so with `parity.mjs` renamed or deleted, `forge verify` would have run it,
    seen exit 0 and reported `ts.parity  yes`, and the only check in the repo
    that can catch a JS/Python divergence would have been passing by not
    running. Anything but the real runner is now a failure.
    """
    tools = tmp_repo.root / "web" / "tools"
    tools.mkdir(parents=True)
    # Exits 0 without scoring anything, exactly like a successful pack-check.
    (tools / "pack-check.mjs").write_text("process.exit(0);\n", encoding="utf-8")

    report = verify_all(tmp_repo, cfg, run_node=True)
    ok, detail = dict((n, (ok, d)) for n, ok, d in report.checks if n == "ts.parity")["ts.parity"]
    assert not ok, detail
    assert "web/tools/parity.mjs" in detail
    assert "pack-check" not in detail


def test_verify_catches_a_stale_expected_z(
    tmp_repo: Paths, real_paths: Paths, cfg: SynthIDConfig
) -> None:
    """§4.5 lists four detector expectations; check 5 recomputed three.

    `expected_z` is the one the BROWSER asserts on load — it recomputes the
    pristine passage and, on a mismatch, refuses local detection behind a
    banner. It is also the one that goes stale on its own: it is a function of
    `data/assets/thresholds.v1.json` as well as of the text, so recalibrating
    after packing invalidates it while score, n_scored and g_digest all still
    reproduce.
    """
    shipped = sorted(real_paths.passages.glob("*.public.json"))
    if not shipped:
        pytest.skip("no packed passage to copy into the temp tree")
    data = json.loads(shipped[0].read_text(encoding="utf-8"))
    data["detector"]["expected_z"] += 0.25
    tmp_repo.ensure(tmp_repo.passages)
    (tmp_repo.passages / shipped[0].name).write_text(json.dumps(data), encoding="utf-8")

    report = verify_all(tmp_repo, cfg, run_node=False)
    failed = dict((name, detail) for name, ok, detail in report.checks if not ok)
    assert "passages.rederived" in failed
    assert "!= packed" in failed["passages.rederived"]


def test_a_level_naming_a_calibration_that_does_not_exist_fails(
    tmp_repo: Paths, cfg: SynthIDConfig
) -> None:
    """L5 named `calibration = "code"`; thresholds.v1.json has only `default`.

    `parse_thresholds` raises `KeyError` on an unknown bucket set, so that was a
    500 at PLAY on the level's win condition — the failure ARCHITECTURE.md §8
    says must happen at boot instead. `load_levels` cannot catch it: it checks
    that params are READ, never that their values name something that exists.
    The two files have to be read together.
    """
    text = tmp_repo.levels_toml.read_text(encoding="utf-8")
    tmp_repo.levels_toml.write_text(
        text.replace(
            '{ check = "detector_threshold", params = { max_z = 0.0 } },',
            '{ check = "detector_threshold", params = { max_z = 0.0, calibration = "code" } },',
            1,
        ),
        encoding="utf-8",
    )

    report = verify_all(tmp_repo, cfg, run_node=False)
    failed = dict((name, detail) for name, ok, detail in report.checks if not ok)
    assert "config.calibrations" in failed
    assert "'code'" in failed["config.calibrations"]
    assert "['default']" in failed["config.calibrations"]


def test_the_shipped_levels_only_name_calibrations_that_exist(
    tmp_repo: Paths, cfg: SynthIDConfig
) -> None:
    report = verify_all(tmp_repo, cfg, run_node=False)
    named = {name: (ok, detail) for name, ok, detail in report.checks}
    assert named["config.calibrations"][0], named["config.calibrations"]


def test_a_table_digest_that_is_not_recorded_is_a_failure_not_a_skip(
    tmp_repo: Paths, cfg: SynthIDConfig
) -> None:
    """`if k in recorded` made an absent digest silently pass.

    The sampling table is key material and every shipped passage was scored with
    it. Deleting the four hashes from `[table]` used to leave the check with
    nothing to compare and the report still read "4 digests match".
    """
    text = tmp_repo.watermark_toml.read_text(encoding="utf-8")
    stripped = "\n".join(
        line for line in text.splitlines() if not line.startswith(("sha256_", "blake3_"))
    )
    tmp_repo.watermark_toml.write_text(stripped + "\n", encoding="utf-8")

    report = verify_all(tmp_repo, cfg, run_node=False)
    ok, detail = dict((n, (ok, d)) for n, ok, d in report.checks if n == "assets.sampling_table")[
        "assets.sampling_table"
    ]
    assert not ok, detail
    assert "sha256_packed: not recorded" in detail
    assert "blake3_unpacked: not recorded" in detail
