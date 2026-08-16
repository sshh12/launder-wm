"""The packer must reproduce the committed blob BYTE FOR BYTE.

`asset_bundle_id` hashes `data/assets/gemma3-tok.v1.bin.br`, so a packer that
produces a valid-but-different blob silently invalidates every passage. This is
the test that makes that impossible, and it compares bytes rather than
semantics on purpose.

Skips when `data/build/tokenizer.json` is absent — it is 33 MB of gitignored
build input, so CI on a bare checkout cannot run this. The skip message says so
rather than passing quietly.
"""

from __future__ import annotations

import hashlib

import pytest

from launder_forge.packtok import (
    _R,
    _W,
    SPM_BYTE,
    SPM_CONTROL,
    SPM_UNKNOWN,
    brotli_compress,
    brotli_decompress,
    derive_pieces,
    pack_tokenizer,
    unpack_blob,
)
from launder_forge.paths import Paths

#: Measured. `data/config/watermark.toml` records the same number.
COMMITTED_BLOB_BYTES = 1_192_944
COMMITTED_RAW_BYTES = 3_634_910


def _needs_build(paths: Paths) -> None:
    if not paths.tokenizer_json.exists():
        pytest.skip(
            "data/build/tokenizer.json is absent (33 MB, gitignored build input). "
            "Fetch it with `huggingface-cli download google/gemma-3-4b-it tokenizer.json "
            "tokenizer_config.json --local-dir data/build` to run the packer test."
        )


def test_varint_roundtrip() -> None:
    """The LEB128 + zigzag pair, in isolation, including the negative deltas
    that the merge permutation is made of."""
    w = _W()
    values = [0, 1, 127, 128, 300, 65535, 262144, 514906]
    signed = [0, -1, 1, -300, 300, -514906, 514906]
    for v in values:
        w.v(v)
    for s in signed:
        w.sv(s)
    r = _R(w.buf())
    assert [r.v() for _ in values] == values
    assert [r.sv() for _ in signed] == signed


def test_brotli_backends_agree(real_paths: Paths) -> None:
    """Whichever backend is available must reproduce the committed bytes.

    The committed blob was produced by Node's zlib at quality 11 / lgwin 24;
    the Python `brotli` binding at the same settings emits identical bytes.
    """
    committed = real_paths.tokenizer_blob.read_bytes()
    assert len(committed) == COMMITTED_BLOB_BYTES
    raw = brotli_decompress(committed)
    assert len(raw) == COMMITTED_RAW_BYTES
    assert brotli_compress(raw) == committed


def test_blob_unpacks_to_a_usable_tokenizer_spec(real_paths: Paths) -> None:
    raw = brotli_decompress(real_paths.tokenizer_blob.read_bytes())
    spec = unpack_blob(raw)
    assert len(spec["model"]["vocab"]) == 262144
    assert len(spec["model"]["merges"]) == 514906
    assert spec["model"]["byte_fallback"] is True
    types = spec["_spm_types"]
    assert types[3] == SPM_UNKNOWN
    assert [types[i] for i in (0, 1, 2)] == [SPM_CONTROL] * 3
    assert sum(1 for t in types if t == SPM_BYTE) == 256


def test_pack_reproduces_the_committed_blob_byte_for_byte(real_paths: Paths) -> None:
    _needs_build(real_paths)
    res = pack_tokenizer(real_paths.tokenizer_json, real_paths.tokenizer_config_json)
    committed = real_paths.tokenizer_blob.read_bytes()

    assert res.roundtrip_errors == 0, "the blob does not reconstruct tokenizer.json"
    assert len(res.raw) == COMMITTED_RAW_BYTES
    assert res.compressed == committed, (
        f"packed blob differs from the committed asset "
        f"({len(res.compressed)} vs {len(committed)} bytes); "
        f"sha256 {hashlib.sha256(res.compressed).hexdigest()} vs "
        f"{hashlib.sha256(committed).hexdigest()}"
    )


def test_control_symbols_are_not_taken_from_tokenizer_config(real_paths: Paths) -> None:
    """The bug that cost two bytes and was found by byte comparison.

    On `gemma-3-4b-it` the config's `eos_token` is `<end_of_turn>` (an artifact
    of the instruct tune) while SentencePiece types `<eos>` CONTROL and
    `<end_of_turn>` USER_DEFINED.
    """
    _needs_build(real_paths)
    import json

    hf = json.loads(real_paths.tokenizer_json.read_text(encoding="utf-8"))
    cfg = json.loads(real_paths.tokenizer_config_json.read_text(encoding="utf-8"))
    pieces, types = derive_pieces(hf, cfg)
    eos_id = pieces.index("<eos>")
    eot_id = pieces.index("<end_of_turn>")
    assert types[eos_id] == SPM_CONTROL
    assert types[eot_id] != SPM_CONTROL
