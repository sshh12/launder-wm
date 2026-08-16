"""The §4.2 arithmetic, against the §4.5 published constants and against core.

Two independent standards are applied on purpose:

* the constants TECH_PLAN.md publishes, so a reader can check them by eye;
* `launder_core`, so the forge's vectorised port and the shipped reference can
  never drift.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from launder_core.schemas import SynthIDConfig
from launder_forge.config import load_watermark_config
from launder_forge.numerics import (
    accumulate_hash,
    compute_context_hashes,
    compute_context_repetition_mask,
    compute_eos_mask,
    compute_g_values,
    depth_weights,
    load_sampling_table,
    ripple_span,
    sample_index,
    score_ids,
    sigma_null,
    weighted_mean_score,
)
from launder_forge.paths import Paths

# TECH_PLAN.md §4.5 golden case #1.
GOLDEN_NGRAM = [1, 235280, 2121, 576, 573]
GOLDEN_DEPTH_KEYS = [
    -6504205589568445072,
    318672039786673866,
    8070454580555681646,
    8340369110341966617,
    5852124156879677502,
]
GOLDEN_CONTEXT_HASHES = [
    -4495156567014905553,
    -8623669253612157437,
    55752042759906448,
    6202789958854077295,
]


@pytest.fixture(scope="module")
def cfg(real_paths: Paths) -> SynthIDConfig:
    return load_watermark_config(real_paths)


@pytest.fixture(scope="module")
def table(real_paths: Paths) -> np.ndarray:
    return load_sampling_table(real_paths.sampling_table)


def test_depth_keys_match_the_published_constants(cfg: SynthIDConfig) -> None:
    base = accumulate_hash(1, GOLDEN_NGRAM)
    got = [accumulate_hash(base, [k]) for k in cfg.keys[:5]]
    assert got == GOLDEN_DEPTH_KEYS


def test_context_hashes_match_the_published_constants() -> None:
    tokens = [*GOLDEN_NGRAM, 2121, 576, 573]
    got = compute_context_hashes(np.asarray(tokens, dtype=np.int64), ngram_len=5)
    assert [int(x) for x in got[:4]] == GOLDEN_CONTEXT_HASHES


def test_negative_modulo_is_python_remainder() -> None:
    """§4.3 divergence #1, the single most likely one and the least loud."""
    assert sample_index(-12345678901234567, 65536) == 46201
    assert sample_index(-1, 65536) == 65535
    assert sample_index(-65536, 65536) == 0


def test_masked_index_equals_the_double_remainder(cfg: SynthIDConfig, table: np.ndarray) -> None:
    """`h & (N-1)` on the uint64 representation IS `((h % N) + N) % N`."""
    rng = np.random.default_rng(7)
    ids = rng.integers(0, 262144, size=80).astype(np.int64)
    g = compute_g_values(ids, keys=cfg.keys, ngram_len=cfg.ngram_len, table=table)
    for row in range(0, g.shape[0], 17):
        base = accumulate_hash(1, ids[row : row + cfg.ngram_len].tolist())
        for depth in (0, 7, 29):
            hl = accumulate_hash(base, [cfg.keys[depth]])
            assert g[row, depth] == table[sample_index(hl, cfg.sampling_table_size)]


def test_int64_wraparound_is_two_complement() -> None:
    """Every add AND every multiply wraps; the result is a signed int64."""
    h = accumulate_hash(1, [262143] * 40)
    assert -(2**63) <= h < 2**63


def test_repetition_mask_history_starts_full_of_zeros() -> None:
    """A context hash of exactly 0 is a repeat on its first appearance, because
    HF pre-fills the ring buffer with zeros. Easy to get wrong, invisible when
    wrong."""
    ctx = np.asarray([0, 5, 5, 7], dtype=np.int64)
    mask = compute_context_repetition_mask(ctx, context_history_size=1024)
    assert [int(x) for x in mask] == [0, 1, 0, 1]


def test_repetition_mask_fast_path_agrees_with_the_loop() -> None:
    rng = np.random.default_rng(11)
    ctx = rng.integers(-(2**62), 2**62, size=500).astype(np.int64)
    fast = compute_context_repetition_mask(ctx)
    ctx_with_dupe = ctx.copy()
    ctx_with_dupe[400] = ctx_with_dupe[10]
    slow = compute_context_repetition_mask(ctx_with_dupe)
    assert fast.sum() == 500
    assert slow.sum() == 499
    assert slow[400] == 0


def test_eos_mask_zeroes_from_the_first_eos_and_is_sliced() -> None:
    ids = [5, 6, 7, 8, 9, 10, 1, 11, 12, 13, 14]
    mask = compute_eos_mask(np.asarray(ids, dtype=np.int64), eos_token_id=1, ngram_len=5)
    assert [int(x) for x in mask] == [1, 1, 0, 0, 0, 0, 0]


def test_ripple_is_exactly_ngram_len_rows(cfg: SynthIDConfig, table: np.ndarray) -> None:
    """§4.4, as an executable assertion a transformers upgrade cannot break."""
    rng = np.random.default_rng(3)
    before = rng.integers(0, 262144, size=60).astype(np.int64)
    after = before.copy()
    after[30] = (int(after[30]) + 12345) % 262144
    gb = compute_g_values(before, keys=cfg.keys, ngram_len=5, table=table)
    ga = compute_g_values(after, keys=cfg.keys, ngram_len=5, table=table)
    differing = [int(i) for i in np.nonzero((gb != ga).any(axis=1))[0]]
    assert differing == [26, 27, 28, 29, 30]
    assert len(differing) == cfg.ngram_len
    assert ripple_span(30, ngram_len=5, n_rows=gb.shape[0]) == (26, 31)


def test_weights_sum_to_m() -> None:
    w = depth_weights(30)
    assert w.shape == (30,)
    assert abs(float(w.sum()) - 30.0) < 1e-12
    assert w[0] > w[-1]


def test_sigma_null_matches_core_closed_form() -> None:
    from launder_core.detect.calibration import sigma_closed_form

    for n in (10, 57, 157, 400):
        assert abs(sigma_null(n, 30) - sigma_closed_form(n, 30)) < 1e-15


def test_score_of_a_passage_is_reproducible(
    real_paths: Paths, cfg: SynthIDConfig, table: np.ndarray
) -> None:
    from launder_forge.tokenizer import load_tokenizer

    text = (real_paths.root / "data" / "dev" / "passage.txt").read_text(encoding="utf-8").strip()
    ids = load_tokenizer(real_paths).encode(text)
    a = score_ids(
        ids,
        keys=cfg.keys,
        ngram_len=cfg.ngram_len,
        table=table,
        context_history_size=cfg.context_history_size,
    )
    b = score_ids(
        ids,
        keys=cfg.keys,
        ngram_len=cfg.ngram_len,
        table=table,
        context_history_size=cfg.context_history_size,
    )
    assert a.score == b.score
    assert a.digest == b.digest
    assert a.n_scored == len(ids) - cfg.ngram_len + 1 - int((a.mask == 0).sum())


# ---------------------------------------------------------------------------
# parity with the reference implementation
# ---------------------------------------------------------------------------


def _core_or_skip(name: str) -> Any:
    from launder_forge.corebridge import core_symbol

    fn = core_symbol(name)
    if fn is None:
        pytest.skip(f"launder_core.{name} has not landed yet; parity unverified")
    return fn


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_g_values_match_core(seed: int, cfg: SynthIDConfig, table: np.ndarray) -> None:
    core_g = _core_or_skip("compute_g_values")
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, 262144, size=120).astype(np.int64)
    mine = compute_g_values(ids, keys=cfg.keys, ngram_len=cfg.ngram_len, table=table)
    theirs = np.asarray(core_g(ids, cfg, table))
    assert np.array_equal(mine, theirs)


def test_context_hashes_and_masks_match_core(cfg: SynthIDConfig) -> None:
    core_ctx = _core_or_skip("compute_context_hashes")
    core_rep = _core_or_skip("compute_context_repetition_mask")
    core_eos = _core_or_skip("compute_eos_mask")
    ids = np.asarray(
        [7, 8, 9, 10, 11, 12, 13, 7, 8, 9, 10, 11, 99, 7, 8, 9, 10, 11, 12, 13], dtype=np.int64
    )
    mine_ctx = compute_context_hashes(ids, ngram_len=cfg.ngram_len)
    assert np.array_equal(mine_ctx, np.asarray(core_ctx(ids, cfg.ngram_len)))
    mine_rep = compute_context_repetition_mask(
        mine_ctx, context_history_size=cfg.context_history_size
    )
    assert np.array_equal(mine_rep.astype(bool), np.asarray(core_rep(ids, cfg)))
    mine_eos = compute_eos_mask(ids, eos_token_id=13, ngram_len=cfg.ngram_len)
    assert np.array_equal(mine_eos.astype(bool), np.asarray(core_eos(ids, 13, cfg)))


def test_weighted_mean_matches_core(cfg: SynthIDConfig, table: np.ndarray) -> None:
    core_score = _core_or_skip("weighted_mean_score")
    rng = np.random.default_rng(5)
    ids = rng.integers(0, 262144, size=200).astype(np.int64)
    g = compute_g_values(ids, keys=cfg.keys, ngram_len=cfg.ngram_len, table=table)
    ctx = compute_context_hashes(ids, ngram_len=cfg.ngram_len)
    mask = compute_context_repetition_mask(ctx, context_history_size=cfg.context_history_size)
    mine, n_scored = weighted_mean_score(g, mask, depth_weights(cfg.depth))
    theirs = core_score(g, mask.astype(bool))
    assert abs(mine - float(theirs.score)) < 1e-12
    assert n_scored == int(theirs.n_scored)


def test_accumulate_hash_matches_core() -> None:
    core_acc = _core_or_skip("accumulate_hash")
    rng = np.random.default_rng(9)
    for _ in range(20):
        data = rng.integers(0, 262144, size=int(rng.integers(1, 12))).tolist()
        assert accumulate_hash(1, data) == core_acc(1, data)
