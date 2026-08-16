"""The watermark, asserted against the two authorities: golden case #1 and transformers.

Every number in this file is either quoted from TECH_PLAN.md §4 or was produced
by stock `transformers.SynthIDTextWatermarkLogitsProcessor` on CPU and pasted
here. Nothing is a recording of our own output taken on faith — that is the
point of §4.5's "regenerating goldens is never a way to turn a red test green".
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np
import pytest

from launder_core.schemas import CANONICAL_KEYS
from launder_core.watermark import (
    CANONICAL_CONFIG,
    HASH_IV,
    LCG_MULTIPLIER,
    MASK64,
    SynthIDConfig,
    accumulate_hash,
    accumulate_hash_batch,
    changed_span,
    compute_context_hashes,
    compute_context_repetition_mask,
    compute_depth_hashes,
    compute_eos_mask,
    compute_frame,
    compute_g_values,
    compute_ngram_hashes,
    data_dir,
    load_sampling_table,
    load_watermark_config,
    pack_table,
    ripple_span,
    ripple_span_for_retokenization,
    sample_index,
    sample_indices,
    sampling_table_digest,
    unpack_table,
    wrap_int64,
)
from launder_core.watermark.table import (
    SAMPLING_TABLE_BLAKE3_PACKED,
    SAMPLING_TABLE_BLAKE3_UNPACKED,
    SAMPLING_TABLE_ONES,
    SAMPLING_TABLE_SHA256_PACKED,
    SAMPLING_TABLE_SHA256_UNPACKED,
    build_table_cpu,
    digest_algorithm,
)

# ---------------------------------------------------------------------------
# Golden case #1 (TECH_PLAN.md §4.5, table row 1)
# ---------------------------------------------------------------------------

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

#: A fixed 50-id sequence in Gemma-3's id range. `numpy.random.default_rng(20260815)
#: .integers(0, 262144, size=50)`, pasted as a literal so the test does not depend
#: on numpy's generator staying stable across versions.
IDS50 = [
    215610, 138091, 219223, 171859, 23395, 99265, 163097, 203079, 173775, 3970,
    203991, 257260, 244918, 41944, 44352, 94167, 75494, 41821, 212637, 214940,
    173176, 189772, 187056, 91825, 130257, 138554, 188913, 136055, 80489, 20932,
    78594, 7631, 84085, 192785, 204535, 180241, 168342, 252411, 239777, 139929,
    176981, 206645, 235157, 160504, 114980, 83406, 120557, 27751, 46984, 202851,
]  # fmt: skip

#: Golden case #4: the full 0/1 matrix on IDS50, recorded as its digest plus the
#: first row and the total. Produced by transformers 5.15.0 `compute_g_values`
#: on CPU, and asserted bit-identical against it below whenever torch is present.
G50_BLAKE3 = "327030418ccb2283fb83cf33905bf9e517103c0ed58a0dc418050c0c4d48a01e"
G50_ROW0 = [0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0]  # fmt: skip
G50_SUM = 710


def test_accumulate_hash_golden_case_1_depth_keys() -> None:
    """The FULL n-gram is hashed first, then each key is folded in as one more step."""
    h = accumulate_hash(HASH_IV, GOLDEN_NGRAM)
    got = [accumulate_hash(h, [k]) for k in CANONICAL_KEYS[:5]]
    assert got == GOLDEN_DEPTH_KEYS


def test_accumulate_hash_golden_case_1_context_hashes() -> None:
    """ctx[i] = accumulate_hash(1, t[i .. i+n-2]) — the n-1 LEADING tokens."""
    got = [accumulate_hash(HASH_IV, GOLDEN_NGRAM[i : i + 4]) for i in range(2)]
    assert got == GOLDEN_CONTEXT_HASHES[:2]


def test_compute_depth_hashes_matches_golden() -> None:
    """The vectorized depth-key path reproduces golden case #1."""
    ngram = np.array(GOLDEN_NGRAM, dtype=np.int64)
    ngram_hash = compute_ngram_hashes(ngram, 5)
    assert ngram_hash.shape == (1,)
    depth = compute_depth_hashes(ngram_hash, CANONICAL_KEYS[:5])
    assert depth.shape == (1, 5)
    assert depth[0].tolist() == GOLDEN_DEPTH_KEYS


def test_context_hashes_over_a_longer_sequence() -> None:
    """All four of golden case #1's context hashes, from one sliding sequence.

    The plan publishes four context hashes but only the five-token n-gram, and
    five tokens yield two 4-grams. The two remaining tokens were recovered by
    INVERTING the LCG: the multiplier is odd, so it is invertible mod 2**64, and
    `tok = (target - 1) * M^-1 - h` is unique. It returns 2121 and 576 — tokens
    that already appear in the published n-gram, which is the kind of coincidence
    that does not happen by accident.

    So windows 0-1 test the published values independently; windows 2-3 test
    that a single consistent sequence reproduces all four.
    """
    # The trailing 573 is arbitrary and deliberately so: `compute_context_hashes`
    # drops the last token before unfolding, so an eighth token is needed to get
    # a fourth context and its VALUE cannot affect any of them.
    ids = np.array([*GOLDEN_NGRAM, 2121, 576, 573], dtype=np.int64)
    ctx = compute_context_hashes(ids, 5)
    assert ctx.tolist() == GOLDEN_CONTEXT_HASHES


# ---------------------------------------------------------------------------
# §4.3 #2 — int64 wraparound
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, 0),
        (2**63 - 1, 2**63 - 1),
        (2**63, -(2**63)),
        (2**64, 0),
        (2**64 - 1, -1),
        (-(2**63) - 1, 2**63 - 1),
        (LCG_MULTIPLIER * LCG_MULTIPLIER, 7520897724310334953),
    ],
)
def test_wrap_int64(value: int, expected: int) -> None:
    assert wrap_int64(value) == expected


def test_accumulate_hash_wraps_rather_than_growing() -> None:
    """A Python int would reach ~10^60 after 5 steps; int64 keeps it in range."""
    h = accumulate_hash(HASH_IV, [2**62, 2**62, 2**62, 2**62, 2**62])
    assert -(2**63) <= h < 2**63
    # And the wrap is genuinely mod 2^64, not saturation or truncation.
    manual = 1
    for tok in [2**62] * 5:
        manual = ((manual + tok) * LCG_MULTIPLIER + 1) & MASK64
    assert h == wrap_int64(manual)


def test_hash_is_prefix_composable() -> None:
    """f(x, data[:T]) == f(f(x, data[:T-1]), data[T]) — why keys can be folded in."""
    data = [7, 11, 13, 17, 19, 23]
    assert accumulate_hash(HASH_IV, data) == accumulate_hash(
        accumulate_hash(HASH_IV, data[:4]), data[4:]
    )


def test_scalar_and_vectorized_hash_agree() -> None:
    """The fast numpy path can never drift from the readable scalar one."""
    rng = np.random.default_rng(11)
    data = rng.integers(-(2**40), 2**40, size=(64, 7), dtype=np.int64)
    batch = accumulate_hash_batch(HASH_IV, data)
    scalar = [accumulate_hash(HASH_IV, row.tolist()) for row in data]
    assert batch.tolist() == scalar
    assert batch.dtype == np.int64


def test_vectorized_hash_accepts_per_row_seeds() -> None:
    seeds = np.array([1, -5, 2**62], dtype=np.int64)
    data = np.array([[3, 4], [5, 6], [7, 8]], dtype=np.int64)
    got = accumulate_hash_batch(seeds, data)
    want = [accumulate_hash(int(s), row.tolist()) for s, row in zip(seeds, data, strict=True)]
    assert got.tolist() == want


# ---------------------------------------------------------------------------
# §4.3 #1 — the negative-modulo trap, in isolation (golden case #2)
# ---------------------------------------------------------------------------

#: `torch.remainder(-12345678901234567, 65536)`. JS `BigInt %` gives -19335 for
#: the same input, which is the single most likely port divergence and produces
#: a detector that reads ~0.5 on everything.
NEGATIVE_MODULO_CASE = (-12345678901234567, 46201, -19335)


def test_sample_index_negative_modulo() -> None:
    value, torch_result, js_truncating_result = NEGATIVE_MODULO_CASE
    assert sample_index(value, 65536) == torch_result
    assert sample_index(value, 65536) != js_truncating_result
    # The C/JS truncating remainder, spelled out, so the trap is visible here.
    truncating = value - int(value / 65536) * 65536 if value >= 0 else -((-value) % 65536)
    assert truncating == js_truncating_result


def test_sample_index_matches_torch_remainder() -> None:
    torch = pytest.importorskip("torch", reason="reference remainder needs torch")
    rng = np.random.default_rng(3)
    values = rng.integers(-(2**63), 2**63 - 1, size=512, dtype=np.int64)
    want = torch.remainder(torch.tensor(values), 65536).numpy()
    assert sample_indices(values, 65536).tolist() == want.tolist()
    assert [sample_index(int(v), 65536) for v in values[:32]] == want[:32].tolist()


def test_sample_indices_are_always_in_range() -> None:
    rng = np.random.default_rng(5)
    values = rng.integers(-(2**63), 2**63 - 1, size=4096, dtype=np.int64)
    idx = sample_indices(values, 65536)
    assert idx.min() >= 0
    assert idx.max() < 65536


# ---------------------------------------------------------------------------
# the sampling table — key material
# ---------------------------------------------------------------------------


def test_sampling_table_digest_and_shape() -> None:
    table = load_sampling_table()
    assert table.shape == (65536,)
    assert table.dtype == np.uint8
    assert set(np.unique(table).tolist()) <= {0, 1}
    assert int(table.sum()) == SAMPLING_TABLE_ONES
    assert table[:32].tolist() == [
        0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0,
        0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0,
    ]  # fmt: skip
    expected = (
        SAMPLING_TABLE_BLAKE3_UNPACKED
        if digest_algorithm() == "blake3"
        else SAMPLING_TABLE_SHA256_UNPACKED
    )
    assert sampling_table_digest(table) == expected


def test_sampling_table_is_read_only_and_cached() -> None:
    a = load_sampling_table()
    b = load_sampling_table()
    assert a is b, "the table is a constant; handing out per-call copies invites corruption"
    with pytest.raises(ValueError):
        a[0] = 1 - a[0]


def test_packed_bytes_round_trip() -> None:
    packed = (data_dir() / "assets" / "sampling_table.v1.bin").read_bytes()
    assert len(packed) == 8192
    table = unpack_table(packed, 65536)
    assert pack_table(table) == packed


def test_watermark_toml_agrees_with_the_hard_coded_digests() -> None:
    """Two copies of the same fact must not be allowed to drift."""
    with (data_dir() / "config" / "watermark.toml").open("rb") as fh:
        raw = tomllib.load(fh)
    tbl = raw["table"]
    assert tbl["sha256_packed"] == SAMPLING_TABLE_SHA256_PACKED
    assert tbl["sha256_unpacked"] == SAMPLING_TABLE_SHA256_UNPACKED
    assert tbl["blake3_packed"] == SAMPLING_TABLE_BLAKE3_PACKED
    assert tbl["blake3_unpacked"] == SAMPLING_TABLE_BLAKE3_UNPACKED
    assert tbl["ones"] == SAMPLING_TABLE_ONES
    assert tbl["packed_bytes"] == 8192
    assert tbl["format"] == "packbits_msb_first"


def test_corrupt_table_fails_loudly(tmp_path: Path) -> None:
    packed = bytearray((data_dir() / "assets" / "sampling_table.v1.bin").read_bytes())
    packed[17] ^= 0x01
    victim = tmp_path / "sampling_table.v1.bin"
    victim.write_bytes(bytes(packed))
    with pytest.raises(ValueError, match="digest mismatch"):
        load_sampling_table(victim)
    # ... and the escape hatch still works, for the L6 wm_other_key control.
    assert load_sampling_table(victim, verify=False).shape == (65536,)


def test_missing_table_names_the_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="sampling table not found"):
        load_sampling_table(tmp_path / "nope.bin")


def test_load_watermark_config_matches_the_canonical_object() -> None:
    cfg = load_watermark_config()
    assert cfg == CANONICAL_CONFIG
    assert cfg.depth == 30
    assert cfg.keys == CANONICAL_KEYS
    assert cfg.wm_config_id.startswith("wm1:")


# ---------------------------------------------------------------------------
# g-values (golden case #4)
# ---------------------------------------------------------------------------


def test_g_values_shape_and_golden_digest() -> None:
    g = compute_g_values(IDS50)
    assert g.shape == (50 - 5 + 1, 30)
    assert g.dtype == np.uint8
    assert g[0].tolist() == G50_ROW0
    assert int(g.sum()) == G50_SUM
    if digest_algorithm() == "blake3":
        assert sampling_table_digest(g.reshape(-1)) == G50_BLAKE3


def test_g_values_are_position_independent() -> None:
    """Row i depends only on ids[i:i+n]. This is why insertions do not shift
    downstream g-values, and why the ripple genuinely stops (§4.4)."""
    ids = np.array(IDS50, dtype=np.int64)
    g = compute_g_values(ids)
    shifted = compute_g_values(np.concatenate([[999, 998], ids]))
    assert np.array_equal(g, shifted[2:])


def test_g_values_short_sequences_are_empty_not_an_error() -> None:
    """A four-token textarea mid-edit is a legitimate state, not a crash."""
    for n_tokens in range(0, 5):
        g = compute_g_values(np.arange(n_tokens, dtype=np.int64))
        assert g.shape == (0, 30)
    assert compute_g_values(np.arange(5, dtype=np.int64)).shape == (1, 30)


def test_a_different_key_set_gives_uncorrelated_g_values() -> None:
    """The L6 `wm_other_key` control: same ids, different key, detector reads clean."""
    other = SynthIDConfig(keys=tuple(k + 1 for k in CANONICAL_KEYS))
    a = compute_g_values(IDS50)
    b = compute_g_values(IDS50, other)
    assert a.shape == b.shape
    agreement = float((a == b).mean())
    assert 0.4 < agreement < 0.6, "different keys must not correlate"


# ---------------------------------------------------------------------------
# masks
# ---------------------------------------------------------------------------

#: `[11,22,33,44,55]` appears twice, so the 4-gram contexts at rows 8 and 9
#: repeat rows 0 and 1. Verified against transformers below.
REPEATED_IDS = [11, 22, 33, 44, 55, 99, 98, 97, 11, 22, 33, 44, 55, 96, 95]
REPEATED_MASK = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1]


def test_repetition_mask_with_a_deliberately_repeated_4gram() -> None:
    mask = compute_context_repetition_mask(REPEATED_IDS)
    assert mask.astype(int).tolist() == REPEATED_MASK
    ctx = compute_context_hashes(REPEATED_IDS)
    assert int(ctx[8]) == int(ctx[0])
    assert int(ctx[9]) == int(ctx[1])


def test_repetition_mask_is_causal() -> None:
    """History only ever reaches rightward: the FIRST occurrence always scores."""
    mask = compute_context_repetition_mask(REPEATED_IDS)
    assert bool(mask[0]) and bool(mask[1])


def test_repetition_mask_pushes_even_when_repeated() -> None:
    """Three occurrences: the 2nd and 3rd are both masked, and the ring advanced
    three times. Skipping the push on a repeat would desync us from the generator."""
    ids = [*[1, 2, 3, 4, 5], *[9, 8, 7], *[1, 2, 3, 4, 5], *[9, 8, 7], *[1, 2, 3, 4, 5]]
    mask = compute_context_repetition_mask(ids)
    ctx = compute_context_hashes(ids)
    first = int(ctx[0])
    hits = [i for i, h in enumerate(ctx.tolist()) if h == first]
    assert len(hits) == 3
    assert bool(mask[hits[0]]) and not bool(mask[hits[1]]) and not bool(mask[hits[2]])


def test_repetition_history_is_bounded() -> None:
    """A repeat further back than context_history_size scores again."""
    cfg = SynthIDConfig(context_history_size=4)
    ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5]
    ctx = compute_context_hashes(ids, cfg.ngram_len)
    mask = compute_context_repetition_mask(ids, cfg)
    hits = [i for i, h in enumerate(ctx.tolist()) if h == int(ctx[0])]
    assert len(hits) == 2
    assert bool(mask[hits[1]]), "the first context fell out of the 4-deep ring"


def test_eos_mask_kills_the_tail() -> None:
    ids = np.array(IDS50, dtype=np.int64)
    ids[40] = 1
    mask = compute_eos_mask(ids, 1)
    assert mask.shape == (46,)
    # Row i's current token is i+4, so rows 0..35 survive and 36.. die.
    assert mask[:36].all()
    assert not mask[36:].any()
    assert compute_eos_mask(ids, None).all()


def test_frame_is_internally_consistent() -> None:
    frame = compute_frame(REPEATED_IDS)
    assert frame.n_rows == len(REPEATED_IDS) - 4
    assert frame.mask.tolist() == (frame.repetition_mask & frame.eos_mask).tolist()
    assert frame.n_scored == sum(REPEATED_MASK)
    assert frame.masked_fraction == pytest.approx(2 / 11)
    assert frame.token_index(0) == 4
    assert frame.row_index(4) == 0


# ---------------------------------------------------------------------------
# §4.4 — the ripple blast radius. The claim, proved.
# ---------------------------------------------------------------------------


def test_ripple_is_exactly_ngram_len_wide() -> None:
    """TECH_PLAN.md §4.4, executable: 60 random tokens, mutate index 30, the
    differing g-value ROWS are [26,27,28,29,30] and the affected current-token
    indices are [30,31,32,33,34] — the edited token itself plus the next 4."""
    rng = np.random.default_rng(4242)
    a = rng.integers(0, 262144, size=60, dtype=np.int64)
    b = a.copy()
    b[30] = int((b[30] + 12345) % 262144)

    ga = compute_g_values(a)
    gb = compute_g_values(b)
    diff_rows = np.flatnonzero((ga != gb).any(axis=1)).tolist()
    assert diff_rows == [26, 27, 28, 29, 30]
    assert [r + 4 for r in diff_rows] == [30, 31, 32, 33, 34]

    span = ripple_span(30, n_tokens=60, ngram_len=5)
    assert list(span.row_indices) == diff_rows
    assert list(span.token_indices) == [30, 31, 32, 33, 34]
    assert span.width == 5

    # Row 31 onward is BIT-IDENTICAL: g-values are a pure function of content.
    assert np.array_equal(ga[31:], gb[31:])
    assert np.array_equal(ga[:26], gb[:26])


def test_ripple_holds_for_every_interior_index() -> None:
    """Not a lucky index: assert the invariant across the whole sequence."""
    rng = np.random.default_rng(99)
    a = rng.integers(0, 262144, size=40, dtype=np.int64)
    ga = compute_g_values(a)
    for j in range(len(a)):
        b = a.copy()
        b[j] = int((b[j] + 7777) % 262144)
        diff = np.flatnonzero((ga != compute_g_values(b)).any(axis=1)).tolist()
        assert diff == list(ripple_span(j, n_tokens=40, ngram_len=5).row_indices), f"index {j}"


def test_ripple_span_clips_at_the_edges() -> None:
    assert ripple_span(0, n_tokens=60).rows == (0, 1)
    assert ripple_span(0, n_tokens=60).tokens == (0, 5)
    assert ripple_span(59, n_tokens=60).rows == (55, 56)
    assert ripple_span(59, n_tokens=60).tokens == (59, 60)
    assert ripple_span(58, n_tokens=60).width == 2
    with pytest.raises(IndexError):
        ripple_span(60, n_tokens=60)


def test_changed_span_diffs_by_common_prefix_and_suffix() -> None:
    old = [1, 2, 3, 4, 5, 6]
    assert changed_span(old, old) == (6, 6, 6)
    assert changed_span(old, [1, 2, 99, 4, 5, 6]) == (2, 3, 3)
    assert changed_span(old, [1, 2, 4, 5, 6]) == (2, 3, 2)  # deletion
    assert changed_span(old, [1, 2, 7, 8, 3, 4, 5, 6]) == (2, 2, 4)  # insertion


def test_retokenization_ripple_covers_the_real_difference() -> None:
    """§4.4(a): one word edit is 1-3 token edits. Diff, then extend by n-1.

    The claim being tested is the one that licenses the solver's incremental
    g-value cache: OUTSIDE the returned span, every row of the new sequence is
    bit-identical to a row of the old one — the ones before the edit at the same
    index, the ones after it shifted by the length delta. g-values are a pure
    function of token content, so an insertion or a deletion re-indexes rows but
    does not recompute them.
    """
    rng = np.random.default_rng(2718)
    a = rng.integers(0, 262144, size=48, dtype=np.int64)
    cases = [
        ("substitute 2 -> 3", np.concatenate([a[:20], [123, 456, 789], a[22:]])),
        ("delete 3", np.concatenate([a[:20], a[23:]])),
        ("insert 1", np.concatenate([a[:20], [55], a[20:]])),
        ("substitute 1", np.concatenate([a[:20], [777], a[21:]])),
    ]
    for label, new in cases:
        span = ripple_span_for_retokenization(a, new, 5)
        _prefix, old_stop, new_stop = changed_span(a, new)
        delta = old_stop - new_stop  # suffix alignment offset
        ga = compute_g_values(a)
        gb = compute_g_values(new)
        row_start, row_stop = span.rows

        for i in range(row_start):
            assert np.array_equal(gb[i], ga[i]), f"{label}: row {i} before the span moved"
        for i in range(row_stop, gb.shape[0]):
            assert np.array_equal(gb[i], ga[i + delta]), f"{label}: row {i} after the span moved"

    identical = ripple_span_for_retokenization(a, a.copy(), 5)
    assert identical.width == 0
    assert identical.rows == (0, 0)


# ---------------------------------------------------------------------------
# THE cross-check: stock transformers, bit for bit
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def hf_processor() -> object:
    """Stock `SynthIDTextWatermarkLogitsProcessor` on CPU with the canonical config."""
    pytest.importorskip("torch", reason="cross-check needs torch")
    lp = pytest.importorskip(
        "transformers.generation.logits_process",
        reason="cross-check needs transformers (it lives in launder-forge, not core)",
    )
    import torch

    return lp.SynthIDTextWatermarkLogitsProcessor(
        ngram_len=CANONICAL_CONFIG.ngram_len,
        keys=list(CANONICAL_CONFIG.keys),
        sampling_table_size=CANONICAL_CONFIG.sampling_table_size,
        sampling_table_seed=CANONICAL_CONFIG.sampling_table_seed,
        context_history_size=CANONICAL_CONFIG.context_history_size,
        device=torch.device("cpu"),
        skip_first_ngram_calls=CANONICAL_CONFIG.skip_first_ngram_calls,
    )


def test_committed_table_equals_the_hf_cpu_table(hf_processor: object) -> None:
    """§4.3 #3, answered: on CPU, HF's own table IS the committed file.

    The override in the generator ships regardless — this test says the CPU
    branch of the device question is settled, not that the override is
    unnecessary. The CUDA branch cannot be tested here and is exactly why the
    generator overwrites `proc.sampling_table` unconditionally.
    """
    hf_table = np.asarray(hf_processor.sampling_table.numpy()).astype(np.uint8)  # type: ignore[attr-defined]
    assert np.array_equal(hf_table, load_sampling_table())
    assert sampling_table_digest(hf_table) == sampling_table_digest(load_sampling_table())


def test_g_values_are_bit_identical_to_transformers(hf_processor: object) -> None:
    import torch

    ids = np.array(IDS50, dtype=np.int64)
    hf_g = hf_processor.compute_g_values(torch.tensor(ids[None, :]))[0]  # type: ignore[attr-defined]
    assert np.array_equal(compute_g_values(ids), hf_g.numpy().astype(np.uint8))


def test_masks_are_bit_identical_to_transformers(hf_processor: object) -> None:
    """Including the combined mask, which is `rep * eos[:, ngram_len-1:]` in
    `SynthIDTextWatermarkDetector.__call__`."""
    import torch

    ids = np.array(IDS50, dtype=np.int64)
    ids[40] = 1
    t = torch.tensor(ids[None, :])

    hf_rep = hf_processor.compute_context_repetition_mask(t)[0].numpy().astype(bool)  # type: ignore[attr-defined]
    hf_eos = hf_processor.compute_eos_token_mask(t, eos_token_id=1)[0].numpy().astype(bool)  # type: ignore[attr-defined]
    hf_eos_sliced = hf_eos[CANONICAL_CONFIG.ngram_len - 1 :][: len(ids) - 4]

    frame = compute_frame(ids, eos_token_id=1)
    assert np.array_equal(frame.repetition_mask, hf_rep)
    assert np.array_equal(frame.eos_mask, hf_eos_sliced)
    assert np.array_equal(frame.mask, hf_rep * hf_eos_sliced)


def test_repeated_context_mask_matches_transformers(hf_processor: object) -> None:
    import torch

    ids = np.array(REPEATED_IDS, dtype=np.int64)
    hf_rep = hf_processor.compute_context_repetition_mask(torch.tensor(ids[None, :]))[0]  # type: ignore[attr-defined]
    assert np.array_equal(compute_context_repetition_mask(ids), hf_rep.numpy().astype(bool))


def test_build_table_cpu_reproduces_the_committed_bytes() -> None:
    """The regeneration path forge uses to verify the key material."""
    pytest.importorskip("torch", reason="build_table_cpu needs torch, which core does not have")
    rebuilt = build_table_cpu(65536, 0)
    assert np.array_equal(rebuilt, load_sampling_table())
    assert pack_table(rebuilt) == (data_dir() / "assets" / "sampling_table.v1.bin").read_bytes()
