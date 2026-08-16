"""The watermark: hash, sampling table, g-values, masks, ripple.

This subpackage is THE reference implementation of the SynthID sampling-table
variant (TECH_PLAN.md §4.1). The TypeScript port in `web/src/detector/` and the
`forge` generator are both required to reproduce it bit-for-bit; `data/golden/`
is the arbiter.

Import order note: nothing here imports torch. `table.build_table_cpu` imports
torch *inside the function body* precisely so that `import launder_core.watermark`
keeps working in the bare (pydantic + blake3 + numpy) venv that CI enforces.
"""

from __future__ import annotations

from launder_core.watermark.config import (
    CANONICAL_CONFIG,
    GEMMA3_EOS_TOKEN_ID,
    HASH_IV,
    LCG_INCREMENT,
    LCG_MULTIPLIER,
    SCORING_EOS_MASK,
    SCORING_EOS_TOKEN_ID,
    SynthIDConfig,
    assert_scoring_eos,
    data_dir,
    load_watermark_config,
    repo_root,
)
from launder_core.watermark.gvalues import (
    RippleSpan,
    WatermarkFrame,
    changed_span,
    compute_context_hashes,
    compute_context_repetition_mask,
    compute_depth_hashes,
    compute_eos_mask,
    compute_frame,
    compute_g_values,
    compute_ngram_hashes,
    ripple_span,
    ripple_span_for_retokenization,
    sample_index,
    sample_indices,
)
from launder_core.watermark.hash import (
    MASK64,
    accumulate_hash,
    accumulate_hash_batch,
    wrap_int64,
)
from launder_core.watermark.table import (
    SAMPLING_TABLE_BLAKE3_UNPACKED,
    SAMPLING_TABLE_PATH,
    SAMPLING_TABLE_SHA256_UNPACKED,
    build_table_cpu,
    load_sampling_table,
    pack_table,
    sampling_table_digest,
    unpack_table,
)

__all__ = [
    "CANONICAL_CONFIG",
    "GEMMA3_EOS_TOKEN_ID",
    "HASH_IV",
    "LCG_INCREMENT",
    "LCG_MULTIPLIER",
    "MASK64",
    "SAMPLING_TABLE_BLAKE3_UNPACKED",
    "SAMPLING_TABLE_PATH",
    "SAMPLING_TABLE_SHA256_UNPACKED",
    "SCORING_EOS_MASK",
    "SCORING_EOS_TOKEN_ID",
    "RippleSpan",
    "SynthIDConfig",
    "WatermarkFrame",
    "accumulate_hash",
    "accumulate_hash_batch",
    "assert_scoring_eos",
    "build_table_cpu",
    "changed_span",
    "compute_context_hashes",
    "compute_context_repetition_mask",
    "compute_depth_hashes",
    "compute_eos_mask",
    "compute_frame",
    "compute_g_values",
    "compute_ngram_hashes",
    "data_dir",
    "load_sampling_table",
    "load_watermark_config",
    "pack_table",
    "repo_root",
    "ripple_span",
    "ripple_span_for_retokenization",
    "sample_index",
    "sample_indices",
    "sampling_table_digest",
    "unpack_table",
    "wrap_int64",
]
