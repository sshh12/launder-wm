"""The watermark configuration contract.

`SynthIDConfig` is the canonical, hashable description of the watermark. Its
`wm_config_id` is asserted by the browser against every passage on load
(TECH_PLAN.md §4.5 "runtime tripwire"), which is what turns "somebody edited
`watermark.toml` and forgot to regenerate" into a loud failure instead of a
detector that reads plausible numbers that mean nothing.

Canonical id (TECH_PLAN.md §6.6), reproduced exactly:

    wm_config_id = "wm1:" + sha256(canonical_json({
        ngram_len, keys, context_history_size,
        sampling_table_size, sampling_table_seed, skip_first_ngram_calls
    }))

with sorted keys and no whitespace. Nothing else enters the digest — not the
sampling table bytes (those are covered by `asset_bundle_id`), not the model.
"""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Any, Final

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "CANONICAL_KEYS",
    "LCG_MULTIPLIER",
    "WM_CONFIG_ID_PATTERN",
    "SynthIDConfig",
    "canonical_json",
    "wm_config_id_of",
]

#: The int64 LCG multiplier used by `accumulate_hash`. Not configurable, ever
#: (TECH_PLAN.md §12 "what is deliberately NOT modular").
LCG_MULTIPLIER: Final[int] = 6364136223846793005

#: The canonical published 30-key set (TECH_PLAN.md §4.2). DeepMind's
#: `synthid_mixin`, the HF research-projects config and both published Hub
#: detector `config.json` files agree byte-for-byte. `len(keys)` IS the
#: tournament depth m = 30. Do NOT copy the transformers docstring's 9-key
#: example.
CANONICAL_KEYS: Final[tuple[int, ...]] = (
    654, 400, 836, 123, 340, 443, 597, 160, 57, 29,
    590, 639, 13, 715, 468, 990, 966, 226, 324, 585,
    118, 504, 421, 521, 129, 669, 732, 225, 90, 960,
)  # fmt: skip

WM_CONFIG_ID_PATTERN: Final[str] = r"^wm1:[0-9a-f]{64}$"

WmConfigId = Annotated[str, Field(pattern=WM_CONFIG_ID_PATTERN)]
AssetBundleId = Annotated[str, Field(pattern=r"^ab1:[0-9a-f]{64}$")]
Blake3Digest = Annotated[str, Field(pattern=r"^blake3:[0-9a-f]{64}$")]
Sha256Digest = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]


def canonical_json(obj: Any) -> str:
    """Sorted keys, no whitespace, no non-ASCII escapes surprises.

    This exact spelling is what `wm_config_id` and `asset_bundle_id` hash, and
    what the TS port must reproduce. `separators` is explicit because Python's
    default inserts a space after `:` and `,`.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def wm_config_id_of(
    *,
    ngram_len: int,
    keys: tuple[int, ...] | list[int],
    context_history_size: int,
    sampling_table_size: int,
    sampling_table_seed: int,
    skip_first_ngram_calls: bool,
) -> str:
    """Compute `wm1:<sha256>` from the six load-bearing fields.

    Kept as a free function so `forge`, `serve` and the golden-vector generator
    can call it without constructing a model, and so the TS port has one
    obvious thing to mirror.
    """
    payload = canonical_json(
        {
            "context_history_size": context_history_size,
            "keys": list(keys),
            "ngram_len": ngram_len,
            "sampling_table_seed": sampling_table_seed,
            "sampling_table_size": sampling_table_size,
            "skip_first_ngram_calls": skip_first_ngram_calls,
        }
    )
    return "wm1:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


class SynthIDConfig(BaseModel):
    """The exact SynthID configuration, as published in `data/config/watermark.toml`.

    Frozen: the config is a fact about already-generated passages, never a knob
    the runtime turns. Changing any field invalidates every passage, every
    threshold and every golden vector (TECH_PLAN.md §12 row 25).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    ngram_len: int = Field(default=5, ge=2, le=32, description="H+1: context tokens + current")
    keys: tuple[int, ...] = Field(default=CANONICAL_KEYS, min_length=1)
    context_history_size: int = Field(default=1024, ge=1)
    sampling_table_size: int = Field(default=65536, ge=2)
    sampling_table_seed: int = Field(default=0, ge=0)
    skip_first_ngram_calls: bool = Field(default=False)

    @field_validator("keys")
    @classmethod
    def _keys_are_int64(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        for k in v:
            if not (-(2**63) <= k < 2**63):
                raise ValueError(f"watermark key {k} is outside int64")
        return v

    @model_validator(mode="after")
    def _table_size_is_power_of_two(self) -> SynthIDConfig:
        n = self.sampling_table_size
        if n & (n - 1) != 0:
            raise ValueError(
                f"sampling_table_size must be a power of two (got {n}); the g-value index "
                "is `((h % N) + N) % N` and the packed bitmap is N/8 bytes"
            )
        return self

    @property
    def depth(self) -> int:
        """The tournament depth `m`. `len(keys)` IS m — 30 in the canonical set."""
        return len(self.keys)

    @property
    def packed_table_bytes(self) -> int:
        """Size of `sampling_table.v1.bin` implied by this config: 8,192 for 2**16."""
        return self.sampling_table_size // 8

    @property
    def wm_config_id(self) -> str:
        return wm_config_id_of(
            ngram_len=self.ngram_len,
            keys=self.keys,
            context_history_size=self.context_history_size,
            sampling_table_size=self.sampling_table_size,
            sampling_table_seed=self.sampling_table_seed,
            skip_first_ngram_calls=self.skip_first_ngram_calls,
        )

    def assert_id(self, expected: str) -> None:
        """Raise unless `expected` matches this config's canonical id.

        Called at boot by `serve` and on load by the browser's Python-side
        equivalent. A mismatch means the shipped passages were generated under
        a different watermark and every number on screen would be fiction.
        """
        actual = self.wm_config_id
        if actual != expected:
            raise ValueError(
                f"wm_config_id mismatch: config computes {actual}, artifact declares {expected}. "
                "The passages in data/ were generated under a different watermark config."
            )
