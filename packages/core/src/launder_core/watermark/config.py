"""Watermark configuration: the canonical `SynthIDConfig` and where `data/` is.

`SynthIDConfig`, `CANONICAL_KEYS`, `wm_config_id_of` and `canonical_json` are
NOT redeclared here. They live in `launder_core.schemas.watermark` (one source
of truth, TECH_PLAN.md §12) and this module re-exports them so the §3 module
tree's "watermark/config.py — SynthIDConfig + canonical wm_config_id hashing"
address resolves.

What this module *adds* is the two things schemas deliberately does not do,
because schemas must stay free of filesystem knowledge:

* `repo_root()` / `data_dir()` — locating the committed `data/` tree from an
  editable install, an installed wheel, or a container, with an env override.
* `load_watermark_config()` — reading `data/config/watermark.toml` and
  asserting the `wm_config_id` recorded in it against the one the six
  load-bearing fields actually hash to. A config file whose declared id does not
  match its own contents is the exact "somebody edited it and forgot to
  regenerate" failure §4.5 spends a whole tripwire section on; it fails here, at
  boot, loudly.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

from launder_core.schemas.watermark import (
    CANONICAL_KEYS,
    LCG_MULTIPLIER,
    SynthIDConfig,
    canonical_json,
    wm_config_id_of,
)

__all__ = [
    "CANONICAL_CONFIG",
    "CANONICAL_KEYS",
    "GEMMA3_EOS_TOKEN_ID",
    "HASH_IV",
    "LCG_INCREMENT",
    "LCG_MULTIPLIER",
    "SCORING_EOS_MASK",
    "SCORING_EOS_TOKEN_ID",
    "SynthIDConfig",
    "assert_scoring_eos",
    "canonical_json",
    "data_dir",
    "load_watermark_config",
    "repo_root",
    "wm_config_id_of",
]

#: The hash IV. `1`, not a key-derived seed — that is the single bit that
#: separates the canonical HF/PyPI scheme from the bit-incompatible
#: `google-deepmind/synthid-text@main` variant (TECH_PLAN.md §4.1).
HASH_IV: Final[int] = 1

#: The LCG increment (newlib/musl parameters, adapted).
LCG_INCREMENT: Final[int] = 1

#: Gemma-3's `<eos>`. `<pad>`=0, `<eos>`=1, `<bos>`=2, `<unk>`=3.
GEMMA3_EOS_TOKEN_ID: Final[int] = 1

#: **THE SHIPPED SCORING eos POLICY, IN ONE PLACE.**
#:
#: `False` — the scoring pipeline applies NO eos mask, and both runtimes are
#: required to say so explicitly. Three reasons, in order of how much they cost
#: if you get them wrong:
#:
#: 1. **It is an exploit otherwise.** `compute_eos_token_mask_full` zeroes the
#:    first `eos_token_id` *and everything after it*. `add_special_tokens=False`
#:    still maps the literal five-character string ``<eos>`` to id 1, so a player
#:    who types it near the start masks every row downstream, `n_scored`
#:    collapses, `sigma_null` explodes and `z` falls under the notch for free.
#:    The gate would clear a submission that did not launder anything.
#: 2. **The canonical scoring unit has no eos in it** (§6.2): the passage text
#:    alone, `add_special_tokens=False`. There is no generated `<eos>` to mask.
#: 3. The two runtimes DISAGREED here — the browser defaulted to 1, the server to
#:    `None` — and produced z 1.971 vs 0.109 on the same text. One constant, read
#:    by both, recorded in `watermark.toml` and asserted at boot, is the fix.
#:
#: `GEMMA3_EOS_TOKEN_ID` stays exported because `compute_eos_mask` is a faithful
#: port of the transformers detector and is exercised by golden case 5 with an
#: explicit id. Only the *default policy* is off.
SCORING_EOS_MASK: Final[bool] = False

#: The value every scoring call site passes for `eos_token_id`. Derived, so
#: there is exactly one place to flip.
SCORING_EOS_TOKEN_ID: Final[int | None] = GEMMA3_EOS_TOKEN_ID if SCORING_EOS_MASK else None

#: The published configuration, as a constructed object. Every function in this
#: subpackage defaults to it, so ordinary call sites never pass a config at all.
#: It is `frozen=True`, so this really is a constant.
CANONICAL_CONFIG: Final[SynthIDConfig] = SynthIDConfig()

_ENV_DATA_DIR: Final[str] = "LAUNDER_DATA_DIR"
_MARKER: Final[str] = "config/watermark.toml"


@lru_cache(maxsize=1)
def data_dir() -> Path:
    """Absolute path to the committed `data/` tree.

    Resolution order, first hit wins:

    1. `$LAUNDER_DATA_DIR` — the container / test override.
    2. A `data/` directory containing `config/watermark.toml` found by walking
       up from this file (covers the editable workspace install, where
       `__file__` is `packages/core/src/launder_core/watermark/config.py`).
    3. A `data/` directory found by walking up from the current working
       directory (covers `python -c` runs from a checkout with a non-editable
       core installed).

    Raises `FileNotFoundError` naming all three, rather than silently returning
    a path that does not exist — a detector that loads a *missing* sampling
    table and carries on is the failure mode this whole file exists to prevent.
    """
    override = os.environ.get(_ENV_DATA_DIR)
    if override:
        candidate = Path(override).expanduser().resolve()
        if not (candidate / _MARKER).is_file():
            raise FileNotFoundError(
                f"{_ENV_DATA_DIR}={override!r} does not contain {_MARKER}. "
                "Point it at the repository's data/ directory."
            )
        return candidate

    for start in (Path(__file__).resolve(), Path.cwd().resolve() / "_"):
        for parent in start.parents:
            candidate = parent / "data"
            if (candidate / _MARKER).is_file():
                return candidate

    raise FileNotFoundError(
        "cannot locate the data/ tree. Looked at $LAUNDER_DATA_DIR, then every "
        f"parent of {Path(__file__).resolve()}, then every parent of {Path.cwd()}. "
        f"A valid data/ directory contains {_MARKER}."
    )


def repo_root() -> Path:
    """The directory that contains `data/`. Convenience for tools and tests."""
    return data_dir().parent


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        data: dict[str, Any] = tomllib.load(fh)
    return data


def load_watermark_config(path: Path | str | None = None) -> SynthIDConfig:
    """Read `data/config/watermark.toml` into a validated `SynthIDConfig`.

    Asserts the file's own `wm_config_id` against the id its six load-bearing
    fields hash to. Also asserts `hash_iv`, `lcg_multiplier`, `lcg_increment`
    and `scheme` when present, because those four are what §4.1 says people get
    wrong: a file that says `scheme = "iterated_lcg"` describes a watermark this
    code cannot detect, and must not load quietly.
    """
    p = Path(path) if path is not None else data_dir() / "config" / "watermark.toml"
    raw = _load_toml(p)

    cfg = SynthIDConfig(
        ngram_len=int(raw["ngram_len"]),
        keys=tuple(int(k) for k in raw["keys"]),
        context_history_size=int(raw["context_history_size"]),
        sampling_table_size=int(raw["sampling_table_size"]),
        sampling_table_seed=int(raw["sampling_table_seed"]),
        skip_first_ngram_calls=bool(raw["skip_first_ngram_calls"]),
    )

    declared = raw.get("wm_config_id")
    if isinstance(declared, str):
        cfg.assert_id(declared)

    scheme = raw.get("scheme")
    if scheme is not None and scheme != "sampling_table":
        raise ValueError(
            f"{p} declares scheme={scheme!r}. This implementation is the sampling-table "
            "variant, permanently (TECH_PLAN.md sec 4.1). The 'iterated_lcg' variant from "
            "google-deepmind/synthid-text@main is bit-incompatible and CANNOT detect "
            "anything HF transformers generates."
        )
    for name, expected in (
        ("hash_iv", HASH_IV),
        ("lcg_multiplier", LCG_MULTIPLIER),
        ("lcg_increment", LCG_INCREMENT),
    ):
        got = raw.get(name)
        if got is not None and int(got) != expected:
            raise ValueError(
                f"{p} declares {name}={got}, but this implementation hard-codes {expected}. "
                "These are not configurable (TECH_PLAN.md sec 12, 'what is deliberately NOT modular')."
            )
    assert_scoring_eos(raw.get("scoring", {}), str(p))
    return cfg


def assert_scoring_eos(scoring: Mapping[str, Any], where: str) -> None:
    """`[scoring].eos_mask` / `eos_token_id` must match the shipped constants.

    The two runtimes silently disagreed about the eos mask once (browser 1,
    server None) and read z 1.971 against z 0.109 on the same sentence. The file
    now *records* the policy and this asserts it, so drift is a boot failure in
    both the server and `forge`, not a number nobody can reproduce.
    """
    declared_mask = scoring.get("eos_mask")
    if declared_mask is not None and bool(declared_mask) != SCORING_EOS_MASK:
        raise ValueError(
            f"{where} declares [scoring].eos_mask={declared_mask!r}, but "
            f"launder_core.watermark.config.SCORING_EOS_MASK is {SCORING_EOS_MASK!r}. "
            "Both runtimes read the constant; the file records it so the two cannot "
            "drift apart silently (TECH_PLAN.md sec 4.2, sec 6.2)."
        )
    declared_id = scoring.get("eos_token_id")
    if declared_id is not None and int(declared_id) != GEMMA3_EOS_TOKEN_ID:
        raise ValueError(
            f"{where} declares [scoring].eos_token_id={declared_id!r}, but Gemma-3's "
            f"<eos> is {GEMMA3_EOS_TOKEN_ID}. Changing the id would change every mask "
            "the golden vectors pin."
        )
