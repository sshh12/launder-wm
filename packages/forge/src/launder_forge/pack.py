"""`forge pack` — candidate -> `public.json` + `author.json` + `server.json` + MANIFEST.

TECH_PLAN.md §6.3. The public bundle is 3-6 KB and is the enforcement point for
CONCEPT.md's guardrail: `PassagePublic` forbids the six generation-time
optionality fields by name and `extra="forbid"` catches the rest, so a leak is
a load-time exception rather than a subtly wrong heat mirror.

Four hard gates run here, and each of them is cheaper to fail now than later:

1. `encode(text) == token_ids`. A passage whose text does not retokenize to its
   own ids desyncs the browser on keystroke zero. Reject it (§6.2).
2. The four detector expectations are RECOMPUTED, never copied from the
   candidate record. `g_digest` is the highest-value field in the schema and a
   copied one proves nothing.
3. `wm_config_id` is recomputed from `watermark.toml` and must match the asset
   bundle's.
4. The claim list must come from a REVIEWED draft (`forge claims`), because its
   labels are rendered verbatim into rejection copy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from launder_core.schemas import (
    Claim,
    DetectorExpectation,
    PassageAuthor,
    PassagePublic,
    PassageRules,
    SynthIDConfig,
)
from launder_forge.metrics import Difficulty
from launder_forge.numerics import g_digest, score_ids
from launder_forge.paths import Paths

__all__ = ["PackedPassage", "pack_passage", "write_passage"]


@dataclass(slots=True)
class PackedPassage:
    public: PassagePublic
    author: PassageAuthor
    server_sidecar: dict[str, Any]
    paths: dict[str, Path]


def pack_passage(
    *,
    passage_id: str,
    level_id: str,
    text: str,
    token_ids: list[int],
    cfg: SynthIDConfig,
    table: np.ndarray,
    asset_bundle_id: str,
    scoring_version: str,
    tokenizer: Any,
    claims: list[dict[str, Any]],
    par: int,
    par_source: str,
    judge_prompt_id: str,
    inflation: float = 1.0,
    rules: dict[str, Any] | None = None,
    difficulty: Difficulty | None = None,
    author_extra: dict[str, Any] | None = None,
    intro: dict[str, Any] | None = None,
) -> PackedPassage:
    """Build the public/author/server trio. Raises on any of the four gates."""
    encoded = list(tokenizer.encode(text))
    if encoded != list(token_ids):
        first = next(
            (i for i, (a, b) in enumerate(zip(encoded, token_ids, strict=False)) if a != b),
            min(len(encoded), len(token_ids)),
        )
        raise ValueError(
            f"passage {passage_id}: encode(text) != token_ids (first difference at index "
            f"{first}: {encoded[first : first + 3]} vs {list(token_ids)[first : first + 3]}). "
            "A passage that does not retokenize to its own ids desyncs the browser on "
            "keystroke zero and is rejected at pack time (TECH_PLAN.md §6.2)."
        )

    scored = score_ids(
        token_ids,
        keys=cfg.keys,
        ngram_len=cfg.ngram_len,
        table=table,
        context_history_size=cfg.context_history_size,
        inflation=inflation,
    )

    # `expected_z` MUST be the z the shipped detector reports, not forge's.
    #
    # §4.5's runtime tripwire has the client recompute the pristine passage and
    # assert it reproduces `expected_z`; a mismatch makes it refuse local
    # detection and show a banner. `score_ids(inflation=...)` defaults to 1.0,
    # i.e. the CLOSED FORM, while `launder_core` (and therefore the browser and
    # /api/detect) applies the measured kappa from thresholds.v1.json. Packing
    # the closed-form number would have written 1.4239 into a bundle whose
    # detector computes 1.3140 and tripped the banner on every passage, on
    # keystroke zero, in production. So: one calibration, loaded from the same
    # file, and `inflation` stays only as an explicit override for experiments.
    expected_z = scored.z
    if inflation == 1.0:
        from launder_core.detect.calibration import load_thresholds

        expected_z = load_thresholds().z(scored.score, scored.n_scored)

    public = PassagePublic(
        id=passage_id,
        level_id=level_id,
        wm_config_id=cfg.wm_config_id,
        asset_bundle_id=asset_bundle_id,
        scoring_version=scoring_version,
        text=text,
        n_words=len(text.split()),
        detector=DetectorExpectation(
            expected_n_scored=scored.n_scored,
            expected_score=scored.score,
            expected_z=expected_z,
            g_digest=g_digest(scored.g, scored.mask),
        ),
        rules=PassageRules(**(rules or {})),
        claims=tuple(Claim(**c) for c in claims),
        par=par,
        par_source=par_source,  # type: ignore[arg-type]
        judge_prompt_id=judge_prompt_id,
        intro=intro,  # type: ignore[arg-type]
    )

    author_payload: dict[str, Any] = {
        "id": passage_id,
        "claims": public.claims,
        "par": par,
        **(author_extra or {}),
    }
    if difficulty is not None:
        author_payload["difficulty"] = {
            k: v
            for k, v in difficulty.as_dict().items()
            if k
            in {
                "margin_z",
                "n_scored",
                "hot_runs",
                "concentration",
                "upstream_leverage",
                "median_eff_choices",
                "masked_fraction",
                "best_single_edit_drop_z",
                "locked_share",
                "retokenize_ok",
                "judge_baseline_pass",
            }
        }
    author = PassageAuthor(**author_payload)

    return PackedPassage(
        public=public,
        author=author,
        server_sidecar=author.to_server_sidecar().model_dump(by_alias=True),
        paths={},
    )


def write_passage(paths: Paths, packed: PackedPassage) -> dict[str, Path]:
    """Write the three files and return their paths.

    The public bundle is written with `by_alias=True` so `schema_id` lands on
    disk as `"schema"`, which is the wire spelling every reader expects.
    """
    paths.ensure(paths.passages)
    pid = packed.public.id
    out = {
        "public": paths.passages / f"{pid}.public.json",
        "author": paths.passages / f"{pid}.author.json",
        "server": paths.passages / f"{pid}.server.json",
    }
    out["public"].write_text(
        json.dumps(packed.public.model_dump(by_alias=True), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    out["author"].write_text(
        json.dumps(packed.author.model_dump(by_alias=True), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    out["server"].write_text(
        json.dumps(packed.server_sidecar, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    packed.paths.update(out)
    return out
