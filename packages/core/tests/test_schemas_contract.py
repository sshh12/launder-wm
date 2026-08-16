"""The shared contract, regression-tested.

TECH_PLAN.md §12 ("what is deliberately NOT modular") names two of these as
required CI tests rather than documentation:

  * "`extra="forbid"` plus a CI test asserting the entropy fields are absent"
  * "The client never sends a score. Adding a client-supplied number would undo
    §8.3's anti-cheat property in one line."

Both are asserted below against the real config files in `data/`, not against
fixtures, so a config edit that breaks the contract fails here.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import launder_core
from launder_core.schemas import (
    FORBIDDEN_PUBLIC_FIELDS,
    DetectResponse,
    EditOp,
    LevelsFile,
    PassagePublic,
    ScoringConfig,
    SubmitRequest,
    SubmitResponse,
    SynthIDConfig,
)

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "data"


def _toml(rel: str) -> dict[str, Any]:
    with (DATA / rel).open("rb") as fh:
        return tomllib.load(fh)


@pytest.fixture
def public_passage() -> dict[str, Any]:
    return {
        "schema": "launder.passage.public/1",
        "id": "p07",
        "level_id": "L2",
        "wm_config_id": SynthIDConfig().wm_config_id,
        "asset_bundle_id": "ab1:" + "44de" * 16,
        "scoring_version": "sc1",
        "text": "The cutting holds its own weather, and the frost lingers there.",
        "n_words": 11,
        "detector": {
            "expected_n_scored": 179,
            "expected_score": 0.54812,
            "expected_z": 11.83,
            "g_digest": "blake3:" + "7d21" * 16,
        },
        "rules": {"min_words": 50, "max_word_edits": 12, "locked_phrases": ["its own weather"]},
        "claims": [
            {
                "id": "c1",
                "text": "The cutting has its own microclimate.",
                "label": "the cutting's own weather",
                "required": True,
            },
        ],
        "par": 4,
        "par_source": "authored_reference",
        "judge_prompt_id": "judge.observe.v3",
    }


# ---------------------------------------------------------------------------
# The guardrail (§4.6, §12)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field", ["entropy_nats", "eff_choices", "g_mass", "wm_boost", "top1_prob", "topk_alts"]
)
def test_public_passage_rejects_generation_time_fields(
    public_passage: dict[str, Any], field: str
) -> None:
    """The browser cannot recompute these for edited text, so shipping them
    tempts a heat mirror coloured by generation-time entropy — stale instantly,
    and it teaches exactly the heuristic CONCEPT.md forbids."""
    with pytest.raises(ValidationError) as exc:
        PassagePublic.model_validate({**public_passage, field: [0.1, 0.2]})
    message = str(exc.value)
    assert field in message
    assert "author-only" in message


def test_public_passage_rejects_any_unlisted_key(public_passage: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        PassagePublic.model_validate({**public_passage, "reference_solution": "..."})


def test_forbidden_list_covers_the_plan_table() -> None:
    for field in ("entropy_nats", "eff_choices", "g_mass", "wm_boost", "top1_prob", "topk_alts"):
        assert field in FORBIDDEN_PUBLIC_FIELDS


def test_shipped_passages_carry_no_forbidden_keys() -> None:
    """Belt and braces: even a hand-written public.json is checked."""
    for path in sorted((DATA / "passages").glob("*.public.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        assert not FORBIDDEN_PUBLIC_FIELDS.intersection(raw), path
        PassagePublic.model_validate(raw)


def test_public_passage_round_trips(public_passage: dict[str, Any]) -> None:
    model = PassagePublic.model_validate(public_passage)
    assert PassagePublic.model_validate(model.model_dump(by_alias=True, mode="json")) == model


def test_locked_phrase_must_appear_in_the_text(public_passage: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        PassagePublic.model_validate(
            {**public_passage, "rules": {"locked_phrases": ["a phrase that is not there"]}}
        )


# ---------------------------------------------------------------------------
# The submit contract carries no scores (§8.3, §12)
# ---------------------------------------------------------------------------


def _submit() -> dict[str, Any]:
    return {
        "passage_id": "p07",
        "level_id": "L2",
        "text": "rewritten text",
        "client": {"detector": "local", "elapsed_ms": 184320},
    }


@pytest.mark.parametrize("field", ["score", "z", "distance", "cleared", "n_scored", "ops"])
def test_submit_rejects_client_supplied_scores(field: str) -> None:
    with pytest.raises(ValidationError) as exc:
        SubmitRequest.model_validate({**_submit(), field: 1})
    assert "no scores" in str(exc.value)


def test_submit_accepts_the_legitimate_payload() -> None:
    assert SubmitRequest.model_validate(_submit()).client.detector == "local"


def test_rejected_submission_must_name_its_check() -> None:
    with pytest.raises(ValidationError):
        SubmitResponse.model_validate(
            {
                "cleared": False,
                "score": {"distance": 1},
                "detector": {"score": 0.5, "z": 0.0, "z_star": 2.3263, "n_scored": 1},
            }
        )


# ---------------------------------------------------------------------------
# The detect wire shape is exactly what the local detector emits (§5.4, §9.2)
# ---------------------------------------------------------------------------


def test_detect_response_wire_shape_is_exact() -> None:
    payload = {
        "seq": 41,
        "text_hash": "sha256:" + hashlib.sha256(b"x").hexdigest(),
        "score": 0.5321,
        "z": 6.44,
        "z_star": 2.3263,
        "n_scored": 171,
        "n_tokens": 194,
        "tokens": [
            {"s": 0, "e": 3, "heat": 0.61, "masked": False},
            {"s": 12, "e": 16, "heat": 0.0, "masked": True},
        ],
        "preview_distance": 4,
    }
    model = DetectResponse.model_validate(payload)
    assert json.loads(model.model_dump_json()) == payload


def test_detect_response_rejects_more_scored_than_tokens() -> None:
    with pytest.raises(ValidationError):
        DetectResponse.model_validate(
            {
                "seq": 0,
                "text_hash": "sha256:" + "0" * 64,
                "score": 0.5,
                "z": 0.0,
                "z_star": 2.3263,
                "n_scored": 10,
                "n_tokens": 5,
            }
        )


def test_edit_op_serializes_the_reserved_word_key() -> None:
    wire = {"op": "sub", "i": 2, "j": 2, "from": "holds", "to": "keeps"}
    op = EditOp.model_validate(wire)
    assert op.from_ == "holds"
    assert json.loads(op.model_dump_json()) == wire


# ---------------------------------------------------------------------------
# The committed config files satisfy their own schemas
# ---------------------------------------------------------------------------


def test_watermark_toml_matches_its_recorded_id() -> None:
    raw = _toml("config/watermark.toml")
    cfg = SynthIDConfig(
        ngram_len=raw["ngram_len"],
        keys=tuple(raw["keys"]),
        context_history_size=raw["context_history_size"],
        sampling_table_size=raw["sampling_table_size"],
        sampling_table_seed=raw["sampling_table_seed"],
        skip_first_ngram_calls=raw["skip_first_ngram_calls"],
    )
    assert cfg.depth == 30, "len(keys) IS the tournament depth m"
    assert cfg.packed_table_bytes == 8192
    cfg.assert_id(raw["wm_config_id"])


def test_sampling_table_is_the_committed_key_material() -> None:
    table = (DATA / "assets" / "sampling_table.v1.bin").read_bytes()
    assert len(table) == 8192
    assert (
        hashlib.sha256(table).hexdigest()
        == "5151fe795a218d1adf0b7fd707204de23867ef2653fdebb8c9ff28d93f6aa55b"
    )


def test_levels_toml_loads_and_orders_its_checks() -> None:
    levels = LevelsFile.model_validate(_toml("config/levels.toml"))
    assert [x.id for x in levels.levels] == ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9"]

    by_id = levels.by_id()
    # ORDER IS THE SEMANTICS: every deterministic check precedes llm_gate, so a
    # mechanical failure costs zero API calls.
    for level in levels.levels:
        names = [c.check for c in level.checks]
        if "llm_gate" in names:
            assert names[-1] == "llm_gate", level.id
            assert names.index("detector_threshold") == len(names) - 2, level.id

    # Deviation #2: L5 drops llm_gate and gets its own calibration bucket.
    l5 = by_id["L5"]
    assert "llm_gate" not in [c.check for c in l5.checks]
    assert l5.param_for("detector_threshold", "calibration") == "code"

    # defaults merge UNDER the level's own params; the level wins.
    merged = levels.resolved_params(l5, l5.checks[0])
    assert merged["homoglyph_policy"] == "reject_always"
    assert merged["max_combining_marks"] == 2


def test_scoring_toml_pins_the_two_never_switches() -> None:
    raw = _toml("config/scoring.toml")
    cfg = ScoringConfig.model_validate(
        {k: v for k, v in raw.items() if k not in ("words", "distance", "lemma")}
    )
    assert cfg.lowercase is False, "casing is evidence for the judge"
    assert cfg.strip_punctuation is False, "punctuation is sentence structure"
    assert cfg.nfc is True and "nfkc" not in raw, "NFC, not NFKC — NFKC-bait is a must-fail case"


def test_copy_toml_covers_every_check_used_by_levels() -> None:
    """`forge lint-copy` rule 1: adding a check to levels.toml without writing
    its copy fails the build rather than shipping a constraint the player
    cannot interrogate."""
    levels = LevelsFile.model_validate(_toml("config/levels.toml"))
    copy = _toml("config/copy.toml")
    used = {c.check for level in levels.levels for c in level.checks}
    for name in sorted(used):
        block = copy["check"].get(name)
        assert block is not None, f"copy.toml has no [check.{name}] block"
        for key in ("label", "blurb", "reject"):
            assert key in block, f"copy.toml [check.{name}] is missing {key}"


def test_copy_never_claims_the_detector_identifies_a_human() -> None:
    """§10.7 rule 2: absence of a watermark does not prove a human wrote it."""
    readout = _toml("config/copy.toml")["readout"]
    # "AI WATERMARK detected" — the strongest true claim: the detector found the
    # watermark, which is not the same as recognising AI writing.
    assert readout["above"] == "AI watermark detected"
    assert readout["below"] == "Not detected"
    assert "human" not in json.dumps(readout).lower()
    assert "%" not in readout["above"] + readout["below"]


def test_triage_policies_all_parse_and_reject_the_wrong_lesson() -> None:
    """`upstream_leverage < 1.0` — where attacking the hot word beats cutting
    upstream — is a hard reject at every level (§4.6 mechanism 3)."""
    policies = sorted((DATA / "config" / "triage").glob("L*.toml"))
    assert len(policies) == 6
    for path in policies:
        with path.open("rb") as fh:
            raw = tomllib.load(fh)
        assert "accept" in raw and "reject_reasons" in raw, path
        assert raw["accept"]["upstream_leverage_min"] >= 1.0, path


# ---------------------------------------------------------------------------
# The lazy public surface
# ---------------------------------------------------------------------------


def test_unimplemented_public_names_say_which_file_is_missing() -> None:
    try:
        _ = launder_core.accumulate_hash
    except AttributeError as exc:
        assert "watermark/hash.py" in str(exc)
    else:  # pragma: no cover - passes once M1 lands, which is the point
        assert callable(launder_core.accumulate_hash)


def test_unknown_attribute_is_a_plain_attribute_error() -> None:
    with pytest.raises(AttributeError):
        _ = launder_core.definitely_not_a_real_symbol


def test_dir_exposes_the_declared_surface() -> None:
    names = dir(launder_core)
    assert "PassagePublic" in names
    assert "accumulate_hash" in names
    assert "weighted_mean_score" in names
