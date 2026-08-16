"""Level loading — TECH_PLAN.md §7.6.

Every test here asserts the same thing from a different angle: **a bad level
config is a STARTUP error, not a play-time one.** A level that names a check
nobody wrote, or lists the paid judge before the free detector, or sets a param
nobody reads, must never reach a player — the first two are a 500 mid-puzzle
and the third is a checklist row claiming to enforce a rule that is not being
enforced, which is worse because nobody notices.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import pytest

from launder_core.gates import REGISTRY, GateConfigError
from launder_core.levels import load_levels, load_levels_file, validate_level
from launder_core.schemas import CheckSpec, LevelConfig

REPO = Path(__file__).resolve().parents[3]
LEVELS_TOML = REPO / "data" / "config" / "levels.toml"


def write_levels(tmp_path: Path, levels: list[dict[str, Any]], **top: Any) -> Path:
    """Render a levels.toml from Python, so a test can express a broken config
    without a fixture file that somebody later 'fixes'."""
    import json

    payload: dict[str, Any] = {
        "schema": "launder.levels/1",
        "judge_version": "g3",
        "defaults": top.pop("defaults", {}),
        "levels": levels,
    }
    payload.update(top)
    # TOML has no writer in the stdlib; JSON is a subset of what tomllib
    # accepts semantically, so round-trip through a tiny emitter instead.
    path = tmp_path / "levels.toml"
    path.write_text(_emit_toml(payload), encoding="utf-8")
    assert tomllib.loads(path.read_text(encoding="utf-8"))
    del json
    return path


def _emit_toml(payload: dict[str, Any]) -> str:
    def value(v: Any) -> str:
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, str):
            return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
        if isinstance(v, (int, float)):
            return repr(v)
        if isinstance(v, list):
            return "[" + ", ".join(value(x) for x in v) + "]"
        if isinstance(v, dict):
            return "{ " + ", ".join(f"{k} = {value(x)}" for k, x in v.items()) + " }"
        raise TypeError(type(v))

    lines = [
        f"schema = {value(payload['schema'])}",
        f"judge_version = {value(payload['judge_version'])}",
    ]
    for check, params in payload["defaults"].items():
        lines.append(f"[defaults.{check}]")
        lines.extend(f"{k} = {value(v)}" for k, v in params.items())
    for level in payload["levels"]:
        lines.append("[[levels]]")
        for key, v in level.items():
            lines.append(f"{key} = {value(v)}")
    return "\n".join(lines) + "\n"


def a_level(**overrides: Any) -> dict[str, Any]:
    level: dict[str, Any] = {
        "id": "L1",
        "name": "Clean it",
        "teaches": "the loop",
        "par_source": "authored",
        "checks": [
            {
                "check": "unicode_sanitation",
                "params": {
                    "reject_categories": ["zero_width"],
                    "homoglyph_policy": "reject_unless_in_original",
                    "max_combining_marks": 2,
                },
            },
            {"check": "word_floor", "params": {"min_words": 50}},
            {"check": "detector_threshold", "params": {"max_z": 0.0}},
        ],
    }
    level.update(overrides)
    return level


# ---------------------------------------------------------------------------
# The shipped config
# ---------------------------------------------------------------------------
def test_every_level_loads_and_validates() -> None:
    levels = load_levels()
    # L1-L6 are CONCEPT.md's original six; L7-L9 are the campaign's own,
    # added after the shipped levels were measured and found to be
    # scriptable end to end.
    assert sorted(levels) == ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9"]
    for level in levels.values():
        assert isinstance(level, LevelConfig)
        assert level.checks[0].check == "unicode_sanitation"
        for spec in level.checks:
            assert spec.check in REGISTRY


def test_defaults_are_merged_before_the_checks_ever_see_them() -> None:
    """`run_gate` and the checks never learn that [defaults.*] exists — the
    level's own params win and the merge happens once, at boot."""
    levels = load_levels()
    sanitation = levels["L5"].checks[0]
    assert sanitation.params["homoglyph_policy"] == "reject_always", "the level wins"
    assert sanitation.params["max_combining_marks"] == 2, "the default fills the rest"
    assert levels["L6"].checks[1].params["min_words"] == 120
    assert levels["L1"].checks[1].params["min_words"] == 50


def test_every_shipped_level_runs_the_detector_and_ends_with_the_paid_step() -> None:
    for level in load_levels().values():
        names = [c.check for c in level.checks]
        assert "detector_threshold" in names, f"{level.id} has no win condition"
        if "llm_gate" in names:
            assert names[-1] == "llm_gate"
            assert names[-2] == "detector_threshold"


def test_l5_drops_the_judge_and_keeps_the_unit_test() -> None:
    """Deviation from CONCEPT.md #2 (§7.6)."""
    names = [c.check for c in load_levels()["L5"].checks]
    assert "unit_test" in names
    assert "llm_gate" not in names


def test_the_shipped_file_and_the_loader_agree_on_level_count() -> None:
    raw = tomllib.loads(LEVELS_TOML.read_text(encoding="utf-8"))
    assert len(raw["levels"]) == len(load_levels()) == 9
    assert load_levels_file().judge_version == raw["judge_version"]


# ---------------------------------------------------------------------------
# Everything that must fail AT BOOT
# ---------------------------------------------------------------------------
def test_an_unregistered_check_raises_at_load_time(tmp_path: Path) -> None:
    broken = a_level()
    broken["checks"] = [*broken["checks"], {"check": "vibe_check", "params": {}}]
    path = write_levels(tmp_path, [broken])
    with pytest.raises(GateConfigError) as exc:
        load_levels(path)
    message = str(exc.value)
    assert "vibe_check" in message
    assert "gates/checks/vibe_check.py" in message, "the error must name the file to write"


def test_a_level_without_unicode_sanitation_raises(tmp_path: Path) -> None:
    broken = a_level(
        checks=[
            {"check": "word_floor", "params": {"min_words": 50}},
            {"check": "detector_threshold", "params": {"max_z": 0.0}},
        ]
    )
    with pytest.raises(GateConfigError, match=r"ctx\.normalized"):
        load_levels(write_levels(tmp_path, [broken]))


def test_a_check_out_of_pipeline_order_raises(tmp_path: Path) -> None:
    """Order is the semantics: a deterministic check behind the paid one means
    a mechanical failure starts costing an API call."""
    broken = a_level(
        checks=[
            {
                "check": "unicode_sanitation",
                "params": {
                    "reject_categories": ["zero_width"],
                    "homoglyph_policy": "reject_always",
                    "max_combining_marks": 2,
                },
            },
            {"check": "detector_threshold", "params": {"max_z": 0.0}},
            {"check": "word_floor", "params": {"min_words": 50}},
        ]
    )
    with pytest.raises(GateConfigError, match="Order is the semantics"):
        load_levels(write_levels(tmp_path, [broken]))


def test_the_judge_may_not_be_reached_without_a_detector_pass(tmp_path: Path) -> None:
    """The single largest cost saver in the product (§7.1): only submissions
    that already beat the watermark ever reach the LLM. A level that skips the
    detector sends every submission, including the hopeless ones, to a paid
    provider."""
    broken = a_level(
        checks=[
            {
                "check": "unicode_sanitation",
                "params": {
                    "reject_categories": ["zero_width"],
                    "homoglyph_policy": "reject_always",
                    "max_combining_marks": 2,
                },
            },
            {"check": "word_floor", "params": {"min_words": 50}},
            {
                "check": "llm_gate",
                "params": {"require_claims": "all", "max_missing_claims": 0, "max_added_claims": 0},
            },
        ]
    )
    with pytest.raises(GateConfigError, match="immediately before"):
        load_levels(write_levels(tmp_path, [broken]))


def test_the_paid_step_may_not_run_before_a_free_one(tmp_path: Path) -> None:
    broken = a_level(
        checks=[
            {
                "check": "unicode_sanitation",
                "params": {
                    "reject_categories": ["zero_width"],
                    "homoglyph_policy": "reject_always",
                    "max_combining_marks": 2,
                },
            },
            {
                "check": "llm_gate",
                "params": {"require_claims": "all", "max_missing_claims": 0, "max_added_claims": 0},
            },
            {"check": "detector_threshold", "params": {"max_z": 0.0}},
        ]
    )
    with pytest.raises(GateConfigError, match="Order is the semantics"):
        load_levels(write_levels(tmp_path, [broken]))


def test_a_missing_required_param_raises(tmp_path: Path) -> None:
    broken = a_level()
    broken["checks"] = [
        *broken["checks"][:2],
        {"check": "edit_budget", "params": {}},
        broken["checks"][2],
    ]
    with pytest.raises(GateConfigError, match="max_word_distance"):
        load_levels(write_levels(tmp_path, [broken]))


def test_a_param_nobody_reads_raises(tmp_path: Path) -> None:
    """Most often a rename that only landed in one of the two files. Silently
    ignoring it ships a rule that is not enforced."""
    broken = a_level()
    broken["checks"] = [
        *broken["checks"][:2],
        {"check": "edit_budget", "params": {"max_word_distance": 12, "max_word_edits": 12}},
        broken["checks"][2],
    ]
    with pytest.raises(GateConfigError, match="not read by that check"):
        load_levels(write_levels(tmp_path, [broken]))


def test_defaults_for_an_unregistered_check_raise(tmp_path: Path) -> None:
    path = write_levels(tmp_path, [a_level()], defaults={"vibe_check": {"strength": 3}})
    with pytest.raises(GateConfigError, match="silently apply to nothing"):
        load_levels(path)


def test_a_duplicate_check_in_one_level_is_rejected_by_the_schema(tmp_path: Path) -> None:
    broken = a_level()
    broken["checks"] = [*broken["checks"], {"check": "word_floor", "params": {"min_words": 5}}]
    with pytest.raises(Exception, match="same check twice"):
        load_levels(write_levels(tmp_path, [broken]))


def test_validate_level_is_callable_on_its_own() -> None:
    """`forge doctor` and any future config linter want the validator without
    the file loading."""
    level = LevelConfig(
        id="L9",
        name="synthetic",
        teaches="",
        checks=(
            CheckSpec(check="word_floor", params={"min_words": 1}),
            CheckSpec(check="detector_threshold", params={"max_z": 0.0}),
        ),
        par_source="authored",
    )
    with pytest.raises(GateConfigError, match="unicode_sanitation"):
        validate_level(level)
