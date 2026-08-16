"""§10.7's two lint rules, including the failure each exists to catch."""

from __future__ import annotations

import copy as copy_module
from typing import Any

import pytest

from launder_forge.config import load_copy
from launder_forge.lintcopy import REQUIRED_KEYS, lint_copy
from launder_forge.paths import Paths


@pytest.fixture
def shipped(real_paths: Paths) -> dict[str, Any]:
    return load_copy(real_paths)


def test_the_shipped_copy_file_passes(real_paths: Paths, shipped: dict[str, Any]) -> None:
    report = lint_copy(real_paths, shipped)
    assert report.ok, report.as_dict()
    assert report.checked_templates > 50


def test_rule_1_catches_a_deliberately_removed_blurb(
    real_paths: Paths, shipped: dict[str, Any]
) -> None:
    """Adding a check to levels.toml without writing its copy must fail the
    build rather than ship a constraint the player cannot interrogate."""
    broken = copy_module.deepcopy(shipped)
    del broken["check"]["edit_budget"]["blurb"]
    report = lint_copy(real_paths, broken)
    assert not report.ok
    assert "[check.edit_budget].blurb" in report.missing_keys


@pytest.mark.parametrize("key", REQUIRED_KEYS)
def test_rule_1_catches_every_required_key(
    real_paths: Paths, shipped: dict[str, Any], key: str
) -> None:
    broken = copy_module.deepcopy(shipped)
    broken["check"]["locked_phrase"][key] = "   "
    report = lint_copy(real_paths, broken)
    assert f"[check.locked_phrase].{key}" in report.missing_keys


def test_rule_1_catches_a_whole_missing_block(real_paths: Paths, shipped: dict[str, Any]) -> None:
    broken = copy_module.deepcopy(shipped)
    del broken["check"]["word_floor"]
    report = lint_copy(real_paths, broken)
    assert "word_floor" in report.missing_blocks


def test_rule_2_catches_a_renamed_param(real_paths: Paths, shipped: dict[str, Any]) -> None:
    """A renamed param silently blanking a rejection message is exactly the bug
    this rule exists for."""
    broken = copy_module.deepcopy(shipped)
    broken["check"]["edit_budget"]["reject"] = (
        "You changed {edit_distance} words. This level allows {max_word_distance}."
    )
    report = lint_copy(real_paths, broken)
    assert not report.ok
    assert any("{edit_distance}" in u for u in report.unresolved)


def test_rule_2_ignores_escaped_braces(real_paths: Paths, shipped: dict[str, Any]) -> None:
    ok_copy = copy_module.deepcopy(shipped)
    ok_copy["check"]["word_floor"]["blurb"] = "A literal {{brace}} is not a placeholder."
    report = lint_copy(real_paths, ok_copy)
    assert report.ok, report.as_dict()


def test_llm_gate_sub_blocks_are_linted(real_paths: Paths, shipped: dict[str, Any]) -> None:
    """`reject_by_code` and `reject_by_unnatural_kind` are nested tables; their
    templates are rendered the same way and must be checked the same way."""
    broken = copy_module.deepcopy(shipped)
    broken["check"]["llm_gate"]["reject_by_code"]["meaning_drift"] = "You dropped {claim_name}."
    report = lint_copy(real_paths, broken)
    assert any("claim_name" in u for u in report.unresolved)
