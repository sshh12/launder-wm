"""Boot assertions that had no home: the production judge and the campaign map.

ARCHITECTURE.md §8: "Fail at BOOT, never at play." Both assertions here are
config errors that the process used to accept silently and pay for at play time
— one by turning the meaning half of the gate off in production, the other by
recording a clear against a level nobody played.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import SecretStr

from launder_serve.content import ProgressionFile, load_content
from launder_serve.settings import PAID_PROVIDERS, Settings

# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------


def test_production_refuses_an_offline_judge() -> None:
    """`JUDGE_PROVIDER` defaults to `fake`, and `fake` clears everything.

    A Railway variable panel missing `JUDGE_PROVIDER` produced a server that
    booted clean and healthchecked green with `FakeJudge` behind `llm_gate` —
    which reports `natural_prose=True` with every claim present, so word salad
    and dropped claims both pass. `Settings`' own docstring has always claimed
    production "refuses to boot" when it did not override the development
    defaults; nothing enforced it.
    """
    for provider in ("fake", "cassette"):
        settings = Settings(
            env="production",
            judge_provider=provider,
            judge_cassette_path=Path("cassette.jsonl"),
        )
        with pytest.raises(RuntimeError, match="never contacts a model"):
            settings.assert_keys_present()


def test_production_with_the_real_judge_boots() -> None:
    settings = Settings(
        env="production", judge_provider="openai", openai_api_key=SecretStr("sk-test")
    )
    assert settings.assert_keys_present() == []
    assert settings.judge_provider in PAID_PROVIDERS


@pytest.mark.parametrize("env", ["development", "test", "pr"])
def test_every_other_environment_still_runs_the_fake_judge(env: str) -> None:
    """§11.5: sealed variables do not propagate to PR environments, so PR envs
    run `fake` with no keys ON PURPOSE and must still boot. That is a feature,
    and it is the reason the assertion above is scoped to `production` alone."""
    assert Settings(env=env, judge_provider="fake").assert_keys_present() == []  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# progression.toml
# ---------------------------------------------------------------------------


def _progression(*passage_ids: str) -> dict[str, object]:
    return {
        "schema": "launder.progression/1",
        "strict": True,
        "level": [
            {"n": i, "passage_id": pid, "rules": "L1"} for i, pid in enumerate(passage_ids, start=1)
        ],
    }


def test_two_levels_cannot_share_a_passage() -> None:
    """A passage IS a level's identity, because the mapping is used backwards.

    `/api/submit` takes `passage_id` from the request and refuses to take
    `level_n`, so the server derives the campaign position with
    `Content.level_n_of` — which returns the FIRST match. With a passage listed
    twice, a player clearing level 3 has the clear recorded against level 1 and
    unlocks level 2, and level 3 can never be submitted to at all. The file was
    checked for contiguous numbering and not for this.
    """
    with pytest.raises(ValueError, match="both level 1 and level 3"):
        ProgressionFile.model_validate(_progression("p01", "p02", "p01"))


def test_distinct_passages_are_fine() -> None:
    prog = ProgressionFile.model_validate(_progression("p01", "p02", "p03"))
    assert [spec.n for spec in prog.levels] == [1, 2, 3]


def test_the_shipped_campaign_maps_one_passage_to_one_level(
    real_data_root: Path,
) -> None:
    """The assertion above, run against `data/config/progression.toml` itself."""
    prog = ProgressionFile.load(real_data_root / "config" / "progression.toml")
    ids = [spec.passage_id for spec in prog.levels]
    assert len(set(ids)) == len(ids)


# ---------------------------------------------------------------------------
# levels.toml -> thresholds.v1.json
# ---------------------------------------------------------------------------


def test_a_level_naming_an_unmeasured_calibration_fails_at_boot(data_root: Path) -> None:
    """A bucket set that was never measured must stop the server, not a player.

    L5 shipped `detector_threshold.calibration = "code"` against a thresholds
    file defining only `default`. `load_levels` accepts it — `calibration` is a
    declared `config_param` and core has no reason to know which curves were
    ever fitted — so the first thing to object was `parse_thresholds`, inside
    `read_tokens`, at submit time: a 500 on the level's WIN CONDITION.

    It survived because L5 is out of the campaign AND is dropped at boot for a
    missing `unit_tests` dependency. Masked by two accidents, one `[[level]]`
    block from shipping. So the check runs over every ruleset in the file, and
    this test names a ruleset the campaign does not play.
    """
    levels = (data_root / "config" / "levels.toml").read_text(encoding="utf-8")
    # Comment lines are skipped: the file explains at length why `calibration =
    # "code"` was removed, and a naive substring search finds the explanation.
    live = "\n".join(ln for ln in levels.splitlines() if not ln.lstrip().startswith("#"))
    assert 'calibration = "' not in live, (
        "a level has started naming a calibration again; if it was measured, add it to the "
        "allow-list of this test rather than deleting the assertion"
    )
    patched = levels.replace(
        '{ check = "detector_threshold" }',
        '{ check = "detector_threshold", params = { calibration = "never_measured" } }',
        1,
    )
    assert patched != levels, "levels.toml no longer has a bare detector_threshold to patch"
    (data_root / "config" / "levels.toml").write_text(patched, encoding="utf-8")

    with pytest.raises(KeyError, match="never_measured"):
        load_content(data_root, is_production=False)


def test_the_shipped_levels_only_name_calibrations_that_exist(real_data_root: Path) -> None:
    """The assertion above, run against the real config: boot must not raise."""
    load_content(real_data_root, is_production=False)
