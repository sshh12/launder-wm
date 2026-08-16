"""Fixtures: a real `data/` tree, a real app, in-memory repositories, no network.

The data root is the repo's own `data/config` and `data/assets` — the real
watermark config, the real levels, the real copy — plus synthetic passages
written into a tmp dir. Using the real configs is deliberate: these tests are
the only thing standing between a typo in `levels.toml` and a 500 in
production, and a hand-rolled fixture config would test the fixture.
"""

from __future__ import annotations

import json
import shutil
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from launder_core.schemas import PassagePublic
from launder_serve.content import Content, load_content
from launder_serve.judge.fake import FakeJudge
from launder_serve.judge.gate import CachingJudge
from launder_serve.limits import TokenBucketLimiter
from launder_serve.main import RepoSet, create_app
from launder_serve.settings import Settings, repo_root

# `--import-mode=importlib` with no package marker means sibling test helpers are
# not importable by default, and `packages/*/tests` would collide as `tests.*`
# anyway. One explicit path entry is cheaper than a package layout.
sys.path.insert(0, str(Path(__file__).parent))

from support import StubDetector

PASSAGE_TEXT = (
    "The team followed four hundred households for twelve years, and the pattern held "
    "in every region they examined. Families that moved before a child turned ten "
    "earned more as adults, and the effect grew with each additional year of exposure. "
    "The researchers matched siblings against one another to rule out the possibility "
    "that ambitious parents simply moved more often. They found the same gap. Critics "
    "note that the sample skews toward larger cities, and the authors agree that the "
    "rural picture remains thin. Even so, the study has changed how housing vouchers "
    "are argued about in a dozen states, and two federal programs now cite it directly."
)


def _passage(
    wm_config_id: str, asset_bundle_id: str, passage_id: str, level_id: str
) -> dict[str, Any]:
    n_words = len(PASSAGE_TEXT.split())
    return {
        "schema": "launder.passage.public/1",
        "id": passage_id,
        "level_id": level_id,
        "wm_config_id": wm_config_id,
        "asset_bundle_id": asset_bundle_id,
        "scoring_version": "sc1",
        "text": PASSAGE_TEXT,
        "n_words": n_words,
        "detector": {
            "expected_n_scored": 140,
            "expected_score": 0.53,
            "expected_z": 6.44,
            "g_digest": "blake3:" + "ab" * 32,
        },
        "rules": {"min_words": 50, "max_word_edits": 12, "locked_phrases": []},
        "claims": [
            {
                "id": "c1",
                "text": "The study followed four hundred households for twelve years.",
                "label": "the 12-year study window",
                "required": True,
            },
            {
                "id": "c2",
                "text": "Children who moved before age ten earned more as adults.",
                "label": "the earnings effect for children who moved early",
                "required": True,
            },
            {
                "id": "c3",
                "text": "The researchers compared siblings to control for parental ambition.",
                "label": "the sibling comparison",
                "required": False,
            },
        ],
        "par": 4,
        "par_source": "authored_reference",
        "judge_prompt_id": "judge.observe.v3",
    }


@pytest.fixture(scope="session")
def real_data_root() -> Path:
    root = repo_root() / "data"
    if not (root / "config" / "watermark.toml").is_file():
        pytest.skip("data/config is not present in this checkout")
    return root


@pytest.fixture
def data_root(tmp_path: Path, real_data_root: Path) -> Path:
    """A copy of the committed `data/` with synthetic passages written in."""
    root = tmp_path / "data"
    (root / "passages").mkdir(parents=True)
    shutil.copytree(real_data_root / "config", root / "config")
    if (real_data_root / "assets").is_dir():
        shutil.copytree(real_data_root / "assets", root / "assets")

    # Load once with no passages to learn the ids the passages must declare.
    probe = load_content(root, is_production=False)
    for slot in probe.schedule.days:
        payload = _passage(
            probe.wm_config_id, probe.asset_bundle_id, slot.passage_id, slot.level_id
        )
        (root / "passages" / f"{slot.passage_id}.public.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        # The packed sidecar the image actually ships (§11.3): claims + par only.
        (root / "passages" / f"{slot.passage_id}.server.json").write_text(
            json.dumps(
                {
                    "schema": "launder.passage.server/1",
                    "id": slot.passage_id,
                    "claims": payload["claims"],
                    "par": payload["par"],
                }
            ),
            encoding="utf-8",
        )
    return root


@pytest.fixture
def settings(data_root: Path) -> Settings:
    return Settings(
        env="test",
        data_dir=data_root,
        judge_provider="fake",
        database_url="",
        git_sha="testsha0",
        rate_limit_judge_per_hour=10,
        rate_limit_judge_burst=3,
    )


@pytest.fixture
def content(settings: Settings) -> Content:
    return load_content(settings.data_path, is_production=False)


@pytest.fixture
def repos() -> RepoSet:
    return RepoSet.in_memory()


@pytest.fixture
def judge() -> FakeJudge:
    return FakeJudge()


@pytest.fixture
def limiter() -> TokenBucketLimiter:
    return TokenBucketLimiter(per_hour=10, burst=3)


@pytest.fixture
def app(
    settings: Settings,
    content: Content,
    repos: RepoSet,
    judge: FakeJudge,
    limiter: TokenBucketLimiter,
) -> Any:
    gate = CachingJudge(
        provider=judge,
        failover=None,
        cache=repos.judge_cache,
        spend=repos.spend,
        limiter=limiter,
        judge_version=content.judge_version,
        prompt_hash="test-prompt-hash",
        scoring_version=content.scoring_version,
        lru_entries=64,
        daily_usd_cap=settings.judge_daily_usd_cap,
    )
    application = create_app(
        settings=settings,
        content=content,
        repos=repos,
        detector=StubDetector(),
        judge=gate,
        mount_static=False,
    )
    application.state.launder.limiter = limiter
    return application


@pytest.fixture
async def client(app: Any) -> AsyncIterator[httpx.AsyncClient]:
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
            yield http


@pytest.fixture
def scheduled_day(content: Content) -> Any:
    return content.schedule.days[0]


@pytest.fixture
def passage_id(scheduled_day: Any) -> str:
    return str(scheduled_day.passage_id)


@pytest.fixture
def level_id(scheduled_day: Any) -> str:
    return str(scheduled_day.level_id)


@pytest.fixture
def public_passage(content: Content, passage_id: str) -> PassagePublic:
    bundle = content.passage(passage_id)
    assert bundle is not None
    return bundle.public
