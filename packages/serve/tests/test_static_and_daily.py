"""Static serving (§11.4) and the daily/leaderboard endpoints (§9.1, §9.4)."""

from __future__ import annotations

import gzip
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import brotli
import httpx
import pytest

from launder_serve.content import Content
from launder_serve.main import create_app
from launder_serve.settings import Settings
from launder_serve.static import IMMUTABLE, cache_control_for


@pytest.mark.parametrize(
    "path,expected",
    [
        ("/index.html", "no-cache"),
        ("/", "no-cache"),
        ("/data/assets/gemma3-tok.v1.bin.br", IMMUTABLE),
        ("/assets/index-a1b2c3.js", IMMUTABLE),
        ("/data/passages/p_2026-09-01.public.json", "public, max-age=300"),
    ],
)
def test_the_cache_policy_table(path: str, expected: str) -> None:
    assert cache_control_for(path) == expected


@pytest.fixture
def static_app(tmp_path: Path, settings: Settings, content: Content, repos: Any) -> Any:
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>Launder</title>", encoding="utf-8")
    payload = b"console.log('launder');" * 40
    (dist / "assets" / "app-a1b2c3.js").write_bytes(payload)
    (dist / "assets" / "app-a1b2c3.js.gz").write_bytes(gzip.compress(payload))
    # REAL brotli, not a stand-in string. `launder-serve` now depends on the
    # `brotli` wheel (the server tokenizer reads a .br asset), which means httpx
    # transparently DECODES `Content-Encoding: br` — so a sibling holding
    # arbitrary bytes fails at the client with "decoder failed" rather than
    # proving anything. Compressing for real makes this assert the whole loop:
    # the committed sibling is served verbatim AND a real client can read it.
    (dist / "assets" / "app-a1b2c3.js.br").write_bytes(brotli.compress(payload, quality=11))

    tweaked = settings.model_copy(update={"web_dist_dir": dist})
    return create_app(settings=tweaked, content=content, repos=repos, mount_static=True)


@pytest.fixture
async def static_client(static_app: Any) -> Any:
    async with static_app.router.lifespan_context(static_app):
        transport = httpx.ASGITransport(app=static_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
            yield http


async def test_a_brotli_sibling_is_served_when_accepted(static_client: httpx.AsyncClient) -> None:
    response = await static_client.get(
        "/assets/app-a1b2c3.js", headers={"Accept-Encoding": "br, gzip"}
    )
    assert response.status_code == 200
    assert response.headers["content-encoding"] == "br"
    assert response.headers["vary"] == "Accept-Encoding"
    assert response.headers["cache-control"] == IMMUTABLE
    # No runtime compression: the bytes are the committed sibling's, verbatim
    # (httpx decodes Content-Encoding: br, so `.content` is the payload back).
    assert response.content == b"console.log('launder');" * 40


async def test_gzip_is_the_fallback(static_client: httpx.AsyncClient) -> None:
    response = await static_client.get("/assets/app-a1b2c3.js", headers={"Accept-Encoding": "gzip"})
    assert response.headers["content-encoding"] == "gzip"


async def test_a_client_that_accepts_nothing_gets_the_identity_body(
    static_client: httpx.AsyncClient,
) -> None:
    response = await static_client.get(
        "/assets/app-a1b2c3.js", headers={"Accept-Encoding": "identity"}
    )
    assert "content-encoding" not in response.headers
    assert response.headers["vary"] == "Accept-Encoding"
    assert b"console.log" in response.content


async def test_q0_is_honoured(static_client: httpx.AsyncClient) -> None:
    response = await static_client.get(
        "/assets/app-a1b2c3.js", headers={"Accept-Encoding": "br;q=0, gzip"}
    )
    assert response.headers["content-encoding"] == "gzip"


async def test_the_answer_key_is_never_served(
    static_client: httpx.AsyncClient, data_root: Path
) -> None:
    """§2.3: `data/passages/*.author.json` — Browser: NEVER. ENFORCED, not documented.

    The whole `data/passages` directory was mounted with no filename filter, so
    `reference_solution`, the solver's `par_upper` / `clears_at_k`, the
    optionality and entropy arrays, `topk_alts` and the generation provenance
    came back HTTP 200 on any deployment run from source. The only thing keeping
    them out of production was a `.dockerignore` line, which is a build rule and
    not a serving rule.
    """
    secret = '{"schema": "launder.passage.author/1", "reference_solution": "the answer"}'
    names = ("p_leak.author.json", "p_leak.claims.draft.json", "p_leak.author.json.br")
    for name in names:
        (data_root / "passages" / name).write_text(secret, encoding="utf-8")

    for name in names:
        response = await static_client.get(f"/data/passages/{name}")
        assert response.status_code == 404, name
        assert "reference_solution" not in response.text

    # ... and the public bundle beside it is still served.
    public = sorted((data_root / "passages").glob("*.public.json"))[0]
    assert (await static_client.get(f"/data/passages/{public.name}")).status_code == 200


async def test_index_html_is_no_cache(static_client: httpx.AsyncClient) -> None:
    response = await static_client.get("/index.html")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers.get("etag")


async def test_the_api_router_wins_over_the_static_catch_all(
    static_client: httpx.AsyncClient,
) -> None:
    """THE ORDER THE WHOLE FILE IS ABOUT: a StaticFiles mount at "/" swallows
    everything beneath it, so /healthz must be registered first."""
    response = await static_client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["ok"] is True


# ---------------------------------------------------------------------------
# /api/daily and /api/leaderboard
# ---------------------------------------------------------------------------


@pytest.fixture
def frozen_day(monkeypatch: pytest.MonkeyPatch, content: Content) -> date:
    day = content.schedule.days[0].date
    monkeypatch.setattr("launder_serve.api.daily.today_utc", lambda: day)
    return day


async def test_daily_reports_the_scheduled_passage(
    client: httpx.AsyncClient, content: Content, frozen_day: date
) -> None:
    response = await client.get("/api/daily")
    assert response.status_code == 200
    body = response.json()
    slot = content.schedule.days[0]
    assert body["day"] == frozen_day.isoformat()
    assert body["passage_id"] == slot.passage_id
    assert body["level_id"] == slot.level_id
    assert body["passage_url"] == f"/data/passages/{slot.passage_id}.public.json"
    assert body["asset_bundle_id"] == content.asset_bundle_id
    # puzzle_number is a pure function of the date, so the share string cannot
    # drift if a day is inserted or removed.
    assert body["puzzle_number"] == content.schedule.puzzle_number(frozen_day)
    assert response.headers["cache-control"] == "public, max-age=60"


async def test_daily_404s_off_schedule(
    client: httpx.AsyncClient, content: Content, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No local-time scheme: rollover is UTC midnight and a day with no
    scheduled passage is a 404, not an invented puzzle.

    The off-schedule day is CONSTRUCTED (one past the last scheduled date)
    rather than assumed. This test used to just call /api/daily and rely on the
    real calendar being off the seeded schedule — which meant it asserted the
    right thing only while `data/passages/` was empty, and started failing the
    moment the repo got real content scheduled for today.
    """
    off_schedule = max(slot.date for slot in content.schedule.days) + timedelta(days=1)
    assert all(slot.date != off_schedule for slot in content.schedule.days)
    monkeypatch.setattr("launder_serve.api.daily.today_utc", lambda: off_schedule)

    response = await client.get("/api/daily")
    assert response.status_code == 404


async def test_the_leaderboard_publishes_diffs(
    client: httpx.AsyncClient, repos: Any, content: Content, frozen_day: date
) -> None:
    from datetime import UTC, datetime

    from launder_core.schemas import EditOp
    from launder_serve.repo.protocol import SubmissionRecord

    slot = content.schedule.days[0]
    for i, distance in enumerate((7, 3)):
        await repos.submissions.record(
            SubmissionRecord(
                day=frozen_day,
                passage_id=slot.passage_id,
                level_id=slot.level_id,
                text_hash=str(i) * 64,
                text="…",
                cleared=True,
                distance=distance,
                detector_score=0.5,
                detector_z=1.0,
                n_scored=100,
                masked_fraction=0.1,
                scoring_version="sc1",
                wm_config_id=content.wm_config_id,
                created_at=datetime.now(UTC),
                ops=(EditOp(op="sub", i=1, j=1, **{"from": "holds"}, to="keeps"),),
            )
        )

    response = await client.get(
        f"/api/leaderboard/{frozen_day.isoformat()}", params={"level_id": slot.level_id}
    )
    assert response.status_code == 200
    body = response.json()
    assert [row["rank"] for row in body["rows"]] == [1, 2]
    assert [row["distance"] for row in body["rows"]] == [3, 7]
    # The diff IS the brag, and it is what makes the board self-verifying.
    assert body["rows"][0]["ops"][0]["from"] == "holds"
    # No names, no session ids, no identity.
    assert set(body["rows"][0]) == {"rank", "distance", "ops", "elapsed_ms", "at"}
    assert response.headers["cache-control"] == "public, max-age=30"
