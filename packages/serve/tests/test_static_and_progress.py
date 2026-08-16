"""Static serving (§11.4) and `GET /api/progress` (§9.1)."""

from __future__ import annotations

import gzip
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
        ("/data/passages/p07.public.json", "public, max-age=300"),
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
# /api/progress
# ---------------------------------------------------------------------------


async def test_an_unknown_session_gets_a_playable_answer_not_a_404(
    client: httpx.AsyncClient, content: Content
) -> None:
    """A first visit, a cleared cache and a typo are indistinguishable here.

    All three deserve "nothing cleared, level 1 unlocked" rather than an error
    on a player who has done nothing wrong.
    """
    response = await client.get("/api/progress", params={"session_id": "s_never_seen"})
    assert response.status_code == 200
    assert response.json() == {
        "cleared": [],
        "unlocked": 1,
        "level_count": content.level_count,
    }
    assert response.headers["cache-control"] == "no-store"


async def test_no_session_id_at_all_is_the_same_answer(client: httpx.AsyncClient) -> None:
    body = (await client.get("/api/progress")).json()
    assert body["cleared"] == []
    assert body["unlocked"] == 1


async def test_recorded_clears_come_back_with_the_next_level_unlocked(
    client: httpx.AsyncClient, repos: Any, content: Content
) -> None:
    for level_n, distance in ((1, 4), (2, 6), (3, 5)):
        await repos.progress.record("s_abc", level_n, distance)

    body = (await client.get("/api/progress", params={"session_id": "s_abc"})).json()
    assert body["cleared"] == [1, 2, 3]
    # `unlocked = max(cleared) + 1`, so the player resumes where they stopped.
    assert body["unlocked"] == 4
    assert body["level_count"] == content.level_count


async def test_progress_is_per_session(client: httpx.AsyncClient, repos: Any) -> None:
    """`session_id` is a key, not an identity — but it still has to separate."""
    await repos.progress.record("s_one", 1, 4)
    assert (await client.get("/api/progress", params={"session_id": "s_two"})).json()[
        "cleared"
    ] == []


async def test_unlocked_never_runs_past_the_end_of_the_campaign(
    client: httpx.AsyncClient, repos: Any, content: Content
) -> None:
    """Clearing the LAST level unlocks the last level, not level 16.

    `min(max + 1, level_count)`: the all-clear state is the client's business,
    and an `unlocked` past the end would send it to a level that cannot render.
    """
    await repos.progress.record("s_done", content.level_count, 3)
    body = (await client.get("/api/progress", params={"session_id": "s_done"})).json()
    assert body["unlocked"] == content.level_count
