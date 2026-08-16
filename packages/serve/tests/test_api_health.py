"""`/healthz` is the deploy gate, so what it says has to be true (§9.5)."""

from __future__ import annotations

from typing import Any

import httpx

from launder_serve.content import Content


async def test_healthz_exposes_wm_config_id(client: httpx.AsyncClient, content: Content) -> None:
    response = await client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    # The one-curl answer to "is the needle lying?": this is the id the browser
    # asserts against every passage it loads.
    assert body["wm_config_id"] == content.wm_config_id
    assert body["wm_config_id"].startswith("wm1:")
    assert body["asset_bundle_id"].startswith("ab1:")
    assert body["judge_version"] == content.judge_version
    assert body["scoring_version"] == content.scoring_version
    assert body["sha"]


async def test_healthz_is_never_cached(client: httpx.AsyncClient) -> None:
    response = await client.get("/healthz")
    assert response.headers["cache-control"] == "no-store"


async def test_healthz_is_503_when_the_database_is_unreachable(
    app: Any, client: httpx.AsyncClient
) -> None:
    """A 200 promotes the deploy. A dead database must not get one."""

    class DeadEngine:
        def connect(self) -> Any:
            raise OSError("connection refused")

    app.state.launder.engine = DeadEngine()
    response = await client.get("/healthz")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "network"


async def test_healthz_is_registered(client: httpx.AsyncClient) -> None:
    """Mount order: the API router must win over the static catch-all (§11.4).

    Structural assertions on `app.routes` do not survive a FastAPI upgrade —
    included routers are wrapped, not flattened — so the order is asserted the
    way it actually matters, in `test_static_and_daily.py`: with the catch-all
    mounted, `/healthz` still answers.
    """
    assert (await client.get("/healthz")).status_code == 200
