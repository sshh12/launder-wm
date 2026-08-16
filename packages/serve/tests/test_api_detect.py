"""`/api/detect` — the documented shape, and the body cap (§9.2).

The shape assertions are the load-bearing ones: the local TS detector emits
this exact object, so one renderer serves both paths and the SERVER->LOCAL
handover is a no-op in the view layer (§5.4). A field added or renamed here is
a field the browser silently stops rendering.
"""

from __future__ import annotations

import hashlib

import httpx

from launder_core.schemas import MAX_TEXT_BYTES, PassagePublic


async def test_detect_returns_the_documented_shape(
    client: httpx.AsyncClient, passage_id: str, public_passage: PassagePublic
) -> None:
    response = await client.post(
        "/api/detect",
        json={"passage_id": passage_id, "text": public_passage.text, "seq": 41},
    )
    assert response.status_code == 200
    body = response.json()

    assert set(body) == {
        "seq",
        "text_hash",
        "score",
        "z",
        "z_star",
        "n_scored",
        "n_tokens",
        "tokens",
        "preview_distance",
    }
    assert body["seq"] == 41
    # The hash is over the EXACT text the reading describes, so the client can
    # drop a response that no longer matches the textarea (§10.2).
    expected = "sha256:" + hashlib.sha256(public_passage.text.encode("utf-8")).hexdigest()
    assert body["text_hash"] == expected
    assert 0.0 <= body["score"] <= 1.0
    assert body["n_scored"] <= body["n_tokens"]

    token = body["tokens"][0]
    assert set(token) == {"s", "e", "heat", "masked"}
    # CHAR offsets, not token indices: the mirror needs no client tokenizer.
    assert token["s"] == 0
    assert token["e"] > token["s"]
    assert public_passage.text[token["s"] : token["e"]] == public_passage.text.split()[0]
    assert any(t["masked"] for t in body["tokens"])
    assert all(0.0 <= t["heat"] <= 1.0 for t in body["tokens"])


async def test_detect_is_never_cached(
    client: httpx.AsyncClient, passage_id: str, public_passage: PassagePublic
) -> None:
    response = await client.post(
        "/api/detect", json={"passage_id": passage_id, "text": "hello there", "seq": 0}
    )
    assert response.headers["cache-control"] == "no-store"


async def test_detect_rejects_oversize_bodies_with_413(
    client: httpx.AsyncClient, passage_id: str
) -> None:
    """Body cap 64 KB (§9.2), enforced in the ASGI layer before buffering."""
    huge = "word " * ((MAX_TEXT_BYTES // 5) + 200)
    assert len(huge.encode()) > MAX_TEXT_BYTES
    response = await client.post(
        "/api/detect", json={"passage_id": passage_id, "text": huge, "seq": 1}
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "too_long"


async def test_detect_404s_on_an_unknown_passage(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/api/detect", json={"passage_id": "p_not_today", "text": "hello", "seq": 1}
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "unknown_passage"


async def test_detect_rejects_unknown_fields(client: httpx.AsyncClient, passage_id: str) -> None:
    """`extra="forbid"`: a client that invents a field learns about it at once."""
    response = await client.post(
        "/api/detect",
        json={"passage_id": passage_id, "text": "hello", "seq": 1, "z": 0.0},
    )
    assert response.status_code == 400
