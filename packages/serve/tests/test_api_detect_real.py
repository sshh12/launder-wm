"""`/api/detect` with the REAL detector — no stub anywhere in the path.

`packages/serve/tests/support.py::StubDetector` is deliberate fiction: it exists
so the gate and solver tests can control which words carry signal. It is the
right tool for those tests and the wrong one for this question, which is whether
the endpoint the browser falls back to reports the same numbers the browser
computes locally.

So this file wires `CoreDetector` — the shipped tokenizer, the shipped
sampling table, the shipped calibration — into the real app and asserts the
response against `data/golden/vectors.json`, the same file vitest reads. If the
handover in §5.4 is ever a no-op in the view layer but a jump in the needle,
this is the test that fails.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from launder_serve.content import Content
from launder_serve.engine import CoreDetector
from launder_serve.main import RepoSet, create_app
from launder_serve.settings import Settings

REPO_ROOT = Path(__file__).resolve().parents[3]
GOLDEN = REPO_ROOT / "data" / "golden" / "vectors.json"
DEV_PASSAGE = REPO_ROOT / "data" / "dev" / "passage.txt"


def _tokenizer_available() -> str | None:
    """None when the real detector can run here; otherwise why it cannot."""
    from launder_core.tokenizer import TokenizerUnavailable, load_tokenizer

    try:
        load_tokenizer()
    except TokenizerUnavailable as exc:
        return str(exc)
    return None


_UNAVAILABLE = _tokenizer_available()
pytestmark = pytest.mark.skipif(
    _UNAVAILABLE is not None,
    reason=f"the server tokenizer cannot be built here: {_UNAVAILABLE}",
)


@pytest.fixture
def golden_case() -> dict[str, Any]:
    if not GOLDEN.is_file() or not DEV_PASSAGE.is_file():
        pytest.skip("data/golden/vectors.json or data/dev/passage.txt is absent")
    data = json.loads(GOLDEN.read_text(encoding="utf-8"))
    text = DEV_PASSAGE.read_text(encoding="utf-8").removesuffix("\n")
    sha = "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
    case = next((c for c in data["cases"]["score"]["cases"] if c.get("text_sha256") == sha), None)
    if case is None:
        pytest.skip("data/dev/passage.txt is in no golden score case")
    return {"text": text, **case}


@pytest.fixture
def real_app(settings: Settings, content: Content, repos: RepoSet) -> Any:
    return create_app(
        settings=settings,
        content=content,
        repos=repos,
        detector=CoreDetector(),
        mount_static=False,
    )


@pytest.fixture
async def real_client(real_app: Any) -> AsyncIterator[httpx.AsyncClient]:
    async with real_app.router.lifespan_context(real_app):
        transport = httpx.ASGITransport(app=real_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
            yield http


async def test_detect_reproduces_the_golden_reading(
    real_client: httpx.AsyncClient, passage_id: str, golden_case: dict[str, Any]
) -> None:
    """The number the server reports IS the number the browser computes."""
    response = await real_client.post(
        "/api/detect",
        json={"passage_id": passage_id, "text": golden_case["text"], "seq": 7},
    )
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["seq"] == 7
    assert body["text_hash"] == golden_case["text_sha256"]
    assert body["n_tokens"] == golden_case["n_tokens"]
    assert body["n_scored"] == golden_case["n_scored"]
    assert abs(body["score"] - golden_case["score"]) < 1e-12
    assert abs(body["z"] - golden_case["z"]) < 1e-9
    assert abs(body["z_star"] - golden_case["z_star"]) < 1e-12


async def test_the_token_array_is_per_token_and_covers_the_text(
    real_client: httpx.AsyncClient, passage_id: str, golden_case: dict[str, Any]
) -> None:
    """Per TOKEN, not per row (§9.2, and vectors.json's `heat_note`).

    Row `i` covers ids[i..i+n-1] and its heat belongs to the CURRENT token
    `i+n-1`, so the leading `ngram_len-1` tokens carry neutral 0.5 and
    `masked: true`. Zipping char offsets against rows shifts every span four
    tokens left, which paints heat on the wrong words — plausibly.
    """
    text = golden_case["text"]
    response = await real_client.post(
        "/api/detect", json={"passage_id": passage_id, "text": text, "seq": 1}
    )
    tokens = response.json()["tokens"]

    assert len(tokens) == golden_case["n_tokens"] == len(golden_case["heat"])
    assert "".join(text[t["s"] : t["e"]] for t in tokens) == text
    for i, t in enumerate(tokens):
        assert abs(t["heat"] - golden_case["heat"][i]) < 1e-12, f"heat[{i}]"
        assert int(t["masked"]) == golden_case["masked"][i], f"masked[{i}]"
    assert all(t["masked"] for t in tokens[:4]), "the leading n-1 tokens are in no window"


async def test_an_edited_passage_moves_the_needle(
    real_client: httpx.AsyncClient, passage_id: str, golden_case: dict[str, Any]
) -> None:
    """A real edit produces a real, different reading — not a cached one."""
    text = golden_case["text"]
    edited = text.replace("committee", "panel", 1)
    assert edited != text
    body = (
        await real_client.post(
            "/api/detect", json={"passage_id": passage_id, "text": edited, "seq": 2}
        )
    ).json()
    assert body["text_hash"] != golden_case["text_sha256"]
    assert body["z"] != golden_case["z"]


async def test_empty_text_reads_neutral_rather_than_innocent(
    real_client: httpx.AsyncClient, passage_id: str
) -> None:
    """No evidence either way is z = 0, not z = -inf and not nan."""
    body = (
        await real_client.post("/api/detect", json={"passage_id": passage_id, "text": "", "seq": 3})
    ).json()
    assert body["n_scored"] == 0
    assert body["z"] == 0.0
    assert body["score"] == 0.5
    assert body["tokens"] == []
