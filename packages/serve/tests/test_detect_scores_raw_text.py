"""The detector scores RAW text, never the normalized form.

Regression test for a divergence that survived the parallel build, the
integrator AND three adversarial verifiers — because every one of them
exercised `data/dev/passage.txt`, a file with no double spaces, where
`normalize()` happens to be the identity.

`/api/detect` used to score `scorer.normalize(body.text)` while returning
character offsets the mirror maps onto the raw textarea. On any text where
normalization actually changes something (a double space, an NBSP, a smart
quote) that produced two independent failures at once:

* every offset after the first collapsed run shifted left, so heat landed on
  the wrong words and the error accumulated down the passage; and
* the tokenization changed (measured: 329 tokens raw vs 321 normalized on the
  first real daily), so the live reading could never reproduce the passage's
  own `expected_z` / `g_digest` — §4.5's runtime tripwire would have fired on
  every passage, on keystroke zero, in production.

The browser worker encodes `msg.text` raw. These tests pin the server to the
same rule from both ends: the arithmetic and the geometry.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from itertools import pairwise
from typing import Any

import httpx
import pytest

from launder_core.scoring import normalize
from launder_core.tokenizer import TokenizerUnavailable, load_tokenizer
from launder_serve.content import Content
from launder_serve.engine import CoreDetector
from launder_serve.main import RepoSet, create_app
from launder_serve.settings import Settings


def _tokenizer_available() -> str | None:
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

# Passage-shaped text whose normalization is NOT the identity: a double space
# after the first sentence. Long enough to clear the ngram window many times.
RAW = (
    "Flood insurance provides crucial financial protection against property damage "
    "caused by flooding.  The National Flood Insurance Program sets the terms under "
    "which most policies are written, and the rate a household pays depends on the "
    "elevation of the structure relative to the base flood elevation established for "
    "its zone. Private carriers have re-entered the market in recent years, which has "
    "widened the range of available coverage without simplifying the process of "
    "choosing between policies."
)


@pytest.fixture
def real_app(settings: Settings, content: Content, repos: RepoSet) -> Any:
    """The real app with the REAL detector — a stub cannot answer this question."""
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


def test_the_fixture_actually_exercises_the_bug() -> None:
    """Guard the guard: if normalize became the identity here, the rest is vacuous."""
    assert normalize(RAW) != RAW, "fixture no longer exercises normalization"
    assert len(normalize(RAW)) < len(RAW)


async def test_offsets_index_the_raw_text_and_cover_it_exactly(
    real_client: httpx.AsyncClient, passage_id: str
) -> None:
    response = await real_client.post(
        "/api/detect", json={"passage_id": passage_id, "text": RAW, "seq": 3}
    )
    assert response.status_code == 200, response.text
    tokens: list[dict[str, Any]] = response.json()["tokens"]

    # Contiguous, gapless, reassembling the RAW string exactly. Under the bug
    # this diverged from the first collapsed space onward.
    assert "".join(RAW[t["s"] : t["e"]] for t in tokens) == RAW
    assert tokens[0]["s"] == 0
    assert tokens[-1]["e"] == len(RAW)
    for a, b in pairwise(tokens):
        assert a["e"] == b["s"], "spans must be contiguous"
    assert RAW[tokens[0]["s"] : tokens[0]["e"]] == "Flood"


async def test_reading_uses_the_raw_tokenization_not_the_normalized_one(
    real_client: httpx.AsyncClient, passage_id: str
) -> None:
    from launder_core.tokenizer import encode_with_offsets

    raw_ids, _ = encode_with_offsets(RAW)
    norm_ids, _ = encode_with_offsets(normalize(RAW))
    assert len(raw_ids) != len(norm_ids), "fixture must distinguish the two paths"

    response = await real_client.post(
        "/api/detect", json={"passage_id": passage_id, "text": RAW, "seq": 4}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["n_tokens"] == len(raw_ids)
    assert body["n_tokens"] != len(norm_ids)
