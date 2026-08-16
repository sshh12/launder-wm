"""`POST /api/detect` — the fallback path and the parity oracle (§9.2).

Two properties make this endpoint cheap enough to keep forever:

1. **Its response shape is exactly what the local TS detector emits** — char
   offsets, per-token heat, masked flags. One renderer serves both paths, so
   the SERVER->LOCAL handover is a no-op in the view layer and M2 is playable
   before a line of the TS detector exists (§5.4).
2. **It is not rate limited.** A player hammering the detector costs one
   CPU-millisecond; throttling it would make the game feel broken. The body cap
   (64 KB) is the only bound, and it is enforced by the ASGI layer before the
   body is read as well as by the schema.

`Cache-Control: no-store`. 0.5% of calls are sampled into the parity-oracle log
(`PARITY_SAMPLE_RATE`), which is what turns "the client and server disagree"
from a bug report into a dataset.
"""

from __future__ import annotations

import hashlib
import logging

from fastapi import APIRouter, Request, Response

from launder_core.schemas import DetectRequest, DetectResponse
from launder_serve.api.deps import state_of
from launder_serve.engine import text_hash

__all__ = ["router"]

_log = logging.getLogger("launder.parity")
router = APIRouter(prefix="/api", tags=["detect"])


@router.post("/detect", response_model=DetectResponse, summary="Server-side detector")
async def detect(request: Request, body: DetectRequest, response: Response) -> DetectResponse:
    state = state_of(request)
    bundle = state.passage_or_404(body.passage_id)

    # THE DETECTOR SCORES THE RAW TEXT, NEVER THE NORMALIZED FORM.
    #
    # Two independent reasons, either one sufficient:
    #
    # 1. `tokens[].s/.e` are character offsets the mirror maps onto the RAW
    #    textarea. Normalization collapses runs of whitespace, so every offset
    #    after the first collapsed run is shifted left by the number of
    #    characters removed — the heat lands on the wrong words, and the error
    #    accumulates down the passage.
    # 2. The watermark lives in the token ids the model actually emitted.
    #    `forge pack` computes `expected_z`/`g_digest` from `encode(raw text)`,
    #    so scoring the normalized form produces a different tokenization
    #    (measured: 329 raw vs 321 normalized on p02) and a z that can
    #    never reproduce the passage's own conformance record — §4.5's runtime
    #    tripwire would fire on every passage, on keystroke zero.
    #
    # The browser worker encodes `msg.text` raw. This line is what keeps the
    # SERVER and LOCAL readings the same number. Normalization is a SCORING
    # concern (§8.1) and belongs to the word floor, the edit budget and the
    # distance — never to detection.
    readout = state.detector.read_tokens(body.text, bundle.public)

    # Preview only, and labelled as such: the authoritative distance is the one
    # /api/submit computes and persists (§8.3).
    preview = state.scorer.score(bundle.public.text, body.text)

    result = DetectResponse(
        seq=body.seq,
        # The hash is over the EXACT text this reading describes — the raw
        # textarea contents, not the normalized form. The client compares it
        # against what is on screen, and what is on screen is the raw text.
        text_hash=text_hash(body.text),
        score=readout.reading.score,
        z=readout.reading.z,
        z_star=readout.reading.z_star,
        n_scored=readout.reading.n_scored,
        n_tokens=readout.n_tokens,
        tokens=readout.tokens,
        preview_distance=preview.distance,
    )
    response.headers["Cache-Control"] = "no-store"
    _sample_parity(state.settings.parity_sample_rate, body, result)
    return result


def _sample_parity(rate: float, body: DetectRequest, result: DetectResponse) -> None:
    """Deterministic 0.5% sample: hash the text, don't roll a die.

    A random sample would make the log non-reproducible for exactly the request
    somebody is trying to explain. Hashing the text means the same submission
    is always sampled or always not, and re-running a suspicious request
    reproduces its log line.
    """
    if rate <= 0.0:
        return
    digest = hashlib.sha256(body.text.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:4], "big") / 0xFFFFFFFF
    if bucket >= rate:
        return
    _log.info(
        "parity passage_id=%s text_hash=%s score=%.6f z=%.4f n_scored=%d n_tokens=%d",
        body.passage_id,
        result.text_hash,
        result.score,
        result.z,
        result.n_scored,
        result.n_tokens,
    )
