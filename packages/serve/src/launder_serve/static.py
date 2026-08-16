"""Precompressed-sibling static files + the §11.4 cache policy.

**No `GZipMiddleware`.** Brotli-11 on a 3.6 MB blob, per request, is exactly the
CPU we are paying not to spend: the assets are compressed once at build time and
committed, and this subclass simply prefers the `.br` (then `.gz`) sibling when
`Accept-Encoding` allows, sets `Content-Encoding` and `Vary` itself, and never
invokes runtime compression.

| asset class                              | header                                        |
|------------------------------------------|-----------------------------------------------|
| `web/dist/assets/*` (content-hashed)     | `public, max-age=31536000, immutable`         |
| `data/assets/*` (blob, table, thresholds)| same, plus `Vary: Accept-Encoding`            |
| `data/passages/*.public.json`            | `public, max-age=300`                         |
| `index.html`                             | `no-cache` (ETag -> 304)                      |

**Egress is the real cost line and it is asset-shaped** (§11.4): 50k uniques x
1.28 MB is ~$3.20, but the same traffic against an unpacked 33 MB
`tokenizer.json` is ~$82.50. Content-hashed immutable assets plus Cloudflare in
front of a custom domain collapses it to near zero — and keeping the origin
headers authoritative is what lets that work.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Final

from starlette.datastructures import Headers
from starlette.exceptions import HTTPException
from starlette.responses import FileResponse, Response
from starlette.staticfiles import StaticFiles
from starlette.types import Receive, Scope, Send

__all__ = [
    "AUTHOR_ONLY_SUFFIXES",
    "IMMUTABLE",
    "PrecompressedStaticFiles",
    "cache_control_for",
]

#: Filename suffixes under `data/passages/` that MUST NOT be served, ever.
#:
#: §2.3's table says `data/passages/*.author.json` — Browser: "never", and
#: `schemas/passage.py` says "repo only, never served". Nothing enforced it: the
#: whole directory was mounted with no filter, so the answer key (the reference
#: solution, the solver's `par_upper` and `clears_at_k`, the optionality and
#: entropy arrays, `topk_alts`, the generation provenance) came back 200 on any
#: deployment run from source. The only thing keeping it out of production was a
#: build-time `.dockerignore` line, which is not a serving rule.
#:
#: `*.claims.draft.json` is here too: it is an unreviewed model-written claim
#: list, i.e. the answer key in prose form.
AUTHOR_ONLY_SUFFIXES: tuple[str, ...] = (".author.json", ".claims.draft.json")

IMMUTABLE: Final[str] = "public, max-age=31536000, immutable"
_NO_CACHE: Final[str] = "no-cache"
_PASSAGE: Final[str] = "public, max-age=300"
_DEFAULT: Final[str] = "public, max-age=3600"

#: `.br` first: it is both smaller and the one the build actually produces at
#: level 11. `.gz` is the fallback for the small number of clients that still
#: do not advertise br over TLS.
_ENCODINGS: Final[tuple[tuple[str, str], ...]] = ((".br", "br"), (".gz", "gzip"))

_CONTENT_TYPES: Final[dict[str, str]] = {
    ".br": "application/octet-stream",
    ".gz": "application/octet-stream",
}


def cache_control_for(path: str) -> str:
    """The §11.4 table, as a function of the request path.

    Backslashes are folded first: starlette hands `get_response` an OS-native
    path, so on Windows the sub-path arrives as `assets\app-a1b2c3.js` and a
    naive `"/assets/" in path` silently drops every hashed asset to the default
    one-hour policy — on the exact platform the developer is testing on.
    """
    lowered = path.replace("\\", "/").lower()
    if lowered.endswith((".html", "/")) or not lowered:
        return _NO_CACHE
    if "/data/passages/" in lowered:
        return _PASSAGE
    if "/data/assets/" in lowered or "/assets/" in lowered:
        return IMMUTABLE
    return _DEFAULT


def _accepts(header: str, encoding: str) -> bool:
    """True when `Accept-Encoding` allows `encoding` and does not q=0 it."""
    for part in header.split(","):
        token = part.strip()
        if not token:
            continue
        name, _, params = token.partition(";")
        name = name.strip().lower()
        if name not in (encoding, "*"):
            continue
        for param in params.split(";"):
            key, _, value = param.partition("=")
            if key.strip().lower() == "q":
                try:
                    if float(value.strip()) == 0.0:
                        return False
                except ValueError:
                    return False
        return True
    return False


class PrecompressedStaticFiles(StaticFiles):
    """Serves `<file>.br` / `<file>.gz` in place of `<file>` when allowed.

    `deny_suffixes` is an ENFORCED serving rule, not a build convention. See
    `AUTHOR_ONLY_SUFFIXES`.
    """

    def __init__(
        self,
        *args: Any,
        cache_control: str | None = None,
        deny_suffixes: tuple[str, ...] = (),
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._forced_cache_control = cache_control
        self._deny_suffixes = tuple(s.lower() for s in deny_suffixes)

    def _denied(self, path: str) -> bool:
        name = path.replace("\\", "/").rsplit("/", 1)[-1].lower()
        # Strip a precompressed sibling's extension too: `x.author.json.br` is
        # the same secret with three more bytes on it.
        for suffix, _ in _ENCODINGS:
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        return any(name.endswith(s) for s in self._deny_suffixes)

    def file_response(
        self,
        full_path: str | os.PathLike[str],
        stat_result: os.stat_result,
        scope: Scope,
        status_code: int = 200,
    ) -> Response:
        response = super().file_response(full_path, stat_result, scope, status_code)
        path = str(scope.get("path", ""))
        response.headers["Cache-Control"] = self._forced_cache_control or cache_control_for(path)
        # `Vary` unconditionally, not just on the compressed branch: a shared
        # cache that stored the identity body under a key with no Vary would
        # then serve it to a client that asked for br, and vice versa.
        response.headers["Vary"] = "Accept-Encoding"
        return response

    async def get_response(self, path: str, scope: Scope) -> Response:
        # THE ANSWER KEY IS NOT SERVABLE. Checked here, before any filesystem
        # lookup, so it applies to the identity body and to every precompressed
        # sibling. `data/passages` was mounted with no filename filter at all,
        # which made `*.author.json` — reference_solution, the solver trace with
        # par_upper and clears_at_k, the optionality arrays, topk_alts and the
        # generation provenance — publicly readable over HTTP from any
        # source-run deployment. `.dockerignore` kept it out of the IMAGE, so
        # the guardrail existed only at build time on one deployment path.
        if self._denied(path):
            raise HTTPException(status_code=404)
        request_headers = Headers(scope=scope)
        accept = request_headers.get("accept-encoding", "")
        full_path = Path(self.directory or ".") / path

        for suffix, encoding in _ENCODINGS:
            if not _accepts(accept, encoding):
                continue
            candidate = Path(str(full_path) + suffix)
            try:
                stat_result = candidate.stat()
            except OSError:
                continue
            if not candidate.is_file():
                continue
            media_type = self._media_type_of(path)
            response = FileResponse(candidate, stat_result=stat_result, media_type=media_type)
            response.headers["Content-Encoding"] = encoding
            response.headers["Vary"] = "Accept-Encoding"
            response.headers["Cache-Control"] = self._forced_cache_control or cache_control_for(
                "/" + path
            )
            # The ETag starlette derived is the *sibling's*; distinguish it from
            # the identity body's so a cache cannot cross them.
            etag = response.headers.get("etag")
            if etag:
                response.headers["ETag"] = (
                    etag[:-1] + f'-{encoding}"' if etag.endswith('"') else etag
                )
            return response

        return await super().get_response(path, scope)

    @staticmethod
    def _media_type_of(path: str) -> str:
        import mimetypes

        guessed, _ = mimetypes.guess_type(path)
        return guessed or _CONTENT_TYPES.get(Path(path).suffix, "application/octet-stream")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        await super().__call__(scope, receive, send)
