"""HTTP error envelope and the exceptions the API raises.

TECH_PLAN.md §9.3: **a rejection is a game outcome, not an HTTP error.** Real
HTTP errors are reserved for real errors:

| status | meaning                                   |
|--------|-------------------------------------------|
| 400    | malformed body                            |
| 404    | unknown `passage_id` / `level_id`         |
| 413    | body > 64 KB                              |
| 429    | rate limited (carries `Retry-After`)      |
| 503    | DB unavailable                            |

Every message a player can see comes from `data/config/copy.toml` `[errors]`,
never from a string literal in this file — the same rule `gates/feedback.py`
follows in core. `ApiError.code` is the copy key.

**Every one of them is `Cache-Control: no-store`.** §11.4 gives `/api/detect` and
`/api/submit` that header and the endpoints set it on their own responses — but
an `ApiError` never reaches the endpoint's `Response` object: the handler builds
a fresh `JSONResponse` from `error_payload` plus `err.headers`, so the header was
dropped on exactly the responses that must not be stored. 404 and 413 are
heuristically cacheable statuses (RFC 9111 §4.2.2), `GET /` raises `NotFound`
when there is nothing to render, and there is a CDN in front of this origin
(§11.4) — a shared cache that pinned one of those would serve it to everybody.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "ApiError",
    "BodyTooLarge",
    "CoreUnavailable",
    "DatabaseUnavailable",
    "NotFound",
    "RateLimited",
    "error_payload",
]


class ApiError(Exception):
    """A real HTTP error. `code` keys into `[errors]` in copy.toml."""

    def __init__(
        self,
        status_code: int,
        code: str,
        *,
        message: str = "",
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(f"{status_code} {code}: {message}")
        self.status_code = status_code
        self.code = code
        self.message = message
        self.params: dict[str, Any] = dict(params or {})
        # `no-store` FIRST, so a subclass that genuinely needs another policy has
        # to say so rather than inherit one by accident. See the module docstring
        # for why an error response cannot rely on the endpoint's own header.
        self.headers: dict[str, str] = {"Cache-Control": "no-store", **dict(headers or {})}


class NotFound(ApiError):
    def __init__(self, code: str = "unknown_passage", **params: Any) -> None:
        super().__init__(404, code, params=params)


class BodyTooLarge(ApiError):
    def __init__(self, limit: int) -> None:
        super().__init__(413, "too_long", params={"limit": limit})


class RateLimited(ApiError):
    def __init__(self, retry_after_s: int) -> None:
        super().__init__(
            429,
            "rate_limited",
            params={"retry_after_s": retry_after_s},
            headers={"Retry-After": str(retry_after_s)},
        )


class DatabaseUnavailable(ApiError):
    def __init__(self, detail: str = "") -> None:
        super().__init__(503, "network", params={}, headers={"Retry-After": "5"})
        self.detail = detail


class CoreUnavailable(RuntimeError):
    """A `launder_core` module this endpoint needs has not been written yet.

    Deliberately NOT an `ApiError`: this is a build defect, not a runtime
    condition, and it must surface as a 500 with a message naming the exact
    file. Never catch this to serve a plausible-looking fallback — a detector
    that returns invented numbers is the precise failure mode TECH_PLAN.md
    §4.3 and §4.5 exist to prevent.
    """

    def __init__(self, symbol: str, module_path: str, why: str = "") -> None:
        tail = f" {why}" if why else ""
        super().__init__(
            f"launder_serve needs `launder_core.{symbol}`, which lives in "
            f"packages/core/src/launder_core/{module_path} and does not exist yet."
            f"{tail} See TECH_PLAN.md §3 for what belongs in it."
        )
        self.symbol = symbol
        self.module_path = module_path


def error_payload(err: ApiError) -> dict[str, Any]:
    """The JSON body for a real HTTP error."""
    body: dict[str, Any] = {"error": {"code": err.code, "message": err.message}}
    if err.params:
        body["error"]["params"] = err.params
    return body
