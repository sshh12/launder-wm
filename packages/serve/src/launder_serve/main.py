"""The app factory (TECH_PLAN.md §9, §11.4).

**THE API ROUTER IS MOUNTED BEFORE THE STATIC CATCH-ALL.** A `StaticFiles`
mount at `/` swallows every path beneath it, so a router registered afterwards
is dead code that passes every unit test and 404s every real request. The two
`app.include_router` / `app.mount` calls at the bottom of `create_app()` are in
the order they are in for that reason and no other.

Boot order, and why:

1. Load `data/` and assert its invariants (`content.load_content`) — a
   `wm_config_id` that disagrees with the passages must stop the process, not
   produce a server that renders confident, meaningless numbers.
2. Assert the judge's `prompt_hash` against `judge.toml`.
3. Build the repositories, the judge (one provider, wrapped in the abuse
   ladder) and the engine seams.
4. Mount routes.

Anything that can be wrong is wrong here, loudly, before a player sees it.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from launder_core.levels import unsatisfiable_levels
from launder_core.schemas import MAX_TEXT_BYTES
from launder_serve.api import api_router
from launder_serve.api.deps import AppState
from launder_serve.boot import BootRenderer
from launder_serve.content import Content, error_message, load_content
from launder_serve.engine import CoreDetector, CoreScorer, ServerDetector
from launder_serve.errors import ApiError, BodyTooLarge, NotFound, error_payload
from launder_serve.judge import (
    CachingJudge,
    JudgeProvider,
    assert_prompt_hash,
    build_provider,
    resolve_model,
    resolve_prompt_hash,
)
from launder_serve.limits import TokenBucketLimiter
from launder_serve.repo.memory import (
    MemoryJudgeCacheRepo,
    MemoryProgressRepo,
    MemorySpendRepo,
    MemorySubmissionRepo,
)
from launder_serve.repo.protocol import JudgeCacheRepo, ProgressRepo, SpendRepo, SubmissionRepo
from launder_serve.settings import Settings, get_settings
from launder_serve.static import AUTHOR_ONLY_SUFFIXES, IMMUTABLE, PrecompressedStaticFiles

__all__ = ["LEVEL_COOKIE", "RepoSet", "create_app", "resolve_level_n", "run"]

_log = logging.getLogger("launder")

#: Hard body cap for the two POST endpoints (§9.2, §9.3). Enforced in the ASGI
#: layer as well as by the schema so a 40 MB body is refused before it is read
#: into memory, not after.
MAX_BODY_BYTES = MAX_TEXT_BYTES

#: Written by the CLIENT only, on a clear (`min(level_n + 1, level_count)`). It
#: exists so `GET /` can render the right level with zero API calls and no
#: flash of level 1 — the server reads it and never sets it.
LEVEL_COOKIE = "launder_level"


@dataclass
class RepoSet:
    submissions: SubmissionRepo
    judge_cache: JudgeCacheRepo
    spend: SpendRepo
    progress: ProgressRepo
    engine: Any | None = None

    @classmethod
    def in_memory(cls) -> RepoSet:
        return cls(
            submissions=MemorySubmissionRepo(),
            judge_cache=MemoryJudgeCacheRepo(),
            spend=MemorySpendRepo(),
            progress=MemoryProgressRepo(),
        )


class BodySizeLimitMiddleware:
    """413 for `/api/*` bodies over the cap (§9.2, §9.3), before the app sees them.

    `Content-Length` is checked first because it is free, but it is only a
    claim: a chunked request carries none, and a lying one is exactly the
    request this exists to stop. So the body is read here, bounded — one byte
    past the cap is enough to decide, and the read stops there rather than
    buffering a 40 MB upload to measure it.

    Buffering is affordable precisely because the cap is 64 KB. If that number
    ever grows past a megabyte, this becomes a streaming check instead.
    """

    def __init__(self, app: ASGIApp, *, max_bytes: int = MAX_BODY_BYTES) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not str(scope.get("path", "")).startswith("/api/"):
            await self.app(scope, receive, send)
            return

        declared = _content_length(scope)
        if declared is not None and declared > self.max_bytes:
            await _send_error(send, BodyTooLarge(self.max_bytes))
            return

        chunks: list[bytes] = []
        seen = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body = bytes(message.get("body", b""))
            seen += len(body)
            if seen > self.max_bytes:
                await _send_error(send, BodyTooLarge(self.max_bytes))
                return
            chunks.append(body)
            if not message.get("more_body", False):
                break

        replayed: list[Message] = [
            {"type": "http.request", "body": b"".join(chunks), "more_body": False},
            {"type": "http.disconnect"},
        ]

        async def replay() -> Message:
            # After the body, every further receive() is a disconnect — an
            # endpoint that reads twice must not block forever waiting on a
            # stream this middleware already drained.
            return replayed.pop(0) if len(replayed) > 1 else replayed[0]

        await self.app(scope, replay, send)


def _content_length(scope: Scope) -> int | None:
    for key, value in scope.get("headers", []):
        if key.lower() == b"content-length":
            try:
                return int(value)
            except ValueError:
                return None
    return None


async def _send_error(send: Send, err: ApiError) -> None:
    response = JSONResponse(status_code=err.status_code, content=error_payload(err))
    for name, value in err.headers.items():
        response.headers[name] = value
    await response({"type": "http"}, _empty_receive, send)


async def _empty_receive() -> Message:  # pragma: no cover - starlette needs the callable
    return {"type": "http.disconnect"}


def create_app(
    *,
    settings: Settings | None = None,
    content: Content | None = None,
    repos: RepoSet | None = None,
    scorer: CoreScorer | None = None,
    detector: ServerDetector | None = None,
    judge: JudgeProvider | None = None,
    mount_static: bool = True,
) -> FastAPI:
    """Build the app. Every seam is injectable; every default is the real thing."""
    settings = settings or get_settings()
    warnings: list[str] = []

    content = content or load_content(
        settings.data_path,
        asset_bundle_id=settings.asset_bundle_id,
        is_production=settings.is_production,
    )
    warnings.extend(content.warnings)
    warnings.extend(settings.assert_keys_present())
    settings.assert_model_pinned(resolve_model(content, settings))
    warnings.extend(assert_prompt_hash(content, settings))
    prompt_hash = resolve_prompt_hash(content, settings)

    repos = repos or _default_repos(settings)
    limiter = TokenBucketLimiter(
        per_hour=settings.rate_limit_judge_per_hour,
        burst=settings.rate_limit_judge_burst,
    )

    http: httpx.AsyncClient | None = None
    if settings.judge_is_paid:
        timeout = settings.judge_timeout_s
        http = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=min(2.0, timeout), read=timeout, write=timeout),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )

    if judge is None:
        judge = CachingJudge(
            provider=build_provider(content, settings, http=http),
            cache=repos.judge_cache,
            spend=repos.spend,
            limiter=limiter,
            judge_version=content.judge_version,
            prompt_hash=prompt_hash,
            scoring_version=content.scoring_version,
            separator=content.judge.cache.separator,
            lru_entries=content.judge.cache.lru_entries,
            cache_failures=content.judge.cache.cache_failures,
            daily_usd_cap=settings.judge_daily_usd_cap,
            est_usd_per_call=settings.judge_est_usd_per_call,
            max_retries=content.judge.max_retries,
        )

    # THE DEPENDENCIES THE GATE WILL ACTUALLY HAVE (see api/submit.py's `Deps`).
    #
    # `unit_tests` is NOT among them: core declares `UnitTestRunner` and refuses
    # to implement it (running player-authored code needs a real sandbox, and
    # `memory_mb` has no portable implementation), and serve does not build one
    # either. `load_levels` validated check NAMES and not the DEPENDENCIES those
    # checks declare, so L5 booted perfectly clean and then raised
    # `GateDependencyError` as an unhandled 500 on its first submit.
    #
    # An unserviceable ruleset is DROPPED here, loudly, so requests naming it
    # get the ordinary `unknown_level` 404 instead of a 500 — and running a
    # CAMPAIGN level under one is a hard boot failure, because that is a level
    # of the campaign nobody could ever get past.
    content = _drop_unserviceable_levels(content, warnings)

    state = AppState(
        settings=settings,
        content=content,
        prompt_hash=prompt_hash,
        scorer=scorer or CoreScorer(content.scoring),
        detector=detector or CoreDetector(),
        judge=judge,
        submissions=repos.submissions,
        progress=repos.progress,
        limiter=limiter,
        engine=repos.engine,
        http=http,
        boot_warnings=tuple(warnings),
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        for warning in state.boot_warnings:
            _log.warning("boot: %s", warning)
        _log.info(
            "launder-serve up: env=%s sha=%s wm_config_id=%s judge=%s levels=%d passages=%d",
            settings.env,
            settings.sha,
            content.wm_config_id,
            settings.judge_provider,
            content.level_count,
            len(content.passages),
        )
        await _prepare_database(settings, repos)
        try:
            yield
        finally:
            if http is not None:
                await http.aclose()
            if repos.engine is not None:
                await repos.engine.dispose()

    app = FastAPI(
        title="Launder WM",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/api/docs" if not settings.is_production else None,
        redoc_url=None,
        openapi_url="/api/openapi.json" if not settings.is_production else None,
    )
    app.state.launder = state
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=MAX_BODY_BYTES)

    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError) -> JSONResponse:
        message = exc.message or error_message(state.content.copy, exc.code, **exc.params)
        payload = error_payload(exc)
        payload["error"]["message"] = message
        return JSONResponse(status_code=exc.status_code, content=payload, headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # 400, not 422: §9.3 enumerates the API's error codes and 422 is not one
        # of them. The detail is kept because a rejected client-supplied score
        # field explains itself in that message.
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "code": "malformed_body",
                    "message": error_message(state.content.copy, "malformed_body")
                    or "The request body is not valid.",
                    "detail": _safe_errors(exc),
                }
            },
        )

    # ---- MOUNT ORDER: API FIRST, INDEX, STATIC CATCH-ALL LAST (§11.4) ------
    app.include_router(api_router)
    if mount_static:
        _mount_index(app, state, settings)
        _mount_static(app, settings)
    return app


def resolve_level_n(query: str | None, cookie: str | None, level_count: int) -> int:
    """Which level `GET /` renders: `?level=`, then the cookie, then 1.

    `?level=` is the TEST ESCAPE HATCH. It renders that level regardless of
    progress and does NOT write the cookie, so a scripted run can reach level 12
    without playing eleven levels first and without leaving the browser
    convinced it belongs there.

    Anything that is not an integer inside `1..level_count` falls back to 1
    rather than being clamped into range: clamping would render level 15 for a
    typo'd `?level=150`, and a test meaning "that level does not exist" would
    then pass against the wrong page.
    """
    for raw in (query, cookie):
        if raw is None:
            continue
        try:
            n = int(raw.strip())
        except ValueError:
            return 1
        if 1 <= n <= level_count:
            return n
        # A present-but-unusable value ENDS the search rather than falling
        # through to the next source: a bad `?level=` quietly rendering whatever
        # the cookie says would make the escape hatch untrustworthy exactly when
        # a test is relying on it.
        return 1
    return 1


def _mount_index(app: FastAPI, state: AppState, settings: Settings) -> None:
    """`GET /` renders `index.html`. **BEFORE the static catch-all** (§11.4).

    This route is the whole of §5.4 step 1. Without it `StaticFiles(html=True)`
    served the checked-in DEV FIXTURE verbatim — `"dev": true`, `"passage_id":
    "p_dev"`, `"assets": null` — so `main.ts`'s `upgrade()` bailed on the first
    line, the local detector never loaded, and the level's passage was never
    delivered to the client at all.
    """
    renderer = BootRenderer(settings.web_dist_path, state.content, state.detector)
    if not renderer.available():
        return
    missing = renderer.missing_regions()
    if missing:
        # A dist that is not the launder page (a placeholder, somebody else's
        # SPA). Falling through to the static mount is right for it — but it is
        # also exactly the state that shipped the dev fixture to players, so it
        # is a WARNING with the consequence spelled out, never silence.
        _log.warning(
            "boot: %s carries no %s marker(s), so GET / falls through to the static "
            "mount and will serve that file VERBATIM. If that file is web/index.html, "
            "the page ships the checked-in dev fixture (passage_id p_dev, assets null) "
            "and every /api/detect it makes will 404.",
            renderer.index,
            ", ".join(missing),
        )
        return

    @app.get("/", include_in_schema=False)
    @app.get("/index.html", include_in_schema=False)
    async def index(request: Request) -> Response:
        level_n = resolve_level_n(
            request.query_params.get("level"),
            request.cookies.get(LEVEL_COOKIE),
            state.content.level_count,
        )
        html = renderer.render(level_n)
        if html is None:
            # Nothing to render: no packed passages AND no dev fixture. Serving
            # the raw template would hand the player a page whose every API call
            # 404s, so say so instead.
            raise NotFound("unknown_passage", level_n=level_n)
        return HTMLResponse(
            content=html,
            headers={
                # `private`, because the page the cookie selected is this
                # player's level and a shared cache handing it to the next
                # visitor would drop them into somebody else's campaign.
                # `no-cache` rather than `no-store`: revalidation is cheap and
                # the ETag makes the common case a 304 (§11.4).
                "Cache-Control": "private, no-cache",
                # `Cookie` is load-bearing: the body depends on `launder_level`,
                # so a cache keyed on the URL alone would serve level 1 to
                # everyone who ever got there first.
                "Vary": "Cookie, Accept-Encoding",
            },
        )


#: Field names of `launder_core.gates.registry.Deps` that `api/submit.py` fills.
#: Keep in step with the `Deps(...)` construction there.
WIRED_DEPS: frozenset[str] = frozenset({"detector", "judge"})


def _drop_unserviceable_levels(content: Content, warnings: list[str]) -> Content:
    """Remove rulesets whose checks need a dependency this process cannot provide."""
    from dataclasses import replace as dataclass_replace

    unserviceable = unsatisfiable_levels(content.levels, WIRED_DEPS)
    if not unserviceable:
        return content

    for entry in content.campaign:
        if entry.level_id in unserviceable:
            missing = ", ".join(sorted(unserviceable[entry.level_id]))
            raise RuntimeError(
                f"progression.toml runs level {entry.n} under rules {entry.level_id}, whose "
                f"checks require Deps.{missing} — which this process does not provide. That "
                "level would be a 500 on the first submit, and because the campaign is "
                "linear it would strand every player who reached it. Wire the dependency, "
                "or run that level under a different ruleset."
            )

    for level_id, missing_deps in sorted(unserviceable.items()):
        missing = ", ".join(sorted(missing_deps))
        warnings.append(
            f"level {level_id} is UNSERVICEABLE and has been dropped: its checks require "
            f"Deps.{missing}, which this process does not wire. Requests naming it now get "
            "404 unknown_level instead of an unhandled 500. To restore it, implement the "
            f"dependency and add it to launder_serve.main.WIRED_DEPS."
        )
    kept = {k: v for k, v in content.levels.items() if k not in unserviceable}
    return dataclass_replace(content, levels=kept)


def _safe_errors(exc: RequestValidationError) -> list[dict[str, Any]]:
    """Validation detail without echoing the submitted text back at the client."""
    out: list[dict[str, Any]] = []
    for err in exc.errors()[:5]:
        out.append(
            {
                "loc": [str(part) for part in err.get("loc", ())],
                "type": err.get("type", ""),
                "msg": str(err.get("msg", ""))[:400],
            }
        )
    return out


def _mount_static(app: FastAPI, settings: Settings) -> None:
    data_root = settings.data_path
    assets = data_root / "assets"
    passages = data_root / "passages"
    dist = settings.web_dist_path

    if assets.is_dir():
        app.mount(
            "/data/assets",
            PrecompressedStaticFiles(directory=assets, cache_control=IMMUTABLE),
            name="assets",
        )
    if passages.is_dir():
        app.mount(
            "/data/passages",
            PrecompressedStaticFiles(
                directory=passages,
                cache_control="public, max-age=300",
                # ENFORCED, not documented: see static.AUTHOR_ONLY_SUFFIXES.
                deny_suffixes=AUTHOR_ONLY_SUFFIXES,
            ),
            name="passages",
        )
    if dist.is_dir():
        # `html=True` makes this the SPA catch-all. It is LAST on purpose.
        app.mount("/", PrecompressedStaticFiles(directory=dist, html=True), name="web")
    else:
        _log.warning(
            "web/dist does not exist at %s: the API is up but there is no game to serve. "
            "Run `npm run build` in web/, or set WEB_DIST_DIR.",
            dist,
        )


def _default_repos(settings: Settings) -> RepoSet:
    from launder_serve.repo.sqlalchemy import (
        SqlJudgeCacheRepo,
        SqlProgressRepo,
        SqlSpendRepo,
        SqlSubmissionRepo,
        make_engine,
    )

    engine = make_engine(settings.resolved_database_url())
    return RepoSet(
        submissions=SqlSubmissionRepo(engine),
        judge_cache=SqlJudgeCacheRepo(engine),
        spend=SqlSpendRepo(engine),
        progress=SqlProgressRepo(engine),
        engine=engine,
    )


async def _prepare_database(settings: Settings, repos: RepoSet) -> None:
    """Create the dev SQLite schema. Production runs `alembic upgrade head`.

    §9.8: `preDeployCommand` runs the migration between build and deploy, and a
    non-zero exit blocks the deployment. Creating tables from metadata at boot
    in production would silently paper over a migration that failed to apply.
    """
    if repos.engine is None or settings.is_production:
        return
    if repos.engine.dialect.name != "sqlite":
        return
    from launder_serve.repo.sqlalchemy import create_all

    await create_all(repos.engine)


def __getattr__(name: str) -> Any:
    """`uvicorn launder_serve.main:app` builds the app; importing does not.

    Boot assertions read `data/` and can raise; a bare `import launder_serve.main`
    (by a test, by alembic, by `--help`) must not trigger them.
    """
    if name == "app":
        application = create_app()
        globals()["app"] = application
        return application
    raise AttributeError(f"module 'launder_serve.main' has no attribute {name!r}")


def run() -> None:
    """`launder-serve` console script. Railway uses the uvicorn command directly."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "launder_serve.main:app",
        host="0.0.0.0",
        port=settings.port,
        workers=1,  # load-bearing: the bucket and the LRU are process-local (§9.6)
        proxy_headers=True,
        forwarded_allow_ips="*",
    )
