"""The API router. Assembled here, mounted BEFORE the static catch-all.

Order is stated in `main.py` too, because getting it wrong produces a server
that passes every unit test and 404s every API call in production: a
`StaticFiles` mount at "/" swallows everything below it.
"""

from __future__ import annotations

from fastapi import APIRouter

from launder_serve.api import detect, health, progress, submit

__all__ = ["api_router"]

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(detect.router)
api_router.include_router(submit.router)
api_router.include_router(progress.router)
