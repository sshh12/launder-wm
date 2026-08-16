# syntax=docker/dockerfile:1.7
#
# TECH_PLAN.md §11.2. node build stage -> python deps stage -> runtime.
#
# `--package launder-serve` is what keeps torch out: launder-forge is never
# resolved, so neither is torch, and the image stays ~330 MB instead of ~8 GB.
# CI asserts `pip list | grep -i torch` is empty and measures the image against
# the §2.3 budget. Both are build gates, not warnings.
#
# **THE RUNTIME STAGE HAS NO uv.** It used to run on
# `ghcr.io/astral-sh/uv:python3.12-bookworm-slim` and call `uv sync` again, which
# carried a 52.5 MB uv binary into the final image for a tool the container never
# invokes after build. The workspace install now happens entirely in the deps
# stage and the runtime is stock `python:3.12-slim-bookworm` — the SAME base the
# uv image is built on, so the venv's interpreter symlinks resolve unchanged.

# EVERY `--mount=type=cache` CARRIES AN EXPLICIT `id=`. Local BuildKit defaults
# the id to the target path and accepts the flag without one; Railway's Metal
# builder does not, and rejects the Dockerfile outright with "flag
# '--mount=type=cache,target=...' is missing an id argument". That is a build
# which passes `docker build` on a laptop and fails every deploy, so these ids
# are not optional decoration.

# ---------- stage 1: frontend ----------
FROM node:24-bookworm-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN --mount=type=cache,id=launder-npm,target=/root/.npm npm ci
# The build reads the committed assets and the golden vectors: vitest runs the
# SAME data/golden/vectors.json pytest reads, and pack-check.mjs asserts the
# packed tokenizer blob round-trips to zero mismatches.
COPY data/assets/ /data/assets/
COPY data/golden/ /data/golden/
# The shipped passages too, so `npm run parity` re-derives every one of them
# from `token_ids` and fails the BUILD on a passage the browser detector reads
# differently — not the player's first keystroke. `.dockerignore` keeps
# *.author.json out, which is also what makes this safe. `data/passages/.gitkeep`
# is load-bearing: git cannot commit an empty directory, so without it this COPY
# fails on a fresh clone (GitHub Actions checkout, Railway build) before any
# layer is built.
COPY data/passages/ /data/passages/
COPY web/ ./
# `npm run parity` runs the SHIPPED detector modules over data/golden/vectors.json
# and every data/passages/*.public.json, so an image cannot be built from a
# browser detector that disagrees with the committed goldens. The cross-language
# half (against launder_core) runs in CI's python job, which has both runtimes.
RUN npm run build \
 && node tools/pack-check.mjs \
 && npm run slop-check \
 && npm run parity \
 && npm run size-gate

# ---------- stage 2: python deps AND the workspace install ----------
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS deps
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY packages/core/pyproject.toml packages/core/
COPY packages/serve/pyproject.toml packages/serve/
# The forge MANIFEST only, never its source (.dockerignore excludes
# `packages/forge/` and re-admits exactly this one file). `uv sync --locked`
# validates the lockfile against the whole workspace, so a declared member with
# no pyproject.toml is a hard error. `--package launder-serve` is what keeps
# torch out: launder-forge is never resolved, only acknowledged.
COPY packages/forge/pyproject.toml packages/forge/
# Third-party wheels first, on their own layer, so a source-only change does not
# re-resolve them.
RUN --mount=type=cache,id=launder-uv,target=/root/.cache/uv \
    uv sync --locked --no-dev --package launder-serve --no-install-project
COPY packages/core/ ./packages/core/
COPY packages/serve/ ./packages/serve/
RUN --mount=type=cache,id=launder-uv,target=/root/.cache/uv \
    uv sync --locked --no-dev --package launder-serve

# ---------- stage 3: runtime ----------
# Stock python, NOT the uv image: nothing here runs uv.
FROM python:3.12-slim-bookworm AS runtime
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"
WORKDIR /app
COPY --from=deps /app/.venv /app/.venv
# The venv installs launder-core and launder-serve from /app/packages, so the
# sources must land at the SAME path they were installed from.
COPY --from=deps /app/packages /app/packages
COPY pyproject.toml uv.lock alembic.ini ./
# REQUIRED: we redistribute the Gemma-3 tokenizer, so the Gemma Terms of Use
# notice ships with the artifact that contains it (§6.1).
COPY NOTICE ./
COPY alembic/ ./alembic/
# LAST: level and threshold tuning happens far more often than code changes, so
# a config-only edit rebuilds one thin layer.
COPY data/ ./data/
COPY --from=web /web/dist ./web/dist
USER 1000:1000
CMD ["sh", "-c", "uvicorn launder_serve.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --proxy-headers --forwarded-allow-ips '*'"]
