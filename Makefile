# Makefile — a thin shim. TECH_PLAN.md §3, §6.4.
#
# Every target is `uv run ...` so it works identically on Windows, macOS and
# Linux, and so nothing depends on a venv being "activated". `make` itself is
# optional: every line below can be pasted into a shell.
#
# uv is not on PATH in a fresh Windows shell. Either add it once:
#   PowerShell:  $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
#   Git Bash:    export PATH="/c/Users/$USER/.local/bin:$PATH"
# or set UV explicitly:  make UV=C:/Users/you/.local/bin/uv.exe test

UV      ?= uv
PY      := $(UV) run python
FORGE   := $(UV) run python -m launder_forge.cli
PORT    ?= 8000

.DEFAULT_GOAL := help

# ---------------------------------------------------------------------------
.PHONY: help
help:
	@echo "setup      install the workspace (no torch extra)"
	@echo "forge-cpu  install the CPU torch path  (CI, any non-CUDA checkout)"
	@echo "forge-gpu  install the cu130 torch path (RTX 5090 / Blackwell sm_120)"
	@echo "check      ruff + format check + mypy"
	@echo "test       pytest"
	@echo "parity     THE GATE: the TS detector against the Python one, bit for bit"
	@echo "verify     forge verify: recompute every checksum, cross-check the TS detector"
	@echo "serve      run the API on http://127.0.0.1:$(PORT)"
	@echo "migrate    alembic upgrade head"
	@echo "docker     build the image Railway will build, then run its two assertions"

# --- environment ------------------------------------------------------------
.PHONY: setup
setup:
	$(UV) sync

# CPU torch (PyPI wheels, torch==2.13.0). Use this in CI and on any machine
# without a CUDA 13 driver. Same pin as forge-gpu; only the index differs.
.PHONY: forge-cpu
forge-cpu:
	$(UV) sync --extra cpu

# cu130 torch (torch==2.13.0+cu130). cu128/cu129 wheels were REMOVED in 2.13;
# cu130 is the Blackwell path. Never `uv pip install torch` ad hoc — the
# lockfile is the record.
.PHONY: forge-gpu
forge-gpu:
	$(UV) sync --extra cuda

.PHONY: lock
lock:
	$(UV) lock

# --- quality ----------------------------------------------------------------
.PHONY: check
check: lint typecheck

.PHONY: lint
lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .

.PHONY: fmt
fmt:
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

# ONE INVOCATION PER PACKAGE, not `mypy packages`. Both packages/serve/tests
# and packages/forge/tests hold a `conftest.py`, and neither tests directory is
# an importable package, so a single run dies with "Duplicate module named
# conftest" before it checks anything. Adding __init__.py to the test
# directories would fix mypy and break pytest's `from support import ...`, so
# the split is the cheap correct answer. alembic/ rides with serve because
# env.py imports launder_serve.repo.models.
.PHONY: typecheck
typecheck:
	$(UV) run mypy packages/core
	$(UV) run mypy packages/serve alembic
	$(UV) run mypy packages/forge

.PHONY: test
test:
	$(UV) run pytest

.PHONY: test-golden
test-golden:
	$(UV) run pytest -m golden

# The hard invariant of §2.1: `import launder_core` must succeed in a venv
# containing ONLY pydantic, blake3 and numpy. This is what keeps torch out of
# the Railway image, and it only stays true if something checks.
.PHONY: test-bare-core
test-bare-core:
	$(UV) run --isolated --no-project --python 3.12 \
		--with pydantic --with blake3 --with numpy \
		--with-editable packages/core \
		python -c "import launder_core; print('bare core ok:', launder_core.__version__)"

# --- the forge (local only; needs the GPU and the model for `gen`) ----------
.PHONY: doctor
doctor:
	$(FORGE) doctor

.PHONY: bench
bench:
	$(FORGE) bench

.PHONY: table
table:
	$(FORGE) table build

.PHONY: tok-pack
tok-pack:
	$(FORGE) tok pack

.PHONY: gen
gen:
	$(FORGE) gen --n 400 --batch 48

.PHONY: analyze
analyze:
	$(FORGE) analyze

.PHONY: solve
solve:
	$(FORGE) solve --max-edits 10

.PHONY: triage
triage:
	$(FORGE) triage --policy data/config/triage/L2.toml --report

.PHONY: calibrate
calibrate:
	$(FORGE) calibrate --fpr 1e-2 --n 20000 --buckets 40,60,80,120,180,260,400 --out data/assets/thresholds.v1.json

.PHONY: vectors
vectors:
	$(FORGE) vectors

.PHONY: verify
verify:
	$(FORGE) verify

.PHONY: lint-copy
lint-copy:
	$(FORGE) lint-copy

# --- the judge eval set (§7.8) ---------------------------------------------
.PHONY: eval-judge
eval-judge:
	$(UV) run python -m launder_serve.eval.judge --runs 3

.PHONY: eval-judge-fake
eval-judge-fake:
	JUDGE_PROVIDER=cassette $(UV) run python -m launder_serve.eval.judge --runs 1

.PHONY: eval-judge-compare
eval-judge-compare:
	$(UV) run python -m launder_serve.eval.judge --compare

.PHONY: eval-judge-record
eval-judge-record:
	$(UV) run python -m launder_serve.eval.judge --record

.PHONY: eval-judge-promote
eval-judge-promote:
	$(UV) run python -m launder_serve.eval.judge --promote

# --- server -----------------------------------------------------------------
.PHONY: serve
serve:
	$(UV) run uvicorn launder_serve.main:app --host 127.0.0.1 --port $(PORT) --reload

.PHONY: migrate
migrate:
	$(UV) run alembic upgrade head

.PHONY: migration
migration:
	$(UV) run alembic revision --autogenerate -m "$(m)"

# --- web --------------------------------------------------------------------
.PHONY: web-install
web-install:
	cd web && npm ci

.PHONY: web-dev
web-dev:
	cd web && npm run dev

.PHONY: web-build
web-build:
	cd web && npm run build && node tools/pack-check.mjs && npm run slop-check && npm run parity && npm run size-gate

# THE PARITY GATE (§4.5): the shipped TypeScript detector against the shipped
# Python one, on the same inputs. `npm run parity` is the TS half alone;
# `parity` below is the cross-language comparison and is the one that matters.
.PHONY: parity
parity:
	cd web && npm run parity
	$(UV) run pytest -m golden packages/forge/tests/test_parity_ts.py -v

.PHONY: web-test
web-test:
	cd web && npm run test

# --- docker (the same image Railway builds) ---------------------------------
.PHONY: docker
docker:
	docker build -t launder:ci .
	@echo "--- asserting no .env in the image ---"
	! docker run --rm launder:ci sh -c 'ls -a /app | grep -qE "^\.env"'
	@echo "--- asserting no torch in the venv ---"
	! docker run --rm launder:ci sh -c 'pip list 2>/dev/null | grep -i torch'
	@echo "--- image size (budget: 250 MB) ---"
	docker image inspect launder:ci --format '{{.Size}}'

.PHONY: clean
clean:
	$(PY) -c "import shutil,pathlib; [shutil.rmtree(p, ignore_errors=True) for p in ['.pytest_cache','.ruff_cache','.mypy_cache','.hypothesis']]"
