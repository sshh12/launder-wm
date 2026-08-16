# Makefile — a thin shim. .build-docs/TECH_PLAN.md §3, §6.4.
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

# The three authoring commands that read a FILE rather than the whole tree, so
# each takes an argument. They used to be spelled without one and could not run
# at all: `make analyze` exited 2 on a missing `candidates` argument, and the
# help text above still advertised them as if they worked. CANDIDATES / TEXT /
# POLICY are overridable on the command line — `make solve TEXT=data/dev/passage.txt`.
CANDIDATES ?= data/candidates/latest.jsonl
TEXT       ?= data/dev/passage.txt
POLICY     ?= data/config/triage/L2.toml

.PHONY: analyze
analyze:
	$(FORGE) analyze $(CANDIDATES)

.PHONY: solve
solve:
	$(FORGE) solve $(TEXT) --max-edits 10

.PHONY: triage
triage:
	$(FORGE) triage --in $(CANDIDATES) --policy $(POLICY) --report

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
#
# THESE ARE THE SAME THREE ASSERTIONS .github/workflows/ci.yml MAKES, spelled
# the same way. Two of them used to be spelled differently here, and both of the
# local spellings were broken:
#
#   * `pip list | grep -i torch` reported the SYSTEM python's packages. The venv
#     has no pip in it, so the command printed nothing whatever the venv
#     contained and the leading `!` turned that empty output into a pass — the
#     "no torch in the image" gate could not see the thing it was gating. It
#     looks at /app/.venv directly now, and then proves the path it listed is a
#     REAL venv, because `ls` of a typo'd directory is also empty.
#   * `docker image inspect --format '{{.Size}}'` means different things on
#     different daemons: the COMPRESSED content-store size under the containerd
#     snapshotter, the uncompressed total under overlay2 — 135 MB here and
#     584 MB in CI for one image. It also only PRINTED a number under a heading
#     claiming a 250 MB budget, comparing it to nothing. `du -sm /` is the same
#     number everywhere; the ceiling is 400 MB and the gap to §2.3's 250 MB
#     target is a warning, exactly as CI has it.
.PHONY: docker
docker:
	docker build -t launder:ci --build-arg GIT_SHA=$$(git rev-parse HEAD) .
	@echo "--- asserting the image knows which commit it is ---"
	docker run --rm --entrypoint sh launder:ci -c 'test -n "$$GIT_SHA"'
	@echo "--- asserting no .env in the image ---"
	! docker run --rm launder:ci sh -c 'ls -a /app | grep -qE "^\.env"'
	@echo "--- asserting no torch in the venv ---"
	! docker run --rm --entrypoint sh launder:ci -c 'ls /app/.venv/lib/python3.12/site-packages' | grep -iE '^torch'
	docker run --rm --entrypoint sh launder:ci -c 'ls -d /app/.venv/lib/python3.12/site-packages/launder_core-*.dist-info >/dev/null'
	@echo "--- asserting no answer keys in the image ---"
	! docker run --rm launder:ci sh -c 'ls /app/data/passages 2>/dev/null | grep -q "author.json"'
	@echo "--- image rootfs (ceiling 400 MB; TECH_PLAN §2.3 target 250 MB) ---"
	@SIZE_MB=$$(docker run --rm --entrypoint sh launder:ci -c 'du -sm / 2>/dev/null | tail -1 | cut -f1'); \
	echo "image rootfs: $$SIZE_MB MB"; \
	test "$$SIZE_MB" -le 400 || { echo "image over the 400 MB ceiling: $$SIZE_MB MB"; exit 1; }; \
	test "$$SIZE_MB" -le 250 || echo "WARNING: above TECH_PLAN §2.3's 250 MB target"

.PHONY: clean
clean:
	$(PY) -c "import shutil,pathlib; [shutil.rmtree(p, ignore_errors=True) for p in ['.pytest_cache','.ruff_cache','.mypy_cache','.hypothesis']]"
