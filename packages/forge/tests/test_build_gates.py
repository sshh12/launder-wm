"""The build gates, gated.

A gate that can pass by not running is not a gate — and three of the four fixed
here could. These are the regression tests for the ones that live in shell and
in JavaScript, which pytest cannot exercise the way it exercises `verify_all`:

* the Makefile's `docker` target is `sh` inside a container, so what is asserted
  is the SPELLING, on the grounds that the spelling is precisely what was wrong
  (`pip list` in a venv with no pip) and that the identical bug already shipped
  once in `.github/workflows/ci.yml` before being fixed there and left here;
* `web/tools/size-gate.mjs` is run for real, from a temp tree with the tokenizer
  blob missing, because that is the exact condition under which it used to
  print "size-gate: OK" while skipping 93% of the budget;
* `web/tools/pack-check.mjs`'s digest constant is compared against the committed
  blob, because the check that quotes it used to assert only that a sha256 hex
  string is 64 characters long.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from launder_forge.paths import Paths

_NODE = shutil.which("node")


@pytest.fixture(scope="module")
def makefile(real_paths: Paths) -> str:
    return (real_paths.root / "Makefile").read_text(encoding="utf-8")


def _docker_target(makefile: str) -> str:
    body = makefile.split("\ndocker:\n", 1)
    assert len(body) == 2, "the Makefile has no `docker:` target"
    return body[1].split("\n.PHONY", 1)[0]


def test_the_no_torch_gate_looks_inside_the_venv(makefile: str) -> None:
    """`pip list` reported the SYSTEM python's packages — an empty list.

    The runtime image installs the workspace with uv and the venv has no pip in
    it, so `pip list | grep -i torch` printed nothing whatever the venv
    contained and the leading `!` turned that into a pass. The gate could not
    see the thing it was gating. CI was fixed; `make docker`, which the help
    text calls "the image Railway will build, then run its two assertions", was
    not.
    """
    target = _docker_target(makefile)
    assert "pip list" not in target
    assert "/app/.venv/lib/python3.12/site-packages" in target
    # ... and it must prove the path it listed is a real venv, because `ls` of a
    # directory that does not exist is also empty.
    assert "launder_core-*.dist-info" in target


def test_the_image_size_gate_measures_the_rootfs_and_compares_it(makefile: str) -> None:
    """`docker image inspect --format '{{.Size}}'` means two different things.

    Under the containerd snapshotter it is the COMPRESSED content-store size and
    under overlay2 the uncompressed total — 135 MB in development and 584 MB in
    CI for one image (ARCHITECTURE.md §14). Worse, the Makefile only PRINTED it,
    under a heading naming a 250 MB budget it compared to nothing.
    """
    target = _docker_target(makefile)
    assert "docker image inspect" not in target
    assert "du -sm /" in target
    assert "-le 400" in target, "the rootfs ceiling is not asserted"


def test_pack_check_pins_the_digest_of_the_committed_blob(real_paths: Paths) -> None:
    """The line named "sha256 of the committed .br" measured a string length.

    `createHash("sha256").update(br).digest("hex").length === 64` is true of
    every input on every run. The blob's bytes are hashed into
    `asset_bundle_id`, so a swapped or re-compressed blob invalidates every
    passage; this is the digest that says so.
    """
    source = (real_paths.web / "tools" / "pack-check.mjs").read_text(encoding="utf-8")
    assert "brSha256 === EXPECTED.brSha256" in source, "the digest is not compared to anything"

    blob = real_paths.tokenizer_blob.read_bytes()
    sha = re.search(r'brSha256:\s*"([0-9a-f]{64})"', source)
    assert sha is not None, "pack-check.mjs records no expected sha256"
    assert sha.group(1) == hashlib.sha256(blob).hexdigest()

    # `brBlake3` is not computable in node; it is here to be read against the
    # two places Python records it. Asserting that keeps it from rotting into a
    # constant nobody checks.
    b3 = re.search(r'brBlake3:\s*"([0-9a-f]{64})"', source)
    assert b3 is not None
    manifest = json.loads(real_paths.manifest.read_text(encoding="utf-8"))
    entry = manifest["files"][real_paths.rel(real_paths.tokenizer_blob)]
    assert entry["blake3"] == f"blake3:{b3.group(1)}"


def test_the_image_can_say_which_commit_it_is(real_paths: Paths, makefile: str) -> None:
    """`/healthz` answered `"sha": "unknown"` in production.

    It is `railway.json`'s `healthcheckPath`, so it is the deploy gate, and
    `Settings.sha` reads RAILWAY_GIT_COMMIT_SHA, then GIT_SHA, then
    SOURCE_COMMIT — none of which existed in the container. A deploy gate that
    cannot name its own build cannot answer "is the new build live", which is
    the only question anyone asks it.

    The image now takes a `GIT_SHA` build arg and exports it as an env var
    (`Settings.sha` reads that name second), and the two build paths this repo
    controls pass the real value. An EMPTY default is deliberate: "unknown" is
    the honest answer when the sha is genuinely unknown.
    """
    dockerfile = (real_paths.root / "Dockerfile").read_text(encoding="utf-8")
    assert re.search(r"^ARG GIT_SHA=", dockerfile, re.M), "the image takes no GIT_SHA build arg"
    assert re.search(r"^ENV GIT_SHA=", dockerfile, re.M), "GIT_SHA never becomes an env var"
    assert "--build-arg GIT_SHA=" in _docker_target(makefile)

    workflow = (real_paths.root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "GIT_SHA=${{ github.sha }}" in workflow
    # An unused ARG is silently ignored, so CI must also read it back out.
    assert "printenv GIT_SHA" in workflow


def test_every_build_input_the_dockerfile_copies_is_watched_by_railway(
    real_paths: Paths,
) -> None:
    """`watchPatterns` decides whether a push rebuilds the image.

    `packages/forge/pyproject.toml` is COPYed by the deps stage — `uv sync
    --locked` validates the lockfile against every declared workspace member, so
    a member with no pyproject is a hard error — and it was in no watch pattern,
    because `packages/forge/**` is deliberately excluded from the image. A push
    that touched only that file therefore rebuilt nothing, and the running
    container stopped being the thing the repository describes. The rule is
    mechanical: if the Dockerfile reads it from the context, Railway watches it.
    """
    dockerfile = (real_paths.root / "Dockerfile").read_text(encoding="utf-8")
    patterns = json.loads((real_paths.root / "railway.json").read_text(encoding="utf-8"))["build"][
        "watchPatterns"
    ]

    def watched(src: str) -> bool:
        src = src.rstrip("/")
        for pat in patterns:
            if pat == src:
                return True
            if pat.endswith("/**") and (src == pat[:-3] or src.startswith(pat[:-3] + "/")):
                return True
        return False

    for line in dockerfile.splitlines():
        stripped = line.strip()
        if not stripped.startswith("COPY ") or "--from=" in stripped:
            continue  # a stage-to-stage copy reads no build context
        # The last token is the destination inside the image.
        sources = stripped.split()[1:-1]
        for src in sources:
            assert watched(src), (
                f"Dockerfile COPYs {src!r} from the build context, but no railway.json "
                f"watchPattern matches it: a change to it would not redeploy. Patterns: {patterns}"
            )


@pytest.mark.skipif(_NODE is None, reason="node is not installed")
def test_size_gate_fails_when_the_tokenizer_blob_is_not_where_it_looks(tmp_path: Path) -> None:
    """1.19 MB of the 1.4 MB budget used to be able to vanish silently.

    The blob is 93% of the total wire budget and lives in `data/assets/`, which
    the Docker web stage copies to `/data/assets/`. `if (existsSync(blob))
    gate(...)` meant a moved path or a forgotten COPY dropped it out of the
    measurement and the gate still printed "size-gate: OK".
    """
    paths = Paths(tmp_path)
    tools = tmp_path / "web" / "tools"
    tools.mkdir(parents=True)
    shutil.copy(
        Path(__file__).resolve().parents[3] / "web" / "tools" / "size-gate.mjs",
        tools / "size-gate.mjs",
    )
    dist = tmp_path / "web" / "dist"
    dist.mkdir()
    (dist / "app.js").write_text("console.log(1);\n", encoding="utf-8")
    assert not (paths.root / "data" / "assets").exists()

    proc = subprocess.run(
        [str(_NODE), str(tools / "size-gate.mjs")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "tokenizer blob" in (proc.stdout + proc.stderr)
