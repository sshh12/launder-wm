"""`data/MANIFEST.json` must verify on a FRESH CLONE, not just on this machine.

THE BUG THIS EXISTS FOR. `forge manifest --write` hashes the bytes on disk. On
Windows, a tool that rewrites a config file through Python's text mode — which
is every `pathlib.Path.write_text` without `newline=""` — turns every `\\n` into
`\\r\\n`. `.gitattributes` marks `data/config/**` as `text eol=lf`, so Git stores
the LF form and hands it to CI, while the manifest was computed over the CRLF
form sitting in the author's working tree.

The result verifies locally, forever, and can never verify anywhere else.
`data/config/copy.toml` shipped exactly that way, and nothing caught it: `forge
verify` ran only on the author's machine, because it had never been added to CI.
It surfaced the same hour CI started running it.

So this compares the manifest against `git show HEAD:<path>` — the bytes a clone
actually receives — rather than against the working tree, which is the thing
`forge verify` already checks and the thing that was wrong.
"""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from launder_forge.paths import repo_paths

_GIT = shutil.which("git")


def _blob(rel: str) -> bytes | None:
    """The committed bytes for a path, or None if it is not tracked."""
    proc = subprocess.run(
        [str(_GIT), "show", f"HEAD:{rel}"],
        capture_output=True,
        cwd=repo_paths().root,
        check=False,
    )
    return proc.stdout if proc.returncode == 0 else None


@pytest.mark.skipif(_GIT is None, reason="git is not on PATH")
def test_every_manifested_file_is_byte_identical_to_what_git_stores() -> None:
    paths = repo_paths()
    if not paths.manifest.is_file():
        pytest.skip("data/MANIFEST.json has not been written yet")
    if subprocess.run(
        [str(_GIT), "rev-parse", "--verify", "HEAD"],
        capture_output=True,
        cwd=paths.root,
        check=False,
    ).returncode:
        pytest.skip("no HEAD commit in this checkout")

    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    files = manifest.get("files", manifest)

    drifted: list[str] = []
    for rel in files:
        on_disk = paths.root / rel
        if not on_disk.is_file():
            continue
        committed = _blob(rel)
        # Untracked (a locally generated asset) is not this test's business —
        # `forge verify` covers the working tree.
        if committed is None:
            continue
        if committed != on_disk.read_bytes():
            drifted.append(rel)

    assert not drifted, (
        "these manifested files differ between the working tree and what Git stores, so the "
        "blake3 in data/MANIFEST.json describes bytes no clone will ever see: "
        + ", ".join(drifted)
        + ". Almost always CRLF: a tool wrote them through Python's text mode on Windows while "
        ".gitattributes stores them as LF. Restore them (`git checkout --`) and re-run "
        "`forge manifest --write`; do not regenerate the manifest over the local bytes."
    )
