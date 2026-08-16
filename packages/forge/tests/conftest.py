"""Shared fixtures.

`tmp_repo` is a throwaway copy of the parts of `data/` that the forge reads,
plus the two root markers `find_repo_root` looks for. Tests that mutate
artifacts (corrupting a manifest digest, deleting a copy blurb) work there so
they can never damage the real checkout — which matters more than usual here,
because several of those artifacts are key material.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from launder_forge.paths import Paths, repo_paths

#: Copied into every temp repo. `data/build/` is deliberately absent: it is
#: 33 MB of gitignored input and only `tok pack` needs it.
_COPIED = ("data/config", "data/assets", "data/golden", "data/dev")


@pytest.fixture(scope="session")
def real_paths() -> Paths:
    return repo_paths()


@pytest.fixture
def tmp_repo(tmp_path: Path, real_paths: Paths) -> Paths:
    (tmp_path / "TECH_PLAN.md").write_text("# stub root marker\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("# stub root marker\n", encoding="utf-8")
    for rel in _COPIED:
        src = real_paths.root / rel
        if src.exists():
            shutil.copytree(src, tmp_path / rel)
    return Paths(tmp_path)
