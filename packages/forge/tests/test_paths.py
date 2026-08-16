"""Repo-root discovery — the walk that every other test depends on.

`_ROOT_MARKERS` once listed TECH_PLAN.md, which then moved to `.build-docs/`.
A marker that exists nowhere makes the walk run past the checkout to the
filesystem root and raise, so every forge command and 55 of these 80 tests died
at fixture setup. That failure is silent in review and total at runtime, which
is why the marker set gets its own test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import launder_forge
from launder_forge.paths import find_repo_root


def test_finds_the_checkout_root_from_inside_the_package(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LAUNDER_REPO_ROOT", raising=False)
    root = find_repo_root(Path(launder_forge.__file__))
    assert (root / "data" / "config" / "copy.toml").is_file()
    assert (root / "pyproject.toml").is_file()


def test_does_not_stop_at_the_forge_package(monkeypatch: pytest.MonkeyPatch) -> None:
    """`packages/forge/` has a `pyproject.toml` of its own and no `data/`.

    A one-marker rule would stop the walk there and point the whole pipeline at
    a directory with no config, no assets and no golden file — so the second
    marker is load-bearing, not decoration.
    """
    monkeypatch.delenv("LAUNDER_REPO_ROOT", raising=False)
    forge_package = Path(launder_forge.__file__).parents[2]
    assert (forge_package / "pyproject.toml").is_file()
    assert find_repo_root(Path(launder_forge.__file__)) != forge_package
