"""Repo-root discovery and the canonical `data/` layout.

Every forge command takes its paths from here, so a command run from any
working directory reads and writes the same files, and so the tests can point
the whole pipeline at a temporary tree with one object.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

__all__ = ["Paths", "find_repo_root", "repo_paths"]

#: Files that together identify the repository root. TECH_PLAN.md and the root
#: pyproject both live there and neither exists anywhere else in the tree.
_ROOT_MARKERS: Final[tuple[str, ...]] = ("TECH_PLAN.md", "pyproject.toml")


def find_repo_root(start: Path | None = None) -> Path:
    """Walk up from `start` (default: this file) until every marker is present.

    Also honours `LAUNDER_REPO_ROOT`, which is how the tests and the Makefile
    point forge at a scratch tree.
    """
    env = os.environ.get("LAUNDER_REPO_ROOT")
    if env:
        root = Path(env).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"LAUNDER_REPO_ROOT={env!r} is not a directory")
        return root
    here = (start or Path(__file__)).resolve()
    for candidate in (here, *here.parents):
        if candidate.is_dir() and all((candidate / m).exists() for m in _ROOT_MARKERS):
            return candidate
    raise FileNotFoundError(
        "could not find the repository root: walked up from "
        f"{here} looking for a directory containing {list(_ROOT_MARKERS)}. "
        "Run forge from inside the checkout, or set LAUNDER_REPO_ROOT."
    )


@dataclass(frozen=True, slots=True)
class Paths:
    """The `data/` layout of TECH_PLAN.md §3, as resolved paths."""

    root: Path

    # --- inputs that are gitignored build material -------------------------
    @property
    def build(self) -> Path:
        return self.root / "data" / "build"

    @property
    def tokenizer_json(self) -> Path:
        return self.build / "tokenizer.json"

    @property
    def tokenizer_config_json(self) -> Path:
        return self.build / "tokenizer_config.json"

    # --- shipped assets ----------------------------------------------------
    @property
    def assets(self) -> Path:
        return self.root / "data" / "assets"

    @property
    def sampling_table(self) -> Path:
        return self.assets / "sampling_table.v1.bin"

    @property
    def tokenizer_blob(self) -> Path:
        return self.assets / "gemma3-tok.v1.bin.br"

    @property
    def wm_config_asset(self) -> Path:
        return self.assets / "wm_config.v1.json"

    @property
    def thresholds(self) -> Path:
        return self.assets / "thresholds.v1.json"

    # --- config ------------------------------------------------------------
    @property
    def config(self) -> Path:
        return self.root / "data" / "config"

    @property
    def watermark_toml(self) -> Path:
        return self.config / "watermark.toml"

    @property
    def levels_toml(self) -> Path:
        return self.config / "levels.toml"

    @property
    def scoring_toml(self) -> Path:
        return self.config / "scoring.toml"

    @property
    def copy_toml(self) -> Path:
        return self.config / "copy.toml"

    @property
    def judge_toml(self) -> Path:
        return self.config / "judge.toml"

    @property
    def schedule_toml(self) -> Path:
        return self.config / "schedule.toml"

    @property
    def triage_dir(self) -> Path:
        return self.config / "triage"

    @property
    def prompts_dir(self) -> Path:
        return self.config / "prompts"

    # --- pipeline outputs --------------------------------------------------
    @property
    def candidates(self) -> Path:
        return self.root / "data" / "candidates"

    @property
    def passages(self) -> Path:
        return self.root / "data" / "passages"

    @property
    def regen(self) -> Path:
        return self.root / "data" / "regen"

    @property
    def runs(self) -> Path:
        return self.root / "data" / "runs"

    @property
    def reports(self) -> Path:
        return self.root / "data" / "reports"

    @property
    def golden(self) -> Path:
        return self.root / "data" / "golden"

    @property
    def vectors_json(self) -> Path:
        return self.golden / "vectors.json"

    @property
    def tokenizer_golden(self) -> Path:
        return self.golden / "tokenizer_golden.json"

    @property
    def checksum(self) -> Path:
        return self.golden / "CHECKSUM"

    @property
    def manifest(self) -> Path:
        return self.root / "data" / "MANIFEST.json"

    @property
    def web(self) -> Path:
        return self.root / "web"

    def rel(self, path: Path) -> str:
        """Repo-relative POSIX spelling — the form used inside MANIFEST.json."""
        return path.resolve().relative_to(self.root).as_posix()

    def ensure(self, path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        return path


def repo_paths(start: Path | None = None) -> Paths:
    return Paths(find_repo_root(start))
