"""Rendering `(code, params)` against `data/config/copy.toml` — §7.4, §10.7.

**THIS MODULE HOLDS NO PLAYER-FACING STRINGS.** Not one. Every word a player
reads lives in `data/config/copy.toml`, because rejection wording, check labels,
check blurbs and the primer are the surface most worth tuning from playtesting
and none of it should require a code change. §10.7 supersedes the `feedback.py`
template table implied by §12 row 13 for exactly that reason. The only strings
in this file are structural keys (`label`, `blurb`, `reject`) and error text for
developers.

Labels vs. blurbs — the distinction this module ENFORCES
--------------------------------------------------------
* A **label** is templated from a level's params. `"Budget {max_word_distance}
  words"` renders per level, which is why the checklist can state the actual
  rule the player is under.
* A **blurb** is static per check TYPE. It explains the KIND of constraint, so
  it must not embed a level's numbers: "you may change only this many words"
  reads correctly at every budget, while a blurb hardcoding `12` would silently
  lie on the level tuned to 6. `lint_copy` fails the build if a blurb contains
  any placeholder at all.
* A **reject** is templated from `CheckResult.params` at submit time.

The two lint rules (§10.7), run in CI, at boot and by `forge lint-copy`:

1. Every check name in `REGISTRY` has a `[check.<name>]` block with all three
   of `label`, `blurb`, `reject`.
2. Every `{placeholder}` in a template resolves against that check's declared
   params. A renamed param silently blanking a rejection message is exactly the
   bug this catches.
"""

from __future__ import annotations

import os
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

from launder_core.gates.registry import META_COPY_KEY, META_COPY_TABLE, REGISTRY, GateCheck
from launder_core.schemas import CheckResult, GateFailure

__all__ = [
    "CopyBook",
    "CopyError",
    "assert_copy_complete",
    "data_dir",
    "filter_notes",
    "lint_copy",
    "load_copy",
    "placeholders",
    "render_feedback",
]

_REQUIRED_KEYS: Final[tuple[str, ...]] = ("label", "blurb", "reject")
_PLACEHOLDER: Final[re.Pattern[str]] = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


class CopyError(RuntimeError):
    """copy.toml and the registry disagree. Raised at BOOT, never at play."""


# ---------------------------------------------------------------------------
# Where `data/` is
# ---------------------------------------------------------------------------
def data_dir() -> Path:
    """The repo's `data/` directory. THE one place this is resolved.

    `LAUNDER_DATA_DIR` wins if set — the Docker image copies `data/` to a
    different prefix than the source tree. Otherwise walk up from this file
    looking for a directory that actually contains the config, rather than
    counting `parents[n]`, which breaks the moment the package moves.
    """
    override = os.environ.get("LAUNDER_DATA_DIR")
    if override:
        candidate = Path(override)
        if not (candidate / "config").is_dir():
            raise CopyError(f"LAUNDER_DATA_DIR={override!r} has no config/ directory in it")
        return candidate
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "data"
        if (candidate / "config" / "copy.toml").is_file():
            return candidate
    raise CopyError(
        "could not locate the repo's data/ directory from "
        f"{here}. Set LAUNDER_DATA_DIR to point at it."
    )


def placeholders(template: str) -> frozenset[str]:
    """The `{name}` keys a template interpolates."""
    return frozenset(_PLACEHOLDER.findall(template))


@dataclass(frozen=True)
class CopyBook:
    """Parsed `copy.toml`. Immutable, cached, and the only source of copy."""

    raw: Mapping[str, Any]
    path: Path

    def block(self, check: str) -> Mapping[str, Any]:
        blocks: Mapping[str, Any] = self.raw.get("check", {})
        block: Mapping[str, Any] | None = blocks.get(check)
        if block is None:
            raise CopyError(
                f"copy.toml ({self.path}) has no [check.{check}] block. Every registered check "
                "needs one — a constraint the player cannot interrogate is a constraint they "
                "will assume is arbitrary (lint rule 1, TECH_PLAN.md §10.7)."
            )
        return block

    def label(self, check: str, params: Mapping[str, Any]) -> str:
        """Templated from the LEVEL's params (the checklist, before submit)."""
        return self._format(check, "label", self.block(check)["label"], params)

    def blurb(self, check: str) -> str:
        """Static per check type. Never templated — see the module docstring."""
        return str(self.block(check)["blurb"])

    def reject(self, result: CheckResult) -> str:
        """Templated from `CheckResult.params` (after submit).

        Which template: `meta["copy.table"]` names a sub-table and
        `meta["copy.key"]` names the entry, both set by the check. That is how
        `llm_gate` picks between five unnatural-prose lines without this module
        knowing that any of them exist. Missing variants fall back to `reject`,
        so a check can emit a more specific code before its copy is written.
        """
        block = self.block(result.check)
        table: Mapping[str, Any] = block
        table_name = result.meta.get(META_COPY_TABLE)
        if table_name is not None:
            sub = block.get(str(table_name))
            if isinstance(sub, Mapping):
                table = sub
        key = str(result.meta.get(META_COPY_KEY, "reject"))
        template = table.get(key)
        if template is None:
            template = block["reject"]
            key = "reject"
        return self._format(result.check, key, str(template), result.params)

    def _format(self, check: str, key: str, template: str, params: Mapping[str, Any]) -> str:
        missing = placeholders(template) - set(params)
        if missing:
            raise CopyError(
                f"copy.toml [check.{check}].{key} interpolates {sorted(missing)}, which the "
                f"check did not supply (it supplied {sorted(params)}). A rejection message "
                "with a blank in it is the bug lint rule 2 exists to prevent — either the "
                "param was renamed or the template was."
            )
        return template.format(**params)


@lru_cache(maxsize=4)
def load_copy(path: Path | None = None) -> CopyBook:
    resolved = path if path is not None else data_dir() / "config" / "copy.toml"
    with resolved.open("rb") as fh:
        raw = tomllib.load(fh)
    return CopyBook(raw=raw, path=resolved)


# ---------------------------------------------------------------------------
# The two lint rules
# ---------------------------------------------------------------------------
def lint_copy(
    book: CopyBook | None = None, registry: Mapping[str, GateCheck] | None = None
) -> list[str]:
    """Return every violation as a human-readable line. Empty means clean."""
    book = book if book is not None else load_copy()
    registry = registry if registry is not None else REGISTRY
    problems: list[str] = []
    blocks = book.raw.get("check", {})

    for name in sorted(registry):
        check = registry[name]
        block = blocks.get(name)
        if block is None:
            problems.append(
                f"rule 1: [check.{name}] is missing from copy.toml. Adding a check to "
                "levels.toml without writing its copy must fail the build."
            )
            continue
        for key in _REQUIRED_KEYS:
            if key not in block:
                problems.append(f"rule 1: [check.{name}] has no `{key}`")

        # Rule 2, split three ways because the three template kinds draw their
        # values from three different places (see the module docstring).
        label = block.get("label")
        if isinstance(label, str):
            unknown = placeholders(label) - check.config_params
            if unknown:
                problems.append(
                    f"rule 2: [check.{name}].label interpolates {sorted(unknown)}, which are "
                    f"not level params of {name} ({sorted(check.config_params)}). A label is "
                    "rendered from levels.toml before the player submits anything."
                )
        blurb = block.get("blurb")
        if isinstance(blurb, str) and placeholders(blurb):
            problems.append(
                f"rule 2: [check.{name}].blurb interpolates "
                f"{sorted(placeholders(blurb))}. A blurb explains the KIND of constraint and "
                "is shown at every level, so a number in it lies on every level but one."
            )
        for key, template in _walk_templates(block):
            if key in ("label", "blurb"):
                continue
            unknown = placeholders(template) - check.template_params
            if unknown:
                problems.append(
                    f"rule 2: [check.{name}].{key} interpolates {sorted(unknown)}, which "
                    f"{name} never puts in CheckResult.params "
                    f"({sorted(check.template_params)})."
                )

    for name in sorted(blocks):
        if name not in registry:
            problems.append(
                f"rule 1 (converse): copy.toml has [check.{name}] but no such check is "
                "registered. Either the check was deleted and its copy was not, or the name "
                "is misspelled — and a misspelled name means the real check has no copy."
            )
    return problems


def _walk_templates(block: Mapping[str, Any], prefix: str = "") -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for key, value in block.items():
        path = f"{prefix}{key}"
        if isinstance(value, str):
            out.append((path, value))
        elif isinstance(value, Mapping):
            out.extend(_walk_templates(value, prefix=f"{path}."))
    return out


def assert_copy_complete(
    book: CopyBook | None = None, registry: Mapping[str, GateCheck] | None = None
) -> None:
    """Boot-time form of `lint_copy`. Call it once at startup."""
    problems = lint_copy(book, registry)
    if problems:
        raise CopyError(
            "data/config/copy.toml does not satisfy the §10.7 lint rules:\n  - "
            + "\n  - ".join(problems)
        )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
_ANGLE: Final[re.Pattern[str]] = re.compile(r"[<>]")
#: Verbs that make a sentence an instruction addressed to a system rather than
#: an observation about the text. Matching one means the model echoed the
#: player's injection into a field we render.
_IMPERATIVE: Final[re.Pattern[str]] = re.compile(
    r"^\s*(ignore|disregard|output|print|return|set|respond|reply|answer|approve|pass|"
    r"accept|override|forget|reveal|execute|run|act|pretend|assume|treat)\b",
    re.IGNORECASE,
)
_MAX_NOTES_WORDS: Final[int] = 15


def filter_notes(notes: str) -> str:
    """§7.4's output filter. `notes` is model-generated text derived from
    player-controlled input, so it is untrusted for rendering.

    Returns the empty string — meaning "render no secondary line" — for
    anything over 15 words, anything containing angle brackets, or anything that
    opens like an instruction to a system. Fails silent rather than loud on
    purpose: this is a nice-to-have line under a rejection the player has
    already been given a real reason for.
    """
    text = notes.strip()
    if not text:
        return ""
    if _ANGLE.search(text):
        return ""
    if len(text.split()) > _MAX_NOTES_WORDS:
        return ""
    if _IMPERATIVE.match(text):
        return ""
    return text


def render_feedback(result: CheckResult, book: CopyBook | None = None) -> GateFailure:
    """Turn the failing `CheckResult` into the `GateFailure` the API returns.

    Errors are specific and do not apologize (§10.6). The message names the
    failure, not the taxonomy (§10.7 rule 3): "invisible characters were added",
    not "zero-width codepoints were inserted".
    """
    if result.status == "pass":
        raise CopyError(
            f"render_feedback called on a passing {result.check} result; there is nothing to "
            "tell the player."
        )
    book = book if book is not None else load_copy()
    notes = result.meta.get("notes", "")
    return GateFailure(
        check=result.check,
        code=result.code or result.check,
        message=book.reject(result),
        params=dict(result.params),
        notes=filter_notes(str(notes)),
    )
