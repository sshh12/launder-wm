"""`forge lint-copy` — the two §10.7 rules, run in CI and at boot.

> 1. Every check name in `REGISTRY` has a `[check.<name>]` block with all three
>    of `label`, `blurb`, `reject`. Adding a check to `levels.toml` without
>    writing its copy fails the build rather than shipping a constraint the
>    player cannot interrogate.
> 2. Every `{placeholder}` in a template resolves against that check's declared
>    `params` keys. A renamed param silently blanking a rejection message is
>    exactly the bug this catches.

Where "declared params" comes from, in priority order:

1. `REGISTRY[name].declared_params` — a class attribute on the check. This is
   the right source and the one to standardise on: the check that *emits*
   `CheckResult.params` is the only thing that knows their names.
2. `[check.<name>].params` in `copy.toml`, if a copy author declared them.
3. `DERIVED_PARAMS[name]` below, unioned with the param keys the check is given
   in `levels.toml`.

Source 3 is a fallback with a cost, and the linter says so in its output rather
than pretending it checked something it did not: a param invented by a check
and absent from the table would be reported as unresolvable, and a param
*removed* from a check would not be caught at all. Landing source 1 retires
this table.

Rule 1 is checked against `REGISTRY` when core's registry is importable and
against the union of `levels.toml`'s check names otherwise — a level cannot
reference a check that does not exist, so the union is a strict subset and the
weaker check is still sound in the direction that matters.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass, field
from typing import Any

from launder_forge.config import load_copy, load_toml
from launder_forge.paths import Paths

__all__ = ["DERIVED_PARAMS", "REQUIRED_KEYS", "CopyLintReport", "lint_copy"]

#: The three keys every check block must carry (§10.7 rule 1).
REQUIRED_KEYS: tuple[str, ...] = ("label", "blurb", "reject")


def _requested_keys(name: str) -> frozenset[str]:
    """Copy keys the CHECK can select, from `REGISTRY[name].copy_keys`.

    Rule 1 checked the three keys every block must have — i.e. the keys that
    EXIST, not the keys a check can ASK FOR. `close_paraphrase` emitted five
    distinct failure codes with copy for three of them, so a distance or
    alignment failure rendered the RETENTION message (quoting retention numbers
    that were fine) while the lint reported "OK 77 templates checked".
    """
    try:
        from launder_core import REGISTRY
        from launder_core.gates.registry import copy_keys_of

        check = REGISTRY.get(name)
        if check is not None:
            return copy_keys_of(check)
    except Exception:
        pass
    return frozenset({"reject"})


#: Params each check computes at runtime and puts in `CheckResult.params`.
#: Fallback source only — see the module docstring. Keys are check names.
DERIVED_PARAMS: dict[str, frozenset[str]] = {
    "unicode_sanitation": frozenset({"count", "categories", "sample"}),
    "word_floor": frozenset({"n_words", "min_words"}),
    "edit_budget": frozenset({"distance", "max_word_distance"}),
    "locked_phrase": frozenset({"phrase", "phrases"}),
    "close_paraphrase": frozenset(
        {
            "content_word_retention_pct",
            "min_content_word_retention_pct",
            "word_distance_ratio_pct",
            "max_word_distance_ratio_pct",
            "length_ratio_pct",
            "length_ratio_min_pct",
            "length_ratio_max_pct",
            "max_sentence_count_delta",
            "sentence_count_delta",
            "sentence_alignment_pct",
            "min_sentence_alignment_pct",
        }
    ),
    "unit_test": frozenset({"test_name", "timeout_ms", "failed", "passed"}),
    # `z_target_display` is `z_star + max_z`, the EFFECTIVE bar. See
    # detector_threshold.py: the rejection used to name z* while the test was
    # `z - z_star > max_z`.
    "detector_threshold": frozenset(
        {"z_display", "z_star_display", "z_target_display", "z", "z_star"}
    ),
    "llm_gate": frozenset({"claim_label", "added", "code", "kind"}),
}

#: Non-check sections and the context each is rendered against. `[primer]`,
#: `[gate]` and `[disclosure]` are static prose and take no placeholders.
SECTION_PARAMS: dict[str, frozenset[str]] = {
    # `level_n`/`level_count` belong here because the two share strings are
    # rendered in the same context as the readout they summarise: one names the
    # level just cleared, the other the whole campaign (`total` moves, spread
    # over `per_level`, plus the `url` to play it). They replace the old
    # `puzzle_number`, which was the daily's number and no longer exists
    # anywhere in the product — leaving it listed here would let a share string
    # that still interpolates it lint clean and then render "Launder #{}".
    "readout": frozenset(
        {
            "z_display",
            "z_star_display",
            "distance",
            "n_words",
            "level_n",
            "level_count",
            "total",
            "per_level",
            "url",
        }
    ),
    # `n`/`total` are the level rail's "3 / 15": the screen chrome names the
    # position, not the puzzle.
    "screen": frozenset({"distance", "max_word_distance", "n_words", "min_words", "n", "total"}),
    "errors": frozenset({"retry_after_s"}),
    "intro": frozenset({"n_words", "z_display"}),
    "gate": frozenset(),
    "primer": frozenset(),
    "disclosure": frozenset(),
}

_FORMATTER = string.Formatter()

#: Matches a `{name}` that `str.format` would substitute, ignoring `{{`/`}}`.
_PLACEHOLDER = re.compile(r"(?<!\{)\{([a-zA-Z_][a-zA-Z0-9_]*)(?:[^{}]*)\}")


@dataclass(slots=True)
class CopyLintReport:
    missing_blocks: list[str] = field(default_factory=list)
    missing_keys: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    orphan_blocks: list[str] = field(default_factory=list)
    checked_templates: int = 0
    param_source: str = "fallback-table"
    registry_source: str = "levels.toml"
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.missing_blocks or self.missing_keys or self.unresolved)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "checked_templates": self.checked_templates,
            "registry_source": self.registry_source,
            "param_source": self.param_source,
            "missing_blocks": self.missing_blocks,
            "missing_keys": self.missing_keys,
            "unresolved_placeholders": self.unresolved,
            "orphan_blocks": self.orphan_blocks,
            "notes": self.notes,
        }


def _placeholders(template: str) -> set[str]:
    """Names `str.format` would look up. Uses the stdlib parser, not a regex,
    so `{{literal}}` and `{value:>8}` behave exactly as they will at render."""
    names: set[str] = set()
    try:
        for _, field_name, _, _ in _FORMATTER.parse(template):
            if field_name:
                names.add(field_name.split(".")[0].split("[")[0])
    except ValueError:
        # An unbalanced brace is itself a copy bug; fall back to the regex so
        # the linter reports the placeholder rather than dying on the file.
        names.update(m.group(1) for m in _PLACEHOLDER.finditer(template))
    return names


def _registry_names(paths: Paths) -> tuple[set[str], str]:
    try:
        from launder_core import REGISTRY

        if REGISTRY:
            return set(REGISTRY), "launder_core.gates.registry.REGISTRY"
    except Exception:
        pass
    levels = load_toml(paths.levels_toml)
    names: set[str] = set(levels.get("defaults", {}))
    for level in levels.get("levels", []):
        for spec in level.get("checks", []):
            names.add(str(spec["check"]))
    return names, "data/config/levels.toml (REGISTRY not importable)"


def _declared_params(paths: Paths, name: str, block: dict[str, Any]) -> tuple[set[str], str]:
    try:
        from launder_core import REGISTRY

        check = REGISTRY.get(name)
        declared = getattr(check, "declared_params", None)
        if declared:
            return set(declared), "REGISTRY.declared_params"
    except Exception:
        pass
    if isinstance(block.get("params"), list):
        return {str(p) for p in block["params"]}, "copy.toml [check.*].params"

    names = set(DERIVED_PARAMS.get(name, frozenset()))
    levels = load_toml(paths.levels_toml)
    names.update(levels.get("defaults", {}).get(name, {}))
    for level in levels.get("levels", []):
        for spec in level.get("checks", []):
            if spec.get("check") == name:
                names.update(spec.get("params", {}))
    return names, "fallback-table"


def _walk_templates(block: Any, prefix: str) -> list[tuple[str, str]]:
    """Every string leaf under a copy block, with a dotted path for reporting."""
    out: list[tuple[str, str]] = []
    if isinstance(block, str):
        out.append((prefix, block))
    elif isinstance(block, dict):
        for k, v in block.items():
            out.extend(_walk_templates(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(block, list):
        for i, v in enumerate(block):
            out.extend(_walk_templates(v, f"{prefix}[{i}]"))
    return out


def lint_copy(paths: Paths, copy: dict[str, Any] | None = None) -> CopyLintReport:
    data = copy if copy is not None else load_copy(paths)
    report = CopyLintReport()
    checks = data.get("check", {})

    registry, source = _registry_names(paths)
    report.registry_source = source

    # --- rule 1 ------------------------------------------------------------
    for name in sorted(registry):
        block = checks.get(name)
        if not isinstance(block, dict):
            report.missing_blocks.append(name)
            continue
        # Rule 1: the three keys every block must carry, PLUS every key this
        # check can select at runtime via `meta[META_COPY_KEY]`.
        for key in sorted(set(REQUIRED_KEYS) | _requested_keys(name)):
            if not isinstance(block.get(key), str) or not block[key].strip():
                report.missing_keys.append(f"[check.{name}].{key}")
    for name in sorted(checks):
        if name not in registry:
            report.orphan_blocks.append(name)

    # --- rule 2 ------------------------------------------------------------
    sources: set[str] = set()
    for name, block in sorted(checks.items()):
        if not isinstance(block, dict):
            continue
        allowed, param_source = _declared_params(paths, name, block)
        sources.add(param_source)
        for path, template in _walk_templates(block, ""):
            if path == "params":
                continue
            report.checked_templates += 1
            for ph in sorted(_placeholders(template)):
                if ph not in allowed:
                    report.unresolved.append(
                        f"[check.{name}].{path}: {{{ph}}} is not a declared param "
                        f"(declared: {sorted(allowed)})"
                    )
    if sources:
        report.param_source = ", ".join(sorted(sources))

    for section, section_allowed in SECTION_PARAMS.items():
        block = data.get(section)
        if not isinstance(block, dict):
            continue
        for path, template in _walk_templates(block, ""):
            report.checked_templates += 1
            for ph in sorted(_placeholders(template)):
                if ph not in section_allowed:
                    report.unresolved.append(
                        f"[{section}].{path}: {{{ph}}} is not in this section's render context "
                        f"(declared: {sorted(section_allowed)})"
                    )

    if report.param_source == "fallback-table":
        report.notes.append(
            "declared params came from launder_forge.lintcopy.DERIVED_PARAMS, not from the "
            "checks themselves. Add a `declared_params` class attribute to each GateCheck and "
            "this table can be deleted — see the module docstring."
        )
    if report.orphan_blocks:
        report.notes.append(
            f"copy blocks with no registered check: {report.orphan_blocks}. Not an error (a "
            "check may land later), but dead copy is copy nobody maintains."
        )
    return report
