"""Loaders for `data/config/*.toml`.

Everything the forge reads from disk lands in a pydantic model owned by
`launder_core.schemas` wherever one exists (`SynthIDConfig`, `LevelsFile`,
`ScoringConfig`). Forge only adds the authoring-time shapes core has no reason
to know about: the triage accept policy and the copy file.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from launder_core.schemas import LevelsFile, ScoringConfig, SynthIDConfig
from launder_forge.paths import Paths, repo_paths

__all__ = [
    "PromptPack",
    "TriagePolicy",
    "load_copy",
    "load_levels_file",
    "load_prompt_pack",
    "load_scoring_config",
    "load_toml",
    "load_triage_policy",
    "load_watermark_config",
    "watermark_asset_digests",
]


def load_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing config file: {path}")
    with path.open("rb") as fh:
        return tomllib.load(fh)


# ---------------------------------------------------------------------------
# watermark.toml
# ---------------------------------------------------------------------------

#: Keys in watermark.toml that are documentation/provenance rather than part of
#: the six fields that enter `wm_config_id`.
_WM_NON_CONFIG = {
    "schema",
    "wm_config_id",
    "scheme",
    "hash_iv",
    "lcg_multiplier",
    "lcg_increment",
    "table",
    "scoring",
    "seeds",
    "model",
}


def load_watermark_config(paths: Paths | None = None) -> SynthIDConfig:
    """Build the frozen `SynthIDConfig` and assert the recorded id matches.

    The recorded `wm_config_id` is not trusted: it is *checked*. A hand-edited
    key list with a stale id is exactly the silent failure §4.5 exists to stop.
    """
    p = paths or repo_paths()
    raw = load_toml(p.watermark_toml)
    fields = {k: v for k, v in raw.items() if k not in _WM_NON_CONFIG}
    cfg = SynthIDConfig(**fields)
    recorded = raw.get("wm_config_id")
    if recorded is not None:
        cfg.assert_id(str(recorded))
    scheme = raw.get("scheme")
    if scheme != "sampling_table":
        raise ValueError(
            f"watermark.toml declares scheme={scheme!r}; this project is the sampling-table "
            "variant permanently (TECH_PLAN.md §4.1). The iterated-LCG variant is "
            "bit-incompatible and cannot detect anything HF generates."
        )
    return cfg


def watermark_asset_digests(paths: Paths | None = None) -> dict[str, Any]:
    """The `[table]` block: the four recorded digests of the sampling table."""
    p = paths or repo_paths()
    raw = load_toml(p.watermark_toml)
    table = raw.get("table")
    if not isinstance(table, dict):
        raise ValueError(f"{p.watermark_toml} has no [table] block")
    return table


def watermark_seeds(paths: Paths | None = None) -> dict[str, int]:
    p = paths or repo_paths()
    raw = load_toml(p.watermark_toml)
    seeds = raw.get("seeds", {})
    return {str(k): int(v) for k, v in seeds.items()}


def watermark_model_block(paths: Paths | None = None) -> dict[str, Any]:
    p = paths or repo_paths()
    raw = load_toml(p.watermark_toml)
    block = raw.get("model", {})
    return dict(block) if isinstance(block, dict) else {}


# ---------------------------------------------------------------------------
# levels.toml / scoring.toml / copy.toml
# ---------------------------------------------------------------------------


def load_levels_file(paths: Paths | None = None) -> LevelsFile:
    p = paths or repo_paths()
    return LevelsFile(**load_toml(p.levels_toml))


def load_scoring_config(paths: Paths | None = None) -> ScoringConfig:
    p = paths or repo_paths()
    return ScoringConfig(**load_toml(p.scoring_toml))


def load_copy(paths: Paths | None = None) -> dict[str, Any]:
    p = paths or repo_paths()
    return load_toml(p.copy_toml)


# ---------------------------------------------------------------------------
# triage/L*.toml — authoring-time only, so it lives here and not in core
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TriagePolicy:
    """One `data/config/triage/L<n>.toml` (TECH_PLAN.md §6.5).

    `accept` is a map from a difficulty-metric name to a constraint. Three
    constraint spellings are supported, all of which appear in the shipped L2
    policy:

    * ``<metric>_min = x``      -> metric >= x
    * ``<metric>_max = x``      -> metric <= x
    * ``<metric> = {min=a, max=b}`` -> a <= metric <= b
    * ``<metric> = true/false`` -> metric is exactly that boolean

    `reject_reasons` maps a reason name to a human-readable *expression string*
    evaluated against the candidate's metrics. Keeping it a string in TOML is
    the point: a new reject reason is a config edit, not a code change.
    """

    level_id: str
    path: Path
    accept: dict[str, Any]
    reject_reasons: dict[str, str]
    derive: dict[str, str]
    raw: dict[str, Any]

    @property
    def calibration_bucket(self) -> str:
        return str(self.raw.get("calibration_bucket", "prose"))


def load_triage_policy(path: Path) -> TriagePolicy:
    raw = load_toml(path)
    level_id = str(raw.get("level_id") or path.stem)
    return TriagePolicy(
        level_id=level_id,
        path=path,
        accept=dict(raw.get("accept", {})),
        reject_reasons={str(k): str(v) for k, v in raw.get("reject_reasons", {}).items()},
        derive={str(k): str(v) for k, v in raw.get("derive", {}).items()},
        raw=raw,
    )


# ---------------------------------------------------------------------------
# data/config/prompts/*.yaml — §6.2, §12 row 21
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PromptPack:
    """One `data/config/prompts/<id>.yaml`: system, templates, slots, gen_params.

    §12 row 21 lists generation prompts as tunable "without a code change", and
    they were not: `forge gen` built them from a hardcoded f-string over a
    module-level `_NULL_TOPICS` tuple, `Paths.prompts_dir` was defined and never
    called, and the directory did not exist. This is that config, loaded.

    `render(n, start)` is ROUND-ROBIN over templates x slot values rather than
    random, so a run covers every pair evenly and per-template yield numbers are
    comparable. It is also a pure function of `(start, i)`, which keeps a
    resumed or re-sharded run reproducible.
    """

    id: str
    path: Path
    system: str
    templates: tuple[str, ...]
    slots: dict[str, tuple[str, ...]]
    gen_params: dict[str, Any]

    def render(self, n: int, start: int = 0) -> list[str]:
        out: list[str] = []
        for i in range(start, start + n):
            template = self.templates[i % len(self.templates)]
            values = {
                name: options[(i // max(1, len(self.templates))) % len(options)]
                for name, options in self.slots.items()
                if options
            }
            try:
                out.append(template.format(**values))
            except KeyError as exc:
                raise ValueError(
                    f"{self.path}: template {template!r} uses slot {exc.args[0]!r}, which the "
                    f"`slots` block does not define (it has {sorted(self.slots)})."
                ) from exc
        return out


def load_prompt_pack(paths: Paths | None = None, name: str = "high_entropy") -> PromptPack:
    """Read `data/config/prompts/<name>.yaml`. Raises naming the file and the fix."""
    import yaml

    resolved = paths or repo_paths()
    path = resolved.prompts_dir / f"{name}.yaml"
    if not path.is_file():
        available = (
            ", ".join(sorted(p.stem for p in resolved.prompts_dir.glob("*.yaml")))
            if resolved.prompts_dir.is_dir()
            else "(the directory does not exist)"
        )
        raise FileNotFoundError(
            f"{path} does not exist. `forge gen --prompts <name>` reads a prompt pack from "
            f"data/config/prompts/. Available: {available}"
        )
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    schema = str(raw.get("schema", ""))
    if schema and schema != "launder.prompts/1":
        raise ValueError(f"{path} declares schema {schema!r}, expected 'launder.prompts/1'")
    templates = tuple(str(t) for t in raw.get("templates", ()))
    if not templates:
        raise ValueError(f"{path} declares no `templates`; there is nothing to generate from.")
    slots = {str(k): tuple(str(x) for x in (v or ())) for k, v in (raw.get("slots") or {}).items()}
    return PromptPack(
        id=str(raw.get("id", name)),
        path=path,
        system=str(raw.get("system", "")),
        templates=templates,
        slots=slots,
        gen_params=dict(raw.get("gen_params") or {}),
    )
