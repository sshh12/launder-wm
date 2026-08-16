"""`forge tune` — sweep triage parameters, print yield x par tables.

TECH_PLAN.md §6.5:

    uv run forge tune --policy data/config/triage/L2.toml \
                      --sweep margin_z_min=4,6,8 upstream_leverage_min=1.0,1.25,1.5 \
                      --corpus data/candidates/*.jsonl

The point is not the numbers; it is that changing a threshold is a config
sweep rather than a code change. Every cell of the output is a full re-run of
`triage` over the same candidate corpus with one `[accept]` key overridden, so
a cell can never disagree with what `forge triage` would do with that file.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field, replace
from typing import Any

from launder_forge.config import TriagePolicy
from launder_forge.triage import CandidateRecord, triage_run

__all__ = ["SweepCell", "SweepResult", "parse_sweep", "sweep"]


def parse_sweep(specs: list[str]) -> dict[str, list[Any]]:
    """`["margin_z_min=4,6,8", "hot_runs_min=1,2"]` -> `{key: [values]}`.

    Values are parsed as int, then float, then left as strings; `true`/`false`
    become booleans so a boolean gate can be swept too.
    """
    out: dict[str, list[Any]] = {}
    for spec in specs:
        key, _, raw = spec.partition("=")
        if not raw:
            raise ValueError(f"--sweep entry {spec!r} must be key=v1,v2,v3")
        values: list[Any] = []
        for token in raw.split(","):
            token = token.strip()
            low = token.lower()
            if low in {"true", "false"}:
                values.append(low == "true")
                continue
            try:
                values.append(int(token))
                continue
            except ValueError:
                pass
            try:
                values.append(float(token))
            except ValueError:
                values.append(token)
        out[key.strip()] = values
    return out


@dataclass(slots=True)
class SweepCell:
    overrides: dict[str, Any]
    total: int
    accepted: int
    yield_pct: float
    par_histogram: dict[str, int] = field(default_factory=dict)
    median_par: float | None = None
    top_constraint: str | None = None


@dataclass(slots=True)
class SweepResult:
    policy_path: str
    keys: list[str]
    cells: list[SweepCell] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy_path,
            "swept_keys": self.keys,
            "cells": [
                {
                    "overrides": c.overrides,
                    "total": c.total,
                    "accepted": c.accepted,
                    "yield_pct": round(c.yield_pct, 3),
                    "median_par": c.median_par,
                    "par_histogram": c.par_histogram,
                    "top_constraint": c.top_constraint,
                }
                for c in self.cells
            ],
        }


def sweep(
    candidates: list[CandidateRecord], policy: TriagePolicy, grid: dict[str, list[Any]]
) -> SweepResult:
    keys = list(grid)
    result = SweepResult(policy_path=str(policy.path), keys=keys)
    for combo in itertools.product(*(grid[k] for k in keys)):
        overrides = dict(zip(keys, combo, strict=True))
        accept = dict(policy.accept)
        accept.update(overrides)
        variant = replace(policy, accept=accept)
        report = triage_run(candidates, variant)

        pars = [
            int(o.derived.get("par_upper") or c.metrics.get("par_upper") or 0)
            for o, c in zip(report.outcomes, candidates, strict=True)
            if o.accepted and (o.derived.get("par_upper") or c.metrics.get("par_upper"))
        ]
        histogram: dict[str, int] = {}
        for p in pars:
            histogram[str(p)] = histogram.get(str(p), 0) + 1
        median = None
        if pars:
            ordered = sorted(pars)
            median = float(ordered[len(ordered) // 2])
        top = None
        if report.constraint_failures:
            top = max(report.constraint_failures.items(), key=lambda kv: kv[1])[0]
        result.cells.append(
            SweepCell(
                overrides=overrides,
                total=report.total,
                accepted=report.accepted,
                yield_pct=report.yield_pct,
                par_histogram=histogram,
                median_par=median,
                top_constraint=top,
            )
        )
    return result
