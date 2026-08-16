"""`forge triage` — apply a level policy to a candidate run.

TECH_PLAN.md §6.5, including the warning that matters more than the code:

> **Every threshold in that file is a hypothesis nobody has measured.** The
> mitigation is structural: triage is a config sweep, not a code change, and
> the first 400-candidate run is a calibration experiment whose output is the
> thresholds, not the passages.

So this module contains no thresholds. It contains an evaluator for the three
constraint spellings that appear in `data/config/triage/L*.toml` and a tiny
sandboxed expression language for `[reject_reasons]`, whose entries are
*named* rejections — the name is what shows up in the report, which is what
makes a run's yield diagnosable instead of merely low.

Reject reasons that are prose rather than an expression (L5's
`original_fails_tests = "the candidate's own unit test suite does not pass"`)
are carried through as documentation and reported as non-evaluable rather than
silently ignored or crashed on.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from launder_forge.config import TriagePolicy

__all__ = [
    "CandidateRecord",
    "TriageOutcome",
    "TriageReport",
    "apply_policy",
    "evaluate_expression",
    "load_candidates",
    "triage_run",
]


# ---------------------------------------------------------------------------
# a very small, very restricted expression evaluator
# ---------------------------------------------------------------------------

_ALLOWED_NODES = (
    ast.Expression,
    ast.BoolOp,
    ast.UnaryOp,
    ast.BinOp,
    ast.Compare,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.And,
    ast.Or,
    ast.Not,
    ast.USub,
    ast.UAdd,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.Is,
    ast.IsNot,
)


class NotAnExpression(ValueError):
    """Raised when a `[reject_reasons]` entry is prose, not a predicate."""


def evaluate_expression(expr: str, variables: dict[str, Any]) -> bool:
    """Evaluate a `[reject_reasons]` predicate against a candidate's metrics.

    `null` is accepted as a spelling of `None` because the policies are TOML
    written by someone thinking in JSON. Anything outside the allow-list — a
    call, an attribute, a subscript, an import — raises rather than evaluating,
    because these strings come from a config file and config files get edited.
    """
    src = expr.replace("null", "None")
    try:
        tree = ast.parse(src, mode="eval")
    except SyntaxError as exc:
        raise NotAnExpression(f"{expr!r} is not a predicate: {exc.msg}") from exc
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise NotAnExpression(
                f"{expr!r} uses {type(node).__name__}, which is not allowed in a triage "
                "predicate. Allowed: names, literals, and/or/not, + - * /, comparisons."
            )
        if isinstance(node, ast.Name) and node.id not in variables and node.id not in {"None"}:
            raise NotAnExpression(
                f"{expr!r} refers to {node.id!r}, which is not a metric on this candidate. "
                f"Known metrics: {sorted(variables)}"
            )
    return bool(eval(compile(tree, "<triage>", "eval"), {"__builtins__": {}}, dict(variables)))


# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CandidateRecord:
    """One line of `data/candidates/<run_id>.jsonl`."""

    candidate_id: str
    text: str
    token_ids: list[int]
    metrics: dict[str, Any]
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, rec: dict[str, Any]) -> CandidateRecord:
        metrics: dict[str, Any] = {}
        for block in ("difficulty", "solver"):
            value = rec.get(block)
            if isinstance(value, dict):
                metrics.update(value)
        for key in ("n_words", "z_total", "margin_z"):
            if key in rec:
                metrics.setdefault(key, rec[key])
        return cls(
            candidate_id=str(rec.get("candidate_id") or rec.get("id") or "?"),
            text=str(rec.get("text", "")),
            token_ids=[int(x) for x in rec.get("token_ids", [])],
            metrics=metrics,
            raw=rec,
        )


@dataclass(slots=True)
class TriageOutcome:
    candidate_id: str
    accepted: bool
    reject_reasons: list[str] = field(default_factory=list)
    failed_constraints: list[str] = field(default_factory=list)
    derived: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TriageReport:
    level_id: str
    policy_path: str
    total: int = 0
    accepted: int = 0
    outcomes: list[TriageOutcome] = field(default_factory=list)
    constraint_failures: dict[str, int] = field(default_factory=dict)
    reason_counts: dict[str, int] = field(default_factory=dict)
    non_evaluable: dict[str, str] = field(default_factory=dict)

    @property
    def yield_pct(self) -> float:
        return 0.0 if self.total == 0 else 100.0 * self.accepted / self.total

    def as_dict(self) -> dict[str, Any]:
        return {
            "level_id": self.level_id,
            "policy": self.policy_path,
            "total": self.total,
            "accepted": self.accepted,
            "yield_pct": round(self.yield_pct, 3),
            "constraint_failures": self.constraint_failures,
            "reason_counts": self.reason_counts,
            "non_evaluable_reject_reasons": self.non_evaluable,
            "outcomes": [
                {
                    "candidate_id": o.candidate_id,
                    "accepted": o.accepted,
                    "reject_reasons": o.reject_reasons,
                    "failed_constraints": o.failed_constraints,
                    "derived": o.derived,
                }
                for o in self.outcomes
            ],
        }


#: Keys that may appear in `[accept]` without being a metric constraint.
_NON_METRIC_ACCEPT_KEYS = {"calibration_bucket"}


def _check_constraint(key: str, spec: Any, metrics: dict[str, Any]) -> tuple[bool, str | None]:
    """Return `(ok, failure_label)` for one `[accept]` entry."""
    if key in _NON_METRIC_ACCEPT_KEYS:
        return True, None
    if isinstance(spec, dict):
        if set(spec) <= {"min", "max"}:
            value = metrics.get(key)
            if value is None:
                return False, f"{key} is missing"
            if "min" in spec and value < spec["min"]:
                return False, f"{key} = {value} < min {spec['min']}"
            if "max" in spec and value > spec["max"]:
                return False, f"{key} = {value} > max {spec['max']}"
            return True, None
        return True, None  # a nested sub-table (e.g. [accept.unit_test]); not a metric
    if key.endswith("_min"):
        metric = key[:-4]
        value = metrics.get(metric)
        if value is None:
            return False, f"{metric} is missing"
        return (value >= spec, None if value >= spec else f"{metric} = {value} < {spec}")
    if key.endswith("_max"):
        metric = key[:-4]
        value = metrics.get(metric)
        if value is None:
            return False, f"{metric} is missing"
        return (value <= spec, None if value <= spec else f"{metric} = {value} > {spec}")
    value = metrics.get(key)
    if value is None:
        return False, f"{key} is missing"
    return (value == spec, None if value == spec else f"{key} = {value!r}, want {spec!r}")


def apply_policy(candidate: CandidateRecord, policy: TriagePolicy) -> TriageOutcome:
    outcome = TriageOutcome(candidate_id=candidate.candidate_id, accepted=True)
    metrics = dict(candidate.metrics)

    for key, spec in policy.accept.items():
        ok, label = _check_constraint(key, spec, metrics)
        if not ok:
            outcome.accepted = False
            outcome.failed_constraints.append(label or key)

    for name, expr in policy.reject_reasons.items():
        try:
            if evaluate_expression(expr, metrics):
                outcome.accepted = False
                outcome.reject_reasons.append(name)
        except NotAnExpression as exc:
            outcome.notes.append(f"{name}: {exc}")

    for target, expr in policy.derive.items():
        if expr.strip() in {"null", "None"}:
            outcome.derived[target] = None
            continue
        try:
            value = eval(
                compile(ast.parse(expr, mode="eval"), "<derive>", "eval"),
                {"__builtins__": {}},
                metrics,
            )
            outcome.derived[target] = value
        except Exception as exc:
            outcome.notes.append(f"derive {target}: {type(exc).__name__}: {exc}")
    return outcome


def load_candidates(path: Path) -> list[CandidateRecord]:
    out: list[CandidateRecord] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(CandidateRecord.from_json(json.loads(line)))
    return out


def triage_run(candidates: list[CandidateRecord], policy: TriagePolicy) -> TriageReport:
    report = TriageReport(level_id=policy.level_id, policy_path=str(policy.path))
    for cand in candidates:
        outcome = apply_policy(cand, policy)
        report.total += 1
        report.accepted += int(outcome.accepted)
        report.outcomes.append(outcome)
        for label in outcome.failed_constraints:
            metric = label.split(" ")[0]
            report.constraint_failures[metric] = report.constraint_failures.get(metric, 0) + 1
        for name in outcome.reject_reasons:
            report.reason_counts[name] = report.reason_counts.get(name, 0) + 1
        for note in outcome.notes:
            key, _, msg = note.partition(":")
            report.non_evaluable.setdefault(key.strip(), msg.strip())
    return report
