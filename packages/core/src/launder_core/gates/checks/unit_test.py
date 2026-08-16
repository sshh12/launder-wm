"""Check 5 — TECH_PLAN.md §7.1 row 5, §7.6. L5 only.

**The unit test IS the meaning check for code.** That is why L5 drops
`llm_gate` (deviation from CONCEPT.md #2): a passing suite is a strictly
stronger statement about preserved meaning than any prose-naturalness judge
could make, and asking a prose judge about Python produces noise.

WHY THIS FILE DOES NOT CONTAIN A SANDBOX
----------------------------------------
It runs player-authored code. `launder_core` is imported by the API server, and
a `subprocess.run([sys.executable, ...])` here would look sandboxed to every
reviewer while being a remote-code-execution hole — the level config asks for
`memory_mb`, and there is no portable way to honour it (`RLIMIT_AS` is POSIX
only and absent on the Windows box this repo is developed on). A limit that
silently does not apply is worse than no limit, because it is documented.

So core declares `UnitTestRunner` and refuses to implement it. The runner is
`serve`'s to build, in whatever isolation the deployment actually has, and this
check raises `GateDependencyError` — loudly, at runtime, naming the missing
dependency — if it is asked to run without one. It never passes by default.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from launder_core.gates.checks.locked_phrase import resolve_ref
from launder_core.gates.registry import (
    META_COPY_KEY,
    GateContext,
    GateDataError,
    GateDependencyError,
    Phase,
    register,
)
from launder_core.schemas import CheckResult

__all__ = ["UnitTest"]


@register
class UnitTest:
    name: ClassVar[str] = "unit_test"
    phase: ClassVar[int] = Phase.EXECUTION
    fail_open: ClassVar[bool] = False
    config_params: ClassVar[frozenset[str]] = frozenset({"suite_ref", "timeout_ms", "memory_mb"})
    required_params: ClassVar[frozenset[str]] = config_params
    template_params: ClassVar[frozenset[str]] = frozenset({"test_name", "timeout_ms"})
    required_deps: ClassVar[frozenset[str]] = frozenset({"unit_tests"})
    copy_keys: ClassVar[frozenset[str]] = frozenset({"reject", "reject_timeout"})

    def __call__(self, ctx: GateContext, params: Mapping[str, Any]) -> CheckResult:
        runner = ctx.deps.unit_tests
        timeout_ms = int(params["timeout_ms"])
        if runner is None:
            raise GateDependencyError(
                "unit_test ran with no UnitTestRunner in Deps. This check is the ONLY meaning "
                "check on L5 (the level drops llm_gate), so passing it by default would clear "
                "any code that beats the detector, working or not. Wire a sandboxed runner "
                "into Deps.unit_tests."
            )
        suite_id = resolve_ref(str(params["suite_ref"]), ctx)
        if not suite_id:
            raise GateDataError(
                f"passage {ctx.passage.id} runs level {ctx.level.id}, which includes unit_test, "
                "but the passage declares no rules.unit_test_id. There is nothing to run and "
                "nothing to check meaning with."
            )

        outcome = runner.run(
            suite_id=str(suite_id),
            source=ctx.normalized,
            timeout_ms=timeout_ms,
            memory_mb=int(params["memory_mb"]),
        )
        rendered = {"test_name": outcome.failed_test, "timeout_ms": timeout_ms}
        if outcome.timed_out:
            return CheckResult(
                status="fail",
                check=self.name,
                code=f"{self.name}_timeout",
                params=rendered,
                meta={META_COPY_KEY: "reject_timeout"},
            )
        if not outcome.passed:
            return CheckResult(status="fail", check=self.name, code=self.name, params=rendered)
        return CheckResult(status="pass", check=self.name, params=rendered)
