"""The judge's provider contract. **Core owns the types; serve owns the wires.**

`Observation`, `ClaimObservation`, `JudgeProvider` and `JudgeUnavailable` all
live in `launder_core.gates.registry`: core's `llm_gate` check calls
`deps.judge.observe(passage, normalized, nonce)` and computes the verdict from
what comes back (§7.3, §7.4). Re-declaring any of them here would give the
project two definitions of the thing the whole gate turns on.

What serve adds is everything core deliberately refuses to know about — HTTP,
SDKs, API keys, retries, failover, token accounting:

* `JudgeError` is the `JudgeUnavailable` core catches, carrying whether the
  failure is worth a retry. §7.5 makes retry and failover the PROVIDER's job,
  so by the time an exception reaches core it should already be one of these.
* `JudgeUsage` is cost instrumentation. It is not part of the model's
  structured output and core's `Observation` is frozen with `extra="forbid"`,
  so providers return it alongside via `observe_with_usage()` rather than
  smuggling it into the schema.
"""

from __future__ import annotations

from typing import ClassVar, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from launder_core.gates.registry import (
    ClaimObservation,
    JudgeProvider,
    JudgeUnavailable,
    Observation,
)
from launder_core.schemas import PassagePublic

__all__ = [
    "ClaimObservation",
    "JudgeError",
    "JudgeProvider",
    "JudgeUnavailable",
    "JudgeUsage",
    "Observation",
    "UsageReportingProvider",
]


class JudgeError(JudgeUnavailable):
    """A provider failed. `retryable` decides retry-then-failover vs failover now.

    A 5xx, a 429 or a timeout is worth one retry; a 400 will be a 400 again, and
    burning the retry budget on it costs latency the player pays for.
    """

    def __init__(self, provider: str, message: str, *, retryable: bool = True) -> None:
        super().__init__(f"{provider}: {message}")
        self.provider = provider
        self.retryable = retryable


class JudgeUsage(BaseModel):
    """Cost instrumentation. Never part of the structured output."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    latency_ms: int = 0

    def cost_usd(self, input_per_mtok: float, output_per_mtok: float) -> float:
        billable_input = max(self.input_tokens - self.cached_input_tokens, 0)
        return (
            billable_input * input_per_mtok + self.output_tokens * output_per_mtok
        ) / 1_000_000.0


@runtime_checkable
class UsageReportingProvider(Protocol):
    """A `JudgeProvider` that can also report what the call cost."""

    name: ClassVar[str]

    async def observe(self, passage: PassagePublic, normalized: str, nonce: str) -> Observation: ...

    async def observe_with_usage(
        self, passage: PassagePublic, normalized: str, nonce: str
    ) -> tuple[Observation, JudgeUsage]: ...
