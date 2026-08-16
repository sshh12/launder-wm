"""Anthropic judge provider — the §7.5 failover leg. Official SDK, explicit timeouts.

**Failover, not consensus.** This provider is not called on every submit; it is
called when OpenAI has already failed a retry. Running both every time would
double cost and latency for a gate whose hard cases are already deterministic.

Two shape differences from the OpenAI leg, both required rather than cosmetic:

1. `output_config.format` takes `{"type": "json_schema", "schema": ...}` — no
   `name`, no `strict` (strictness is implied). §7.5 asks for "an equivalent
   schema", and `anthropic_schema()` below is that equivalence: JSON Schema's
   `type: ["string", "null"]` union is rewritten as `anyOf`, which is in the
   documented supported subset.
2. `max_retries=0` on the client. Retry and failover policy lives in
   `gate.py`, which knows about the spend ledger and the player waiting; an SDK
   retrying underneath it would silently double the wall-clock budget the
   6-second timeout exists to bound.

No `thinking` parameter is sent. The failover model is Haiku-class (§7.5), which
does not think unless asked — and §7.5's "non-reasoning is non-negotiable
regardless of model" is why nothing here ever asks. **The API key is passed to
the client constructor and never logged.**
"""

from __future__ import annotations

import json
import time
from typing import Any, ClassVar

import anthropic
import httpx
from anthropic import AsyncAnthropic
from anthropic.types import Message, TextBlock
from pydantic import SecretStr, ValidationError

from launder_core.schemas import PassagePublic
from launder_serve.judge.prompt import OBSERVATION_SCHEMA, SYSTEM_PROMPT, render_user_message
from launder_serve.judge.protocol import JudgeError, JudgeUsage, Observation

__all__ = ["AnthropicJudge", "anthropic_schema", "make_anthropic_client"]


def anthropic_schema(node: Any) -> Any:
    """Rewrite `type: [T, "null"]` unions as `anyOf`, recursively.

    Anthropic's structured-output subset supports `anyOf`, `enum`, `const` and
    `additionalProperties: false`, but not JSON Schema's list-valued `type`.
    Everything else in §7.4's schema passes through untouched, so the two
    providers really are answering the same question.
    """
    if isinstance(node, list):
        return [anthropic_schema(item) for item in node]
    if not isinstance(node, dict):
        return node

    out: dict[str, Any] = {k: anthropic_schema(v) for k, v in node.items() if k != "type"}
    node_type = node.get("type")

    if isinstance(node_type, list):
        non_null = [t for t in node_type if t != "null"]
        nullable = "null" in node_type
        enum = out.pop("enum", None)
        branches: list[dict[str, Any]] = []
        for t in non_null:
            branch: dict[str, Any] = {"type": t}
            if enum is not None:
                values = [v for v in enum if v is not None]
                if values:
                    branch["enum"] = values
            branches.append(branch)
        if nullable:
            branches.append({"type": "null"})
        out["anyOf"] = branches
        return out

    if node_type is not None:
        out["type"] = node_type
    return out


def make_anthropic_client(api_key: SecretStr, *, timeout_s: float) -> AsyncAnthropic:
    """One client, explicit per-phase timeouts, no SDK-level retries.

    `connect` is tight because a slow DNS/TLS handshake is never going to
    recover inside a submit; `read` carries the whole budget because that is
    where a real generation spends it. The total must stay under
    `railway.json`'s `drainingSeconds: 15` or a deploy chops an in-flight call
    in half (§11.1).
    """
    if timeout_s <= 0:
        raise ValueError("judge timeout must be positive; an unbounded judge call blocks a submit")
    return AsyncAnthropic(
        api_key=api_key.get_secret_value(),
        timeout=httpx.Timeout(
            timeout_s, connect=min(2.0, timeout_s), read=timeout_s, write=timeout_s
        ),
        max_retries=0,
    )


class AnthropicJudge:
    """`JUDGE_PROVIDER=anthropic`, and the failover leg for `openai`."""

    name: ClassVar[str] = "anthropic"

    def __init__(
        self,
        client: AsyncAnthropic,
        *,
        model: str,
        max_output_tokens: int = 300,
        temperature: float | None = 0.0,
    ) -> None:
        self._client = client
        self.model = model
        self.max_output_tokens = max_output_tokens
        #: `None` omits the field. Current frontier models reject `temperature`
        #: outright; a Haiku-class failover model accepts it. Neither is
        #: hardcoded, because `JUDGE_FAILOVER_MODEL` is pinned at build time.
        self.temperature = temperature
        self._schema: dict[str, Any] = anthropic_schema(OBSERVATION_SCHEMA)

    async def observe(self, passage: PassagePublic, normalized: str, nonce: str) -> Observation:
        obs, _ = await self.observe_with_usage(passage, normalized, nonce)
        return obs

    async def observe_with_usage(
        self, passage: PassagePublic, normalized: str, nonce: str
    ) -> tuple[Observation, JudgeUsage]:
        user = render_user_message(
            passage_id=passage.id,
            original_text=passage.text,
            claims=passage.claims,
            normalized=normalized,
            nonce=nonce,
        )
        kwargs: dict[str, Any] = {}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature

        started = time.monotonic()
        try:
            message: Message = await self._client.messages.create(
                model=self.model,
                max_tokens=self.max_output_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user}],
                output_config={"format": {"type": "json_schema", "schema": self._schema}},
                **kwargs,
            )
        except anthropic.APITimeoutError as exc:
            raise JudgeError(self.name, "timeout") from exc
        except anthropic.APIConnectionError as exc:
            raise JudgeError(self.name, f"connection error: {type(exc).__name__}") from exc
        except anthropic.APIStatusError as exc:
            raise JudgeError(
                self.name,
                f"HTTP {exc.status_code}",
                retryable=exc.status_code in (408, 409, 429) or exc.status_code >= 500,
            ) from exc

        latency_ms = int((time.monotonic() - started) * 1000)

        # Check stop_reason BEFORE reading content: a safety decline returns a
        # normal 200 with empty or partial content, and indexing content[0]
        # would raise IndexError on a path that is not an error at all.
        if message.stop_reason == "refusal":
            raise JudgeError(
                self.name,
                "the model declined the request (stop_reason=refusal)",
                retryable=False,
            )

        # `content` is a union of a dozen block types and only `text` carries
        # the structured output; narrowing by isinstance keeps that fact
        # checkable rather than hoping for an attribute.
        text = next((block.text for block in message.content if isinstance(block, TextBlock)), None)
        if not text:
            raise JudgeError(
                self.name,
                f"no text block in response (stop_reason={message.stop_reason!r})",
                retryable=message.stop_reason == "max_tokens",
            )

        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise JudgeError(
                self.name, f"structured output was not JSON: {exc}", retryable=False
            ) from exc
        try:
            obs = Observation.model_validate(raw)
        except ValidationError as exc:
            raise JudgeError(
                self.name,
                f"structured output did not match launder_gate_observation: "
                f"{exc.errors(include_url=False)[:3]}",
                retryable=False,
            ) from exc

        return obs, JudgeUsage(
            provider=self.name,
            model=self.model,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            cached_input_tokens=int(getattr(message.usage, "cache_read_input_tokens", 0) or 0),
            latency_ms=latency_ms,
        )
