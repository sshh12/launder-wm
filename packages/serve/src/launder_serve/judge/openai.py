"""OpenAI judge provider — Responses API over `httpx`, with explicit timeouts.

Raw REST rather than the SDK, deliberately: the resolved `openai` package is a
major version ahead of the examples in TECH_PLAN.md §7.5, and the one thing this
call must be is *stable under a dependency bump*. The wire shape of
`POST /v1/responses` is versioned by `openai-version`, the request is four
fields, and the failure modes we care about (429, 5xx, timeout) are HTTP status
codes rather than SDK exception classes. `httpx` also gives the per-phase
timeout §7.5 requires: connect, read and write are bounded separately so a
half-open socket cannot outlive `drainingSeconds`.

**The API key is read from `SecretStr` at request time, placed in one header,
and never logged.** No error path in this module includes the header dict.
"""

from __future__ import annotations

import json
import time
from typing import Any, ClassVar, Final

import httpx
from pydantic import SecretStr, ValidationError

from launder_core.schemas import PassagePublic
from launder_serve.judge.prompt import OBSERVATION_SCHEMA, SYSTEM_PROMPT, render_user_message
from launder_serve.judge.protocol import JudgeError, JudgeUsage, Observation

__all__ = ["OpenAIJudge"]

_DEFAULT_BASE_URL: Final[str] = "https://api.openai.com/v1"
#: Statuses worth one retry then failover. Everything else fails over at once —
#: a 400 will be a 400 again, and burning the retry budget on it costs latency
#: the player pays for.
_RETRYABLE: Final[frozenset[int]] = frozenset({408, 409, 429, 500, 502, 503, 504, 529})


class OpenAIJudge:
    """`JUDGE_PROVIDER=openai`."""

    name: ClassVar[str] = "openai"

    def __init__(
        self,
        api_key: SecretStr,
        *,
        model: str,
        client: httpx.AsyncClient,
        max_output_tokens: int = 300,
        temperature: float | None = 0.0,
        reasoning_effort: str | None = "none",
        schema_name: str = "launder_gate_observation",
        base_url: str = _DEFAULT_BASE_URL,
    ) -> None:
        self._api_key = api_key
        self.model = model
        self._client = client
        self.max_output_tokens = max_output_tokens
        #: `None` omits the field. Reasoning models reject `temperature`;
        #: non-reasoning models reject `reasoning`. Both are therefore optional
        #: rather than hardcoded, and §7.5's "non-reasoning is non-negotiable"
        #: is enforced by the VALUE (`"none"`), not by omitting the field.
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.schema_name = schema_name
        self.base_url = base_url.rstrip("/")

    def _body(self, passage: PassagePublic, normalized: str, nonce: str) -> dict[str, Any]:
        user = render_user_message(
            passage_id=passage.id,
            original_text=passage.text,
            claims=passage.claims,
            normalized=normalized,
            nonce=nonce,
        )
        body: dict[str, Any] = {
            "model": self.model,
            "instructions": SYSTEM_PROMPT,
            "input": [{"role": "user", "content": [{"type": "input_text", "text": user}]}],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": self.schema_name,
                    "strict": True,
                    "schema": OBSERVATION_SCHEMA,
                }
            },
            "max_output_tokens": self.max_output_tokens,
            # The submission is untrusted player text. Nothing about it should
            # be retained on a third party's servers a moment longer than the
            # request takes.
            "store": False,
        }
        if self.reasoning_effort is not None:
            body["reasoning"] = {"effort": self.reasoning_effort}
        if self.temperature is not None:
            body["temperature"] = self.temperature
        return body

    async def observe(self, passage: PassagePublic, normalized: str, nonce: str) -> Observation:
        obs, _ = await self.observe_with_usage(passage, normalized, nonce)
        return obs

    async def observe_with_usage(
        self, passage: PassagePublic, normalized: str, nonce: str
    ) -> tuple[Observation, JudgeUsage]:
        started = time.monotonic()
        try:
            response = await self._client.post(
                f"{self.base_url}/responses",
                json=self._body(passage, normalized, nonce),
                headers={
                    "Authorization": f"Bearer {self._api_key.get_secret_value()}",
                    "Content-Type": "application/json",
                },
            )
        except httpx.TimeoutException as exc:
            raise JudgeError(self.name, f"timeout after {type(exc).__name__}") from exc
        except httpx.HTTPError as exc:
            raise JudgeError(self.name, f"transport error: {type(exc).__name__}") from exc

        if response.status_code >= 400:
            raise JudgeError(
                self.name,
                f"HTTP {response.status_code}: {_safe_error(response)}",
                retryable=response.status_code in _RETRYABLE,
            )

        latency_ms = int((time.monotonic() - started) * 1000)
        payload = response.json()
        text = _extract_output_text(payload)
        if text is None:
            raise JudgeError(
                self.name,
                f"no output_text in response (status={payload.get('status')!r}, "
                f"incomplete={payload.get('incomplete_details')!r})",
                retryable=False,
            )
        return _parse(self.name, self.model, text, payload.get("usage") or {}, latency_ms)


def _safe_error(response: httpx.Response) -> str:
    """The provider's error message, never the request that produced it."""
    try:
        body = response.json()
    except ValueError:
        return response.text[:200]
    err = body.get("error") if isinstance(body, dict) else None
    if isinstance(err, dict):
        return f"{err.get('type', '')} {err.get('message', '')}".strip()[:200]
    return str(body)[:200]


def _extract_output_text(payload: dict[str, Any]) -> str | None:
    direct = payload.get("output_text")
    if isinstance(direct, str) and direct:
        return direct
    for item in payload.get("output", []) or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for block in item.get("content", []) or []:
            if isinstance(block, dict) and block.get("type") == "output_text":
                value = block.get("text")
                if isinstance(value, str) and value:
                    return value
    return None


def _parse(
    provider: str, model: str, text: str, usage: dict[str, Any], latency_ms: int
) -> tuple[Observation, JudgeUsage]:
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise JudgeError(
            provider, f"structured output was not JSON: {exc}", retryable=False
        ) from exc
    try:
        obs = Observation.model_validate(raw)
    except ValidationError as exc:
        raise JudgeError(
            provider,
            f"structured output did not match launder_gate_observation: "
            f"{exc.errors(include_url=False)[:3]}",
            retryable=False,
        ) from exc
    details = usage.get("input_tokens_details") or {}
    return obs, JudgeUsage(
        provider=provider,
        model=model,
        input_tokens=int(usage.get("input_tokens", 0) or 0),
        output_tokens=int(usage.get("output_tokens", 0) or 0),
        cached_input_tokens=int(details.get("cached_tokens", 0) or 0),
        latency_ms=latency_ms,
    )
