"""The judge: observations in, a derived verdict out (TECH_PLAN.md §7.3-§7.5).

Nothing outside this package may call a provider directly; `LlmGate` owns the
cache, the rate limiter, the spend ledger and the failover chain, and skipping
it skips all four.

`build_provider()` is the only place `JUDGE_PROVIDER` is interpreted. It also
refuses to build a live provider whose `prompt_hash` disagrees with
`data/config/judge.toml` — a forgotten `judge_version` bump is the single most
likely way to serve stale verdicts after a prompt edit (§7.5).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import httpx

from launder_serve.content import Content
from launder_serve.judge.cassette import CassetteJudge, RecordingJudge
from launder_serve.judge.fake import FakeJudge
from launder_serve.judge.gate import (
    EST_USD_PER_CALL,
    RATE_LIMITED,
    REQUEST_CLIENT_KEY,
    CachingJudge,
)
from launder_serve.judge.openai import OpenAIJudge
from launder_serve.judge.prompt import (
    OBSERVATION_SCHEMA,
    PROMPT_ID,
    SYSTEM_PROMPT,
    cache_key,
    compute_prompt_hash,
)
from launder_serve.judge.protocol import (
    ClaimObservation,
    JudgeError,
    JudgeProvider,
    JudgeUnavailable,
    JudgeUsage,
    Observation,
)

if TYPE_CHECKING:
    from launder_serve.settings import Settings

__all__ = [
    "EST_USD_PER_CALL",
    "OBSERVATION_SCHEMA",
    "PROMPT_ID",
    "RATE_LIMITED",
    "REQUEST_CLIENT_KEY",
    "SYSTEM_PROMPT",
    "CachingJudge",
    "CassetteJudge",
    "ClaimObservation",
    "FakeJudge",
    "JudgeError",
    "JudgeProvider",
    "JudgeUnavailable",
    "JudgeUsage",
    "Observation",
    "OpenAIJudge",
    "RecordingJudge",
    "assert_prompt_hash",
    "build_provider",
    "cache_key",
    "compute_prompt_hash",
    "resolve_prompt_hash",
]

_log = logging.getLogger("launder.judge")


def resolve_prompt_hash(content: Content, settings: Settings) -> str:
    """The hash this build actually computes, over the model that will be used."""
    model = settings.judge_model or content.judge.model
    return compute_prompt_hash(model_id=model, reasoning_effort=content.judge.reasoning_effort)


def assert_prompt_hash(content: Content, settings: Settings) -> list[str]:
    """Boot assertion 3. Returns warnings; raises for a provider that spends money.

    The placeholder in `judge.toml` is treated as "not yet pinned", not as
    "matches": offline providers get a loud warning naming the value to paste,
    paid providers get a boot failure. That asymmetry is deliberate — PR
    environments run `fake` with no keys by design (§11.5) and must still boot.
    """
    actual = resolve_prompt_hash(content, settings)
    declared = content.judge.prompt_hash
    if not content.judge.prompt_hash_enforced:
        return [
            "judge.toml sets prompt_hash_enforced = false. A prompt edit can now change "
            "verdicts without changing judge_version, and the cache will serve the old "
            "ones (TECH_PLAN.md §7.5)."
        ]
    if content.judge.prompt_hash_is_placeholder:
        message = (
            f"data/config/judge.toml still carries the prompt_hash placeholder "
            f"{declared!r}. The value this build computes is:\n\n    prompt_hash = "
            f'"{actual}"\n\nPaste it into judge.toml.'
        )
        if settings.judge_is_paid:
            raise RuntimeError(message)
        return [message]
    if declared != actual:
        raise RuntimeError(
            f"prompt_hash mismatch: judge.toml records {declared}, this build computes "
            f"{actual}. Either the system prompt, the schema, JUDGE_MODEL or "
            f"reasoning_effort changed. Bump judge_version in judge.toml AND levels.toml "
            "in the same commit, then record the new hash — otherwise the cache serves "
            "verdicts produced by the old prompt (TECH_PLAN.md §7.5, §12 row 12)."
        )
    return []


def build_provider(
    content: Content, settings: Settings, *, http: httpx.AsyncClient | None = None
) -> tuple[JudgeProvider, JudgeProvider | None]:
    """`(primary, failover)` for `JUDGE_PROVIDER`. Failover is `None` when unusable."""
    prompt_hash = resolve_prompt_hash(content, settings)
    provider = settings.judge_provider

    if provider == "fake":
        return FakeJudge(), None

    if provider == "cassette":
        assert settings.judge_cassette_path is not None  # asserted in Settings
        return CassetteJudge(settings.judge_cassette_path, prompt_hash=prompt_hash), None

    if http is None:
        raise RuntimeError(
            f"JUDGE_PROVIDER={provider} needs an httpx client; the app factory owns its "
            "lifetime so connections are pooled and closed on shutdown."
        )

    primary: JudgeProvider
    if provider == "openai":
        primary = OpenAIJudge(
            settings.openai_api_key,
            model=settings.judge_model or content.judge.model,
            client=http,
            max_output_tokens=content.judge.max_output_tokens,
            temperature=content.judge.temperature,
            reasoning_effort=content.judge.reasoning_effort or None,
            schema_name=content.judge.schema_out.name,
        )
        failover = _anthropic_leg(content, settings)
    else:
        leg = _anthropic_leg(content, settings, primary=True)
        if leg is None:
            raise RuntimeError("JUDGE_PROVIDER=anthropic but no ANTHROPIC_API_KEY is set.")
        primary, failover = leg, None

    if settings.judge_record and settings.judge_cassette_path is not None:
        primary = RecordingJudge(primary, settings.judge_cassette_path, prompt_hash=prompt_hash)
    return primary, failover


def _anthropic_leg(
    content: Content, settings: Settings, *, primary: bool = False
) -> JudgeProvider | None:
    if not settings.anthropic_api_key.get_secret_value():
        return None
    model = (
        (settings.judge_model or content.judge.model)
        if primary
        else (settings.judge_failover_model or content.judge.failover_model)
    )
    if not model or model == "REPLACE_AT_BUILD_TIME":
        _log.warning(
            "no Anthropic model pinned (JUDGE_FAILOVER_MODEL is unset and judge.toml "
            "carries the build-time sentinel): the §7.5 failover leg is disabled."
        )
        return None
    # Imported here so `import launder_serve.judge` does not pull the SDK into a
    # process that will only ever run the fake provider.
    from launder_serve.judge.anthropic import AnthropicJudge, make_anthropic_client

    client = make_anthropic_client(settings.anthropic_api_key, timeout_s=settings.judge_timeout_s)
    return AnthropicJudge(
        client,
        model=model,
        max_output_tokens=content.judge.max_output_tokens,
        temperature=content.judge.temperature,
    )
