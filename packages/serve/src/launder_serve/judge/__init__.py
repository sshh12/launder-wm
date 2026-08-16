"""The judge: observations in, a derived verdict out (TECH_PLAN.md §7.3-§7.5).

Nothing outside this package may call a provider directly; `CachingJudge` owns
the cache, the rate limiter and the spend ledger, and skipping it skips all
three.

There is ONE paid provider: OpenAI. The policy is one provider, one retry, then
fail open provisional — `build_provider()` returns a single `JudgeProvider`, and
a provider error becomes a provisional clear rather than a call to a second
vendor. `fake` and `cassette` are offline stand-ins for tests and CI, not
alternatives that spend money.

`build_provider()` is the only place `JUDGE_PROVIDER` is interpreted. It also
refuses to build a live provider whose `prompt_hash` disagrees with
`data/config/judge.toml` — a forgotten `judge_version` bump is the single most
likely way to serve stale verdicts after a prompt edit (§7.5).
"""

from __future__ import annotations

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
    "resolve_model",
    "resolve_prompt_hash",
]


def resolve_model(content: Content, settings: Settings) -> str:
    """The model this build will send. `JUDGE_MODEL` overrides judge.toml's pin.

    One definition, because three things have to agree on it or the cache is
    wrong: the request body, the `prompt_hash` (which covers the model id) and
    the boot assertion that the model is pinned at all.
    """
    return settings.judge_model or content.judge.model


def resolve_prompt_hash(content: Content, settings: Settings) -> str:
    """The hash this build actually computes, over the model that will be used."""
    return compute_prompt_hash(
        model_id=resolve_model(content, settings),
        reasoning_effort=content.judge.reasoning_effort,
    )


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
) -> JudgeProvider:
    """THE provider for `JUDGE_PROVIDER`. One provider, never a chain.

    A provider error is not routed to a second vendor: `CachingJudge` retries
    once and then raises, and core turns that into a provisional clear (§7.5).
    """
    prompt_hash = resolve_prompt_hash(content, settings)
    provider = settings.judge_provider

    if provider == "fake":
        return FakeJudge()

    if provider == "cassette":
        assert settings.judge_cassette_path is not None  # asserted in Settings
        return CassetteJudge(settings.judge_cassette_path, prompt_hash=prompt_hash)

    if http is None:
        raise RuntimeError(
            f"JUDGE_PROVIDER={provider} needs an httpx client; the app factory owns its "
            "lifetime so connections are pooled and closed on shutdown."
        )

    live: JudgeProvider = OpenAIJudge(
        settings.openai_api_key,
        model=resolve_model(content, settings),
        client=http,
        max_output_tokens=content.judge.max_output_tokens,
        temperature=content.judge.temperature,
        reasoning_effort=content.judge.reasoning_effort or None,
        schema_name=content.judge.schema_out.name,
    )

    if settings.judge_record and settings.judge_cassette_path is not None:
        live = RecordingJudge(live, settings.judge_cassette_path, prompt_hash=prompt_hash)
    return live
