"""`CachingJudge` — the abuse ladder wrapped around a provider (§7.5).

Core's `llm_gate` check calls `deps.judge.observe(passage, normalized, nonce)`
and derives the verdict from what comes back. Everything between "a submission
reached check 8" and "a model was actually asked" is serve's, and it is this
class: the process LRU, the global cache, the per-IP bucket, the daily spend
ledger, then ONE provider with one retry, then fail open provisional.

| rung | mechanism                               | cost of an abusive request |
|------|-----------------------------------------|----------------------------|
| 2    | process LRU, then the global cache repo | one indexed SELECT         |
| 3    | per-IP token bucket, **on misses only** | in-process, free           |
| 4    | atomic daily spend ledger               | one UPDATE                 |

Two design points worth stating because they are easy to get subtly wrong:

**The cache key covers the passage's DATA, not just its id.** A verdict is a
function of (system prompt, original text, claim list, submission).
`prompt_hash` covers the prompt and `sha256(normalized)` the submission; the
text and the claims were covered only by `passage_id`, so re-authoring a
passage's claims under the same id kept serving verdicts derived from the old
ones — and `DELETE FROM judge_cache` did not help, because the process LRU in
front of the repo holds the same key and survives a database wipe.
`prompt.passage_digest` closes it.

**The cache stores the observation, and its key omits `level_id`.** §7.5's key
includes the level, but the model is never told what level it is judging — it
reports observations about (passage, text) and *code* derives the verdict from
the level's params. Keying on the level would mean paying twice for the same
API call to learn the same facts, and it would make an L1 answer unavailable to
an L4 submission of identical text. Dropping it is strictly more hits and
exactly the same verdicts. `judge_version`, `prompt_hash` and `scoring_version`
still scope the key, so a prompt or normalization change still orphans it.

**Rate limiting has to reach the HTTP layer, and a raised exception cannot.**
Core catches `JudgeUnavailable` and turns it into a fail-open provisional clear
— correct for a provider outage, wrong for "you have had ten of these this
hour", which §9.3 says is a 429 with `Retry-After`. A `ContextVar` carries that
one fact back out: it is request-scoped under asyncio, so concurrent submits
cannot read each other's.
"""

from __future__ import annotations

import contextvars
import logging
from collections import OrderedDict
from datetime import UTC, date, datetime
from typing import ClassVar, Final

from launder_core.schemas import PassagePublic
from launder_serve.judge.prompt import cache_key, passage_digest
from launder_serve.judge.protocol import (
    JudgeError,
    JudgeProvider,
    JudgeUsage,
    Observation,
    UsageReportingProvider,
)
from launder_serve.limits import SHARED_BUCKET_KEY, TokenBucketLimiter
from launder_serve.repo.protocol import CachedVerdict, JudgeCacheRepo, SpendRepo

__all__ = ["EST_USD_PER_CALL", "RATE_LIMITED", "REQUEST_CLIENT_KEY", "CachingJudge"]

_log = logging.getLogger("launder.judge")

#: What one judge call reserves from the daily ledger, in USD. The DEFAULT for
#: `Settings.judge_est_usd_per_call`, which is what `CachingJudge` is actually
#: given — this constant is the single source of that default and nothing else.
#:
#: Derivation, from the live probe of `gpt-5.6-terra` with this repo's exact
#: request body (a real 250-word passage with a 6-claim list): 1,887 input +
#: 164 output tokens, 0 reasoning tokens at `effort: "none"`.
#:
#: ASSUMPTION, NOT VERIFIED: the price per token. The token counts above were
#: measured; the dollars-per-token they are multiplied by were not checked
#: against a live pricing page. $0.005 is that product rounded up, and it is the
#: number to re-derive first if the bill and the ledger ever disagree.
#:
#: Accuracy matters in BOTH directions, because the ledger reserves this value
#: BEFORE each call and §9.6's `SpendRepo` has no reconciliation call:
#: too low  -> the cap is passed before the ledger notices and the real bill
#:             overshoots what the operator authorised;
#: too high -> the ledger refuses early and levels start clearing provisionally
#:             while most of the authorised budget is still unspent.
EST_USD_PER_CALL: Final[float] = 0.005

#: Set by `CachingJudge` when it refuses a call, read by `/api/submit` to turn
#: the fail-open clear into the 429 §9.3 specifies. Request-scoped.
RATE_LIMITED: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "launder_judge_rate_limited", default=None
)
#: The bucket key for the in-flight request. Set by the endpoint before the gate
#: runs, because core's `Deps` has no place for a client identity and should
#: not grow one — it is an HTTP concern.
REQUEST_CLIENT_KEY: contextvars.ContextVar[str] = contextvars.ContextVar(
    "launder_judge_client_key", default=SHARED_BUCKET_KEY
)


class CachingJudge:
    """A `JudgeProvider` that wraps ONE provider in the §7.5 abuse ladder."""

    name: ClassVar[str] = "caching"

    def __init__(
        self,
        *,
        provider: JudgeProvider,
        cache: JudgeCacheRepo,
        spend: SpendRepo,
        limiter: TokenBucketLimiter,
        judge_version: str,
        prompt_hash: str,
        scoring_version: str,
        separator: str = "\x1f",
        lru_entries: int = 2000,
        cache_failures: bool = True,
        #: DELIBERATELY not the shipped cap. The app factory always passes
        #: `Settings.judge_daily_usd_cap`, so this default is only ever reached by
        #: a hand-constructed judge in a test or a script — and the safe number
        #: for those is a small one, not the production spend authorisation.
        daily_usd_cap: float = 2.0,
        est_usd_per_call: float = EST_USD_PER_CALL,
        max_retries: int = 1,
    ) -> None:
        self.provider = provider
        self.cache = cache
        self.spend = spend
        self.limiter = limiter
        self.judge_version = judge_version
        self.prompt_hash = prompt_hash
        self.scoring_version = scoring_version
        self.separator = separator
        self.cache_failures = cache_failures
        self.daily_usd_cap = daily_usd_cap
        self.est_usd_per_call = est_usd_per_call
        self.max_retries = max_retries
        self._lru: OrderedDict[str, Observation] = OrderedDict()
        self._lru_max = lru_entries
        #: Instrumentation, read by tests and by the M5 eval harness.
        self.stats: dict[str, int] = {
            "lru_hits": 0,
            "repo_hits": 0,
            "calls": 0,
            "rate_limited": 0,
            "spend_refused": 0,
            "errors": 0,
        }

    # -- cache ---------------------------------------------------------------

    def key_for(self, passage: PassagePublic | str, normalized: str) -> str:
        """The key. Takes the PASSAGE, not just its id — see `passage_digest`.

        A string id is still accepted so a caller that genuinely has nothing but
        an id can look one up, but that key deliberately does not match a real
        one: the digest is what stops a re-authored claim list from being served
        an observation derived from the old one.
        """
        if isinstance(passage, str):
            return cache_key(
                judge_version=self.judge_version,
                prompt_hash=self.prompt_hash,
                scoring_version=self.scoring_version,
                passage_id=passage,
                normalized=normalized,
                separator=self.separator,
            )
        return cache_key(
            judge_version=self.judge_version,
            prompt_hash=self.prompt_hash,
            scoring_version=self.scoring_version,
            passage_id=passage.id,
            passage_digest=passage_digest(passage.id, passage.text, passage.claims),
            normalized=normalized,
            separator=self.separator,
        )

    def _lru_get(self, key: str) -> Observation | None:
        obs = self._lru.get(key)
        if obs is not None:
            self._lru.move_to_end(key)
        return obs

    def _lru_put(self, key: str, obs: Observation) -> None:
        self._lru[key] = obs
        self._lru.move_to_end(key)
        while len(self._lru) > self._lru_max:
            self._lru.popitem(last=False)

    # -- the provider contract ----------------------------------------------

    async def observe(
        self, passage: PassagePublic, normalized: str, nonce: str, day: date | None = None
    ) -> Observation:
        key = self.key_for(passage, normalized)

        cached = self._lru_get(key)
        if cached is not None:
            self.stats["lru_hits"] += 1
            return cached

        stored = await self.cache.get(key)
        if stored is not None:
            self.stats["repo_hits"] += 1
            obs = Observation.model_validate(stored.observation)
            self._lru_put(key, obs)
            return obs

        # --- rung 3: per-IP bucket, MISSES ONLY -----------------------------
        # A player resubmitting text that is already cached, or hammering the
        # detector, never reaches this line. Throttling either would make the
        # game feel broken and would save nothing.
        decision = self.limiter.take(REQUEST_CLIENT_KEY.get())
        if not decision.allowed:
            self.stats["rate_limited"] += 1
            RATE_LIMITED.set(decision.retry_after_s)
            raise JudgeError(
                self.name,
                f"per-IP judge budget exhausted; retry in {decision.retry_after_s}s",
                retryable=False,
            )

        # --- rung 4: atomic daily spend ledger ------------------------------
        today = day or datetime.now(UTC).date()
        if not await self.spend.reserve(today, self.est_usd_per_call, self.daily_usd_cap):
            self.stats["spend_refused"] += 1
            _log.warning(
                "judge daily spend cap $%.2f reached; degrading to provisional",
                self.daily_usd_cap,
            )
            raise JudgeError(self.name, "daily spend cap reached", retryable=False)

        obs, usage = await self._observe_with_retry(passage, normalized, nonce)
        self._lru_put(key, obs)
        await self.cache.put(
            key,
            CachedVerdict(
                judge_version=self.judge_version,
                passage_id=passage.id,
                # The observation is level-independent; the level enters at
                # derivation. Recorded for forensics, not used in the key.
                level_id=passage.level_id,
                cleared=obs.verdict_opinion == "pass",
                observation=obs.model_dump(mode="json"),
                provider=usage.provider,
                model=usage.model,
                created_at=datetime.now(UTC),
                feedback=obs.notes or None,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
            ),
        )
        return obs

    async def _observe_with_retry(
        self, passage: PassagePublic, normalized: str, nonce: str
    ) -> tuple[Observation, JudgeUsage]:
        """One provider, one retry on a retryable failure, then raise.

        Raising is the WHOLE fail-open path: core catches `JudgeUnavailable` and
        clears the submission provisionally. There is no second vendor to fall
        through to, and adding one would double cost and latency on a gate whose
        hard cases are already caught deterministically before it runs (§7.5).

        A non-retryable failure does not burn the retry: a 400 will be a 400
        again, and the extra round trip is latency the player pays for.
        """
        attempts = self.max_retries + 1
        last: JudgeError | None = None
        for attempt in range(attempts):
            try:
                self.stats["calls"] += 1
                return await _call(self.provider, passage, normalized, nonce)
            except JudgeError as exc:
                last = exc
                if not exc.retryable or attempt == attempts - 1:
                    break
        self.stats["errors"] += 1
        assert last is not None  # the loop cannot exit without setting it
        raise last


async def _call(
    provider: JudgeProvider, passage: PassagePublic, normalized: str, nonce: str
) -> tuple[Observation, JudgeUsage]:
    if isinstance(provider, UsageReportingProvider):
        return await provider.observe_with_usage(passage, normalized, nonce)
    return await provider.observe(passage, normalized, nonce), JudgeUsage(
        provider=getattr(provider, "name", "unknown")
    )
