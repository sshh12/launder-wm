"""Per-IP token bucket + the daily spend ledger (TECH_PLAN.md §7.5 rungs 3-4).

Three rules here are load-bearing and all three are tested:

1. **The rate-limit key is `X-Real-IP`** — Railway's documented header.
   `X-Forwarded-For` is not in the documented set (§14.2 item 6), so it is
   accepted only as an explicitly-configured fallback, never as the default.
2. **A missing header means ONE SHARED BUCKET, never "unlimited".** The
   obvious implementation — `if ip is None: allow` — turns "the proxy header
   changed name" into "the rate limiter silently switched off", which is the
   failure you find out about from the bill. `SHARED_BUCKET_KEY` exists so the
   degenerate path is a named, greppable thing rather than an omission.

3. **This bucket is a FAIRNESS AND ABUSE brake, not the cost bound.** The cost
   bound is the daily spend ledger, which is atomic and lives in Postgres, so it
   holds across every replica — while this bucket is process-local and each
   extra replica adds another full bucket (railway.json runs 3, so the real
   per-IP ceiling is ~3x this number). That division of labour is what makes a
   generous
   limiter safe, and it is why the defaults are 60/hour with a burst of 10
   rather than the 10/3 they started at: a 15-level campaign is played in ONE
   SITTING, and corporate and mobile NAT put many players behind a single
   `X-Real-IP`, so a tight per-IP bucket throttles legitimate play long before
   it costs an abuser anything.

The bucket is **process-local**, which is why `--workers 1` is load-bearing
(§9.6): two workers silently double the effective limit within one replica. If
throughput ever matters, move the bucket into Postgres *first*.

Rungs 0-2 are NOT rate limited. This limiter is consulted on judge cache
**misses only** — a player hammering the detector, or resubmitting text that is
already in the cache, costs nothing and must never be throttled, because a game
that says "too many checks" while you are typing feels broken.
"""

from __future__ import annotations

import math
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final

__all__ = [
    "SHARED_BUCKET_KEY",
    "RateDecision",
    "TokenBucketLimiter",
    "client_key",
]

#: The bucket every request with no usable client header shares. Named, not
#: implicit: "we could not identify the caller" must cost something.
SHARED_BUCKET_KEY: Final[str] = "__no_real_ip__"

#: Most buckets we keep before evicting the least-recently-used. Bounded so a
#: spray of forged IPs cannot grow the dict without limit.
_MAX_BUCKETS: Final[int] = 20_000


def client_key(
    headers: Mapping[str, str],
    *,
    header: str = "X-Real-IP",
    fallback_header: str | None = None,
) -> str:
    """The bucket key for a request. Missing/blank header -> the shared bucket."""
    value = (headers.get(header) or headers.get(header.lower()) or "").strip()
    if not value and fallback_header:
        raw = (headers.get(fallback_header) or headers.get(fallback_header.lower()) or "").strip()
        # X-Forwarded-For is a list; the client is the first entry.
        value = raw.split(",")[0].strip()
    return value or SHARED_BUCKET_KEY


@dataclass(frozen=True)
class RateDecision:
    allowed: bool
    remaining: float
    retry_after_s: int


@dataclass
class _Bucket:
    tokens: float
    updated: float


class TokenBucketLimiter:
    """`per_hour` sustained, `burst` instantaneous. In-process, LRU-bounded."""

    def __init__(
        self,
        *,
        per_hour: int = 60,
        burst: int = 10,
        clock: Callable[[], float] | None = None,
        max_buckets: int = _MAX_BUCKETS,
    ) -> None:
        if per_hour < 1 or burst < 1:
            raise ValueError("per_hour and burst must both be >= 1")
        self.per_hour = per_hour
        self.burst = float(burst)
        self.refill_per_s = per_hour / 3600.0
        self._clock = clock or time.monotonic
        self._max_buckets = max_buckets
        self._buckets: OrderedDict[str, _Bucket] = OrderedDict()

    def _bucket(self, key: str, now: float) -> _Bucket:
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = _Bucket(tokens=self.burst, updated=now)
            self._buckets[key] = bucket
            if len(self._buckets) > self._max_buckets:
                self._buckets.popitem(last=False)
        else:
            self._buckets.move_to_end(key)
            elapsed = max(now - bucket.updated, 0.0)
            bucket.tokens = min(self.burst, bucket.tokens + elapsed * self.refill_per_s)
            bucket.updated = now
        return bucket

    def peek(self, key: str) -> float:
        return self._bucket(key, self._clock()).tokens

    def take(self, key: str, cost: float = 1.0) -> RateDecision:
        now = self._clock()
        bucket = self._bucket(key, now)
        if bucket.tokens >= cost:
            bucket.tokens -= cost
            return RateDecision(allowed=True, remaining=bucket.tokens, retry_after_s=0)
        deficit = cost - bucket.tokens
        retry_after = math.ceil(deficit / self.refill_per_s)
        return RateDecision(
            allowed=False, remaining=bucket.tokens, retry_after_s=max(retry_after, 1)
        )

    def reset(self) -> None:
        self._buckets.clear()
