"""The abuse ladder's rung 3 (TECH_PLAN.md §7.5, §14.2 item 6).

The test that matters most in this file is
`test_a_missing_real_ip_header_shares_one_bucket`. `X-Forwarded-For` is not in
Railway's documented header set; if the header we key on ever stops arriving,
the obvious implementation — "no IP, no limit" — turns a header rename into a
silently disabled rate limiter, and you find out from the bill.
"""

from __future__ import annotations

import httpx
import pytest

from launder_serve.limits import SHARED_BUCKET_KEY, TokenBucketLimiter, client_key


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_a_missing_real_ip_header_shares_one_bucket() -> None:
    """A missing header must mean ONE SHARED BUCKET, never 'unlimited'."""
    assert client_key({}) == SHARED_BUCKET_KEY
    assert client_key({"X-Real-IP": "   "}) == SHARED_BUCKET_KEY
    assert client_key({"X-Forwarded-For": "1.2.3.4"}) == SHARED_BUCKET_KEY

    limiter = TokenBucketLimiter(per_hour=10, burst=3)
    # Three anonymous callers, three different forged headers we do not read,
    # one bucket between them.
    for headers in ({}, {"X-Forwarded-For": "9.9.9.9"}, {"User-Agent": "curl"}):
        assert limiter.take(client_key(headers)).allowed is True
    assert limiter.take(client_key({})).allowed is False


def test_x_forwarded_for_is_only_read_when_explicitly_configured() -> None:
    headers = {"X-Forwarded-For": "203.0.113.7, 70.41.3.18"}
    assert client_key(headers) == SHARED_BUCKET_KEY
    assert client_key(headers, fallback_header="X-Forwarded-For") == "203.0.113.7"


def test_the_documented_header_wins() -> None:
    headers = {"X-Real-IP": "203.0.113.9", "X-Forwarded-For": "1.1.1.1"}
    assert client_key(headers, fallback_header="X-Forwarded-For") == "203.0.113.9"
    assert client_key({"x-real-ip": "203.0.113.9"}) == "203.0.113.9"


def test_burst_then_sustained_refill() -> None:
    clock = Clock()
    limiter = TokenBucketLimiter(per_hour=10, burst=3, clock=clock)

    assert [limiter.take("ip").allowed for _ in range(4)] == [True, True, True, False]

    denied = limiter.take("ip")
    assert denied.retry_after_s >= 1

    # 10/hour == one token every six minutes.
    clock.advance(360)
    assert limiter.take("ip").allowed is True
    assert limiter.take("ip").allowed is False


def test_buckets_are_per_key() -> None:
    limiter = TokenBucketLimiter(per_hour=10, burst=1)
    assert limiter.take("a").allowed is True
    assert limiter.take("a").allowed is False
    assert limiter.take("b").allowed is True


def test_bucket_table_is_bounded() -> None:
    """A spray of forged IPs must not grow the dict without limit."""
    limiter = TokenBucketLimiter(per_hour=10, burst=3, max_buckets=16)
    for i in range(200):
        limiter.take(f"ip-{i}")
    assert len(limiter._buckets) <= 16


@pytest.mark.parametrize("bad", [(0, 3), (10, 0)])
def test_a_nonsense_limit_is_a_construction_error(bad: tuple[int, int]) -> None:
    per_hour, burst = bad
    with pytest.raises(ValueError):
        TokenBucketLimiter(per_hour=per_hour, burst=burst)


async def test_the_detector_is_never_rate_limited(
    client: httpx.AsyncClient, app: object, passage_id: str
) -> None:
    """Rungs 0-2 are not rate limited: throttling the needle would make the
    game feel broken, and a detect call costs one CPU-millisecond."""
    limiter = app.state.launder.limiter  # type: ignore[attr-defined]
    limiter.reset()
    for _ in range(30):
        response = await client.post(
            "/api/detect", json={"passage_id": passage_id, "text": "a short probe", "seq": 0}
        )
        assert response.status_code == 200
    # Nothing was ever taken from the bucket.
    assert limiter.peek("anything") == limiter.burst
