"""`cassette` — record/replay, and **raises on a miss** (TECH_PLAN.md §7.5).

The raise is the feature. A cassette that fell back to a live call on a miss
would let a CI run start spending money the first time somebody edited a test
fixture, and nobody would notice until the bill. `JudgeMiss` names the key and
the text hash so the fix is one `make record` away.

Cassette key = `sha256(prompt_hash ⋮ passage_id ⋮ sha256(normalized))`.

It deliberately omits `level_id`, which the §7.5 *verdict* cache key includes:
the model reports observations about (passage, text) and knows nothing about the
level. Two levels over the same submission are one recording and two different
`derive_verdict` outcomes — which is exactly the property that lets all six
levels share one prompt and one eval set.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import ClassVar

from launder_core.schemas import PassagePublic
from launder_serve.judge.protocol import (
    JudgeError,
    JudgeProvider,
    JudgeUsage,
    Observation,
    UsageReportingProvider,
)

__all__ = ["CassetteJudge", "JudgeMiss", "RecordingJudge", "cassette_key"]


class JudgeMiss(JudgeError):
    def __init__(self, key: str, passage_id: str, path: Path) -> None:
        super().__init__(
            "cassette",
            f"no recording for key {key} (passage {passage_id}) in {path}. The cassette "
            "provider raises on a miss by design so CI can never silently start spending "
            "money: re-record with JUDGE_PROVIDER=openai JUDGE_RECORD=1.",
            retryable=False,
        )
        self.key = key


def cassette_key(prompt_hash: str, passage_id: str, normalized: str) -> str:
    text_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return hashlib.sha256("\x1f".join((prompt_hash, passage_id, text_hash)).encode()).hexdigest()


class CassetteJudge:
    """Replays a JSONL cassette. One `{"key": ..., "observation": {...}}` per line."""

    name: ClassVar[str] = "cassette"

    def __init__(self, path: Path, *, prompt_hash: str) -> None:
        self.path = path
        self.prompt_hash = prompt_hash
        self._entries: dict[str, Observation] = {}
        self.hits: int = 0
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            row = json.loads(line)
            self._entries[str(row["key"])] = Observation.model_validate(row["observation"])

    async def observe(self, passage: PassagePublic, normalized: str, nonce: str) -> Observation:
        obs, _ = await self.observe_with_usage(passage, normalized, nonce)
        return obs

    async def observe_with_usage(
        self, passage: PassagePublic, normalized: str, nonce: str
    ) -> tuple[Observation, JudgeUsage]:
        del nonce  # excluded from the key: it is per-request, the recording is not
        key = cassette_key(self.prompt_hash, passage.id, normalized)
        obs = self._entries.get(key)
        if obs is None:
            raise JudgeMiss(key, passage.id, self.path)
        self.hits += 1
        return obs, JudgeUsage(provider=self.name, model="cassette")


class RecordingJudge:
    """Wraps a live provider and appends every observation to a cassette.

    Used by `make record`; never selected by `JUDGE_PROVIDER` directly, because
    "the provider that spends money" should require naming the provider that
    spends money.
    """

    name: ClassVar[str] = "recording"

    def __init__(self, inner: JudgeProvider, path: Path, *, prompt_hash: str) -> None:
        self.inner = inner
        self.path = path
        self.prompt_hash = prompt_hash

    async def observe(self, passage: PassagePublic, normalized: str, nonce: str) -> Observation:
        obs, _ = await self.observe_with_usage(passage, normalized, nonce)
        return obs

    async def observe_with_usage(
        self, passage: PassagePublic, normalized: str, nonce: str
    ) -> tuple[Observation, JudgeUsage]:
        inner = self.inner
        if isinstance(inner, UsageReportingProvider):
            obs, usage = await inner.observe_with_usage(passage, normalized, nonce)
        else:
            obs, usage = await inner.observe(passage, normalized, nonce), JudgeUsage()
        key = cassette_key(self.prompt_hash, passage.id, normalized)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "key": key,
                        "passage_id": passage.id,
                        "observation": obs.model_dump(mode="json"),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        return obs, usage
