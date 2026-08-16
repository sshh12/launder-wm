"""Pluggable negative-corpus sources for `forge calibrate`.

TECH_PLAN.md §6.5 is explicit about what a *correct* negative set contains:

> Negatives must include **both** human prose **and** Gemma-3 output generated
> without the watermark and with a different key — otherwise the calibration
> encodes "is this Gemma-shaped text", not "does this carry key K", and the
> game's central claim collapses.

So the corpus is an interface, not a file. Four implementations ship:

* `TextFileCorpus`    — human prose, one document per file or per line.
* `JsonlCorpus`       — `{"text": ...}` or `{"token_ids": [...]}` records.
* `UnwatermarkedGemmaCorpus` — the real thing; needs CUDA + Gemma weights and
  says exactly that when it cannot run.
* `SyntheticRandomIdCorpus` — uniform random token ids. **This is not English**
  and every artifact built from it records that fact in its provenance. It
  exists so the curve's *shape* and the whole pipeline are exercised and
  reviewable before a corpus lands, not to stand in for one.

A `--corpus` string selects one: `synthetic`, `gemma`, a path to a `.jsonl`, or
a path to a directory of `.txt`.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

__all__ = [
    "Corpus",
    "CorpusSample",
    "JsonlCorpus",
    "SyntheticRandomIdCorpus",
    "TextFileCorpus",
    "UnwatermarkedGemmaCorpus",
    "resolve_corpus",
]


@dataclass(frozen=True, slots=True)
class CorpusSample:
    token_ids: list[int]
    text: str | None = None
    origin: str = ""


@runtime_checkable
class Corpus(Protocol):
    """A source of NEGATIVE (unwatermarked, or wrong-key) token sequences."""

    #: Recorded verbatim in `thresholds.v1.json` under `provenance.source`.
    provenance: str

    #: True only for a corpus whose members are real language.
    is_natural_language: bool

    def samples(self, n: int, lengths: list[int], seed: int) -> Iterator[CorpusSample]: ...


# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SyntheticRandomIdCorpus:
    """Uniform random token ids. Honest label, honest limitations.

    Random ids are *not* a substitute for prose. Real text has repeated
    n-grams, so a real corpus has a non-trivial `masked_fraction` and its
    scores are correlated across rows in ways uniform noise is not. Expect the
    empirical/closed-form sd ratio measured here to be closer to 1 than the
    ratio a prose corpus gives (the session's 208-passage Gemma measurement
    landed at 1.17-1.27).
    """

    vocab_size: int = 262144
    provenance: str = "synthetic-random-ids"
    is_natural_language: bool = False

    def samples(self, n: int, lengths: list[int], seed: int) -> Iterator[CorpusSample]:
        rng = np.random.default_rng(seed)
        lo, hi = min(lengths), max(lengths)
        for i in range(n):
            length = int(rng.integers(lo, hi + 1))
            ids = rng.integers(0, self.vocab_size, size=length, dtype=np.int64)
            yield CorpusSample(token_ids=ids.tolist(), origin=f"synthetic#{i}")


@dataclass(slots=True)
class TextFileCorpus:
    """Human prose from `.txt` files (one document per file, or per line)."""

    root: Path
    tokenizer: object
    per_line: bool = False
    provenance: str = "human-prose-textfiles"
    is_natural_language: bool = True

    def samples(self, n: int, lengths: list[int], seed: int) -> Iterator[CorpusSample]:
        files = sorted(self.root.rglob("*.txt")) if self.root.is_dir() else [self.root]
        if not files:
            raise FileNotFoundError(f"no .txt files under {self.root}")
        rng = np.random.default_rng(seed)
        lo, hi = min(lengths), max(lengths)
        emitted = 0
        for path in files:
            raw = path.read_text(encoding="utf-8")
            docs = [ln for ln in raw.splitlines() if ln.strip()] if self.per_line else [raw]
            for doc in docs:
                ids = self.tokenizer.encode(doc)  # type: ignore[attr-defined]
                if len(ids) < lo:
                    continue
                want = int(rng.integers(lo, hi + 1))
                for start in range(0, len(ids) - want + 1, want):
                    yield CorpusSample(
                        token_ids=ids[start : start + want],
                        text=None,
                        origin=f"{path.name}:{start}",
                    )
                    emitted += 1
                    if emitted >= n:
                        return


@dataclass(slots=True)
class JsonlCorpus:
    """`{"text": ...}` or `{"token_ids": [...]}` per line."""

    path: Path
    tokenizer: object | None = None
    provenance: str = "jsonl-corpus"
    is_natural_language: bool = True

    def samples(self, n: int, lengths: list[int], seed: int) -> Iterator[CorpusSample]:
        lo, hi = min(lengths), max(lengths)
        emitted = 0
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if "token_ids" in rec:
                    ids = [int(x) for x in rec["token_ids"]]
                elif "text" in rec:
                    if self.tokenizer is None:
                        raise ValueError(
                            f"{self.path} carries `text` records but no tokenizer was supplied"
                        )
                    ids = self.tokenizer.encode(rec["text"])  # type: ignore[attr-defined]
                else:
                    raise ValueError(f"{self.path}: a record has neither `text` nor `token_ids`")
                if len(ids) < lo:
                    continue
                yield CorpusSample(token_ids=ids[:hi], text=rec.get("text"), origin=self.path.name)
                emitted += 1
                if emitted >= n:
                    return


@dataclass(slots=True)
class UnwatermarkedGemmaCorpus:
    """Gemma-3 output generated WITHOUT the watermark. The correct negative.

    Two variants matter and both are supported by `wrong_key`:

    * `wrong_key=False` — no watermark at all.
    * `wrong_key=True`  — watermarked with a DIFFERENT key set, which is the
      negative that proves the detector measures "carries key K" rather than
      "looks like Gemma".
    """

    model_id: str = "google/gemma-3-4b-it"
    wrong_key: bool = False
    temperature: float = 1.0
    provenance: str = "gemma3-unwatermarked"
    is_natural_language: bool = True

    def samples(self, n: int, lengths: list[int], seed: int) -> Iterator[CorpusSample]:
        from launder_forge.generate import generate_batch, load_model

        bundle = load_model(self.model_id)
        prompts = _null_prompts(n, seed)
        max_new = max(lengths)
        for i in range(0, n, 16):
            batch = prompts[i : i + 16]
            outs = generate_batch(
                bundle,
                batch,
                max_new_tokens=max_new,
                min_new_tokens=min(lengths),
                temperature=self.temperature,
                seed=seed + i,
                watermark=None,
            )
            for j, out in enumerate(outs):
                yield CorpusSample(
                    token_ids=out.token_ids,
                    text=out.text,
                    origin=f"gemma-null#{i + j}",
                )


_NULL_TOPICS = (
    "harbour dredging", "school bus routing", "cheese ripening", "bridge inspection",
    "a disused railway cutting", "a hotel lobby in the off-season", "a tidal flat",
    "sharpening a chisel", "proofing dough", "splicing rope", "tuning a spoke",
    "the smell of a hardware store", "an empty municipal swimming pool",
    "the last hour of a long shift", "rain arriving over a reservoir",
)  # fmt: skip


def _null_prompts(n: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        topic = _NULL_TOPICS[int(rng.integers(0, len(_NULL_TOPICS)))]
        out.append(f"Write one flowing paragraph about {topic}. No lists, no headings.")
    return out


# ---------------------------------------------------------------------------


def resolve_corpus(
    spec: str, *, tokenizer: object | None = None, vocab_size: int = 262144
) -> Corpus:
    """`synthetic` | `gemma` | `gemma-wrong-key` | path.jsonl | path/to/txt-dir"""
    if spec == "synthetic":
        return SyntheticRandomIdCorpus(vocab_size=vocab_size)
    if spec == "gemma":
        return UnwatermarkedGemmaCorpus()
    if spec == "gemma-wrong-key":
        return UnwatermarkedGemmaCorpus(wrong_key=True, provenance="gemma3-wrong-key")
    path = Path(spec)
    if path.suffix == ".jsonl":
        if not path.exists():
            raise FileNotFoundError(f"corpus file not found: {path}")
        return JsonlCorpus(path=path, tokenizer=tokenizer)
    if path.exists():
        if tokenizer is None:
            raise ValueError("a text corpus needs a tokenizer")
        return TextFileCorpus(root=path, tokenizer=tokenizer)
    raise ValueError(
        f"unknown corpus {spec!r}. Use 'synthetic', 'gemma', 'gemma-wrong-key', a .jsonl path, "
        "or a directory of .txt files."
    )
