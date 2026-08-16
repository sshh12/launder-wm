"""Tokenizer access for the forge.

THE CANONICAL SCORING UNIT, locked down (TECH_PLAN.md §6.2):

    tokenizer(text, add_special_tokens=False)   # on the PASSAGE TEXT ALONE

No BOS, no chat template, no prompt. Never store a score computed over
prompt+completion. Every entry point here goes through `encode()`, which takes
that decision away from the caller.

Two sources, and they must agree:

* `data/build/tokenizer.json` — the original 33 MB HF file. Gitignored build
  material; present on the authoring box.
* `data/assets/gemma3-tok.v1.bin.br` — the 1.1 MB packed blob that ships. The
  browser reconstructs a tokenizer from it.

`load_tokenizer()` prefers the original and falls back to reconstructing one
from the blob, so a checkout without `data/build/` can still score cached
passages. `assert_sources_agree()` runs both over a corpus and is what
`forge verify` calls.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from launder_forge.paths import Paths, repo_paths

__all__ = ["ForgeTokenizer", "assert_sources_agree", "load_tokenizer"]


@dataclass(frozen=True, slots=True)
class ForgeTokenizer:
    """A thin wrapper that pins `add_special_tokens=False` and records provenance."""

    _tok: Any
    source: str

    def encode(self, text: str) -> list[int]:
        return list(self._tok.encode(text, add_special_tokens=False).ids)

    def encode_tokens(self, text: str) -> list[str]:
        return list(self._tok.encode(text, add_special_tokens=False).tokens)

    def decode(self, ids: list[int]) -> str:
        return str(self._tok.decode(ids, skip_special_tokens=False))

    def id_to_token(self, i: int) -> str | None:
        token: str | None = self._tok.id_to_token(i)
        return token

    @property
    def vocab_size(self) -> int:
        return int(self._tok.get_vocab_size(with_added_tokens=True))


def _tokenizers_module() -> Any:
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "the `tokenizers` package is required to tokenize. It is a declared dependency of "
            "launder-forge; run `uv sync` in the workspace root."
        ) from exc
    return Tokenizer


def _from_json_file(path: Path) -> Any:
    Tokenizer = _tokenizers_module()
    return Tokenizer.from_file(str(path))


def _from_blob(blob_path: Path, tokenizer_json: Path | None) -> Any:
    """Reconstruct a tokenizer from the packed blob.

    The blob deliberately omits `post_processor` (a few bytes of JSON the web
    build inlines). We do not need one: the canonical scoring unit never adds
    special tokens, so `post_processor: null` is not merely acceptable, it is
    the configuration that matches how the passage is scored.
    """
    from launder_forge.packtok import brotli_decompress, unpack_blob

    post: Any = None
    if tokenizer_json is not None and tokenizer_json.exists():
        post = json.loads(tokenizer_json.read_text(encoding="utf-8")).get("post_processor")
    spec = unpack_blob(brotli_decompress(blob_path.read_bytes()), post_processor=post)
    spec.pop("_spm_types", None)
    Tokenizer = _tokenizers_module()
    return Tokenizer.from_str(json.dumps(spec, ensure_ascii=False))


@lru_cache(maxsize=4)
def _cached(source: str, root: str) -> ForgeTokenizer:
    paths = Paths(Path(root))
    if source == "build":
        if not paths.tokenizer_json.exists():
            raise FileNotFoundError(
                f"{paths.tokenizer_json} is missing. It is gitignored build material; fetch it "
                "with `huggingface-cli download google/gemma-3-4b-it tokenizer.json "
                "tokenizer_config.json --local-dir data/build`, or pass source='blob' to "
                "reconstruct from the committed 1.1 MB asset instead."
            )
        return ForgeTokenizer(_from_json_file(paths.tokenizer_json), "build")
    if source == "blob":
        if not paths.tokenizer_blob.exists():
            raise FileNotFoundError(f"{paths.tokenizer_blob} is missing (it is committed)")
        return ForgeTokenizer(_from_blob(paths.tokenizer_blob, paths.tokenizer_json), "blob")
    raise ValueError(f"unknown tokenizer source {source!r}; use 'build', 'blob' or 'auto'")


def load_tokenizer(paths: Paths | None = None, source: str = "auto") -> ForgeTokenizer:
    p = paths or repo_paths()
    if source == "auto":
        source = "build" if p.tokenizer_json.exists() else "blob"
    return _cached(source, str(p.root))


def assert_sources_agree(texts: list[str], paths: Paths | None = None) -> tuple[int, int]:
    """Encode every text with both sources; return `(checked, mismatches)`.

    A disagreement here means the shipped blob and the model's own tokenizer
    part company, which desyncs the browser on keystroke zero.
    """
    p = paths or repo_paths()
    a = load_tokenizer(p, "build")
    b = load_tokenizer(p, "blob")
    mismatches = 0
    for text in texts:
        if a.encode(text) != b.encode(text):
            mismatches += 1
    return len(texts), mismatches
