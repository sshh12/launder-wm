"""Text -> token ids + character offsets, from the committed blob.

`/api/detect` addresses tokens by CHARACTER OFFSET so the browser needs no
tokenizer in SERVER mode (§5.4, §9.2), and `detector_threshold` recomputes the
reading server-side (§7.1 row 7). Both need this module.

**The one dependency, and why it is not core's.** Turning the unpacked spec into
an encoder needs the Rust `tokenizers` wheel — the same implementation the 4,031
golden vectors were generated from, so using it is the only way to get server
parity without writing a third BPE. It is imported LAZILY, inside the function,
so `import launder_core` still succeeds in a venv holding only pydantic, blake3
and numpy (§2.1's hard invariant, enforced by `make test-bare-core`). It is
declared as a dependency of `launder-serve`, which is the package that actually
calls this.

Missing dependency, missing asset and unreadable blob each raise a
`TokenizerUnavailable` naming the fix. There is no fallback path: a detector
that guesses at tokenization reports a confident number about a different text.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

from launder_core.tokenizer.blob import METASPACE, unpack_blob
from launder_core.watermark.config import data_dir

__all__ = [
    "BLOB_PATH",
    "TokenizerUnavailable",
    "char_spans",
    "encode_with_offsets",
    "load_tokenizer",
    "tokenizer_spec",
]

#: Path relative to `data/`.
BLOB_PATH: Final[str] = "assets/gemma3-tok.v1.bin.br"

_BYTE_FALLBACK = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")


class TokenizerUnavailable(RuntimeError):
    """The tokenizer cannot be built here, with the reason and the fix."""


def _brotli_decompress(raw: bytes) -> bytes:
    try:
        import brotli
    except ImportError:  # pragma: no cover - exercised only without the wheel
        try:
            import brotlicffi as brotli
        except ImportError as exc:
            raise TokenizerUnavailable(
                "the packed tokenizer is brotli-compressed and neither `brotli` nor "
                "`brotlicffi` is installed. Fix: add `brotli>=1.1` to the environment "
                "(it is already a dependency of launder-serve and launder-forge)."
            ) from exc
    return bytes(brotli.decompress(raw))


@lru_cache(maxsize=2)
def tokenizer_spec(path: str | None = None) -> str:
    """The blob as a HuggingFace `tokenizer.json` string, ready for `from_str`.

    Cached: unpacking derives 514,906 merges out of 262,144 pieces and costs a
    few hundred milliseconds. The file never changes within a process.
    """
    import json

    p = Path(path) if path is not None else data_dir() / BLOB_PATH
    if not p.is_file():
        raise TokenizerUnavailable(
            f"the packed tokenizer {p} does not exist. It is a committed asset; if this is "
            "a container, either copy data/ above the installed package or set "
            "LAUNDER_DATA_DIR. Rebuild it with `uv run forge tok pack`."
        )
    raw = p.read_bytes()
    spec = unpack_blob(_brotli_decompress(raw) if p.suffix == ".br" else raw)
    spec.pop("_spm_types", None)
    return json.dumps(spec, ensure_ascii=False)


@lru_cache(maxsize=2)
def load_tokenizer(path: str | None = None) -> Any:
    """A `tokenizers.Tokenizer` built from the committed blob."""
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise TokenizerUnavailable(
            "the `tokenizers` wheel is not installed, so text cannot be turned into "
            "token ids here. It is the same Rust implementation the 4,031 golden "
            "vectors were generated from, which is why nothing in this repo "
            "reimplements it. Fix: `uv sync` (launder-serve declares "
            "`tokenizers>=0.23,<0.24`)."
        ) from exc
    return Tokenizer.from_str(tokenizer_spec(path))


def char_spans(text: str, tokens: list[str]) -> list[tuple[int, int]]:
    """Map token pieces back to `[start, end)` offsets in `text`.

    **The offsets are UTF-16 code units, not Python code points.** They are
    consumed by `text.slice(s, e)` in the browser (§9.2 `TokenHeat`), and the
    local detector emits the same shape, so a passage containing an astral
    character (an emoji) would otherwise paint heat on the wrong words in
    SERVER mode and the right ones in LOCAL mode — a divergence that only
    appears on some passages, which is the worst kind.

    The trick that makes this exact rather than heuristic: the normalizer is
    `Replace(" " -> U+2581)`, a strict 1-char -> 1-char substitution, so
    character indices in the normalized string are identical to character
    indices in the source. We walk the NORMALIZED text's UTF-8 bytes and never
    try to undo the metaspace substitution — the step that would silently drift
    on a passage containing a literal U+2581.

    Raises rather than guessing if the pieces do not reassemble the text byte
    for byte. A silently wrong offset map paints heat on the wrong words, which
    is precisely the "plausible numbers that mean nothing" failure §4.3 is
    about.
    """
    norm = text.replace(" ", METASPACE)
    norm_bytes = norm.encode("utf-8")

    # byte offset -> UTF-16 code-unit index into `norm` (== index into `text`)
    byte_to_u16 = [0] * (len(norm_bytes) + 1)
    b = 0
    u = 0
    for ch in norm:
        cp = ord(ch)
        nb = 1 if cp < 0x80 else 2 if cp < 0x800 else 3 if cp < 0x10000 else 4
        for k in range(nb):
            byte_to_u16[b + k] = u
        b += nb
        u += 2 if cp > 0xFFFF else 1
    byte_to_u16[len(norm_bytes)] = u

    spans: list[tuple[int, int]] = []
    pos = 0
    for t, piece in enumerate(tokens):
        start = pos
        bf = _BYTE_FALLBACK.match(piece)
        if bf is not None:
            want = int(bf.group(1), 16)
            if pos >= len(norm_bytes) or norm_bytes[pos] != want:
                got = norm_bytes[pos] if pos < len(norm_bytes) else -1
                raise ValueError(
                    f"char_spans: byte-fallback token {piece} at token {t} expects byte "
                    f"0x{want:02x} but the text has 0x{got:02x} at byte {pos}. The token "
                    "stream does not reassemble the text."
                )
            pos += 1
        else:
            pb = piece.encode("utf-8")
            if norm_bytes[pos : pos + len(pb)] != pb:
                raise ValueError(
                    f"char_spans: token {t} {piece!r} does not match the text at byte {pos}. "
                    "The token stream does not reassemble the text (an <unk> token, or a "
                    "normalizer that drops characters, would do this)."
                )
            pos += len(pb)
        spans.append((byte_to_u16[start], byte_to_u16[pos]))

    if pos != len(norm_bytes):
        raise ValueError(
            f"char_spans: consumed {pos} of {len(norm_bytes)} bytes. "
            f"{len(norm_bytes) - pos} bytes of the text are covered by no token."
        )
    return spans


def encode_with_offsets(
    text: str, path: str | None = None
) -> tuple[list[int], list[tuple[int, int]]]:
    """`(token_ids, char_spans)` for SCORING.

    `add_special_tokens=False` is not a preference, it is
    `data/config/watermark.toml [scoring]`: the canonical scoring unit is the
    passage text alone — no BOS, no chat template, no prompt. A BOS here would
    shift every n-gram window by one and change every g-value in the passage.
    """
    enc = load_tokenizer(path).encode(text, add_special_tokens=False)
    return list(enc.ids), char_spans(text, list(enc.tokens))
