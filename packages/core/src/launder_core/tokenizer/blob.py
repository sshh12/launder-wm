"""Reader for the packed Gemma-3 tokenizer blob (`data/assets/gemma3-tok.v1.bin.br`).

TECH_PLAN.md §5.1. The blob is `google/gemma-3-4b-it`'s 33,384,568-byte
`tokenizer.json` compressed to 1,192,944 bytes by throwing away the 93% of it
that is structural overhead: the 514,906 merges are not stored at all, they are
*derived* client-side by enumerating every `left + " " + right` split of every
piece whose halves are both in the vocabulary, and stored as one signed varint
delta into that candidate list.

**Why the reader lives in `launder-core` and not in `launder-forge`.** Three
runtimes have to agree on this format byte for byte: the packer (forge), the
browser (`web/src/tokenizer/blob.ts`) and the server (`/api/detect` must
tokenize the same text the browser tokenizes, or the fallback path scores a
different passage than the local path). Two Python copies of the reader would
be a third implementation of a format whose whole risk is silent divergence —
so `launder_forge.packtok` imports this module rather than repeating it, and
`web/tools/pack-check.mjs` keeps its own deliberate duplicate for the different
job of gating the packer against an independent reader.

It is pure Python: no numpy, no `tokenizers`, nothing that could break the §2.1
bare-venv invariant. Turning the spec it returns into an actual encoder needs
the `tokenizers` wheel and lives in `launder_core.tokenizer`, behind a lazy
import, so `import launder_core` still works with only pydantic + blake3 +
numpy installed.
"""

from __future__ import annotations

from typing import Any, Final

__all__ = [
    "METASPACE",
    "SPM_BYTE",
    "SPM_CONTROL",
    "SPM_NORMAL",
    "SPM_UNKNOWN",
    "SPM_USER_DEFINED",
    "VarintReader",
    "merge_candidates",
    "unpack_blob",
]

SPM_NORMAL: Final[int] = 1
SPM_UNKNOWN: Final[int] = 2
SPM_CONTROL: Final[int] = 3
SPM_USER_DEFINED: Final[int] = 4
SPM_BYTE: Final[int] = 6

#: U+2581 LOWER ONE EIGHTH BLOCK — SentencePiece's space.
METASPACE: Final[str] = "▁"


class VarintReader:
    """LEB128 + zigzag, matching the packer's writer and `blobio.js` exactly."""

    __slots__ = ("b", "o")

    def __init__(self, b: bytes) -> None:
        self.b = b
        self.o = 0

    def v(self) -> int:
        r = 0
        s = 0
        while True:
            x = self.b[self.o]
            self.o += 1
            r |= (x & 0x7F) << s
            s += 7
            if not x & 0x80:
                return r

    def sv(self) -> int:
        """Zigzag: ``-k -> 2k-1``, ``k -> 2k``."""
        u = self.v()
        return -((u + 1) // 2) if u & 1 else u // 2

    def u8(self) -> int:
        x = self.b[self.o]
        self.o += 1
        return x

    def s(self, n: int) -> str:
        """`n` BYTES of UTF-8, decoded strictly.

        Strict, and never `utf-8-sig`: Gemma-3's vocabulary contains pieces that
        *begin with* U+FEFF, and a decoder that strips a leading BOM drops 3 of
        the 514,906 merge candidates — which then shifts every candidate index
        after it and makes the blob read as corrupt. The browser hit exactly
        this (`TextDecoder` defaults to `ignoreBOM: false`); Python's
        `bytes.decode("utf-8")` is already correct and must stay that way.
        """
        out = self.b[self.o : self.o + n].decode("utf-8")
        self.o += n
        return out


def merge_candidates(pieces: list[str]) -> list[str]:
    """Every ``left + " " + right`` split of every piece whose halves are both
    in the vocabulary, in (piece index asc, split index asc) order.

    Python iterates a `str` by code point, which is what the JS original goes
    out of its way to reproduce with `Array.from`.
    """
    in_vocab = set(pieces)
    out: list[str] = []
    append = out.append
    for piece in pieces:
        ln = len(piece)
        if ln < 2:
            continue
        for s in range(1, ln):
            left = piece[:s]
            if left in in_vocab:
                right = piece[s:]
                if right in in_vocab:
                    append(left + " " + right)
    return out


def unpack_blob(buf: bytes, post_processor: Any = None) -> dict[str, Any]:
    """Raw (already brotli-decompressed) blob -> a HuggingFace `tokenizer.json`.

    `post_processor` is NOT in the blob: it is a handful of bytes of JSON that
    the web build inlines, so shipping it inside a 1.1 MB binary would be silly.
    Passing `None` yields the field as `null`, which is what SCORING wants —
    §4.1's canonical unit is `tokenizer(text)` on the passage text alone, with
    `add_special_tokens=false`, so a BOS would shift every n-gram window by one
    and change every g-value in the passage.

    The returned spec's normalizer is `Replace(" " -> U+2581)` and nothing else.
    There is deliberately no NFKC/NFKD/Precompiled step (§14.2 item 8): folding
    `ﬁ`->`fi` at tokenize time would silently repair the NFKC-bait that
    `unicode_sanitation` exists to reject.
    """
    r = VarintReader(buf)
    n = r.v()
    vocab_arr = [r.s(r.v()) for _ in range(n)]
    types = [SPM_NORMAL] * n
    nn = r.v()
    prev = 0
    for _ in range(nn):
        prev += r.v()
        types[prev] = r.u8()

    vocab = {s: i for i, s in enumerate(vocab_arr)}
    cand = merge_candidates(vocab_arr)

    m_len = r.v()
    merges: list[list[str]] = []
    for i in range(m_len):
        left, _, right = cand[i + r.sv()].partition(" ")
        merges.append([left, right])

    a_len = r.v()
    added_tokens: list[dict[str, Any]] = []
    prev = 0
    for _ in range(a_len):
        prev += r.v()
        f = r.u8()
        content = r.s(r.v()) if f & 32 else vocab_arr[prev]
        added_tokens.append(
            {
                "id": prev,
                "content": content,
                "single_word": bool(f & 1),
                "lstrip": bool(f & 2),
                "rstrip": bool(f & 4),
                "normalized": bool(f & 8),
                "special": bool(f & 16),
            }
        )
    if r.o != len(buf):
        raise ValueError(f"blob has {len(buf) - r.o} trailing bytes after the added-token table")

    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added_tokens,
        "normalizer": {"type": "Replace", "pattern": {"String": " "}, "content": METASPACE},
        "pre_tokenizer": {
            "type": "Split",
            "pattern": {"String": " "},
            "behavior": "MergedWithPrevious",
            "invert": False,
        },
        "post_processor": post_processor,
        "decoder": {
            "type": "Sequence",
            "decoders": [
                {"type": "Replace", "pattern": {"String": METASPACE}, "content": " "},
                {"type": "ByteFallback"},
                {"type": "Fuse"},
            ],
        },
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": "<unk>",
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": True,
            "byte_fallback": True,
            "ignore_merges": False,
            "vocab": vocab,
            "merges": merges,
        },
        "_spm_types": types,
    }
