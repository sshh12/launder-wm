"""`forge tok pack` — the Gemma-3 tokenizer blob packer.

A direct port of the measured, working `pack()` in the session scratchpad's
`e2e.js`. The format is not invented here; it is reproduced. The binding
constraint is byte-exactness: `data/assets/gemma3-tok.v1.bin.br` is hashed into
`asset_bundle_id`, so a packer that produces a *valid but different* blob
silently invalidates every passage. `forge tok pack --check` therefore compares
bytes, not semantics, and `packages/forge/tests/test_packtok.py` runs it.

THE FORMAT (all integers LEB128 varints, little-endian groups of 7 bits)
------------------------------------------------------------------------
    v   n_pieces
    n x { v byte_len, raw utf-8 piece }              # id order, dense 0..n-1
    v   n_non_normal
    n x { v delta_id, u8 sentencepiece_type }        # sparse, delta-coded
    v   n_merges
    m x { zigzag_varint (candidate_index - merge_index) }
    v   n_added_tokens
    a x { v delta_id, u8 flags, [v len, raw utf-8] if flags & 32 }

`flags` = single_word|1, lstrip|2, rstrip|4, normalized|8, special|16,
out-of-vocab-content|32.

The merge list — 514,906 pairs, ~13 MB as JSON — is stored as a permutation of
a *canonically derived* candidate list: for every piece in id order, every
split point at which both halves are themselves in the vocabulary. The reader
re-derives the identical candidate list and needs only the offsets. That is the
entire reason the blob is 1.1 MB instead of 33 MB.

WHAT THE ORIGINAL READ THAT WE DO NOT HAVE
------------------------------------------
`e2e.js` read the SentencePiece `tokenizer.model` protobuf for the piece list
and the per-piece SPM type. `tokenizer.model` is not in the repo (it is not
redistributable build material we kept), so both are derived from
`data/build/tokenizer.json` + `tokenizer_config.json` instead:

* pieces        = `model.vocab` inverted; asserted dense and complete.
* type 6 BYTE   = the 256 `<0xHH>` pieces.
* type 2 UNK    = `model.unk_token`.
* type 3 CONTROL= the three SentencePiece control symbols, `SPM_CONTROL_PIECES`.
* type 4 USER   = every other `added_tokens` entry with an in-vocabulary id.
* type 1 NORMAL = everything else.

That derivation is not asserted to be "how SentencePiece would label them" —
it is asserted to reproduce the committed blob byte for byte, which is a
stronger and directly testable claim.

DO NOT take the control set from `tokenizer_config.json`. Measured: on
`gemma-3-4b-it` the config's `eos_token` is `<end_of_turn>` (a chat-template
artifact of the instruct tune), while the SentencePiece model types `<eos>` as
CONTROL and `<end_of_turn>` as USER_DEFINED. Deriving controls from the config
flips exactly those two bytes and the blob stops matching — which is how this
was found.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

__all__ = [
    "PackResult",
    "brotli_compress",
    "brotli_decompress",
    "derive_pieces",
    "pack_blob",
    "pack_tokenizer",
    "unpack_blob",
]

#: SentencePiece piece types, as they appear in the blob's sparse type table.
SPM_NORMAL: Final[int] = 1
SPM_UNKNOWN: Final[int] = 2
SPM_CONTROL: Final[int] = 3
SPM_USER_DEFINED: Final[int] = 4
SPM_BYTE: Final[int] = 6

_BYTE_PIECE = re.compile(r"^<0x[0-9A-F]{2}>$")

#: The SentencePiece *control symbols* of the Gemma-3 model, in id order
#: (0, 1, 2). These are a property of the trained SPM model, NOT of the chat
#: tune, which is why they are a constant here rather than a lookup into
#: `tokenizer_config.json` — see the module docstring.
SPM_CONTROL_PIECES: Final[tuple[str, ...]] = ("<pad>", "<eos>", "<bos>")

#: Node's `zlib.brotliCompressSync` was used to produce the committed blob with
#: quality 11 and a 24-bit window. Python's `brotli` at the same settings
#: produces identical bytes (verified); node is the fallback when the Python
#: binding is absent from the venv.
BROTLI_QUALITY: Final[int] = 11
BROTLI_LGWIN: Final[int] = 24


# ---------------------------------------------------------------------------
# varint writer / reader
# ---------------------------------------------------------------------------


class _W:
    __slots__ = ("_parts",)

    def __init__(self) -> None:
        self._parts: list[bytes] = []

    def v(self, x: int) -> None:
        if x < 0:
            raise ValueError(f"varint must be non-negative, got {x}")
        out = bytearray()
        while x >= 0x80:
            out.append((x & 0x7F) | 0x80)
            x >>= 7
        out.append(x)
        self._parts.append(bytes(out))

    def sv(self, x: int) -> None:
        """Zigzag: ``-k -> 2k-1``, ``k -> 2k``. Matches ``e2e.js`` exactly."""
        self.v(-x * 2 - 1 if x < 0 else x * 2)

    def u8(self, x: int) -> None:
        self._parts.append(bytes((x & 0xFF,)))

    def raw(self, b: bytes) -> None:
        self._parts.append(b)

    def buf(self) -> bytes:
        return b"".join(self._parts)


class _R:
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
        u = self.v()
        return -((u + 1) // 2) if u & 1 else u // 2

    def u8(self) -> int:
        x = self.b[self.o]
        self.o += 1
        return x

    def s(self, n: int) -> str:
        out = self.b[self.o : self.o + n].decode("utf-8")
        self.o += n
        return out


# ---------------------------------------------------------------------------
# brotli, with a node fallback
# ---------------------------------------------------------------------------


def _python_brotli() -> Any | None:
    for name in ("brotli", "brotlicffi"):
        try:
            return __import__(name)
        except ImportError:
            continue
    return None


def _node() -> str:
    from shutil import which

    node = which("node")
    if node is None:
        raise RuntimeError(
            "brotli is unavailable: neither the `brotli` nor `brotlicffi` Python package is "
            "installed in this environment, and `node` is not on PATH either. Install one of "
            "them (`brotli` belongs in packages/forge/pyproject.toml) or install Node 18+."
        )
    return node


def _node_brotli(data: bytes, *, decompress: bool) -> bytes:
    node = _node()
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "in.bin"
        dst = Path(td) / "out.bin"
        src.write_bytes(data)
        op = (
            "zlib.brotliDecompressSync(b)"
            if decompress
            else (
                "zlib.brotliCompressSync(b,{params:{"
                f"[zlib.constants.BROTLI_PARAM_QUALITY]:{BROTLI_QUALITY},"
                f"[zlib.constants.BROTLI_PARAM_LGWIN]:{BROTLI_LGWIN},"
                "[zlib.constants.BROTLI_PARAM_SIZE_HINT]:b.length}})"
            )
        )
        script = (
            "const fs=require('fs'),zlib=require('zlib');"
            f"const b=fs.readFileSync({json.dumps(str(src))});"
            f"fs.writeFileSync({json.dumps(str(dst))},{op});"
        )
        subprocess.run([node, "-e", script], check=True, capture_output=True)
        return dst.read_bytes()


def brotli_compress(data: bytes) -> bytes:
    """Quality 11, 24-bit window, size hint = input length.

    Verified byte-identical between Node 24's zlib and the Python `brotli`
    binding on the 3,634,910-byte Gemma blob.
    """
    mod = _python_brotli()
    if mod is not None:
        return bytes(mod.compress(data, quality=BROTLI_QUALITY, lgwin=BROTLI_LGWIN))
    return _node_brotli(data, decompress=False)


def brotli_decompress(data: bytes) -> bytes:
    mod = _python_brotli()
    if mod is not None:
        return bytes(mod.decompress(data))
    return _node_brotli(data, decompress=True)


# ---------------------------------------------------------------------------
# piece derivation
# ---------------------------------------------------------------------------


def _token_str(value: Any) -> str | None:
    """tokenizer_config spells special tokens either as a bare string or as an
    `AddedToken` dict; accept both and say so if it is neither."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and isinstance(value.get("content"), str):
        return str(value["content"])
    raise ValueError(f"unrecognised special-token spelling in tokenizer_config: {value!r}")


def derive_pieces(hf: dict[str, Any], cfg: dict[str, Any]) -> tuple[list[str], dict[int, int]]:
    """`(pieces_in_id_order, {id: spm_type})` for every non-NORMAL piece."""
    vocab: dict[str, int] = hf["model"]["vocab"]
    n = len(vocab)
    pieces: list[str | None] = [None] * n
    for piece, idx in vocab.items():
        if not 0 <= idx < n:
            raise ValueError(f"vocab id {idx} out of range for a {n}-entry vocabulary")
        if pieces[idx] is not None:
            raise ValueError(f"vocab id {idx} is claimed by two pieces")
        pieces[idx] = piece
    missing = [i for i, p in enumerate(pieces) if p is None]
    if missing:
        raise ValueError(f"vocab is not dense: {len(missing)} ids unassigned, first {missing[:5]}")

    dense: list[str] = [p for p in pieces if p is not None]

    types: dict[int, int] = {}
    for i, piece in enumerate(dense):
        if _BYTE_PIECE.match(piece):
            types[i] = SPM_BYTE

    unk: str | None = hf["model"].get("unk_token") or _token_str(cfg.get("unk_token"))
    controls = set(SPM_CONTROL_PIECES)
    absent = controls.difference(vocab)
    if absent:
        raise ValueError(
            f"control symbols {sorted(absent)} are not in this vocabulary. "
            "SPM_CONTROL_PIECES describes the Gemma-3 SentencePiece model; a different "
            "tokenizer needs its own control set, and the packed blob would change."
        )

    for added in hf.get("added_tokens", []):
        idx = int(added["id"])
        if idx >= n:
            continue  # out-of-vocabulary added token; carried in section 4 instead
        content = added["content"]
        if content == unk:
            types[idx] = SPM_UNKNOWN
        elif content in controls:
            types[idx] = SPM_CONTROL
        else:
            types.setdefault(idx, SPM_USER_DEFINED)

    return [p for p in pieces if p is not None], types


def _candidates(pieces: list[str]) -> list[str]:
    """Every ``left + ' ' + right`` split of every piece whose halves are both
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


# ---------------------------------------------------------------------------
# pack / unpack
# ---------------------------------------------------------------------------


def pack_blob(hf: dict[str, Any], cfg: dict[str, Any]) -> bytes:
    """The raw (uncompressed) blob."""
    pieces, types = derive_pieces(hf, cfg)
    w = _W()

    # 1. vocab strings, id order, length-prefixed UTF-8
    w.v(len(pieces))
    for piece in pieces:
        b = piece.encode("utf-8")
        w.v(len(b))
        w.raw(b)

    # 2. non-normal piece types, delta-coded
    nn = sorted(types.items())
    w.v(len(nn))
    prev = 0
    for idx, t in nn:
        w.v(idx - prev)
        w.u8(t)
        prev = idx

    # 3. merge order as a permutation of the canonical candidate set.
    #    `key` keeps the LAST index for a duplicated candidate string, exactly
    #    as the JS `Map.set` loop does. The pairs are identical either way, so
    #    the reader reconstructs the same merge.
    key = {c: i for i, c in enumerate(_candidates(pieces))}
    merges = hf["model"]["merges"]
    w.v(len(merges))
    for i, merge in enumerate(merges):
        left, right = (merge[0], merge[1]) if isinstance(merge, (list, tuple)) else merge.split(" ")
        k = left + " " + right
        cand_index = key.get(k)
        if cand_index is None:
            raise ValueError(
                f"merge #{i} {k!r} is not derivable from the vocabulary: one of its halves is "
                "not itself a piece. The candidate derivation and the merge list disagree, "
                "which means this tokenizer.json is not the one the format was measured on."
            )
        w.sv(cand_index - i)

    # 4. added tokens
    added_tokens = hf.get("added_tokens", [])
    w.v(len(added_tokens))
    prev = 0
    n_pieces = len(pieces)
    for a in added_tokens:
        idx = int(a["id"])
        w.v(idx - prev)
        prev = idx
        oov = not (idx < n_pieces and pieces[idx] == a["content"])
        flags = (
            (1 if a.get("single_word") else 0)
            | (2 if a.get("lstrip") else 0)
            | (4 if a.get("rstrip") else 0)
            | (8 if a.get("normalized") else 0)
            | (16 if a.get("special") else 0)
            | (32 if oov else 0)
        )
        w.u8(flags)
        if oov:
            b = str(a["content"]).encode("utf-8")
            w.v(len(b))
            w.raw(b)
    return w.buf()


def unpack_blob(buf: bytes, post_processor: Any = None) -> dict[str, Any]:
    """The reader, ported from `blobio.js`. Used by `--check` and by `verify`.

    `post_processor` is NOT in the blob: it is a handful of bytes of JSON that
    the web build inlines, so shipping it inside a 1.1 MB binary would be
    silly. Passing `None` yields the field as `null`.
    """
    r = _R(buf)
    n = r.v()
    vocab_arr = [r.s(r.v()) for _ in range(n)]
    types = [SPM_NORMAL] * n
    nn = r.v()
    prev = 0
    for _ in range(nn):
        prev += r.v()
        types[prev] = r.u8()

    vocab = {s: i for i, s in enumerate(vocab_arr)}
    cand = _candidates(vocab_arr)

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
        "normalizer": {"type": "Replace", "pattern": {"String": " "}, "content": "▁"},
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
                {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
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


@dataclass(frozen=True, slots=True)
class PackResult:
    raw: bytes
    compressed: bytes
    n_pieces: int
    n_merges: int
    n_added: int
    n_non_normal: int
    roundtrip_errors: int


def pack_tokenizer(tokenizer_json: Path, tokenizer_config_json: Path) -> PackResult:
    """Pack, compress, and prove the blob reconstructs the inputs exactly.

    `roundtrip_errors` counts structural disagreements against the source
    `tokenizer.json`: vocab entries, merge pairs and added-token records. §6.4
    requires `forge tok pack` to assert 0.
    """
    for p in (tokenizer_json, tokenizer_config_json):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} is missing. It is gitignored build material (33 MB); fetch it with "
                "`huggingface-cli download google/gemma-3-4b-it tokenizer.json "
                "tokenizer_config.json --local-dir data/build`. The packed output "
                "data/assets/gemma3-tok.v1.bin.br IS committed."
            )
    hf = json.loads(tokenizer_json.read_text(encoding="utf-8"))
    cfg = json.loads(tokenizer_config_json.read_text(encoding="utf-8"))

    raw = pack_blob(hf, cfg)
    compressed = brotli_compress(raw)
    rebuilt = unpack_blob(raw, post_processor=hf.get("post_processor"))

    errors = 0
    for piece, idx in hf["model"]["vocab"].items():
        if rebuilt["model"]["vocab"].get(piece) != idx:
            errors += 1
    src_merges = hf["model"]["merges"]
    if len(src_merges) != len(rebuilt["model"]["merges"]):
        errors += abs(len(src_merges) - len(rebuilt["model"]["merges"]))
    for a, b in zip(src_merges, rebuilt["model"]["merges"], strict=False):
        pair = (a[0], a[1]) if isinstance(a, (list, tuple)) else tuple(a.split(" "))
        if tuple(b) != pair:
            errors += 1
    for src, got in zip(hf.get("added_tokens", []), rebuilt["added_tokens"], strict=False):
        if any(src.get(k) != got.get(k) for k in got):
            errors += 1

    pieces, types = derive_pieces(hf, cfg)
    return PackResult(
        raw=raw,
        compressed=compressed,
        n_pieces=len(pieces),
        n_merges=len(src_merges),
        n_added=len(hf.get("added_tokens", [])),
        n_non_normal=len(types),
        roundtrip_errors=errors,
    )


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - debug entry point
    from launder_forge.paths import repo_paths

    p = repo_paths()
    res = pack_tokenizer(p.tokenizer_json, p.tokenizer_config_json)
    committed = p.tokenizer_blob.read_bytes()
    same = res.compressed == committed
    print(
        f"raw={len(res.raw)} br={len(res.compressed)} committed={len(committed)} identical={same}"
    )
    return 0 if same and res.roundtrip_errors == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
