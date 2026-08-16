"""The seams between `serve` and `launder_core`.

`serve` owns HTTP, persistence, the judge providers and the abuse ladder. It
owns *none* of the maths: normalization, the Damerau-Levenshtein distance, the
verdict derivation and the ordered gate all live in `launder_core`
(TECH_PLAN.md §3), and this module is the thin adapter layer that hands them
what they need.

The one thing core could not supply — **a tokenizer** — now exists at
`launder_core.tokenizer.encode_with_offsets`, reading the same committed blob
the browser reads, through the same Rust `tokenizers` build that generated the
4,031 golden vectors. `packages/forge/tests/test_parity_ts.py` asserts the
server and the browser produce identical token ids AND identical character
offsets on the dev passage, which is what makes the SERVER->LOCAL handover a
no-op in the view layer (§5.4). When the wheel or the asset is missing this
still raises rather than guessing: a detector that guesses at tokenization
reports a confident number about a different text, which is the precise failure
§4.3 and §4.5 exist to prevent.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

from launder_core.detect.calibration import Z_STAR, load_thresholds, z_from_score
from launder_core.detect.weighted_mean import WeightedMeanDetector
from launder_core.schemas import (
    DetectorReading,
    PassagePublic,
    ScoreResult,
    ScoringConfig,
    TokenHeat,
)
from launder_core.scoring import NormalizeConfig, normalize, score, words
from launder_core.watermark import SCORING_EOS_TOKEN_ID, compute_frame
from launder_serve.errors import CoreUnavailable

__all__ = [
    "CoreDetector",
    "CoreScorer",
    "Normalizer",
    "Scorer",
    "ServerDetector",
    "TokenReading",
    "normalize_config_from",
    "text_hash",
]


def text_hash(text: str) -> str:
    """`sha256:<hex>` over the exact text a reading describes.

    The client applies a `/api/detect` reading only if `seq >= lastSeq` AND the
    hash matches its current textarea contents (§10.2) — the one rule that
    kills needle flicker from out-of-order responses, stale worker results and
    the SERVER->LOCAL handover simultaneously.
    """
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_config_from(cfg: ScoringConfig) -> NormalizeConfig:
    """Core's frozen dataclass, built from the on-disk contract.

    `scoring.toml` is the disk contract and `NormalizeConfig` is the runtime
    one; this is the single place they are joined, so neither becomes a second
    source of truth for the other.
    """
    return NormalizeConfig(
        nfc=cfg.nfc,
        collapse_whitespace=cfg.collapse_whitespace,
        fold_smart_quotes=cfg.fold_smart_quotes,
        fold_dashes=cfg.fold_dashes,
        fold_ellipsis=cfg.fold_ellipsis,
        nbsp_to_space=cfg.nbsp_to_space,
        lowercase=cfg.lowercase,
        strip_punctuation=cfg.strip_punctuation,
    )


@dataclass(frozen=True)
class TokenReading:
    """A detector reading *with* per-token heat at CHARACTER offsets.

    `/api/detect` needs the offsets; the gate's `detector_threshold` does not,
    which is why core's `Detector` protocol returns the smaller
    `DetectorReading`. This is the superset, and the extra field is exactly the
    shape the local TS detector emits (§5.4, §9.2) so one renderer serves both
    paths.
    """

    reading: DetectorReading
    tokens: tuple[TokenHeat, ...] = ()
    n_tokens: int = 0


class Normalizer(Protocol):
    def normalize(self, text: str) -> str: ...

    def words(self, normalized: str) -> tuple[str, ...]: ...


class Scorer(Protocol):
    def score(self, original: str, submission: str) -> ScoreResult: ...


class ServerDetector(Protocol):
    """Core's `Detector`, widened with the per-token reading the API needs."""

    def read(
        self, text: str, passage: PassagePublic, calibration: str | None = None
    ) -> DetectorReading: ...

    def read_tokens(
        self, text: str, passage: PassagePublic, calibration: str | None = None
    ) -> TokenReading: ...


class CoreScorer:
    """`launder_core.score` (§8.2) — THE authority.

    `edit_budget` and `close_paraphrase` call the same function through core's
    own checks, so the budget, the fence and the scoreboard cannot disagree.
    """

    def __init__(self, scoring: ScoringConfig) -> None:
        self.config = normalize_config_from(scoring)

    def normalize(self, text: str) -> str:
        return normalize(text, self.config)

    def words(self, normalized: str) -> tuple[str, ...]:
        return words(normalized)

    def score(self, original: str, submission: str) -> ScoreResult:
        return score(original, submission, self.config)


class CoreDetector:
    """The server-side re-check that IS the win condition (§7.1 row 7).

    Tokenize with the committed blob, then hand the ids straight to core. Every
    number below is core's: this class owns the text->ids step and the
    row->token layout, and nothing else.
    """

    def __init__(self, z_star: float = Z_STAR) -> None:
        self.z_star = z_star
        self._detector = WeightedMeanDetector()

    def _encode(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        """text -> (ids, char spans). Raises with the fix rather than guessing."""
        from launder_core.tokenizer import TokenizerUnavailable, encode_with_offsets

        try:
            return encode_with_offsets(text)
        except TokenizerUnavailable as exc:
            raise CoreUnavailable(
                "encode_with_offsets",
                "launder_core/tokenizer/__init__.py",
                f"The server detector cannot tokenize: {exc}",
            ) from exc

    def read(
        self, text: str, passage: PassagePublic, calibration: str | None = None
    ) -> DetectorReading:
        return self.read_tokens(text, passage, calibration).reading

    def read_tokens(
        self, text: str, passage: PassagePublic, calibration: str | None = None
    ) -> TokenReading:
        del passage  # the reading is a function of the text and the config alone
        ids, offsets = self._encode(text)
        # eos EXPLICITLY, not by default. The browser passes the same constant
        # from `web/src/detector/config.ts`; when the two were left to their own
        # defaults (browser 1, server None) the same sentence read z 1.971 in one
        # runtime and z 0.109 in the other, and the parity gate could not see it
        # because it only ever exercised an explicit override.
        frame = compute_frame(ids, eos_token_id=SCORING_EOS_TOKEN_ID)
        result = self._detector.score(frame.g, frame.mask)

        # L5 has its own calibration bucket: code has far fewer scored tokens
        # and far lower optionality, so its sigma(T) curve is fit separately
        # (§7.6 deviation #2). The bucket name comes from the level's params.
        curve = load_thresholds(name=calibration or "default")
        z = z_from_score(result.score, result.n_scored, curve)

        # ROW -> TOKEN. Row `i` covers ids[i .. i+n-1] and its heat belongs to
        # the CURRENT token `i + n - 1`, so the leading `ngram_len - 1` tokens
        # are the final token of no window: they read neutral 0.5 and
        # `masked: true`. That is absence of evidence, not evidence of
        # innocence, and it is the shape the local detector emits — see
        # `web/src/detector/index.ts` and vectors.json's `heat_note`. Zipping
        # offsets against rows directly (which this used to do) shifts every
        # span left by four tokens and drops the last four entirely.
        n = frame.ngram_len
        heat = [0.5] * len(offsets)
        masked = [True] * len(offsets)
        for i in range(frame.n_rows):
            t = i + n - 1
            if t < len(offsets):
                heat[t] = float(result.heat[i])
                masked[t] = not bool(frame.mask[i])
        tokens = tuple(
            TokenHeat(s=int(s), e=int(e), heat=heat[t], masked=masked[t])
            for t, (s, e) in enumerate(offsets)
        )
        return TokenReading(
            reading=DetectorReading(
                score=result.score,
                z=z,
                z_star=curve.z_star,
                n_scored=result.n_scored,
                masked_fraction=result.masked_fraction,
            ),
            tokens=tokens,
            n_tokens=len(tokens),
        )
