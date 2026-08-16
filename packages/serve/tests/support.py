"""The one test double serve still needs, plus two deterministic solvers.

`launder_core` supplies the normalizer, the scorer, the gate registry and the
verdict derivation, so the serve tests exercise the *real* ones. The detector is
real too now — `launder_core.tokenizer.encode_with_offsets` reads the committed
blob and `test_api_detect_real.py` runs `CoreDetector` end to end against
`data/golden/vectors.json`.

`StubDetector` survives that because the gate and solver tests need a detector
whose behaviour they can PREDICT: `test_a_clean_solve_clears` performs an actual
solve, and against the real detector on a fixture passage nobody has triaged,
"which words are hot" is unknown and the test would assert a coincidence. It is
a controlled instrument for those tests, not a stand-in for a missing one.

**It is not a specification of anything.** It models exactly two properties of
the real detector, because the tests depend on them and they are true:

* words from the ORIGINAL carry signal and words the player introduced carry
  almost none — which is what makes "replace a few hot words" a winning move
  and "reorder the same words" a losing one;
* `z` falls as the score falls.

Everything else about it is fiction, and a passing test here says nothing about
§4's maths. `launder`/`keyword_soup` are deterministic solvers *against this
stub*, written so the "a clean solve clears" test performs an actual solve
rather than asserting a hardcoded string that would rot with the fixture.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from typing import Final

from launder_core.detect.calibration import Z_STAR
from launder_core.schemas import DetectorReading, PassagePublic, TokenHeat
from launder_serve.engine import TokenReading

__all__ = ["StubDetector", "keyword_soup", "launder"]

_WORD_SPAN: Final[re.Pattern[str]] = re.compile(r"\S+")


def _key(word: str) -> str:
    """Hash key for a word: case- and punctuation-insensitive.

    Punctuation is stripped *for the hash only*, so repunctuating a word does
    not change its heat. What the real detector does about that move is the
    tokenizer's business, and pretending otherwise would make these tests
    sensitive to a comma.
    """
    return "".join(ch for ch in word.lower() if ch.isalnum())


def _is_hot(word: str) -> bool:
    """~12.5% of words, chosen by hash. Deterministic, never random."""
    return hashlib.sha256(_key(word).encode("utf-8")).digest()[0] < 32


class StubDetector:
    """Structurally a `launder_core.gates.registry.Detector`, plus `read_tokens`."""

    def __init__(self, z_star: float = Z_STAR) -> None:
        self.z_star = z_star

    def read(
        self, text: str, passage: PassagePublic, calibration: str | None = None
    ) -> DetectorReading:
        return self.read_tokens(text, passage, calibration).reading

    def read_tokens(
        self, text: str, passage: PassagePublic, calibration: str | None = None
    ) -> TokenReading:
        del calibration  # one bucket is enough for a stub
        original = {_key(w) for w in passage.text.split()}
        tokens: list[TokenHeat] = []
        total = 0.0
        scored = 0
        for index, match in enumerate(_WORD_SPAN.finditer(text)):
            word = match.group()
            heat = (1.0 if _is_hot(word) else 0.5) if _key(word) in original else 0.1
            # Every fifth token is "masked": a structural free win the player
            # should be able to see, and a shape the renderer must handle.
            masked = index % 5 == 4
            tokens.append(TokenHeat(s=match.start(), e=match.end(), heat=heat, masked=masked))
            if not masked:
                total += heat
                scored += 1
        score = total / scored if scored else 0.5
        z = (score - 0.5) * (scored**0.5) * 8.0
        n_tokens = len(tokens)
        return TokenReading(
            reading=DetectorReading(
                score=score,
                z=z,
                z_star=self.z_star,
                n_scored=scored,
                masked_fraction=1.0 - (scored / n_tokens) if n_tokens else 0.0,
            ),
            tokens=tuple(tokens),
            n_tokens=n_tokens,
        )


def launder(text: str, *, protect: Sequence[str] = (), edits: int = 8) -> str:
    """Replace the hottest unprotected words. A crude solver for a crude detector.

    `protect` keeps claim vocabulary intact so the judge still sees the claims —
    which is the actual shape of the game: lower the reading without losing the
    meaning.
    """
    guard = {_key(w) for phrase in protect for w in phrase.split()}
    words = text.split()
    replaced = 0
    out: list[str] = []
    for index, word in enumerate(words):
        key = _key(word)
        if replaced < edits and _is_hot(word) and key and key not in guard:
            tail = "".join(ch for ch in word if not ch.isalnum() and ch not in "'-")
            out.append(f"item{index}{tail}")
            replaced += 1
        else:
            out.append(word)
    return " ".join(out)


def keyword_soup(text: str) -> str:
    """The same words, none of the sentences. Reordered deterministically.

    The shuffle key includes each word's ORIGINAL POSITION, so repeated words
    scatter instead of clustering. Sorting on the word alone would put ten
    copies of "the" next to each other, which the judge would report as
    `repetition` — a true observation about the wrong exhibit.
    """
    words = [w.strip(".,;:!?") for w in text.split()]
    order = sorted(
        range(len(words)),
        key=lambda i: hashlib.sha256(f"{i}:{words[i]}".encode()).hexdigest(),
    )
    return " ".join(words[i] for i in order)
