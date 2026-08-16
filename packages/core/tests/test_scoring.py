"""Scoring — TECH_PLAN.md §8.

The three properties this file exists to defend:

* `normalize` is idempotent (Hypothesis, §8.1);
* the distance is the UNRESTRICTED Damerau-Levenshtein metric, so a reorder
  costs 1 and "swap two words then change one of them" costs 2, not 3 (§8.2);
* the op list is a faithful edit script — applying it to the original yields the
  submission exactly. The ops ARE the shareable diff, and a diff that does not
  reconstruct the text it claims to describe is a lie on the leaderboard (§8.3).
"""

from __future__ import annotations

import dataclasses

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from launder_core.scoring import (
    DEFAULT_NORMALIZE,
    WHITESPACE_CODEPOINTS,
    NormalizeConfig,
    apply_ops,
    damerau_levenshtein,
    damerau_levenshtein_ops,
    normalize,
    score,
    words,
)

# Characters chosen to hit every branch: the folds, both whitespace flavours,
# a combining mark, and the invisible characters normalization must NOT touch.
NASTY = st.sampled_from(
    [
        "\u200b",  # zero width space   - must survive
        "\ufeff",  # BOM                - must survive (JS `\s` would eat it)
        "\u00ad",  # soft hyphen        - must survive
        "\u00a0",  # nbsp               - folded to a space
        "\u2028",  # line separator     - whitespace
        "",  # vertical tab       - whitespace
        "\u2019",  # right single quote - folded
        "\u201c",  # left double quote  - folded
        "\u2014",  # em dash            - folded
        "\u2026",  # ellipsis           - folded
        "\u0301",  # combining acute    - composes under NFC
        "e",
        "a",
        " ",
        "\r\n",
        "\t",
        "study.",
        "fi",  # the ligature: NFKC bait, NFC must leave it alone
        "\u2460",  # circled one: ditto
    ]
)
TEXTS = st.one_of(st.text(), st.lists(NASTY, max_size=24).map("".join))


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
@settings(max_examples=400)
@given(TEXTS)
def test_normalize_is_idempotent(raw: str) -> None:
    once = normalize(raw)
    assert normalize(once) == once


@settings(max_examples=200)
@given(TEXTS)
def test_normalize_never_strips_the_invisible(raw: str) -> None:
    """§7.2: stripping would let the exploit succeed at the detector while the
    judge sees clean text. Rejection is `unicode_sanitation`'s job, not this
    function's."""
    once = normalize(raw)
    for ch in ("​", "­", "﻿"):
        assert once.count(ch) == raw.count(ch)


@settings(max_examples=200)
@given(TEXTS)
def test_words_round_trips_through_a_single_space(raw: str) -> None:
    assert " ".join(words(normalize(raw))) == normalize(raw)


def test_the_folds_are_exactly_the_declared_ones() -> None:
    assert normalize("\u201cwait\u201d \u2014 he said\u2026") == '"wait" - he said...'
    assert normalize("it\u2019s") == "it's"
    assert normalize("a\u00a0b") == "a b"
    assert normalize("a\r\n\r\nb") == "a b"
    assert normalize("  padded  ") == "padded"


def test_nfc_not_nfkc() -> None:
    """NFKC-bait is a MUST-FAIL eval case; folding it here would let it
    through (§8.1). `unicode_sanitation` rejects these; normalization leaves
    them exactly as the player sent them."""
    assert normalize("\ufb01ve") == "\ufb01ve"  # fi ligature survives
    assert normalize("\u2460") == "\u2460"  # circled digit one survives
    # ...but NFC still composes, which is what makes the function canonical.
    assert normalize("e\u0301") == "\u00e9"


def test_punctuation_stays_attached_to_its_word() -> None:
    """'study.' -> 'study,' costs 1, because it is one thing a player did."""
    assert words(normalize("the study. it ended")) == ("the", "study.", "it", "ended")
    assert score("the study.", "the study,").distance == 1


def test_whitespace_class_excludes_the_bom() -> None:
    """The single most likely place for the TS port to diverge: JavaScript's
    `\\s` matches U+FEFF and Python's does not. Ours matches neither JS's set
    nor Python's — it is written out."""
    assert 0xFEFF not in WHITESPACE_CODEPOINTS
    assert 0x1C not in WHITESPACE_CODEPOINTS
    assert 0xA0 in WHITESPACE_CODEPOINTS


@pytest.mark.parametrize("switch", ["lowercase", "strip_punctuation"])
def test_the_two_never_switches_cannot_be_turned_on(switch: str) -> None:
    with pytest.raises(ValueError, match="must stay False"):
        NormalizeConfig(**{switch: True})


def test_normalize_config_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        DEFAULT_NORMALIZE.nfc = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# The distance
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ([], [], 0),
        (["a"], [], 1),
        ([], ["a"], 1),
        (["a", "b", "c"], ["a", "b", "c"], 0),
        (["a", "b", "c"], ["a", "x", "c"], 1),  # substitute
        (["a", "b", "c"], ["a", "c"], 1),  # delete
        (["a", "c"], ["a", "b", "c"], 1),  # insert
        (["a", "b"], ["b", "a"], 1),  # ADJACENT TRANSPOSITION == 1
        (["the", "quick", "brown"], ["the", "brown", "quick"], 1),
        # The unrestricted property: swap two words, then change one of them.
        # OSA prices this at 3 because it forbids editing between transposed
        # elements. CONCEPT.md requires 2.
        (["a", "b", "c", "d"], ["b", "a", "c", "x"], 2),
        # A transposition with a deletion inside the span.
        (["x", "c", "y"], ["y", "x"], 2),
    ],
)
def test_known_distances(a: list[str], b: list[str], expected: int) -> None:
    assert damerau_levenshtein(a, b) == expected
    assert damerau_levenshtein(b, a) == expected, "the metric is symmetric"


def test_a_reorder_costs_one_not_two() -> None:
    """CONCEPT.md: 'a reorder counts as 1, so reordering is cheap, not free.'"""
    result = score("the committee met on Thursday", "the committee met Thursday on")
    assert result.distance == 1
    assert [op.op for op in result.ops] == ["transpose"]
    op = result.ops[0]
    assert (op.from_, op.to) == ("on", "Thursday")


def test_unrestricted_beats_optimal_string_alignment() -> None:
    """The single case that distinguishes the two algorithms (§8.2)."""
    a = ["swap", "these", "then", "edit"]
    b = ["these", "swap", "then", "changed"]
    assert damerau_levenshtein(a, b) == 2  # OSA would say 3


@pytest.mark.parametrize(
    ("original", "submission"),
    [
        ("the cat sat on the mat", "the cat sat on the mat"),
        ("the cat sat on the mat", "a cat sits on a mat"),
        ("the cat sat", ""),
        ("", "the cat sat"),
        ("one two three four five", "five four three two one"),
        ("a b c d e f g", "g f e d c b a"),
        ("repeat repeat repeat", "repeat repeat"),
        ("The committee met on Thursday.", "On Thursday, the committee met."),
    ],
)
def test_ops_reconstruct_the_submission(original: str, submission: str) -> None:
    result = score(original, submission)
    assert apply_ops(result.a_words, result.ops) == result.b_words
    assert len(result.ops) == result.distance, "every op costs exactly 1"


WORD = st.sampled_from(["the", "cat", "sat", "on", "a", "mat", "study.", "study,", "met"])


@settings(max_examples=400)
@given(st.lists(WORD, max_size=9), st.lists(WORD, max_size=9))
def test_ops_reconstruct_for_any_pair(a: list[str], b: list[str]) -> None:
    distance, ops = damerau_levenshtein_ops(a, b)
    assert apply_ops(a, ops) == tuple(b)
    assert len(ops) == distance


@settings(max_examples=200)
@given(st.lists(WORD, max_size=8), st.lists(WORD, max_size=8))
def test_distance_obeys_the_metric_axioms(a: list[str], b: list[str]) -> None:
    d = damerau_levenshtein(a, b)
    assert d == damerau_levenshtein(b, a)
    assert (d == 0) == (a == b)
    assert d <= max(len(a), len(b))


def test_backtrace_is_deterministic_and_prefers_substitution() -> None:
    """Two clients must render the same diff, so the tie-break is fixed:
    substitute/match > delete > insert > transpose (§8.2)."""
    result = score("alpha beta gamma", "alpha delta gamma")
    assert [op.model_dump() for op in result.ops] == [
        {"op": "sub", "i": 1, "j": 1, "from": "beta", "to": "delta"}
    ]
    # Same distance reachable by delete+insert; substitution must win.
    for _ in range(5):
        assert score("alpha beta gamma", "alpha delta gamma").ops == result.ops


def test_score_normalizes_both_sides() -> None:
    """Whatever the iOS keyboard did to the quotes is not an edit."""
    assert score('he said "no"', "he said “no”").distance == 0


def test_score_is_fast_enough_for_the_submit_path() -> None:
    """~200 words is the shipped passage size.

    MEASURED, and §8.2 IS OPTIMISTIC: 200x200 = 40,000 cells costs ~7.4 ms in
    CPython 3.12 on this machine (~180 ns/cell), not the "sub-millisecond in
    both Python and JS" the plan claims. It is sub-millisecond in JS, where the
    loop JITs. 7.4 ms is still nothing against a p95 latency budget of 2.5 s
    dominated by a 600-2000 ms judge call, so the algorithm stands — but a
    prefix/suffix-trimming optimisation must NOT be added casually: trimming is
    provably safe for plain Levenshtein and NOT obviously safe for the
    unrestricted variant, whose `da` table looks back across the whole of `a`
    for a transposition partner. This module is the port authority; a clever
    optimisation here is a client/server parity bug.

    This test asserts the shape (it completes, correctly), not a wall-clock
    bound, which would be flaky on shared CI."""
    original = " ".join(f"word{i}" for i in range(200))
    edited = " ".join(f"word{i}" if i % 17 else "changed" for i in range(200))
    result = score(original, edited)
    assert result.distance == 12
    assert apply_ops(result.a_words, result.ops) == result.b_words
