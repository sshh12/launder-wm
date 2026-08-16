"""Scoring contracts: the edit op, the score result, and `scoring.toml`.

One implementation, ported once, versioned (TECH_PLAN.md §8). The client
computes a preview off the identical algorithm; the server's number is the one
persisted, ranked and shared. `edit_budget` and `close_paraphrase` call the
same `score()`, so the budget and the scoreboard can never disagree.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_serializer

__all__ = ["EditOp", "EditOpKind", "ScoreResult", "ScoringConfig"]

EditOpKind = Literal["sub", "ins", "del", "transpose"]


class EditOp(BaseModel):
    """One word-level edit, as produced by the Damerau-Levenshtein backtrace.

    The wire spelling uses `from`, which is a Python keyword, so the field is
    `from_` and serialization is pinned by an explicit `model_serializer`.
    Validation accepts either spelling. Do not "simplify" this by dropping the
    serializer: `ops` is the shareable diff and two clients must render it
    identically (TECH_PLAN.md §8.2, golden case #8).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    op: EditOpKind
    i: int = Field(ge=0, description="index into the original word sequence")
    j: int = Field(ge=0, description="index into the submission word sequence")
    from_: str = Field(
        default="",
        validation_alias=AliasChoices("from", "from_"),
        serialization_alias="from",
    )
    to: str = Field(default="")

    @model_serializer(mode="plain")
    def _serialize(self) -> dict[str, Any]:
        return {"op": self.op, "i": self.i, "j": self.j, "from": self.from_, "to": self.to}


class ScoreResult(BaseModel):
    """`score(original, submission, cfg)` — TECH_PLAN.md §8.2.

    `distance` is unrestricted Damerau-Levenshtein (Lowrance-Wagner) over the
    word sequence with unit costs and adjacent transposition = 1. NOT Optimal
    String Alignment: OSA would price "swap two words then change one of them"
    at 3 instead of 2, and CONCEPT.md requires a reorder to cost 1.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    distance: int = Field(ge=0)
    ops: tuple[EditOp, ...] = ()
    a_words: tuple[str, ...] = ()
    b_words: tuple[str, ...] = ()


class ScoringConfig(BaseModel):
    """`data/config/scoring.toml` — normalization switches + `scoring_version`.

    `scoring_version` is hashed into the judge cache key: changing normalization
    changes scores, so it MUST invalidate caches (TECH_PLAN.md §8.3).

    `lowercase` and `strip_punctuation` are declared and pinned to False rather
    than omitted, so that turning them on is a visible diff against a comment
    explaining why you must not.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    schema_id: str = Field(default="launder.scoring/1", alias="schema")
    scoring_version: str = Field(default="sc1", min_length=1)

    nfc: bool = True
    collapse_whitespace: bool = True
    fold_smart_quotes: bool = True
    fold_dashes: bool = True
    fold_ellipsis: bool = True
    nbsp_to_space: bool = True
    lowercase: Literal[False] = False
    strip_punctuation: Literal[False] = False

    # §8.2 backtrace determinism. Fixed order, pinned by golden case #8.
    backtrace_tie_break: tuple[str, ...] = ("substitute", "delete", "insert", "transpose")
