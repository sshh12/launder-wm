"""Passage contracts: what ships, what stays in the repo, and the guardrail.

`PassagePublic` is the single most consequential schema in the project. It is
the enforcement point for CONCEPT.md's critical guardrail — *never teach "spot
the fancy AI word"* — expressed as a hard constraint on the data contract
rather than a copy guideline (TECH_PLAN.md §4.6 mechanism 2).

The generation-time optionality fields (`entropy_nats`, `eff_choices`,
`g_mass`, `wm_boost`, `top1_prob`, `topk_alts`) exist only in `*.author.json`.
If they ever reached the browser somebody would eventually colour the heat
mirror by generation-time entropy — stale after keystroke one, and it teaches
exactly the false heuristic the game exists to refute. `extra="forbid"` plus
the named-field check below make that a load-time failure, and a CI test
asserts none of those keys appears in any `*.public.json`.
"""

from __future__ import annotations

from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from launder_core.schemas.watermark import AssetBundleId, Blake3Digest, WmConfigId

__all__ = [
    "FORBIDDEN_PUBLIC_FIELDS",
    "Claim",
    "DetectorExpectation",
    "IntroConfig",
    "PassageAuthor",
    "PassageDifficulty",
    "PassageOptionality",
    "PassageProvenance",
    "PassagePublic",
    "PassageRules",
    "PassageServer",
    "ReferenceSolution",
    "SolverReport",
]

#: Fields that must NEVER appear in a shipped passage. Design requirement, not
#: bandwidth (TECH_PLAN.md §6.3 table, §4.6 mechanism 2). Imported by the CI
#: test and by `forge pack` so there is exactly one list.
FORBIDDEN_PUBLIC_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "entropy_nats",
        "eff_choices",
        "g_mass",
        "wm_boost",
        "top1_prob",
        "topk_alts",
        # Not guardrail fields, but equally not shippable:
        "g_values",  # stale after keystroke one; the browser recomputes
        "per_token_heat",  # ditto
        "reference_solution",  # the answer key
        "solver",  # the answer key's provenance
        "optionality",  # the author-side container for the six above
        "difficulty",  # authoring metrics, includes median_eff_choices
    }
)


class DetectorExpectation(BaseModel):
    """Four conformance assertions, not display data (TECH_PLAN.md §6.3).

    On load the client recomputes the pristine passage and must reproduce all
    four. Mismatch means the local detector is broken -> refuse local detection,
    fall back to POST /api/detect, show a visible banner. `g_digest` is the
    highest-value field in the whole schema.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    expected_n_scored: int = Field(ge=0)
    expected_score: float = Field(ge=0.0, le=1.0)
    expected_z: float
    g_digest: Blake3Digest


class PassageRules(BaseModel):
    """Per-passage rule payload consumed by the level's checks."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    min_words: int = Field(default=50, ge=0)
    max_word_edits: int | None = Field(default=None, ge=0)
    locked_phrases: tuple[str, ...] = ()
    unit_test_id: str | None = None


class Claim(BaseModel):
    """One atomic factual claim. `label` is authored offline, which is why gate
    feedback can name a dropped claim without ever quoting the player's text
    back at them (TECH_PLAN.md §7.4)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1, max_length=32)
    text: str = Field(min_length=1)
    label: str = Field(min_length=1, description="player-facing noun phrase, authored offline")
    required: bool = False


class IntroConfig(BaseModel):
    """Tutorial passage only: cumulative prefix z of a REAL passage, so the
    length slider is a literal instrument reading and not a cartoon."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    prefix_z: tuple[float, ...] = Field(min_length=2)
    word_index: tuple[int, ...] = ()

    @model_validator(mode="after")
    def _aligned(self) -> IntroConfig:
        if self.word_index and len(self.word_index) != len(self.prefix_z):
            raise ValueError("intro.word_index must align 1:1 with intro.prefix_z")
        return self


class PassagePublic(BaseModel):
    """`data/passages/<id>.public.json` — THIS IS WHAT SHIPS. 3-6 KB.

    `extra="forbid"` is the guardrail. Do not relax it; do not add a field here
    without checking it against the §6.3 "what deliberately does NOT ship"
    table.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    schema_id: Literal["launder.passage.public/1"] = Field(
        default="launder.passage.public/1", alias="schema"
    )
    id: str = Field(min_length=1, max_length=64)
    level_id: str = Field(pattern=r"^L[1-9][0-9]*$")

    wm_config_id: WmConfigId
    asset_bundle_id: AssetBundleId
    scoring_version: str = Field(min_length=1, max_length=32)

    text: str = Field(min_length=1)
    n_words: int = Field(ge=1)

    detector: DetectorExpectation
    rules: PassageRules = PassageRules()
    claims: tuple[Claim, ...] = ()

    par: int = Field(ge=1)
    par_source: Literal["authored_reference", "solver_upper", "first_n_plays"]
    judge_prompt_id: str = Field(min_length=1)

    intro: IntroConfig | None = None

    @model_validator(mode="before")
    @classmethod
    def _reject_author_only_fields(cls, data: Any) -> Any:
        """Fail with a pointed message, before the generic extra="forbid" error.

        `extra="forbid"` alone would say 'Extra inputs are not permitted', which
        is true but does not tell the next person WHY this particular key is a
        product bug rather than a typo.
        """
        if isinstance(data, dict):
            leaked = sorted(FORBIDDEN_PUBLIC_FIELDS.intersection(data))
            if leaked:
                raise ValueError(
                    f"author-only field(s) {leaked} found in a public passage. These are "
                    "generation-time fields; shipping them tempts a heat mirror coloured by "
                    "entropy, which teaches the one heuristic CONCEPT.md forbids. They belong "
                    "in <id>.author.json only. See TECH_PLAN.md §4.6."
                )
        return data

    @field_validator("claims")
    @classmethod
    def _claim_ids_unique(cls, v: tuple[Claim, ...]) -> tuple[Claim, ...]:
        ids = [c.id for c in v]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate claim ids in passage: {ids}")
        return v

    @model_validator(mode="after")
    def _locked_phrases_present(self) -> PassagePublic:
        for phrase in self.rules.locked_phrases:
            if phrase not in self.text:
                raise ValueError(
                    f"locked phrase {phrase!r} does not appear in the passage text; "
                    "the level would be unwinnable"
                )
        return self


class PassageServer(BaseModel):
    """`data/passages/<id>.server.json` — the packed sidecar (TECH_PLAN.md §11.3).

    The Docker image excludes `*.author.json` (it holds the answer key) but the
    server still needs exactly two things from it. `forge pack` writes them here.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    schema_id: Literal["launder.passage.server/1"] = Field(
        default="launder.passage.server/1", alias="schema"
    )
    id: str = Field(min_length=1, max_length=64)
    claims: tuple[Claim, ...] = ()
    par: int = Field(ge=1)


# ---------------------------------------------------------------------------
# Author-side (repo only, never served)
# ---------------------------------------------------------------------------


class ReferenceSolution(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    word_edits: int = Field(ge=0)
    ops: tuple[dict[str, Any], ...] = ()
    resulting_z: float
    judge_verdict: dict[str, Any] = Field(default_factory=dict)
    author_note: str = ""


class SolverReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    move_set_id: str = "M-v2"
    par_upper: int | None = Field(default=None, ge=1)
    #: Exhaustive WITHIN the declared move set M only. Never a theorem about
    #: the game — the move space is unbounded (TECH_PLAN.md §6.4).
    exhaustive_clear_free_upto_k: int = Field(default=0, ge=0)
    best_single_edit_drop_z: float = 0.0
    clears_at_k: dict[str, bool] = Field(default_factory=dict)
    judge_rejected_solutions: int = Field(default=0, ge=0)
    search_seconds: float = Field(default=0.0, ge=0.0)


class PassageDifficulty(BaseModel):
    """The triage metrics (TECH_PLAN.md §6.5). Every threshold applied to these
    is a hypothesis nobody has measured; the first 400-candidate run is the
    experiment that produces them."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    margin_z: float
    n_scored: int = Field(ge=0)
    hot_runs: int = Field(ge=0)
    concentration: float = Field(ge=0.0, le=1.0)
    upstream_leverage: float
    median_eff_choices: float = Field(ge=0.0)
    masked_fraction: float = Field(ge=0.0, le=1.0)
    best_single_edit_drop_z: float | None = None
    locked_share: float | None = Field(default=None, ge=0.0, le=1.0)
    retokenize_ok: bool = True
    judge_baseline_pass: bool = True


class PassageOptionality(BaseModel):
    """The forbidden six, in the one file allowed to hold them."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    entropy_nats: tuple[float, ...] = ()
    eff_choices: tuple[float, ...] = ()
    g_mass: tuple[float, ...] = ()
    wm_boost: tuple[float, ...] = ()
    top1_prob: tuple[float, ...] = ()


class PassageProvenance(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt_rendered: str = ""
    prompt_rendered_sha256: str = ""
    gen_params: dict[str, Any] = Field(default_factory=dict)
    model_id: str = "google/gemma-3-4b-it"
    model_revision: str = ""
    dtype: str = "bfloat16"
    env: dict[str, Any] = Field(default_factory=dict)


class PassageAuthor(BaseModel):
    """`data/passages/<id>.author.json` — repo only, never served.

    Excluded from the Docker image (`.dockerignore`). Holds the answer key, the
    solver trace and the generation-time optionality arrays.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    schema_id: Literal["launder.passage.author/1"] = Field(
        default="launder.passage.author/1", alias="schema"
    )
    id: str = Field(min_length=1, max_length=64)
    candidate_id: str = ""
    run_id: str = ""

    reference_solution: ReferenceSolution | None = None
    solver: SolverReport = SolverReport()
    difficulty: PassageDifficulty | None = None
    optionality: PassageOptionality = PassageOptionality()
    topk_alts: tuple[tuple[int, ...], ...] = ()
    provenance: PassageProvenance = PassageProvenance()

    #: Duplicated from the public bundle so `forge verify` can cross-check them
    #: without loading two files.
    claims: tuple[Claim, ...] = ()
    par: int | None = Field(default=None, ge=1)

    def to_server_sidecar(self) -> PassageServer:
        if self.par is None:
            raise ValueError(f"passage {self.id}: par is required to emit a server sidecar")
        return PassageServer(id=self.id, claims=self.claims, par=self.par)
