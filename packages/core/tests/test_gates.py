"""The gate — TECH_PLAN.md §7.

The load-bearing assertion in this file is the short-circuit one: the pipeline
runs in the order `levels.toml` declares, stops at the FIRST failing check, and
reports that check. That ordering is what makes a mechanical rejection cost zero
API calls, so a test that only checked "the submission was rejected" would pass
happily while the product started paying a judge to read word salad.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import pytest

from launder_core.gates import (
    REGISTRY,
    ClaimObservation,
    Deps,
    GateContext,
    GateDataError,
    GateDependencyError,
    JudgeUnavailable,
    Observation,
    render_feedback,
    run_gate,
)
from launder_core.gates.checks.close_paraphrase import (
    content_lemmas,
    lemma,
    sentence_alignment,
    sentences,
)
from launder_core.gates.checks.detector_threshold import format_z
from launder_core.gates.checks.llm_gate import derive_verdict
from launder_core.gates.checks.unicode_sanitation import classify, homoglyph_target
from launder_core.gates.registry import UnitTestOutcome
from launder_core.levels import load_levels
from launder_core.schemas import Claim, DetectorReading, PassagePublic, SynthIDConfig

# ---------------------------------------------------------------------------
# Fixtures: a real passage, and fakes for the three injected dependencies
# ---------------------------------------------------------------------------
PASSAGE_TEXT = (
    "The committee met on Thursday to consider whether the proposal deserved another year "
    "of funding. The panel had spent twelve years watching the cutting hold its own weather, "
    "and the frost lingers there long after the rest of the valley has thawed. Nobody on the "
    "panel argued that the trial had failed, and the report they filed said so plainly in a "
    "single sentence near the end. The money was renewed for a further year without any "
    "further discussion of the matter at all by anyone present."
)

CLAIMS = (
    Claim(id="c1", text="The trial succeeded.", label="the trial succeeding", required=True),
    Claim(id="c2", text="The study ran twelve years.", label="the 12-year window", required=True),
    Claim(id="c3", text="Funding was renewed.", label="the renewed funding", required=False),
)


def make_passage(text: str = PASSAGE_TEXT, **overrides: Any) -> PassagePublic:
    payload: dict[str, Any] = {
        "id": "p_test",
        "level_id": "L2",
        "wm_config_id": SynthIDConfig().wm_config_id,
        "asset_bundle_id": "ab1:" + "44de" * 16,
        "scoring_version": "sc1",
        "text": text,
        "n_words": len(text.split()),
        "detector": {
            "expected_n_scored": 100,
            "expected_score": 0.54,
            "expected_z": 11.8,
            "g_digest": "blake3:" + "7d21" * 16,
        },
        "rules": {"min_words": 50, "locked_phrases": ["its own weather"]},
        "claims": [c.model_dump() for c in CLAIMS],
        "par": 4,
        "par_source": "authored_reference",
        "judge_prompt_id": "judge.observe.v3",
    }
    payload.update(overrides)
    return PassagePublic.model_validate(payload)


class FakeDetector:
    """Reads `z` off a table keyed by text, so a test can say 'this one clears'
    without owning a watermark."""

    def __init__(self, z: float = 0.0, z_star: float = 2.3263) -> None:
        self.z = z
        self.z_star = z_star
        self.calls = 0

    def read(
        self, text: str, passage: PassagePublic, calibration: str | None = None
    ) -> DetectorReading:
        self.calls += 1
        self.last_calibration = calibration
        return DetectorReading(score=0.5, z=self.z, z_star=self.z_star, n_scored=120)


class FakeJudge:
    name: ClassVar[str] = "fake"

    def __init__(self, observation: Observation | None = None, raises: Exception | None = None):
        self.observation = observation or Observation(natural_prose=True, claims=())
        self.raises = raises
        self.calls = 0
        self.nonces: list[str] = []

    async def observe(self, passage: PassagePublic, normalized: str, nonce: str) -> Observation:
        self.calls += 1
        self.nonces.append(nonce)
        if self.raises is not None:
            raise self.raises
        return self.observation


class FakeRunner:
    def __init__(self, outcome: UnitTestOutcome | None = None) -> None:
        self.outcome = outcome or UnitTestOutcome(passed=True)
        self.calls = 0

    def run(self, suite_id: str, source: str, timeout_ms: int, memory_mb: int) -> UnitTestOutcome:
        self.calls += 1
        return self.outcome


def all_present(claims: tuple[Claim, ...] = CLAIMS) -> Observation:
    return Observation(
        natural_prose=True,
        claims=tuple(ClaimObservation(id=c.id, present=True, how="asserted") for c in claims),
    )


def context(
    level_id: str = "L2",
    text: str = PASSAGE_TEXT,
    raw: str | None = None,
    deps: Deps | None = None,
    passage: PassagePublic | None = None,
) -> GateContext:
    levels = load_levels()
    return GateContext(
        passage=passage if passage is not None else make_passage(text),
        level=levels[level_id],
        raw=raw if raw is not None else text,
        deps=deps if deps is not None else Deps(detector=FakeDetector(), judge=FakeJudge()),
    )


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------
def test_every_check_named_by_the_plan_is_registered() -> None:
    assert set(REGISTRY) == {
        "unicode_sanitation",
        "word_floor",
        "edit_budget",
        "edit_region",
        "locked_phrase",
        "unit_test",
        "close_paraphrase",
        "detector_floor",
        "detector_threshold",
        "llm_gate",
    }


def test_only_the_judge_fails_open() -> None:
    """§7.5. A deterministic check that failed open would clear a submission
    because OUR code raised, which is how a watermark game silently becomes a
    random number generator."""
    assert [name for name, c in REGISTRY.items() if c.fail_open] == ["llm_gate"]


def test_phases_reproduce_the_plans_cost_ladder() -> None:
    order = sorted(REGISTRY, key=lambda n: REGISTRY[n].phase)
    assert order[0] == "unicode_sanitation"
    assert order[1] == "word_floor"
    assert order[-1] == "llm_gate"
    assert order[-2] == "detector_threshold"


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------
async def test_a_clean_submission_clears() -> None:
    judge = FakeJudge(all_present())
    detector = FakeDetector(z=0.0)
    ctx = context(raw=PASSAGE_TEXT, deps=Deps(detector=detector, judge=judge))
    result = await run_gate(ctx)
    assert result.cleared is True
    assert result.provisional is False
    assert result.failure is None
    assert [r.check for r in result.trace] == [c.check for c in ctx.level.checks]


async def test_the_pipeline_short_circuits_at_the_first_failure() -> None:
    """L2 = unicode, floor, budget, detector, judge. This submission busts the
    floor AND the budget AND the detector; the report must name the FLOOR."""
    judge = FakeJudge(all_present())
    detector = FakeDetector(z=99.0)
    ctx = context(raw="far too short", deps=Deps(detector=detector, judge=judge))
    result = await run_gate(ctx)

    assert result.cleared is False
    assert result.failure is not None
    assert result.failure.check == "word_floor"
    assert [r.check for r in result.trace] == ["unicode_sanitation", "word_floor"]
    assert detector.calls == 0, "a rejected submission must not pay for a detector pass"
    assert judge.calls == 0, "and it must certainly not pay for an API call"


async def test_a_mechanical_failure_costs_zero_api_calls_at_every_level() -> None:
    """The claim \u00a77.1 makes about ordering, asserted for all six levels."""
    for level_id in ("L1", "L2", "L3", "L4", "L5", "L6"):
        judge = FakeJudge(all_present())
        detector = FakeDetector(z=99.0)
        ctx = context(
            level_id=level_id,
            raw="one\u200btwo",  # zero-width character: check 1 rejects
            deps=Deps(detector=detector, judge=judge, unit_tests=FakeRunner()),
        )
        result = await run_gate(ctx)
        assert result.failure is not None and result.failure.check == "unicode_sanitation"
        assert judge.calls == 0 and detector.calls == 0, level_id


async def test_only_submissions_that_beat_the_watermark_reach_the_judge() -> None:
    judge = FakeJudge(all_present())
    ctx = context(raw=PASSAGE_TEXT, deps=Deps(detector=FakeDetector(z=42.0), judge=judge))
    result = await run_gate(ctx)
    assert result.failure is not None and result.failure.check == "detector_threshold"
    assert judge.calls == 0


async def test_a_judge_outage_clears_provisionally() -> None:
    """§7.5 fail-open: the deterministic checks already caught every mechanical
    exploit, so a false clear costs one unranked run — it does not advance the
    campaign — while a false rejection costs a player."""
    judge = FakeJudge(raises=JudgeUnavailable("the provider is down"))
    ctx = context(raw=PASSAGE_TEXT, deps=Deps(detector=FakeDetector(z=0.0), judge=judge))
    result = await run_gate(ctx)
    assert result.cleared is True
    assert result.provisional is True
    assert result.trace[-1].status == "error"


async def test_an_injection_attempt_is_never_provisional() -> None:
    judge = FakeJudge(
        Observation(natural_prose=True, contains_embedded_instructions=True, claims=())
    )
    ctx = context(raw=PASSAGE_TEXT, deps=Deps(detector=FakeDetector(z=0.0), judge=judge))
    result = await run_gate(ctx)
    assert result.cleared is False
    assert result.provisional is False
    assert result.failure is not None and result.failure.code == "injection_attempt"


async def test_a_deterministic_error_rejects_rather_than_clearing() -> None:
    ctx = context(raw=PASSAGE_TEXT, deps=Deps(detector=None, judge=FakeJudge()))
    with pytest.raises(GateDependencyError, match="win condition"):
        await run_gate(ctx)


async def test_the_trace_always_returns() -> None:
    """It powers the gate pips, which is what makes the gate legible rather
    than oracular."""
    ctx = context(raw="short", deps=Deps(detector=FakeDetector(), judge=FakeJudge()))
    result = await run_gate(ctx)
    assert len(result.trace) == 2
    assert result.trace[0].status == "pass"
    assert result.trace[1].status == "fail"


# ---------------------------------------------------------------------------
# unicode_sanitation (§7.2)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("payload", "category"),
    [
        ("wea\u200bther", "zero_width"),
        ("wea\u00adther", "soft_hyphen"),
        ("wea\u202ether", "bidi_control"),
        ("wea\ufe0fther", "variation_selector"),
        ("wea\ue000ther", "private_use"),
        ("we\u0430ther", "homoglyph"),  # Cyrillic a
        ("weath\uff45r", "homoglyph"),  # fullwidth e
        ("e\u0301\u0301\u0301ther", "combining_marks"),
    ],
)
def test_unicode_sanitation_rejects_the_whole_family(payload: str, category: str) -> None:
    counts = classify(
        payload,
        reject_categories=frozenset(
            {
                "zero_width",
                "soft_hyphen",
                "bidi_control",
                "variation_selector",
                "private_use",
                "homoglyph",
                "combining_marks",
            }
        ),
        homoglyph_policy="reject_unless_in_original",
        max_combining_marks=2,
        allowed_chars=frozenset("weather"),
    )
    assert category in counts


def test_the_nfkc_ligature_is_a_homoglyph_not_a_normalization() -> None:
    """§8.1 keeps NFKC out of `normalize()` because the eval set has NFKC-bait
    as a must-fail case. This is where that case actually fails."""
    assert homoglyph_target("\ufb01") == "fi"
    assert homoglyph_target("\u2460") == "1"
    assert homoglyph_target("e") is None
    assert homoglyph_target("\u00e9") is None, "a legitimate accent is not a lookalike"
    assert homoglyph_target("\u0434") is None, (
        "a Cyrillic letter with no ASCII twin is not one either"
    )


async def test_unicode_sanitation_rejects_rather_than_strips() -> None:
    """Stripping would let the exploit succeed at the detector while the judge
    sees clean text (§7.2)."""
    ctx = context(raw=PASSAGE_TEXT.replace("weather", "wea\u200bther"))
    result = await run_gate(ctx)
    assert result.failure is not None
    assert result.failure.check == "unicode_sanitation"
    assert result.failure.params["count"] == 1
    assert render_feedback(result.failure).message.endswith(
        "1 invisible or lookalike characters were added."
    )


async def test_a_homoglyph_already_in_the_original_is_allowed() -> None:
    """`reject_unless_in_original`: a passage that legitimately contains a
    lookalike must not become unplayable."""
    text = PASSAGE_TEXT.replace("committee", "committ\u00e9\u00e9")
    ctx = context(text=text, raw=text)
    result = await run_gate(ctx)
    assert result.trace[0].status == "pass"


async def test_unicode_sanitation_fills_the_context_for_everything_downstream() -> None:
    """It is check 1 because `normalized` and `words` do not exist until it
    runs (§7.1 row 1)."""
    ctx = context(raw="  spaced   out  " + PASSAGE_TEXT)
    assert ctx.normalized == "" and ctx.words == ()
    result = await run_gate(ctx)
    floor = next(r for r in result.trace if r.check == "word_floor")
    assert floor.params["n_words"] == len(("spaced out " + PASSAGE_TEXT).split())


def test_l5_tightens_the_homoglyph_policy() -> None:
    levels = load_levels()
    assert levels["L5"].param_for("unicode_sanitation", "homoglyph_policy") == "reject_always"
    assert levels["L1"].param_for("unicode_sanitation", "homoglyph_policy") == (
        "reject_unless_in_original"
    )


# ---------------------------------------------------------------------------
# edit_budget, word_floor, locked_phrase
# ---------------------------------------------------------------------------
async def test_edit_budget_uses_the_same_distance_as_the_scoreboard() -> None:
    from launder_core.scoring import score

    edited = PASSAGE_TEXT.replace("committee", "board").replace("proposal", "plan")
    ctx = context(raw=edited, deps=Deps(detector=FakeDetector(), judge=FakeJudge(all_present())))
    result = await run_gate(ctx)
    budget = next(r for r in result.trace if r.check == "edit_budget")
    assert budget.params["distance"] == score(PASSAGE_TEXT, edited).distance == 2


async def test_edit_budget_rejects_over_budget() -> None:
    scrambled = " ".join(f"word{i}" for i in range(120))
    ctx = context(raw=scrambled, deps=Deps(detector=FakeDetector(), judge=FakeJudge()))
    result = await run_gate(ctx)
    assert result.failure is not None and result.failure.check == "edit_budget"
    assert result.failure.params["max_word_distance"] == 12
    assert "12" in render_feedback(result.failure).message


async def test_locked_phrase_matches_after_whitespace_normalization() -> None:
    kept = PASSAGE_TEXT.replace("its own weather", "its   own\n weather")
    ctx = context(
        level_id="L3", raw=kept, deps=Deps(detector=FakeDetector(), judge=FakeJudge(all_present()))
    )
    result = await run_gate(ctx)
    locked = next(r for r in result.trace if r.check == "locked_phrase")
    assert locked.status == "pass"


async def test_locked_phrase_rejects_a_paraphrase_of_itself() -> None:
    """Never ask an LLM 'does this phrase appear verbatim' — it will
    paraphrase-match (§7.1 row 4)."""
    broken = PASSAGE_TEXT.replace("its own weather", "a weather of its own")
    ctx = context(level_id="L3", raw=broken)
    result = await run_gate(ctx)
    assert result.failure is not None and result.failure.check == "locked_phrase"
    assert result.failure.params["phrase"] == "its own weather"


async def test_a_passage_with_no_locked_phrase_on_l3_is_our_bug_not_the_players() -> None:
    passage = make_passage(text=PASSAGE_TEXT, rules={"min_words": 50, "locked_phrases": []})
    ctx = context(level_id="L3", raw=PASSAGE_TEXT, passage=passage)
    with pytest.raises(GateDataError, match="forge verify"):
        await run_gate(ctx)


# ---------------------------------------------------------------------------
# close_paraphrase (§7.7)
# ---------------------------------------------------------------------------
def test_lemmatization_is_the_committed_table() -> None:
    assert lemma("studies") == "study"
    assert lemma("study.") == "study"
    assert lemma("watching") == "watch"
    assert lemma("children") == "child"
    assert lemma("were") == "be"
    assert lemma("The") == "the"


def test_content_lemmas_drop_the_function_words() -> None:
    assert content_lemmas(("the", "panel", "was", "watching")) == frozenset({"panel", "watch"})


def test_sentence_splitting_is_deterministic() -> None:
    assert sentences("He left. She stayed. Nothing happened.") == (
        "He left.",
        "She stayed.",
        "Nothing happened.",
    )
    assert sentences("") == ()
    assert sentences("No terminator here") == ("No terminator here",)
    # A closing mark belongs to the sentence it closes, not to the next one.
    assert sentences('He said "no." She left.') == ('He said "no."', "She left.")


def test_the_sentence_splitter_is_simple_on_purpose() -> None:
    """It splits inside quoted dialogue and after "Dr.". Both are documented
    limitations, not oversights: the two metrics built on it are a COUNT DELTA
    and a match quality, and both survive a CONSISTENTLY wrong splitter as long
    as the original and the submission are split the same way. Every heuristic
    added here is one more rule the TS port has to reproduce exactly."""
    assert sentences('"Then what?" she asked.') == ('"Then what?"', "she asked.")
    assert len(sentences("Dr. Smith arrived.")) == 2


def test_sentence_alignment_catches_a_merge() -> None:
    """The metric that exists because merging three sentences into one is
    invisible to every corpus-level metric above it (§7.7)."""
    original = "The panel met. The frost lingered. The money was renewed."
    merged = "The panel met, the frost lingered and the money was renewed."
    assert sentence_alignment(original, original) == 1.0
    assert sentence_alignment(original, merged) < 0.4


async def test_close_paraphrase_rejects_a_full_rewrite() -> None:
    rewrite = (
        "Executives assembled midweek to weigh continued financial support. Over more than a "
        "decade they observed an unusual microclimate persisting in one clearing, where ice "
        "remains long past the surrounding thaw. None disputed the experiment's outcome, and "
        "their written summary stated as much briefly. Support continued for twelve more "
        "months with no debate whatsoever among those attending the session that morning."
    )
    ctx = context(level_id="L4", raw=rewrite, deps=Deps(detector=FakeDetector(), judge=FakeJudge()))
    result = await run_gate(ctx)
    assert result.failure is not None and result.failure.check == "close_paraphrase"
    assert (result.failure.code or "").startswith("close_paraphrase_")
    assert "surgery" in render_feedback(result.failure).message


async def test_close_paraphrase_admits_a_surgical_edit() -> None:
    surgical = PASSAGE_TEXT.replace("committee", "board").replace("Nobody", "No one")
    ctx = context(
        level_id="L4",
        raw=surgical,
        deps=Deps(detector=FakeDetector(), judge=FakeJudge(all_present())),
    )
    result = await run_gate(ctx)
    fence = next(r for r in result.trace if r.check == "close_paraphrase")
    assert fence.status == "pass", fence.meta


# ---------------------------------------------------------------------------
# detector_threshold and unit_test
# ---------------------------------------------------------------------------
def test_z_is_formatted_for_a_human() -> None:
    assert format_z(41.7284) == "42"
    assert format_z(2.3263) == "2.3"
    assert format_z(-0.04) == "-0.0"


async def test_l5_asks_for_its_own_calibration_bucket() -> None:
    """Deviation #2: code has far fewer scored tokens and far lower
    optionality, so its sigma(T) curve is fit separately (§7.6)."""
    detector = FakeDetector(z=0.0)
    passage = make_passage(rules={"min_words": 0, "unit_test_id": "reverse_words"})
    ctx = context(
        level_id="L5",
        raw=PASSAGE_TEXT,
        passage=passage,
        deps=Deps(detector=detector, unit_tests=FakeRunner()),
    )
    result = await run_gate(ctx)
    assert result.cleared is True
    assert detector.last_calibration == "code"
    assert "llm_gate" not in [r.check for r in result.trace]


async def test_unit_test_refuses_to_run_without_a_sandbox() -> None:
    passage = make_passage(rules={"min_words": 0, "unit_test_id": "reverse_words"})
    ctx = context(
        level_id="L5", raw=PASSAGE_TEXT, passage=passage, deps=Deps(detector=FakeDetector())
    )
    with pytest.raises(GateDependencyError, match="sandboxed runner"):
        await run_gate(ctx)


async def test_unit_test_reports_the_failing_test_by_name() -> None:
    passage = make_passage(rules={"min_words": 0, "unit_test_id": "reverse_words"})
    runner = FakeRunner(
        UnitTestOutcome(passed=False, failed_test="reverse_words on the empty string")
    )
    ctx = context(
        level_id="L5",
        raw=PASSAGE_TEXT,
        passage=passage,
        deps=Deps(detector=FakeDetector(), unit_tests=runner),
    )
    result = await run_gate(ctx)
    assert result.failure is not None
    assert render_feedback(result.failure).message == (
        "Test failed: reverse_words on the empty string."
    )


async def test_a_unit_test_timeout_gets_its_own_message() -> None:
    passage = make_passage(rules={"min_words": 0, "unit_test_id": "reverse_words"})
    runner = FakeRunner(UnitTestOutcome(passed=False, timed_out=True))
    ctx = context(
        level_id="L5",
        raw=PASSAGE_TEXT,
        passage=passage,
        deps=Deps(detector=FakeDetector(), unit_tests=runner),
    )
    result = await run_gate(ctx)
    assert result.failure is not None
    assert render_feedback(result.failure).message == "The code did not finish in 2000 ms."


# ---------------------------------------------------------------------------
# llm_gate: code computes the verdict, the model does not (§7.3, §7.4)
# ---------------------------------------------------------------------------
L1_PARAMS: Mapping[str, Any] = {
    "require_claims": "required_only",
    "max_missing_claims": 0,
    "max_added_claims": 1,
}
L4_PARAMS: Mapping[str, Any] = {
    "require_claims": "all",
    "max_missing_claims": 0,
    "max_added_claims": 0,
}


def obs(**kwargs: Any) -> Observation:
    base: dict[str, Any] = {
        "natural_prose": True,
        "claims": [{"id": c.id, "present": True, "how": "asserted"} for c in CLAIMS],
    }
    base.update(kwargs)
    return Observation.model_validate(base)


def test_injection_fails_closed_before_anything_else_is_considered() -> None:
    result = derive_verdict(
        obs(contains_embedded_instructions=True, natural_prose=False, verdict_opinion="pass"),
        L1_PARAMS,
        CLAIMS,
    )
    assert result.status == "fail"
    assert result.code == "injection_attempt"


def test_a_dropped_required_claim_is_meaning_drift() -> None:
    claims = [
        {"id": "c1", "present": True, "how": "asserted"},
        {"id": "c2", "present": False, "how": "missing"},
        {"id": "c3", "present": True, "how": "asserted"},
    ]
    result = derive_verdict(obs(claims=claims), L1_PARAMS, CLAIMS)
    assert result.code == "meaning_drift"
    assert result.params["claim_label"] == "the 12-year window"


def test_a_negated_claim_is_an_inversion_not_a_drift() -> None:
    claims = [
        {"id": "c1", "present": False, "how": "negated"},
        {"id": "c2", "present": True, "how": "asserted"},
        {"id": "c3", "present": True, "how": "asserted"},
    ]
    result = derive_verdict(obs(claims=claims), L1_PARAMS, CLAIMS)
    assert result.code == "meaning_inverted"
    assert result.params["claim_label"] == "the trial succeeding"


def test_an_optional_claim_may_be_dropped_at_l1_but_not_at_l4() -> None:
    """This is the tight fence's Layer 2: `require_claims: all` (§7.7)."""
    claims = [
        {"id": "c1", "present": True, "how": "asserted"},
        {"id": "c2", "present": True, "how": "asserted"},
        {"id": "c3", "present": False, "how": "missing"},
    ]
    assert derive_verdict(obs(claims=claims), L1_PARAMS, CLAIMS).status == "pass"
    assert derive_verdict(obs(claims=claims), L4_PARAMS, CLAIMS).code == "meaning_drift"


def test_added_claims_are_budgeted_per_level() -> None:
    added = obs(added_claims=["funding came from the state"])
    assert derive_verdict(added, L1_PARAMS, CLAIMS).status == "pass"
    rejected = derive_verdict(added, L4_PARAMS, CLAIMS)
    assert rejected.code == "meaning_added"
    assert rejected.params["added"] == "funding came from the state"


@pytest.mark.parametrize(
    "kind", ["word_salad", "keyword_soup", "repetition", "non_prose_content", "not_english"]
)
def test_each_unnatural_kind_gets_its_own_line(kind: str) -> None:
    result = derive_verdict(obs(natural_prose=False, unnatural_kind=kind), L1_PARAMS, CLAIMS)
    assert result.code == "not_natural_language"
    assert render_feedback(result).message


def test_verdict_opinion_is_telemetry_only() -> None:
    """A persistent gap between it and the derived verdict is a PROMPT bug
    worth mining, never an input to the answer (§7.4)."""
    passing = derive_verdict(obs(verdict_opinion="fail"), L1_PARAMS, CLAIMS)
    assert passing.status == "pass"
    assert passing.meta["verdict_opinion"] == "fail"


async def test_the_nonce_is_generated_per_request() -> None:
    judge = FakeJudge(all_present())
    for _ in range(3):
        await run_gate(context(raw=PASSAGE_TEXT, deps=Deps(detector=FakeDetector(), judge=judge)))
    assert len(set(judge.nonces)) == 3, "a reused nonce is a forgeable boundary tag"
    assert all(len(n) == 12 for n in judge.nonces)


# ---------------------------------------------------------------------------
# §7.7's committed lemma table — the config keys are REAL now
# ---------------------------------------------------------------------------


def test_the_lemma_tables_are_committed_files_that_are_actually_read() -> None:
    """`scoring.toml [lemma] table_path / stopwords_path` advertised two files
    that did not exist, and nothing read either key: `close_paraphrase` held the
    stopword list and the irregular-forms dict as literals. Anyone editing
    `lemma.v1.tsv` to change an L4 verdict would have changed nothing."""
    import tomllib

    from launder_core.gates.checks.close_paraphrase import (
        LEMMA_TABLE_PATH,
        STOPWORDS_PATH,
        _load_tables,
        assert_tables_match_files,
        irregular_lemmas,
        stopwords,
    )
    from launder_core.watermark.config import data_dir

    root = data_dir()
    with (root / "config" / "scoring.toml").open("rb") as fh:
        scoring = tomllib.load(fh)
    block = scoring["lemma"]
    assert block["table_path"] == f"data/{LEMMA_TABLE_PATH}"
    assert block["stopwords_path"] == f"data/{STOPWORDS_PATH}"
    # NOT "porter2": no Porter2 has ever been in this file, and naming one
    # invites somebody to drop a Porter2 library in and move every L4 verdict.
    assert block["stemmer"] == "suffix_rules_v1"

    assert (root / LEMMA_TABLE_PATH).is_file()
    assert (root / STOPWORDS_PATH).is_file()
    assert _load_tables() is not None, "the committed tables were not loaded"
    assert len(stopwords()) > 100
    assert len(irregular_lemmas()) > 50
    # The in-code fallback IS the file. Two copies of a table that decides
    # verdicts is exactly the drift lint_copy catches for copy.
    assert_tables_match_files()


def test_editing_the_committed_lemma_table_changes_the_lemma() -> None:
    """The file is the source of truth, demonstrated rather than asserted."""
    from launder_core.gates.checks.close_paraphrase import _load_tables, lemma

    _load_tables.cache_clear()
    try:
        assert lemma("children") == "child"
    finally:
        _load_tables.cache_clear()
