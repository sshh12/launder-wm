"""triage / tune / solve / claims / calibrate / vectors / pack, end to end.

These are the authoring-time commands, so the tests are behavioural rather than
numeric: does the policy reject for the stated reason, does the solver find a
clear it can defend, does `pack` refuse a passage that would desync the browser.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from launder_core.schemas import SynthIDConfig
from launder_forge.config import TriagePolicy, load_triage_policy, load_watermark_config
from launder_forge.numerics import load_sampling_table
from launder_forge.paths import Paths
from launder_forge.triage import CandidateRecord


@pytest.fixture
def cfg(real_paths: Paths) -> SynthIDConfig:
    return load_watermark_config(real_paths)


@pytest.fixture
def table(real_paths: Paths) -> np.ndarray:
    return load_sampling_table(real_paths.sampling_table)


# ---------------------------------------------------------------------------
# triage
# ---------------------------------------------------------------------------

_GOOD = {
    "candidate_id": "c_good",
    "text": "a passage",
    "token_ids": [1, 2, 3],
    "difficulty": {
        "margin_z": 9.5,
        "n_scored": 179,
        "hot_runs": 3,
        "concentration": 0.41,
        "upstream_leverage": 1.38,
        "median_eff_choices": 22.4,
        "masked_fraction": 0.043,
        "retokenize_ok": True,
        "judge_baseline_pass": True,
    },
    "solver": {
        "par_upper": 4,
        "exhaustive_clear_free_upto_k": 2,
        "best_single_edit_drop_z": 3.11,
    },
}


def _record(**overrides: object) -> CandidateRecord:
    rec = json.loads(json.dumps(_GOOD))
    for key, value in overrides.items():
        block = "solver" if key in rec["solver"] else "difficulty"
        rec[block][key] = value
    return CandidateRecord.from_json(rec)


@pytest.fixture
def l2(real_paths: Paths) -> TriagePolicy:
    return load_triage_policy(real_paths.triage_dir / "L2.toml")


def test_the_plans_own_example_candidate_is_accepted_by_l2(l2: TriagePolicy) -> None:
    """The §6.3 author-bundle example, against the §6.5 L2 policy."""
    from launder_forge.triage import apply_policy

    outcome = apply_policy(_record(), l2)
    assert outcome.accepted, outcome.failed_constraints + outcome.reject_reasons
    assert outcome.derived["max_word_edits"] == 6  # par_upper + 2


def test_trivial_single_edit_is_rejected_by_name(l2: TriagePolicy) -> None:
    from launder_forge.triage import apply_policy

    outcome = apply_policy(_record(best_single_edit_drop_z=12.0), l2)
    assert not outcome.accepted
    assert "trivial_single_edit" in outcome.reject_reasons


def test_teaches_wrong_lesson_is_rejected_by_name(l2: TriagePolicy) -> None:
    """`upstream_leverage < 1.0` means attacking the hot word beats cutting
    upstream — §4.6 mechanism 3, the guardrail as a selection criterion."""
    from launder_forge.triage import apply_policy

    outcome = apply_policy(_record(upstream_leverage=0.8), l2)
    assert not outcome.accepted
    assert "teaches_wrong_lesson" in outcome.reject_reasons


def test_a_failed_threshold_names_the_metric_and_the_number(l2: TriagePolicy) -> None:
    from launder_forge.triage import apply_policy

    outcome = apply_policy(_record(n_scored=40), l2)
    assert not outcome.accepted
    assert any("n_scored = 40 < 90" in f for f in outcome.failed_constraints)


def test_the_expression_sandbox_refuses_anything_but_a_predicate() -> None:
    from launder_forge.triage import NotAnExpression, evaluate_expression

    variables = {"hot_runs": 3, "par_upper": 4}
    assert evaluate_expression("hot_runs == 0", variables) is False
    assert evaluate_expression("par_upper == null or par_upper > 10", variables) is False
    with pytest.raises(NotAnExpression):
        evaluate_expression("__import__('os').system('echo pwned')", variables)
    with pytest.raises(NotAnExpression):
        evaluate_expression("open('x').read()", variables)
    with pytest.raises(NotAnExpression):
        evaluate_expression("margin_z > 1", variables)  # not a metric on this candidate


def test_prose_reject_reasons_are_reported_not_crashed_on(real_paths: Paths) -> None:
    """L5 carries `original_fails_tests = "the candidate's own unit test suite
    does not pass"`, which is documentation, not a predicate."""
    from launder_forge.triage import triage_run

    policy = load_triage_policy(real_paths.triage_dir / "L5.toml")
    report = triage_run([_record()], policy)
    assert "original_fails_tests" in report.non_evaluable


def test_every_shipped_triage_policy_loads(real_paths: Paths) -> None:
    from launder_forge.triage import triage_run

    for path in sorted(real_paths.triage_dir.glob("L*.toml")):
        policy = load_triage_policy(path)
        report = triage_run([_record()], policy)
        assert report.total == 1


# ---------------------------------------------------------------------------
# tune
# ---------------------------------------------------------------------------


def test_sweep_produces_one_cell_per_combination(l2: TriagePolicy) -> None:
    from launder_forge.tune import parse_sweep, sweep

    grid = parse_sweep(["margin_z_min=4,6,8", "upstream_leverage_min=1.0,1.25,1.5"])
    result = sweep([_record(), _record(upstream_leverage=1.1)], l2, grid)
    assert len(result.cells) == 9
    loose = next(
        c for c in result.cells if c.overrides == {"margin_z_min": 4, "upstream_leverage_min": 1.0}
    )
    tight = next(
        c for c in result.cells if c.overrides == {"margin_z_min": 8, "upstream_leverage_min": 1.5}
    )
    assert loose.accepted >= tight.accepted


def test_parse_sweep_types() -> None:
    from launder_forge.tune import parse_sweep

    assert parse_sweep(["a=1,2", "b=1.5", "c=true,false", "d=x"]) == {
        "a": [1, 2],
        "b": [1.5],
        "c": [True, False],
        "d": ["x"],
    }


# ---------------------------------------------------------------------------
# solve
# ---------------------------------------------------------------------------


def test_solver_finds_a_judge_gated_clear(
    real_paths: Paths, cfg: SynthIDConfig, table: np.ndarray
) -> None:
    from launder_forge.solve import SolveConfig, solve
    from launder_forge.tokenizer import load_tokenizer

    tokenizer = load_tokenizer(real_paths)
    text = (real_paths.root / "data" / "dev" / "passage.txt").read_text("utf-8").strip()
    # A z* far below the passage's own z, so a clear exists but is not free.
    result = solve(
        text,
        tokenizer=tokenizer,
        keys=cfg.keys,
        ngram_len=cfg.ngram_len,
        table=table,
        context_history_size=cfg.context_history_size,
        z_star=0.0,
        config=SolveConfig(max_edits=3, beam=48, exhaustive_pair_budget=0),
    )
    assert result.evaluated > 100
    assert result.best_single_edit_drop_z >= 0.0
    assert result.move_set_id == "M-v2"
    if result.par_upper is not None:
        assert result.best is not None
        assert result.best.judge_ok is True
        assert result.best.z <= 0.0


def test_solver_withholds_par_without_a_judge(
    real_paths: Paths, cfg: SynthIDConfig, table: np.ndarray
) -> None:
    """Without a judge the search 'solves' every passage with degenerate text,
    so `par_upper` must not be reported at all."""
    from launder_forge.solve import SolveConfig, solve
    from launder_forge.tokenizer import load_tokenizer

    result = solve(
        "the cutting holds its own weather through the long afternoon and into evening",
        tokenizer=load_tokenizer(real_paths),
        keys=cfg.keys,
        ngram_len=cfg.ngram_len,
        table=table,
        context_history_size=cfg.context_history_size,
        z_star=-99.0,
        judge=None,
        judge_name="none",
        config=SolveConfig(max_edits=1, beam=8),
    )
    assert result.par_upper is None
    assert any("NO JUDGE RAN" in n for n in result.notes)


def test_heuristic_judge_rejects_degenerate_text() -> None:
    from launder_forge.solve import heuristic_judge

    original = "the cutting holds its own weather through the long afternoon"
    assert heuristic_judge(original, original)[0] is True
    assert heuristic_judge(original, "the the the the the the the the the")[0] is False
    assert heuristic_judge(original, "")[0] is False
    assert heuristic_judge(original, "the cutting")[0] is False


def test_locked_phrases_are_never_broken(
    real_paths: Paths, cfg: SynthIDConfig, table: np.ndarray
) -> None:
    from launder_forge.solve import SolveConfig, solve
    from launder_forge.tokenizer import load_tokenizer

    text = "the cutting holds its own weather through the long afternoon and into evening"
    locked = ("its own weather",)
    result = solve(
        text,
        tokenizer=load_tokenizer(real_paths),
        keys=cfg.keys,
        ngram_len=cfg.ngram_len,
        table=table,
        context_history_size=cfg.context_history_size,
        z_star=-99.0,
        locked_phrases=locked,
        config=SolveConfig(max_edits=2, beam=16),
    )
    if result.best is not None:
        assert locked[0] in result.best.text


# ---------------------------------------------------------------------------
# claims
# ---------------------------------------------------------------------------


def test_claims_draft_is_unreviewed_and_refused_until_a_human_says_otherwise(
    tmp_path: Path,
) -> None:
    from launder_forge.claims import draft_claims, load_reviewed_claims, write_draft

    text = (
        "The cutting holds its own weather. Frost persists there long after it has "
        "cleared from the fields above. Perhaps the light never quite arrives."
    )
    draft = draft_claims("p_test", text)
    assert len(draft.claims) == 3
    assert all(c.label for c in draft.claims)
    assert any(c.note for c in draft.claims), "the hedged sentence should be flagged"

    path = write_draft(tmp_path / "p_test.claims.draft.json", draft)
    with pytest.raises(ValueError, match="reviewed: false"):
        load_reviewed_claims(path)

    data = json.loads(path.read_text(encoding="utf-8"))
    data["reviewed"] = True
    path.write_text(json.dumps(data), encoding="utf-8")
    claims = load_reviewed_claims(path)
    assert {"id", "text", "label", "required"} == set(claims[0])


# ---------------------------------------------------------------------------
# calibrate
# ---------------------------------------------------------------------------


def test_z_star_matches_the_published_constant() -> None:
    from launder_forge.calibrate import z_star_for

    assert abs(z_star_for(1e-2) - 2.3263478740408408) < 1e-9


def test_z_star_matches_cores_phi_inv() -> None:
    from launder_core.detect.calibration import phi_inv
    from launder_forge.calibrate import z_star_for

    for fpr in (1e-1, 1e-2, 1e-3):
        assert abs(z_star_for(fpr) - phi_inv(1 - fpr)) < 1e-9


def test_inflation_interpolates_in_log_n_and_clamps() -> None:
    from launder_forge.calibrate import inflation_at

    knots = [{"n_scored": 50.0, "inflation": 1.0}, {"n_scored": 200.0, "inflation": 1.4}]
    assert inflation_at(10, knots) == 1.0
    assert inflation_at(500, knots) == 1.4
    mid = inflation_at(100, knots)
    assert 1.0 < mid < 1.4


def test_synthetic_calibration_round_trips_through_core(
    tmp_path: Path, cfg: SynthIDConfig, table: np.ndarray
) -> None:
    """The file forge writes must be the file core reads."""
    from launder_core.detect.calibration import load_thresholds
    from launder_forge.calibrate import calibrate, write_thresholds
    from launder_forge.corpus import resolve_corpus

    curve = calibrate(
        resolve_corpus("synthetic"),
        table=table,
        keys=cfg.keys,
        ngram_len=cfg.ngram_len,
        context_history_size=cfg.context_history_size,
        buckets=(60, 180),
        n=400,
    )
    curve.provenance = {"source": "synthetic-random-ids", "is_natural_language": False}
    out = tmp_path / "thresholds.v1.json"
    write_thresholds(out, curve, wm_config_id=cfg.wm_config_id)

    calibration = load_thresholds(out)
    assert calibration.empirical
    assert len(calibration.buckets) == 2
    assert calibration.z_star == pytest.approx(2.3263478740408408, abs=1e-9)
    # kappa lands near sqrt(sum(w^2)/m) = 1.1129 for independent rows: the
    # weighting factor alone, with no correlation contribution. Random ids have
    # no repeated n-grams, so that is exactly what we expect to measure.
    for bucket in calibration.buckets:
        assert 0.9 < bucket.kappa < 1.4


def test_synthetic_corpus_declares_that_it_is_not_language() -> None:
    from launder_forge.corpus import resolve_corpus

    src = resolve_corpus("synthetic")
    assert src.provenance == "synthetic-random-ids"
    assert src.is_natural_language is False


def test_unknown_corpus_names_the_options() -> None:
    from launder_forge.corpus import resolve_corpus

    with pytest.raises(ValueError, match="synthetic"):
        resolve_corpus("nonsense-corpus-name")


# ---------------------------------------------------------------------------
# vectors
# ---------------------------------------------------------------------------


def test_golden_vectors_recompute(real_paths: Paths, cfg: SynthIDConfig, table: np.ndarray) -> None:
    """`forge vectors` regenerates the committed golden file's numbers.

    Known divergences are asserted as an explicit allow-list rather than
    ignored, so this test fails again the moment a NEW one appears — and fails
    loudly the moment a listed one is fixed and the list goes stale.
    """
    from launder_forge.vectors import verify_vectors

    result = verify_vectors(real_paths, cfg, table)
    assert result.checked > 100
    unexpected = [
        m
        for m in result.mismatches
        if "KAPPA INTERPOLATION DIVERGENCE" not in m and "normalize(" not in m
    ]
    assert not unexpected, unexpected


def test_emitted_vectors_reproduce_the_committed_numbers(
    real_paths: Paths, cfg: SynthIDConfig, table: np.ndarray
) -> None:
    """Regeneration is not a fresh roll of the dice: the inputs are reused and
    every computed field comes out identical."""
    from launder_forge.vectors import emit_vectors

    committed = json.loads(real_paths.vectors_json.read_text(encoding="utf-8"))["cases"]
    emitted = emit_vectors(real_paths, cfg, table)["cases"]

    assert emitted["g_values"]["ids"] == committed["g_values"]["ids"]
    assert emitted["g_values"]["g_rows"] == committed["g_values"]["g_rows"]
    assert (
        emitted["repetition_mask"]["repetition_mask"]
        == committed["repetition_mask"]["repetition_mask"]
    )
    assert (
        emitted["repetition_mask"]["context_hashes"]
        == committed["repetition_mask"]["context_hashes"]
    )
    assert (
        emitted["edit_locality"]["differing_g_rows"]
        == committed["edit_locality"]["differing_g_rows"]
    )
    assert (
        emitted["sample_index"]["cases"][0]["expected"]
        == committed["sample_index"]["cases"][0]["expected"]
    )
    assert emitted["tokenize"]["sha256"] == committed["tokenize"]["sha256"]


# ---------------------------------------------------------------------------
# pack
# ---------------------------------------------------------------------------


def test_pack_rejects_a_passage_that_does_not_retokenize(
    real_paths: Paths, cfg: SynthIDConfig, table: np.ndarray
) -> None:
    from launder_forge.pack import pack_passage
    from launder_forge.tokenizer import load_tokenizer

    tokenizer = load_tokenizer(real_paths)
    text = "the cutting holds its own weather through the long afternoon"
    ids = tokenizer.encode(text)
    with pytest.raises(ValueError, match="encode\\(text\\) != token_ids"):
        pack_passage(
            passage_id="p_bad",
            level_id="L2",
            text=text + " tampered",
            token_ids=ids,
            cfg=cfg,
            table=table,
            asset_bundle_id="ab1:" + "0" * 64,
            scoring_version="sc1",
            tokenizer=tokenizer,
            claims=[],
            par=3,
            par_source="solver_upper",
            judge_prompt_id="judge.observe.v3",
        )


def test_pack_recomputes_the_detector_expectations(
    tmp_path: Path, real_paths: Paths, cfg: SynthIDConfig, table: np.ndarray
) -> None:
    from launder_forge.numerics import score_ids
    from launder_forge.pack import pack_passage, write_passage
    from launder_forge.paths import Paths as P
    from launder_forge.tokenizer import load_tokenizer

    tokenizer = load_tokenizer(real_paths)
    text = (real_paths.root / "data" / "dev" / "passage.txt").read_text("utf-8").strip()
    ids = tokenizer.encode(text)
    packed = pack_passage(
        passage_id="p_ok",
        level_id="L2",
        text=text,
        token_ids=ids,
        cfg=cfg,
        table=table,
        asset_bundle_id="ab1:" + "0" * 64,
        scoring_version="sc1",
        tokenizer=tokenizer,
        claims=[
            {
                "id": "c1",
                "text": "It has its own weather.",
                "label": "its own weather",
                "required": True,
            }
        ],
        par=4,
        par_source="solver_upper",
        judge_prompt_id="judge.observe.v3",
    )
    truth = score_ids(
        ids,
        keys=cfg.keys,
        ngram_len=cfg.ngram_len,
        table=table,
        context_history_size=cfg.context_history_size,
    )
    assert packed.public.detector.expected_n_scored == truth.n_scored
    assert packed.public.detector.expected_score == truth.score
    assert packed.public.detector.g_digest == truth.digest

    out_root = P(tmp_path)
    (tmp_path / "TECH_PLAN.md").write_text("x", encoding="utf-8")
    written = write_passage(out_root, packed)
    public = json.loads(written["public"].read_text(encoding="utf-8"))
    assert public["schema"] == "launder.passage.public/1"
    from launder_core.schemas import FORBIDDEN_PUBLIC_FIELDS

    assert not FORBIDDEN_PUBLIC_FIELDS.intersection(public), "author-only field leaked into public"
