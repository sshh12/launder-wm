"""`forge vectors` — the one golden file, three runners (TECH_PLAN.md §4.5).

`data/golden/vectors.json` is consumed by **pytest** (Python core), **vitest**
(the TS port) and **`forge verify`**, and `data/golden/CHECKSUM` is asserted in
CI. Regenerating goldens is therefore always a reviewed diff and never a way to
turn a red test green — which is why the default mode of this module is
`verify`, not `write`.

Two modes:

* `verify_vectors()` — recompute every case from `data/assets/` and the
  committed config and assert the file's numbers. This is the guarantee that
  matters: the file is *regenerable*, and any drift between the assets and the
  golden values is a loud failure.
* `emit_vectors()` — rebuild the file. The random-looking inputs (the 50-id
  g-value sequence, the edit-locality pair) are **read back from the existing
  file** by default, because they are part of the golden data: regenerating
  them from a fresh RNG would produce a different-but-equally-valid file and
  destroy the diff. `--force` is required to overwrite, and there is no path
  that overwrites silently.

Every int64 in the file is a DECIMAL STRING. The values exceed 2^53 and a JSON
number would be silently rounded by every JavaScript reader — which, in a file
whose entire job is to catch a JS/Python divergence, would be an exquisite way
to hide one.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from launder_core.schemas import SynthIDConfig
from launder_forge.numerics import (
    accumulate_hash,
    compute_context_hashes,
    compute_context_repetition_mask,
    compute_eos_mask,
    compute_g_values,
    contributions,
    depth_weights,
    sample_index,
    sigma_null,
    weighted_mean_score,
)
from launder_forge.paths import Paths

__all__ = ["VectorCheck", "emit_vectors", "verify_vectors", "write_checksum"]


@dataclass(slots=True)
class VectorCheck:
    checked: int = 0
    mismatches: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.mismatches


def _i64(value: Any) -> int:
    """Golden int64s are decimal strings; accept a number too, for hand edits."""
    return int(value)


def _rows_to_strings(g: np.ndarray) -> list[str]:
    return ["".join(str(int(v)) for v in row) for row in g]


def _inflation_knots(data: dict[str, Any]) -> list[tuple[float, float]]:
    """`(n_scored, kappa)` pairs from the golden file's calibration block.

    The block is the shipped `thresholds.v1.json` in the shipped shape, so the
    knots are derived exactly as the two loaders derive them:
    `kappa = sigma / sigma_closed_form(n_scored)`.
    """
    cal = data.get("calibration", {})
    depth = int(cal.get("depth", 30))
    knots: list[tuple[float, float]] = []
    for b in cal.get("buckets", []):
        n = float(b["n_scored"])
        closed = sigma_null(int(n), depth)
        knots.append((n, float(b["sigma"]) / closed if closed > 0 else 1.0))
    return sorted(knots)


def _kappa(n_scored: float, knots: list[tuple[float, float]], *, log_scale: bool) -> float:
    """Piecewise-linear kappa, clamped. `log_scale` selects the axis.

    `log_scale=True` is THE convention: `launder_core.detect.calibration`,
    `web/src/detector/calibration.ts` and the shipped file's own
    `sigma_model.inflation.kind` all say `piecewise_linear_in_log_n`. The linear
    spelling survives only so `verify_vectors` can name it by number when a
    golden `z` was produced by the other axis — the two differ by ~0.2% between
    buckets, which is exactly small enough to ship unnoticed.
    """
    if not knots:
        return 1.0
    xs = [np.log(n) if log_scale else n for n, _ in knots]
    ys = [k for _, k in knots]
    x = np.log(n_scored) if log_scale else float(n_scored)
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            t = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
            return float(ys[i - 1] + t * (ys[i] - ys[i - 1]))
    return ys[-1]


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------


def verify_vectors(paths: Paths, cfg: SynthIDConfig, table: np.ndarray) -> VectorCheck:
    """Recompute every case in the committed `vectors.json`."""
    check = VectorCheck()
    data = json.loads(paths.vectors_json.read_text(encoding="utf-8"))
    cases = data.get("cases", {})

    def fail(msg: str) -> None:
        check.mismatches.append(msg)

    # --- ONE CURVE: the golden calibration IS the shipped calibration --------
    # Before the divergence hunt below can mean anything, the two files have to
    # be quoting the same measurement. They were not: vectors.json carried a
    # 208-passage real-prose curve while data/assets/thresholds.v1.json carried
    # the 140,000-sample synthetic one, so the browser and the server were on
    # different scales no `z` comparison would ever have caught.
    golden_cal = data.get("calibration", {})
    if golden_cal and paths.thresholds.exists():
        shipped = json.loads(paths.thresholds.read_text(encoding="utf-8"))
        want = shipped.get("calibrations", {}).get("default", {}).get("buckets", [])
        got = golden_cal.get("buckets", [])
        check.checked += 1
        if [(b["n_scored"], b["sigma"]) for b in got] != [
            (b["n_scored"], b["sigma"]) for b in want
        ]:
            fail(
                "golden calibration block is not data/assets/thresholds.v1.json. The "
                "browser reads one and the server reads the other; two curves means two "
                "needles. Re-copy calibrations.default.buckets into vectors.json."
            )
        check.checked += 1
        if float(golden_cal.get("z_star", 0.0)) != float(shipped.get("z_star", -1.0)):
            fail(f"golden z_star {golden_cal.get('z_star')!r} != shipped {shipped.get('z_star')!r}")

    # --- case 1: accumulate_hash ------------------------------------------
    case = cases.get("accumulate_hash")
    if case:
        for entry in case.get("accumulate", []):
            got = accumulate_hash(_i64(entry["iv"]), entry["data"])
            check.checked += 1
            if got != _i64(entry["expected"]):
                fail(
                    f"accumulate_hash({entry['iv']}, {entry['data']}) = {got}, "
                    f"expected {entry['expected']}"
                )
        for entry in case.get("depth_keys", []):
            base = accumulate_hash(1, entry["ngram"])
            keys = entry.get("keys") or list(cfg.keys)
            for depth, expected in enumerate(entry["expected"]):
                got = accumulate_hash(base, [keys[depth]])
                check.checked += 1
                if got != _i64(expected):
                    fail(f"depth key {depth} = {got}, expected {expected}")
        for entry in case.get("context_hashes", []):
            n = int(entry.get("ngram_len", cfg.ngram_len))
            got_all = compute_context_hashes(
                np.asarray(entry["tokens"], dtype=np.int64), ngram_len=n
            )
            check.checked += 1
            if [str(int(x)) for x in got_all] != [str(_i64(x)) for x in entry["expected"]]:
                fail(f"context hashes of {entry['tokens']} differ from golden")

    # --- case 2: sample_index ---------------------------------------------
    case = cases.get("sample_index")
    if case:
        for entry in case.get("cases", []):
            got = sample_index(_i64(entry["h"]), int(entry["table_size"]))
            check.checked += 1
            if got != int(entry["expected"]):
                fail(f"sample_index({entry['h']}) = {got}, expected {entry['expected']}")

    # --- case 3: tokenize (referenced file) --------------------------------
    case = cases.get("tokenize")
    if case:
        ref = paths.root / case["ref"]
        digest = "sha256:" + hashlib.sha256(ref.read_bytes()).hexdigest()
        check.checked += 1
        if digest != case["sha256"]:
            fail(f"{case['ref']} digest {digest} != golden {case['sha256']}")

    # --- case 4: g_values ---------------------------------------------------
    case = cases.get("g_values")
    if case:
        ids = np.asarray(case["ids"], dtype=np.int64)
        g = compute_g_values(ids, keys=cfg.keys, ngram_len=case["ngram_len"], table=table)
        check.checked += 1
        if _rows_to_strings(g) != case["g_rows"]:
            first = next(
                (
                    i
                    for i, (a, b) in enumerate(
                        zip(_rows_to_strings(g), case["g_rows"], strict=False)
                    )
                    if a != b
                ),
                -1,
            )
            fail(f"g_values matrix differs from golden, first differing row {first}")

    # --- case 5: repetition_mask -------------------------------------------
    case = cases.get("repetition_mask")
    if case:
        ids = np.asarray(case["ids"], dtype=np.int64)
        ctx = compute_context_hashes(ids, ngram_len=case["ngram_len"])
        check.checked += 1
        if [str(int(x)) for x in ctx] != [str(_i64(x)) for x in case["context_hashes"]]:
            fail("repetition_mask: context hashes differ from golden")
        rep = compute_context_repetition_mask(
            ctx, context_history_size=case["context_history_size"]
        )
        check.checked += 1
        if [int(x) for x in rep] != [int(x) for x in case["repetition_mask"]]:
            fail(f"repetition mask {list(map(int, rep))} != golden {case['repetition_mask']}")
        for entry in case.get("eos_cases", []):
            eos = compute_eos_mask(
                np.asarray(entry["ids"], dtype=np.int64),
                eos_token_id=entry.get("eos_token_id"),
                ngram_len=case["ngram_len"],
            )
            check.checked += 1
            if [int(x) for x in eos] != [int(x) for x in entry["expected"]]:
                fail(f"eos mask {list(map(int, eos))} != golden {entry['expected']}")

    # --- case 6: score ------------------------------------------------------
    case = cases.get("score")
    if case:
        tol = float(case.get("tol", 1e-9))
        # The golden file's eos policy MUST be the shipped one. It is not a
        # per-case override: the browser and the server both take this value
        # from a constant, and a golden file quoting a different one would let
        # them diverge again with the gate still green.
        from launder_core.watermark.config import SCORING_EOS_TOKEN_ID

        check.checked += 1
        if case.get("eos_token_id") != SCORING_EOS_TOKEN_ID:
            fail(
                f"golden case 6 declares eos_token_id={case.get('eos_token_id')!r} but the "
                f"shipped policy is {SCORING_EOS_TOKEN_ID!r} "
                "(launder_core.watermark.config.SCORING_EOS_TOKEN_ID / "
                "web/src/detector/config.ts). Two eos policies means two detectors."
            )
        w = depth_weights(len(cfg.keys))
        check.checked += 1
        if max(abs(a - b) for a, b in zip(w, case["weights"], strict=True)) > tol:
            fail("depth weights differ from golden")
        for entry in case.get("cases", []):
            label = entry.get("label", "?")
            ids = np.asarray(entry["ids"], dtype=np.int64)
            g = compute_g_values(ids, keys=cfg.keys, ngram_len=cfg.ngram_len, table=table)
            ctx = compute_context_hashes(ids, ngram_len=cfg.ngram_len)
            rep = compute_context_repetition_mask(
                ctx, context_history_size=cfg.context_history_size
            )
            eos = compute_eos_mask(
                ids, eos_token_id=case.get("eos_token_id"), ngram_len=cfg.ngram_len
            )
            mask = (rep * eos).astype(np.uint8)
            score, n_scored = weighted_mean_score(g, mask, w)
            check.checked += 2
            if abs(score - float(entry["score"])) > tol:
                fail(f"[{label}] score {score!r} != golden {entry['score']!r}")
            if n_scored != int(entry["n_scored"]):
                fail(f"[{label}] n_scored {n_scored} != golden {entry['n_scored']}")
            variant = entry.get("eos_masked_variant")
            if variant:
                # The regression case: the same ids under the OTHER eos policy.
                # Recorded so flipping the policy is a visible diff instead of an
                # 18x change in z that nobody can attribute.
                v_eos = compute_eos_mask(
                    ids, eos_token_id=int(variant["eos_token_id"]), ngram_len=cfg.ngram_len
                )
                v_score, v_n = weighted_mean_score(g, (rep * v_eos).astype(np.uint8), w)
                check.checked += 3
                if v_n != int(variant["n_scored"]):
                    fail(f"[{label}] eos-masked n_scored {v_n} != golden {variant['n_scored']}")
                if abs(v_score - float(variant["score"])) > tol:
                    fail(f"[{label}] eos-masked score {v_score!r} != golden {variant['score']!r}")
                if v_n == n_scored:
                    fail(
                        f"[{label}] carries an eos_masked_variant but the two policies score "
                        "the same number of rows, so the case cannot detect a policy flip"
                    )
            if "z" in entry:
                check.checked += 1
                z_closed = (score - 0.5) / sigma_null(n_scored, len(cfg.keys))
                knots = _inflation_knots(data)
                z_lin = z_closed / _kappa(n_scored, knots, log_scale=False)
                z_log = z_closed / _kappa(n_scored, knots, log_scale=True)
                golden_z = float(entry["z"])
                if min(abs(z_lin - golden_z), abs(z_log - golden_z)) > 1e-6:
                    fail(
                        f"[{label}] z: closed form {z_closed!r}, with kappa "
                        f"(linear in n) {z_lin!r}, (linear in log n) {z_log!r}; "
                        f"golden {golden_z!r} matches neither"
                    )
                elif abs(z_lin - golden_z) <= 1e-6 and abs(z_log - golden_z) > 1e-6:
                    fail(
                        f"[{label}] KAPPA INTERPOLATION DIVERGENCE: the golden z {golden_z!r} "
                        f"was produced by interpolating kappa LINEARLY IN n ({z_lin!r}), but "
                        "launder_core.detect.calibration.Calibration.kappa interpolates "
                        f"linearly in LOG n ({z_log!r}). Two conventions in one repo means the "
                        "needle and the golden file disagree at every length between buckets. "
                        "Pick one and change the other."
                    )
                elif abs(z_log - golden_z) <= 1e-6 and abs(z_lin - golden_z) > 1e-6:
                    pass  # golden agrees with core's log-scale convention

    # --- case 7: edit_locality ----------------------------------------------
    case = cases.get("edit_locality")
    if case:
        before = np.asarray(case["ids_before"], dtype=np.int64)
        after = np.asarray(case["ids_after"], dtype=np.int64)
        n = case["ngram_len"]
        gb = compute_g_values(before, keys=cfg.keys, ngram_len=n, table=table)
        ga = compute_g_values(after, keys=cfg.keys, ngram_len=n, table=table)
        differing = [int(i) for i in np.nonzero((gb != ga).any(axis=1))[0]]
        check.checked += 2
        if differing != [int(x) for x in case["differing_g_rows"]]:
            fail(f"differing g rows {differing} != golden {case['differing_g_rows']}")
        if len(differing) != int(case["expected_row_count"]):
            fail(
                f"the ripple is {len(differing)} rows wide, golden says "
                f"{case['expected_row_count']} (= ngram_len)"
            )
        ins = case.get("insertion_case")
        if ins:
            # Insertions renumber downstream rows but do NOT change them: row
            # j+n and beyond are bit-identical because g-values are a pure
            # function of token CONTENT, not position (§4.4).
            gi = compute_g_values(
                np.asarray(ins["ids_after"], dtype=np.int64),
                keys=cfg.keys,
                ngram_len=n,
                table=table,
            )
            start = int(ins["first_stable_before_row"])
            check.checked += 1
            if not np.array_equal(gi[start + 1 : gb.shape[0] + 1], gb[start:]):
                fail(
                    "insertion case: g_after[i+1] != g_before[i] downstream of "
                    f"row {start}; the ripple does not stop where §4.4 says it does"
                )

    # --- case 8: normalize_and_damerau (core is the authority) --------------
    case = cases.get("normalize_and_damerau")
    if case:
        from launder_forge.corebridge import core_symbol

        normalize = core_symbol("normalize")
        words_fn = core_symbol("words")
        distance = core_symbol("damerau_levenshtein")
        if normalize is None or words_fn is None or distance is None:
            check.skipped.append(
                "normalize_and_damerau: launder_core.scoring has not landed; case unverified"
            )
        else:
            for entry in case.get("normalize", []):
                got = normalize(entry["input"])
                check.checked += 2
                if got != entry["expected"]:
                    fail(f"normalize({entry['input']!r}) = {got!r} != {entry['expected']!r}")
                if normalize(got) != got:
                    fail(f"normalize is not idempotent on {entry['input']!r}")
            for entry in case.get("words", []):
                got_words = list(words_fn(entry["input"]))
                check.checked += 1
                if got_words != list(entry["expected"]):
                    fail(f"words({entry['input']!r}) = {got_words} != {entry['expected']}")
            for entry in case.get("distance", []):
                a = list(words_fn(entry["a"])) if isinstance(entry["a"], str) else list(entry["a"])
                b = list(words_fn(entry["b"])) if isinstance(entry["b"], str) else list(entry["b"])
                got_d = int(distance(a, b))
                check.checked += 1
                if got_d != int(entry["distance"]):
                    fail(
                        f"damerau({entry['a']!r}, {entry['b']!r}) = {got_d} != {entry['distance']}"
                    )

    return check


# ---------------------------------------------------------------------------
# emission
# ---------------------------------------------------------------------------

#: The §4.5 known-good n-gram. Not random: it is the one the plan publishes, so
#: a reader can check case 1 against the document by eye.
GOLDEN_NGRAM: tuple[int, ...] = (1, 235280, 2121, 576, 573)


def emit_vectors(
    paths: Paths,
    cfg: SynthIDConfig,
    table: np.ndarray,
    *,
    reuse_inputs: bool = True,
    seed: int = 20260815,
) -> dict[str, Any]:
    """Rebuild `vectors.json`. Inputs are reused from the existing file by default."""
    existing: dict[str, Any] = {}
    if reuse_inputs and paths.vectors_json.exists():
        existing = json.loads(paths.vectors_json.read_text(encoding="utf-8")).get("cases", {})

    rng = np.random.default_rng(seed)

    def ids_for(case: str, key: str, size: int) -> list[int]:
        prior = existing.get(case, {}).get(key)
        if prior:
            return [int(x) for x in prior]
        return [int(x) for x in rng.integers(0, 262144, size=size)]

    # --- case 1
    base = accumulate_hash(1, GOLDEN_NGRAM)
    accumulate_cases = [
        {"iv": "1", "data": [], "expected": str(accumulate_hash(1, []))},
        {"iv": "1", "data": [0], "expected": str(accumulate_hash(1, [0]))},
        {
            "iv": "1",
            "data": list(GOLDEN_NGRAM),
            "expected": str(base),
        },
    ]
    depth_keys = [
        {
            "ngram": list(GOLDEN_NGRAM),
            "expected": [str(accumulate_hash(base, [k])) for k in cfg.keys[:5]],
        }
    ]
    # `tokens` + the FULL list of context hashes for it — the shape
    # `verify_vectors` reads. It used to emit one `window` per entry with a
    # single scalar `expected`, which the verifier could not read at all
    # (KeyError('tokens')), so `--write` produced a file that failed to load.
    _ctx_all = compute_context_hashes(
        np.asarray(GOLDEN_NGRAM, dtype=np.int64), ngram_len=cfg.ngram_len
    )
    context_hashes = [
        {
            "tokens": list(GOLDEN_NGRAM),
            "ngram_len": cfg.ngram_len,
            "expected": [str(int(x)) for x in _ctx_all],
        }
    ]

    # --- case 2
    sample_cases = [
        {
            "h": str(h),
            "table_size": cfg.sampling_table_size,
            "expected": sample_index(h, cfg.sampling_table_size),
        }
        for h in (
            -12345678901234567,
            -6504205589568445072,
            318672039786673866,
            -1,
            0,
            65536,
            -65536,
        )
    ]

    # --- case 4
    g_ids = ids_for("g_values", "ids", 50)
    g = compute_g_values(
        np.asarray(g_ids, dtype=np.int64), keys=cfg.keys, ngram_len=cfg.ngram_len, table=table
    )

    # --- case 5
    rep_ids = existing.get("repetition_mask", {}).get("ids") or [
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        7,
        8,
        9,
        10,
        11,
        99,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
    ]
    rep_ids = [int(x) for x in rep_ids]
    ctx = compute_context_hashes(np.asarray(rep_ids, dtype=np.int64), ngram_len=cfg.ngram_len)
    rep = compute_context_repetition_mask(ctx, context_history_size=cfg.context_history_size)
    eos = compute_eos_mask(
        np.asarray(rep_ids, dtype=np.int64), eos_token_id=None, ngram_len=cfg.ngram_len
    )
    mask = (rep * eos).astype(np.uint8)

    # --- case 7
    loc_before = ids_for("edit_locality", "ids_before", 60)
    edited_index = int(existing.get("edit_locality", {}).get("edited_index", 30))
    loc_after = list(loc_before)
    loc_after[edited_index] = (loc_after[edited_index] + 12345) % 262144
    gb = compute_g_values(
        np.asarray(loc_before, dtype=np.int64), keys=cfg.keys, ngram_len=cfg.ngram_len, table=table
    )
    ga = compute_g_values(
        np.asarray(loc_after, dtype=np.int64), keys=cfg.keys, ngram_len=cfg.ngram_len, table=table
    )
    differing = [int(i) for i in np.nonzero((gb != ga).any(axis=1))[0]]

    # --- the calibration block, case 6 and case 8 --------------------------
    #
    # These three were MISSING from the emitter while `verify_vectors` required
    # all of them, so `forge vectors --write` silently produced a golden file
    # that its own checker could not read: the score cases, the scoring-parity
    # cases and the ONE-CURVE assertion all vanished, and the next `forge
    # vectors` died on KeyError('tokens'). §4.5 names eight case kinds; the
    # emitter has to emit eight.
    calibration_block = _calibration_block(paths)
    knots = [
        (float(b["n_scored"]), float(b["sigma"]) / sigma_null(int(b["n_scored"]), cfg.depth))
        for b in calibration_block["buckets"]
    ]

    def _z(score: float, n_scored: int) -> float:
        """z the way `launder_core` computes it: closed-form shape, kappa level.

        Interpolated linearly in LOG n — the convention `verify_vectors`
        enforces and the shipped file's own `sigma_model` declares.
        """
        return (
            (score - 0.5)
            / sigma_null(n_scored, cfg.depth)
            / _kappa(n_scored, knots, log_scale=True)
        )

    score_cases = _score_cases(paths, cfg, table, _z, calibration_block["z_star"])

    payload: dict[str, Any] = {
        "schema": "launder.golden/1",
        "note": (
            "The single golden file, consumed by pytest (Python core), vitest (TS port) and "
            "`forge verify`. Regenerating goldens is always a reviewed diff and never a way to "
            "turn a red test green (TECH_PLAN.md §4.5). EVERY int64 is a DECIMAL STRING: the "
            "values exceed 2^53 and would be silently rounded by a JSON number in JavaScript."
        ),
        "generated_by": "forge vectors",
        "wm_config": {
            "ngram_len": cfg.ngram_len,
            "context_history_size": cfg.context_history_size,
            "sampling_table_size": cfg.sampling_table_size,
            "sampling_table_seed": cfg.sampling_table_seed,
            "skip_first_ngram_calls": cfg.skip_first_ngram_calls,
            "keys": list(cfg.keys),
            "wm_config_id": cfg.wm_config_id,
            "scheme": "sampling_table",
        },
        "cases": {
            "accumulate_hash": {
                "kind": "accumulate_hash",
                "asserts": "int64 two's-complement wraparound after EVERY add and multiply.",
                "lcg_multiplier": "6364136223846793005",
                "lcg_increment": "1",
                "hash_iv": "1",
                "accumulate": accumulate_cases,
                "depth_keys": depth_keys,
                "context_hashes": context_hashes,
            },
            "sample_index": {
                "kind": "sample_index",
                "asserts": "the negative-modulo sign trap in isolation: ((h % N) + N) % N",
                "cases": sample_cases,
            },
            "tokenize": {
                "kind": "tokenize",
                "asserts": "4,031 cases from Rust `tokenizers`.",
                "ref": "data/golden/tokenizer_golden.json",
                "count": 4031,
                "sha256": "sha256:"
                + hashlib.sha256(paths.tokenizer_golden.read_bytes()).hexdigest(),
            },
            "g_values": {
                "kind": "g_values",
                "asserts": "full 0/1 matrix on a fixed 50-id sequence.",
                "ngram_len": cfg.ngram_len,
                "m": cfg.depth,
                "ids": g_ids,
                "rows": int(g.shape[0]),
                "g_rows": _rows_to_strings(g),
            },
            "repetition_mask": {
                "kind": "repetition_mask",
                "asserts": "causal mask with a deliberately repeated 4-gram.",
                "context_history_size": cfg.context_history_size,
                "ngram_len": cfg.ngram_len,
                "ids": rep_ids,
                "context_hashes": [str(int(x)) for x in ctx],
                "repetition_mask": [int(x) for x in rep],
                "eos_mask": [int(x) for x in eos],
                "mask": [int(x) for x in mask],
                "n_scored": int(mask.sum()),
            },
            "edit_locality": {
                "kind": "edit_locality",
                "asserts": "the ripple is exactly ngram_len wide.",
                "ngram_len": cfg.ngram_len,
                "ids_before": loc_before,
                "ids_after": loc_after,
                "edited_index": edited_index,
                "differing_g_rows": differing,
                "expected_row_count": cfg.ngram_len,
            },
            "score": score_cases,
            "normalize_and_damerau": _scoring_cases(),
        },
        # TOP-LEVEL, not under `cases`: `verify_vectors` and web/tools/parity.mjs
        # both read `vectors.calibration`, and it must BE
        # `data/assets/thresholds.v1.json`. One curve or two needles.
        "calibration": calibration_block,
    }
    return payload


def _calibration_block(paths: Paths) -> dict[str, Any]:
    """The shipped thresholds document, verbatim, plus a top-level bucket mirror.

    TWO READERS, TWO SHAPES, ONE FILE.

    * `web/tools/parity.mjs` hands this block straight to the TS
      `parseCalibration`, which demands the full thresholds document —
      `schema == "launder.thresholds/1"` and `calibrations.<name>.buckets`.
    * `verify_vectors` (above) reads `calibration.buckets` at the top level to
      assert this block IS `data/assets/thresholds.v1.json`.

    Emitting the document verbatim satisfies the first; mirroring
    `calibrations.default.buckets` to the top level satisfies the second. The
    mirror is the same list object, so the two cannot drift apart.
    """
    shipped: dict[str, Any] = json.loads(paths.thresholds.read_text(encoding="utf-8"))
    block = dict(shipped)
    block["note"] = (
        "data/assets/thresholds.v1.json, verbatim. `forge verify` asserts these buckets ARE "
        "the shipped ones — the browser and the server must read one curve, or the needle "
        "and the golden file disagree at every length between buckets. `buckets` at this "
        "level mirrors calibrations.default.buckets for the Python verifier."
    )
    block["buckets"] = shipped["calibrations"]["default"]["buckets"]
    return block


def _score_cases(
    paths: Paths,
    cfg: SynthIDConfig,
    table: np.ndarray,
    z_of: Any,
    z_star: float,
) -> dict[str, Any]:
    """Case 6: end-to-end weighted mean + z, plus the eos-policy regression."""
    from launder_core.watermark.config import SCORING_EOS_TOKEN_ID

    w = depth_weights(cfg.depth)

    def measure(ids: list[int], eos_token_id: int | None) -> tuple[float, int]:
        arr = np.asarray(ids, dtype=np.int64)
        g = compute_g_values(arr, keys=cfg.keys, ngram_len=cfg.ngram_len, table=table)
        ctx = compute_context_hashes(arr, ngram_len=cfg.ngram_len)
        rep = compute_context_repetition_mask(ctx, context_history_size=cfg.context_history_size)
        eos = compute_eos_mask(arr, eos_token_id=eos_token_id, ngram_len=cfg.ngram_len)
        return weighted_mean_score(g, (rep * eos).astype(np.uint8), w)

    def per_token(ids: list[int]) -> tuple[list[float], list[int]]:
        """`heat` and `masked` PER TOKEN, exactly as `/api/detect` emits them.

        Row `i` covers ids[i..i+n-1] and its heat belongs to the CURRENT token
        `i + n - 1`, so the leading `ngram_len - 1` tokens are the final token
        of no window: neutral 0.5 and masked. Absence of evidence, not evidence
        of innocence.
        """
        arr = np.asarray(ids, dtype=np.int64)
        g = compute_g_values(arr, keys=cfg.keys, ngram_len=cfg.ngram_len, table=table)
        ctx = compute_context_hashes(arr, ngram_len=cfg.ngram_len)
        rep = compute_context_repetition_mask(ctx, context_history_size=cfg.context_history_size)
        eos = compute_eos_mask(arr, eos_token_id=SCORING_EOS_TOKEN_ID, ngram_len=cfg.ngram_len)
        mask = (rep * eos).astype(np.uint8)
        rows = contributions(g, w)
        heat = [0.5] * len(ids)
        masked = [1] * len(ids)
        for i in range(g.shape[0]):
            t = i + cfg.ngram_len - 1
            if t < len(ids):
                heat[t] = float(rows[i])
                masked[t] = 0 if int(mask[i]) else 1
        return heat, masked

    def entry(label: str, ids: list[int], *, text: str | None = None) -> dict[str, Any]:
        score, n_scored = measure(ids, SCORING_EOS_TOKEN_ID)
        heat, masked = per_token(ids)
        out: dict[str, Any] = {
            "label": label,
            "ids": ids,
            "n_tokens": len(ids),
            "score": score,
            "n_scored": n_scored,
            "z": z_of(score, n_scored),
            "z_star": z_star,
            "heat": heat,
            "masked": masked,
        }
        if text is not None:
            out["text_sha256"] = "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
        # The OTHER eos policy, recorded so flipping it is a visible diff rather
        # than an unattributable change in z. Only emitted when the two policies
        # actually score a different number of rows — a variant that does not
        # distinguish them cannot detect a flip, and verify_vectors says so.
        other = None if SCORING_EOS_TOKEN_ID is not None else 1
        if other is not None and other in ids:
            v_score, v_n = measure(ids, other)
            if v_n != n_scored:
                out["eos_masked_variant"] = {
                    "eos_token_id": other,
                    "score": v_score,
                    "n_scored": v_n,
                }
        return out

    cases: list[dict[str, Any]] = []

    # A fixed synthetic sequence: reproducible with no tokenizer at all.
    rng = np.random.default_rng(6060)
    cases.append(entry("fixed_50", [int(x) for x in rng.integers(0, 262144, 50)]))

    # An eos-BEARING sequence, so the eos policy is exercised rather than assumed.
    eos_ids = [int(x) for x in rng.integers(0, 262144, 40)]
    for pos in (9, 23):
        eos_ids[pos] = 1
    cases.append(entry("eos_bearing", eos_ids))

    # The real dev passage, when the tokenizer can be built here. This is the
    # case `packages/serve/tests/test_api_detect_real.py` looks up by
    # `text_sha256`, and the one web/tools/parity.mjs scores end to end.
    try:
        from launder_core.tokenizer import encode_with_offsets

        dev = paths.root / "data" / "dev" / "passage.txt"
        text = dev.read_text(encoding="utf-8").removesuffix("\n")
        ids, _ = encode_with_offsets(text)
        cases.append(entry("dev_passage", [int(x) for x in ids], text=text))
    except Exception:  # the tokenizer is optional at emit time
        pass

    return {
        "kind": "score",
        "asserts": "end-to-end weighted mean + z, tol 1e-9 in float64.",
        "tol": 1e-9,
        # NOT a per-case override: both runtimes read one constant, and
        # verify_vectors fails if this disagrees with it.
        "eos_token_id": SCORING_EOS_TOKEN_ID,
        "weights": [float(x) for x in w],
        "cases": cases,
    }


def _scoring_cases() -> dict[str, Any]:
    """Case 8: normalization + word-level Damerau-Levenshtein, both languages.

    `launder_core.scoring` is the authority; these are recorded from it so the
    TS port has something to fail against.
    """
    from launder_forge.corebridge import core_symbol

    normalize = core_symbol("normalize")
    words_fn = core_symbol("words")
    distance = core_symbol("damerau_levenshtein")
    if normalize is None or words_fn is None or distance is None:
        return {"kind": "normalize_and_damerau", "asserts": "core.scoring absent at emit time"}

    # Written as escapes, not literals: these strings EXIST to carry the exact
    # characters normalization folds, and a source file where a curly quote and
    # a straight one look identical is the last place to rely on the eye.
    norm_inputs = [
        "plain text",
        "double  spaced   words",
        "\u201ccurly\u201d quotes and \u2018singles\u2019",
        "em \u2014 dash and ellipsis \u2026",
        "nbsp\u00a0separated",
        "trailing whitespace   ",
        "  leading whitespace",
        "tabs\tand\nnewlines\n\nhere",
        "combining \u00e9 vs e\u0301",
        "ellipsis... three dots",
        "double--hyphen",
        "mixed \u2013 en dash",
        "ZWSP\u200binside",
        "soft\u00adhyphen",
        "BOM\ufeffstart",
        "\u0130stanbul dotted capital",
        "\ufb01 ligature",
        "\u2460\u2461 circled",
        "CRLF\r\nline",
        "vertical\u000btab",
        "form\ffeed",
        "\u00e9 precomposed",
        "e\u0301 decomposed",
        "\u201cnested \u2018quotes\u2019\u201d",
    ]
    dist_pairs = [
        ("alpha beta gamma", "alpha delta gamma", None),
        ("one two three", "two one three", None),  # a reorder costs 1, not 2
        ("a b c d", "a b c d", None),
        ("the cutting holds its own weather", "the cutting keeps a weather of its own", None),
        ("insert here", "insert a word here", None),
        ("delete one word here", "delete word here", None),
        ("alpha beta gamma delta", "alpha gamma beta delta", None),
        ("one", "one two", None),
        ("one two", "one", None),
        ("a b c", "c b a", None),
        ("keep the claim about funding", "keep the claim regarding funding", None),
        ("swap adjacent words here", "swap words adjacent here", None),
        ("", "", None),
        ("only", "", None),
        ("", "only", None),
        ("repeated repeated repeated", "repeated repeated", None),
    ]
    return {
        "kind": "normalize_and_damerau",
        "asserts": "normalization idempotence + word-level Damerau-Levenshtein + op list.",
        "normalize": [{"input": s, "expected": normalize(s)} for s in norm_inputs],
        "words": [{"input": s, "expected": list(words_fn(normalize(s)))} for s in norm_inputs],
        "distance": [
            {
                "a": a,
                "b": b,
                "distance": int(
                    distance(list(words_fn(normalize(a))), list(words_fn(normalize(b))))
                ),
            }
            for a, b, _ in dist_pairs
        ],
    }


def write_checksum(paths: Paths) -> Path:
    """Rewrite `data/golden/CHECKSUM` over every file in `data/golden/`."""
    lines = [
        "# data/golden/CHECKSUM — TECH_PLAN.md §4.5",
        "#",
        "# sha256 of every file in data/golden/, asserted in CI. Regenerating goldens is",
        "# therefore always a reviewed diff and never a way to turn a red test green.",
        "#",
        "# Verify from inside data/golden/:   sha256sum -c CHECKSUM",
    ]
    for p in sorted(paths.golden.glob("*")):
        if p.name == "CHECKSUM" or not p.is_file():
            continue
        lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}")
    # newline="" so Windows does not translate to CRLF. `sha256sum -c` splits on
    # LF and takes the trailing CR as part of the FILENAME, so a CHECKSUM written
    # on this machine fails on the Linux CI runner with "No such file or
    # directory" — a checksum gate that cannot read its own file.
    with paths.checksum.open("w", encoding="utf-8", newline="") as fh:
        fh.write("\n".join(lines) + "\n")
    return paths.checksum
