"""The `forge` command (TECH_PLAN.md §6.4).

typer + rich. Every command module is imported *inside* its command function,
so `forge lint-copy` and `forge tok pack` run in a checkout without torch, and
`forge doctor` reports that torch is missing instead of dying on an import.

Every target is also a Makefile shim (`uv run python -m launder_forge.cli ...`)
so Windows works without `make`.

Exit codes: 0 success, 1 a check failed, 2 the environment cannot run the
command (missing weights, no CUDA, missing build input). The distinction
matters in CI, where 2 means "not attempted here" and 1 means "broken".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from launder_forge.paths import repo_paths

app = typer.Typer(
    name="forge",
    help="Launder WM authoring studio — generate, analyze, solve, triage, calibrate, pack.",
    no_args_is_help=True,
    add_completion=False,
)
tok_app = typer.Typer(help="Tokenizer blob packing.", no_args_is_help=True)
table_app = typer.Typer(help="Sampling table build and verification.", no_args_is_help=True)
app.add_typer(tok_app, name="tok")
app.add_typer(table_app, name="table")

console = Console()

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_UNAVAILABLE = 2


def _ok(msg: str) -> None:
    console.print(f"[bold green]OK[/]  {msg}")


def _bad(msg: str) -> None:
    console.print(f"[bold red]FAIL[/]  {msg}")


def _info(msg: str) -> None:
    console.print(f"[dim]--[/]  {msg}")


def _assets() -> tuple[Any, Any, Any]:
    """`(paths, SynthIDConfig, sampling_table)` — the three things nearly every
    command needs, loaded once and asserted once."""
    from launder_forge.config import load_watermark_config
    from launder_forge.numerics import load_sampling_table

    paths = repo_paths()
    cfg = load_watermark_config(paths)
    table = load_sampling_table(paths.sampling_table, cfg.sampling_table_size)
    return paths, cfg, table


# ---------------------------------------------------------------------------
# doctor / bench
# ---------------------------------------------------------------------------


def _lost_golden_fields(existing: Path, payload: dict[str, Any]) -> list[str]:
    """Keys present in the committed golden file that `payload` would drop.

    Compared one level into `cases`, which is where the field groups live. Extra
    keys in `payload` are fine — growing the golden file is the point; shrinking
    it silently is the failure mode this catches.
    """
    if not existing.exists():
        return []
    try:
        old = json.loads(existing.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    lost = [k for k in old if k not in payload]
    old_cases = old.get("cases", {})
    new_cases = payload.get("cases", {})
    if isinstance(old_cases, dict) and isinstance(new_cases, dict):
        for name, block in old_cases.items():
            if name not in new_cases:
                lost.append(f"cases.{name}")
                continue
            if isinstance(block, dict) and isinstance(new_cases[name], dict):
                lost += [f"cases.{name}.{k}" for k in block if k not in new_cases[name]]
                # The per-case entries carry the arrays the browser tests read.
                old_entries = block.get("cases")
                new_entries = new_cases[name].get("cases")
                if (
                    isinstance(old_entries, list)
                    and isinstance(new_entries, list)
                    and old_entries
                    and new_entries
                    and isinstance(old_entries[0], dict)
                    and isinstance(new_entries[0], dict)
                ):
                    lost += [
                        f"cases.{name}.cases[].{k}"
                        for k in old_entries[0]
                        if k not in new_entries[0]
                    ]
    return sorted(set(lost))


@app.command()
def doctor(
    write: Annotated[bool, typer.Option(help="Write data/runs/<run_id>.manifest.json")] = False,
    json_out: Annotated[bool, typer.Option("--json", help="Emit the report as JSON")] = False,
    matmul_size: int = 4096,
) -> None:
    """Assert sm_120 and run a real bf16 matmul. Degrades cleanly with no CUDA."""
    from launder_forge.doctor import doctor_report, run_id_now

    report = doctor_report(matmul_size=matmul_size)
    if json_out:
        console.print_json(json.dumps(report))
    else:
        t = Table(show_header=False, box=None)
        for key in (
            "python", "platform", "torch", "torch_cuda_build", "cuda_available",
            "device_name", "capability", "vram_total_mib", "driver",
            "bf16_supported", "matmul_ok", "matmul_ms", "matmul_tflops",
        ):  # fmt: skip
            t.add_row(key, str(report.get(key)))
        console.print(t)
        for note in report["notes"]:
            _info(note)
        for block in report["blocking"]:
            _bad(block)
        (_ok if report["ok"] else _bad)(
            "environment ready" if report["ok"] else "environment not ready for generation"
        )

    if write:
        paths = repo_paths()
        paths.ensure(paths.runs)
        out = paths.runs / f"{run_id_now()}.manifest.json"
        out.write_text(json.dumps({"env": report}, indent=2) + "\n", encoding="utf-8")
        _info(f"wrote {out}")
    raise typer.Exit(EXIT_OK if report["ok"] else EXIT_UNAVAILABLE)


@app.command()
def bench(
    batch: str = "1,8,16,32,48,64",
    tokens: int = 256,
    watermark: str = "on,off",
) -> None:
    """Throughput matrix (batch x watermark on/off). Needs CUDA + weights."""
    from launder_forge.config import load_watermark_config
    from launder_forge.generate import generate_batch, load_model

    paths, cfg, _ = _assets()
    cfg = load_watermark_config(paths)
    try:
        bundle = load_model()
    except RuntimeError as exc:
        _bad(str(exc))
        raise typer.Exit(EXIT_UNAVAILABLE) from exc

    import time

    t = Table("batch", "watermark", "tok/s", "s")
    prompt = "Write one flowing paragraph about a tidal flat at dusk."
    for b in [int(x) for x in batch.split(",")]:
        for wm in watermark.split(","):
            t0 = time.perf_counter()
            outs = generate_batch(
                bundle,
                [prompt] * b,
                max_new_tokens=tokens,
                min_new_tokens=tokens,
                seed=1,
                watermark=cfg if wm == "on" else None,
                table_path=paths.sampling_table if wm == "on" else None,
            )
            dt = time.perf_counter() - t0
            total = sum(len(o.token_ids) for o in outs)
            t.add_row(str(b), wm, f"{total / dt:.0f}", f"{dt:.2f}")
    console.print(t)


# ---------------------------------------------------------------------------
# table
# ---------------------------------------------------------------------------


@table_app.command("build")
def table_build(
    out: Annotated[Path | None, typer.Option(help="Where to write the packed table")] = None,
    seed: int = 0,
    size: int = 65536,
    force: Annotated[bool, typer.Option(help="Overwrite an existing table")] = False,
) -> None:
    """CPU sampling table -> packed bitmap + digests. KEY MATERIAL: refuses to
    overwrite the committed file without --force."""
    from launder_forge.table import build_table, digests_of, pack_table

    paths = repo_paths()
    target = out or paths.sampling_table
    if target.exists() and not force:
        _bad(
            f"{target} exists. It is key material and every shipped passage was scored with it; "
            "rebuilding invalidates all of them. Pass --force if that is genuinely what you want."
        )
        raise typer.Exit(EXIT_FAIL)
    values = build_table(size=size, seed=seed, device="cpu")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(pack_table(values))
    console.print_json(json.dumps(digests_of(values)))
    _ok(f"wrote {target}")


@app.command("check-table")
def check_table_cmd(
    json_out: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """TECH_PLAN §14.2 item 1: is the sampling table device-dependent?

    Compares the committed table against a freshly built CPU table, and — when
    a GPU is present — against a CUDA-built one.
    """
    from launder_forge.table import check_table

    paths, cfg, _ = _assets()
    try:
        result = check_table(
            paths.sampling_table, size=cfg.sampling_table_size, seed=cfg.sampling_table_seed
        )
    except RuntimeError as exc:
        _bad(str(exc))
        raise typer.Exit(EXIT_UNAVAILABLE) from exc

    if json_out:
        console.print_json(
            json.dumps(
                {
                    "cpu_matches": result.cpu_matches,
                    "cpu_first_mismatch": result.cpu_first_mismatch,
                    "cuda_ran": result.cuda_ran,
                    "cuda_matches": result.cuda_matches,
                    "cuda_vs_cpu_agree": result.cuda_vs_cpu_agree,
                    "committed": result.committed_digests,
                    "cpu": result.cpu_digests,
                    "cuda": result.cuda_digests,
                    "notes": result.notes,
                }
            )
        )
    else:
        t = Table("check", "result")
        t.add_row("committed == CPU-built", str(result.cpu_matches))
        t.add_row("CUDA half ran", str(result.cuda_ran))
        t.add_row("committed == CUDA-built", str(result.cuda_matches))
        t.add_row("CUDA == CPU", str(result.cuda_vs_cpu_agree))
        t.add_row("blake3 (unpacked)", str(result.committed_digests["blake3_unpacked"]))
        t.add_row("ones", str(result.committed_digests["ones"]))
        console.print(t)
        for note in result.notes:
            _info(note)
    (_ok if result.ok else _bad)("CPU parity" if result.ok else "CPU parity FAILED")
    raise typer.Exit(EXIT_OK if result.ok else EXIT_FAIL)


# ---------------------------------------------------------------------------
# tok pack
# ---------------------------------------------------------------------------


@tok_app.command("pack")
def tok_pack(
    out: Annotated[Path | None, typer.Option(help="Write the blob here")] = None,
    check: Annotated[bool, typer.Option(help="Compare bytes with the committed blob")] = True,
    write: Annotated[bool, typer.Option(help="Actually write the output")] = False,
) -> None:
    """tokenizer.json -> gemma3-tok.v1.bin(.br); asserts 0 round-trip errors and
    byte-identity with the committed asset."""
    from launder_forge.packtok import pack_tokenizer

    paths = repo_paths()
    try:
        res = pack_tokenizer(paths.tokenizer_json, paths.tokenizer_config_json)
    except FileNotFoundError as exc:
        _bad(str(exc))
        raise typer.Exit(EXIT_UNAVAILABLE) from exc

    t = Table("field", "value")
    t.add_row("pieces", f"{res.n_pieces:,}")
    t.add_row("merges", f"{res.n_merges:,}")
    t.add_row("added tokens", f"{res.n_added:,}")
    t.add_row("non-normal types", f"{res.n_non_normal:,}")
    t.add_row("raw bytes", f"{len(res.raw):,}")
    t.add_row("brotli bytes", f"{len(res.compressed):,}")
    t.add_row("round-trip errors", str(res.roundtrip_errors))
    console.print(t)

    ok = res.roundtrip_errors == 0
    if res.roundtrip_errors:
        _bad(f"{res.roundtrip_errors} round-trip errors — the blob does not reconstruct the input")

    if check and paths.tokenizer_blob.exists():
        committed = paths.tokenizer_blob.read_bytes()
        identical = committed == res.compressed
        (_ok if identical else _bad)(
            f"byte-identical to {paths.rel(paths.tokenizer_blob)}"
            if identical
            else f"DIFFERS from the committed blob ({len(committed):,} vs {len(res.compressed):,} bytes) "
            "— asset_bundle_id would change and every passage would be invalidated"
        )
        ok = ok and identical

    target = out or paths.tokenizer_blob
    if write:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(res.compressed)
        _info(f"wrote {target}")
    raise typer.Exit(EXIT_OK if ok else EXIT_FAIL)


# ---------------------------------------------------------------------------
# generation pipeline
# ---------------------------------------------------------------------------


@app.command()
def gen(
    n: int = 400,
    batch: int = 48,
    out: Annotated[Path | None, typer.Option(help="candidates jsonl")] = None,
    temperature: Annotated[
        float | None, typer.Option(help="overrides the prompt pack's gen_params.temperature")
    ] = None,
    prompts: Annotated[
        str, typer.Option(help="prompt pack id under data/config/prompts/")
    ] = "high_entropy",
    seed: Annotated[int | None, typer.Option(help="overrides seeds.generation_base")] = None,
) -> None:
    """Bulk watermarked candidates + teacher-forced analysis. Needs CUDA + weights.

    Prompts come from `data/config/prompts/<--prompts>.yaml` (§6.2, §12 row 21),
    NOT from a hardcoded f-string. `--temperature` still wins when given, so the
    pack sets the default and the flag is the experiment.
    """
    from launder_forge.config import load_prompt_pack, watermark_seeds
    from launder_forge.doctor import run_id_now
    from launder_forge.generate import generate_batch, load_model
    from launder_forge.numerics import score_ids

    paths, cfg, table = _assets()
    try:
        pack = load_prompt_pack(paths, prompts)
    except (FileNotFoundError, ValueError) as exc:
        _bad(str(exc))
        raise typer.Exit(EXIT_UNAVAILABLE) from exc
    temp = (
        temperature if temperature is not None else float(pack.gen_params.get("temperature", 0.95))
    )
    _info(f"prompt pack {pack.id} ({paths.rel(pack.path)}): {len(pack.templates)} templates")
    try:
        bundle = load_model()
    except RuntimeError as exc:
        _bad(str(exc))
        raise typer.Exit(EXIT_UNAVAILABLE) from exc

    base_seed = seed if seed is not None else watermark_seeds(paths).get("generation_base", 0)
    run_id = run_id_now()
    target = out or (paths.ensure(paths.candidates) / f"{run_id}.jsonl")
    written = 0
    with target.open("w", encoding="utf-8") as fh:
        for start in range(0, n, batch):
            size = min(batch, n - start)
            rendered = pack.render(size, start=start)
            cands = generate_batch(
                bundle,
                rendered,
                temperature=temp,
                seed=base_seed + start,
                watermark=cfg,
                table_path=paths.sampling_table,
            )
            for cand in cands:
                scored = score_ids(
                    cand.token_ids,
                    keys=cfg.keys,
                    ngram_len=cfg.ngram_len,
                    table=table,
                    context_history_size=cfg.context_history_size,
                )
                fh.write(
                    json.dumps(
                        {
                            "candidate_id": cand.candidate_id,
                            "run_id": run_id,
                            "text": cand.text,
                            "token_ids": cand.token_ids,
                            "prompt_rendered": cand.prompt_rendered,
                            "gen_params": cand.gen_params,
                            "seed": cand.seed,
                            "score": scored.score,
                            "n_scored": scored.n_scored,
                            "z_total": scored.z,
                            "masked_fraction": scored.masked_fraction,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                written += 1
            _info(f"{written}/{n}")
    _ok(f"wrote {written} candidates to {target}")


@app.command()
def analyze(
    candidates: Annotated[Path, typer.Argument(help="candidates jsonl")],
    limit: int = 0,
) -> None:
    """Re-run entropy / g_mass / top-k alternatives on existing candidates."""
    from launder_forge.analyze import analyze_ids
    from launder_forge.generate import load_model

    _paths, cfg, table = _assets()
    try:
        bundle = load_model()
    except RuntimeError as exc:
        _bad(str(exc))
        raise typer.Exit(EXIT_UNAVAILABLE) from exc

    lines = [
        json.loads(x) for x in candidates.read_text(encoding="utf-8").splitlines() if x.strip()
    ]
    if limit:
        lines = lines[:limit]
    out = candidates.with_suffix(".analyzed.jsonl")
    with out.open("w", encoding="utf-8") as fh:
        for rec in lines:
            opt = analyze_ids(
                bundle, rec["token_ids"], keys=cfg.keys, ngram_len=cfg.ngram_len, table=table
            )
            rec["optionality"] = {
                "entropy_nats": opt.entropy_nats,
                "eff_choices": opt.eff_choices,
                "g_mass": opt.g_mass,
                "wm_boost": opt.wm_boost,
                "top1_prob": opt.top1_prob,
            }
            rec["topk_alts"] = opt.topk_alts
            rec.setdefault("difficulty", {})["median_eff_choices"] = opt.median_eff_choices
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    _ok(f"wrote {out}")


@app.command("score")
def score_cmd(
    candidates: Annotated[Path, typer.Argument(help="candidates jsonl")],
    inplace: Annotated[bool, typer.Option(help="Rewrite the file")] = False,
) -> None:
    """Recompute detector scores after a threshold or config change."""
    from launder_forge.numerics import score_ids

    _, cfg, table = _assets()
    lines = [
        json.loads(x) for x in candidates.read_text(encoding="utf-8").splitlines() if x.strip()
    ]
    changed = 0
    for rec in lines:
        scored = score_ids(
            rec["token_ids"],
            keys=cfg.keys,
            ngram_len=cfg.ngram_len,
            table=table,
            context_history_size=cfg.context_history_size,
        )
        if rec.get("score") != scored.score:
            changed += 1
        rec["score"] = scored.score
        rec["n_scored"] = scored.n_scored
        rec["z_total"] = scored.z
        rec["masked_fraction"] = scored.masked_fraction
    if inplace:
        candidates.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in lines), encoding="utf-8"
        )
    _ok(f"{len(lines)} candidates rescored, {changed} changed" + ("" if inplace else " (dry run)"))


@app.command()
def solve(
    text_file: Annotated[Path, typer.Argument(help="A file containing the passage text")],
    max_edits: int = 10,
    beam: int = 512,
    json_out: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Minimal-edit beam search -> par_upper (judge-gated, upper bound only)."""
    from launder_forge.solve import SolveConfig
    from launder_forge.solve import solve as run_solve
    from launder_forge.tokenizer import load_tokenizer

    paths, cfg, table = _assets()
    tokenizer = load_tokenizer(paths)
    z_star = _z_star(paths)
    result = run_solve(
        text_file.read_text(encoding="utf-8").strip(),
        tokenizer=tokenizer,
        keys=cfg.keys,
        ngram_len=cfg.ngram_len,
        table=table,
        context_history_size=cfg.context_history_size,
        z_star=z_star,
        config=SolveConfig(max_edits=max_edits, beam=beam),
    )
    if json_out:
        console.print_json(json.dumps(result.as_dict()))
    else:
        t = Table("field", "value")
        t.add_row("par_upper", str(result.par_upper))
        t.add_row("exhaustive_clear_free_upto_k", str(result.exhaustive_clear_free_upto_k))
        t.add_row("best_single_edit_drop_z", f"{result.best_single_edit_drop_z:.4f}")
        t.add_row("clears_at_k", json.dumps(result.clears_at_k))
        t.add_row("judge_rejected_solutions", str(result.judge_rejected_solutions))
        t.add_row("evaluated", f"{result.evaluated:,}")
        t.add_row("search_seconds", f"{result.search_seconds:.2f}")
        console.print(t)
        for note in result.notes:
            _info(note)
        if result.best:
            console.print(f"[bold]best:[/] {result.best.text}")
    _info(
        "par_upper is an UPPER BOUND on the minimum: it proves this many suffice, never that "
        "fewer cannot. No true lower bound exists — the move space is unbounded."
    )


def _z_star(paths: Any) -> float:
    """z* from the shipped thresholds, or the closed-form default."""
    if paths.thresholds.exists():
        data = json.loads(paths.thresholds.read_text(encoding="utf-8"))
        return float(data.get("z_star", 2.3263))
    return 2.3263


@app.command()
def triage(
    in_: Annotated[Path, typer.Option("--in", help="candidates jsonl")],
    policy: Annotated[Path, typer.Option(help="data/config/triage/L*.toml")],
    report: Annotated[bool, typer.Option(help="Print the full report")] = True,
    out: Annotated[Path | None, typer.Option(help="Write the report as JSON")] = None,
) -> None:
    """Apply a level policy; accept/reject with reasons."""
    from launder_forge.config import load_triage_policy
    from launder_forge.triage import load_candidates, triage_run

    pol = load_triage_policy(policy)
    cands = load_candidates(in_)
    rep = triage_run(cands, pol)
    if report:
        t = Table("metric", "count")
        t.add_row("candidates", str(rep.total))
        t.add_row("accepted", str(rep.accepted))
        t.add_row("yield %", f"{rep.yield_pct:.1f}")
        console.print(t)
        if rep.constraint_failures:
            ft = Table("failed constraint", "candidates")
            for k, v in sorted(rep.constraint_failures.items(), key=lambda kv: -kv[1]):
                ft.add_row(k, str(v))
            console.print(ft)
        if rep.reason_counts:
            rt = Table("reject reason", "candidates")
            for k, v in sorted(rep.reason_counts.items(), key=lambda kv: -kv[1]):
                rt.add_row(k, str(v))
            console.print(rt)
        for reason, detail in rep.non_evaluable.items():
            _info(f"reject reason {reason!r} is prose, not a predicate: {detail}")
    if out:
        out.write_text(json.dumps(rep.as_dict(), indent=2) + "\n", encoding="utf-8")
        _info(f"wrote {out}")
    _info(
        "every threshold in that policy is a hypothesis nobody has measured; the first "
        "400-candidate run is a calibration experiment whose output is the thresholds."
    )


@app.command()
def tune(
    policy: Annotated[Path, typer.Option(help="data/config/triage/L*.toml")],
    sweep: Annotated[list[str], typer.Option(help="key=v1,v2,v3 (repeatable)")],
    corpus: Annotated[list[Path], typer.Option(help="candidates jsonl (repeatable)")],
    out: Annotated[Path | None, typer.Option(help="Write the sweep as JSON")] = None,
) -> None:
    """Sweep policy params; yield x par tables."""
    from launder_forge.config import load_triage_policy
    from launder_forge.triage import load_candidates
    from launder_forge.tune import parse_sweep
    from launder_forge.tune import sweep as run_sweep

    pol = load_triage_policy(policy)
    cands = [c for path in corpus for c in load_candidates(path)]
    grid = parse_sweep(sweep)
    result = run_sweep(cands, pol, grid)
    t = Table(*result.keys, "accepted", "yield %", "median par", "top blocker")
    for cell in result.cells:
        t.add_row(
            *[str(cell.overrides[k]) for k in result.keys],
            str(cell.accepted),
            f"{cell.yield_pct:.1f}",
            "-" if cell.median_par is None else f"{cell.median_par:g}",
            cell.top_constraint or "-",
        )
    console.print(t)
    if out:
        out.write_text(json.dumps(result.as_dict(), indent=2) + "\n", encoding="utf-8")
        _info(f"wrote {out}")


@app.command()
def calibrate(
    fpr: float = 1e-2,
    n: int = 20000,
    buckets: str = "40,60,80,120,180,260,400",
    corpus: Annotated[
        str, typer.Option(help="synthetic | gemma | gemma-wrong-key | path")
    ] = "synthetic",
    out: Annotated[Path | None, typer.Option()] = None,
    name: Annotated[str, typer.Option(help="calibration bucket set name")] = "default",
    seed: int = 12345,
    force: Annotated[bool, typer.Option(help="Overwrite an existing thresholds file")] = False,
) -> None:
    """Null-distribution sweep -> data/assets/thresholds.v1.json."""
    from launder_forge.calibrate import calibrate as run_calibrate
    from launder_forge.calibrate import write_thresholds
    from launder_forge.corpus import resolve_corpus

    paths, cfg, table = _assets()
    target = out or paths.thresholds
    if target.exists() and not force:
        _bad(f"{target} exists; pass --force to replace it (it changes every z on screen)")
        raise typer.Exit(EXIT_FAIL)

    tokenizer = None
    if corpus not in {"synthetic", "gemma", "gemma-wrong-key"}:
        from launder_forge.tokenizer import load_tokenizer

        tokenizer = load_tokenizer(paths)
    src = resolve_corpus(corpus, tokenizer=tokenizer)

    bucket_list = tuple(int(x) for x in buckets.split(","))
    with console.status(f"scoring {n:,} negatives over {len(bucket_list)} buckets..."):
        curve = run_calibrate(
            src,
            table=table,
            keys=cfg.keys,
            ngram_len=cfg.ngram_len,
            context_history_size=cfg.context_history_size,
            buckets=bucket_list,
            n=n,
            fpr=fpr,
            seed=seed,
        )
    curve.provenance = {
        "source": src.provenance,
        "is_natural_language": src.is_natural_language,
        "n_requested": n,
        "seed": seed,
        "buckets": list(bucket_list),
        "warning": (
            ""
            if src.is_natural_language
            else "NOT a real corpus. Uniform random token ids exercise the pipeline and give the "
            "curve its shape, but they are not English: real prose has repeated n-grams and "
            "therefore a non-trivial masked fraction and correlated rows. TECH_PLAN.md §6.5 "
            "requires negatives that include BOTH human prose AND unwatermarked Gemma-3 output "
            "with a different key. Replace before shipping."
        ),
    }
    t = Table("target T", "samples", "n_scored", "mean", "sd emp", "sd closed", "kappa", "p(1-fpr)")
    for b in curve.buckets:
        t.add_row(
            str(b.target_tokens), f"{b.n_samples:,}", f"{b.mean_n_scored:.1f}",
            f"{b.mean_score:.6f}", f"{b.sd_empirical:.6f}", f"{b.sd_closed_form:.6f}",
            f"{b.inflation:.4f}", f"{b.p_fpr_score:.6f}",
        )  # fmt: skip
    console.print(t)
    write_thresholds(target, curve, wm_config_id=cfg.wm_config_id, name=name)
    _ok(f"wrote {target}  (z* = {curve.z_star:.4f}, source = {src.provenance})")
    if not src.is_natural_language:
        _bad("provenance is synthetic — this is a first cut, not a calibration")


# ---------------------------------------------------------------------------
# pack — the M4-M7 content path (§6.4)
# ---------------------------------------------------------------------------


def _load_candidate(candidates: Path, candidate_id: str) -> dict[str, Any]:
    from launder_forge.triage import load_candidates

    records = load_candidates(candidates)
    for rec in records:
        if rec.candidate_id == candidate_id:
            return rec.raw
    known = ", ".join(r.candidate_id for r in records[:8])
    raise typer.BadParameter(
        f"{candidates} has no candidate {candidate_id!r}. It holds {len(records)} "
        f"records; the first few are: {known}"
    )


@app.command("pack")
def pack_cmd(
    passage_id: Annotated[str, typer.Argument(help="The id to pack under, e.g. p07")],
    level: Annotated[str, typer.Option(help="Level id from data/config/levels.toml")],
    candidates: Annotated[
        Path | None, typer.Option("--in", help="candidates jsonl holding --candidate")
    ] = None,
    candidate: Annotated[str, typer.Option(help="candidate_id inside --in")] = "",
    text_file: Annotated[
        Path | None, typer.Option(help="Pack a text file instead of a candidate record")
    ] = None,
    claims_file: Annotated[
        Path | None, typer.Option("--claims", help="A REVIEWED forge claims draft")
    ] = None,
    par: Annotated[int, typer.Option(help="Shown par (the number the player sees)")] = 0,
    par_source: Annotated[str, typer.Option()] = "solver_upper",
    force: Annotated[bool, typer.Option(help="Overwrite an existing passage")] = False,
) -> None:
    """candidate -> `<id>.public.json` + `.author.json` + `.server.json` (§6.3).

    `pack.py` has always held the real implementation and the four hard gates;
    it had no entry point, so there was NO SUPPORTED WAY TO ADD A PASSAGE and
    `data/passages/` shipped empty — which is why the server had nothing to
    serve against the committed tree. This is that entry point.

    Packing is HALF of adding a level. The campaign is the ordered
    `[[level]]` list in `data/config/progression.toml`, hand-edited: a packed
    passage nothing points at is never played, and a `[[level]]` block naming a
    passage that was never packed is a boot failure (`strict = true`), which is
    the point — a hole in the 8-level campaign is a broken build, not a skipped
    day. Packing a passage the campaign does not name is not an error: the spare
    authored passages in `data/passages/` outnumber the levels on purpose.

    The four gates are `pack_passage`'s, not this function's: `encode(text) ==
    token_ids`, the detector expectations RECOMPUTED (never copied from the
    candidate), `wm_config_id` recomputed from `watermark.toml`, and a claim list
    that came from a reviewed `forge claims` draft.
    """
    from launder_forge.claims import load_reviewed_claims
    from launder_forge.pack import pack_passage, write_passage
    from launder_forge.tokenizer import load_tokenizer

    paths, cfg, table = _assets()
    tokenizer = load_tokenizer(paths)

    if text_file is not None:
        text = text_file.read_text(encoding="utf-8").rstrip("\n")
        record: dict[str, Any] = {}
        token_ids = list(tokenizer.encode(text))
    elif candidates is not None and candidate:
        record = _load_candidate(candidates, candidate)
        text = str(record["text"])
        token_ids = [int(x) for x in record.get("token_ids") or tokenizer.encode(text)]
    else:
        _bad("pass either --text-file, or --in <candidates.jsonl> --candidate <id>")
        raise typer.Exit(EXIT_UNAVAILABLE)

    target = paths.passages / f"{passage_id}.public.json"
    if target.exists() and not force:
        _bad(
            f"{target} exists. Repacking a live passage changes what every player is "
            "scored against and orphans the judge cache; pass --force if that is genuinely "
            "what you want."
        )
        raise typer.Exit(EXIT_FAIL)

    claim_list: list[dict[str, Any]] = []
    if claims_file is not None:
        claim_list = load_reviewed_claims(claims_file)
    else:
        default_draft = paths.passages / f"{passage_id}.claims.draft.json"
        if default_draft.exists():
            claim_list = load_reviewed_claims(default_draft)

    scoring_version = _scoring_version(paths)
    bundle_id = _asset_bundle_id(paths)
    try:
        packed = pack_passage(
            passage_id=passage_id,
            level_id=level,
            text=text,
            token_ids=token_ids,
            cfg=cfg,
            table=table,
            asset_bundle_id=bundle_id,
            scoring_version=scoring_version,
            tokenizer=tokenizer,
            claims=claim_list,
            par=par or int(record.get("par", 0)) or 4,
            par_source=par_source,
            judge_prompt_id=_judge_prompt_id(paths),
            rules=record.get("rules"),
            author_extra={
                k: record[k]
                for k in ("candidate_id", "run_id", "optionality", "topk_alts", "provenance")
                if k in record
            },
            intro=record.get("intro"),
        )
    except ValueError as exc:
        _bad(str(exc))
        raise typer.Exit(EXIT_FAIL) from exc

    written = write_passage(paths, packed)
    t = Table("field", "value")
    t.add_row("passage_id", passage_id)
    t.add_row("level", level)
    t.add_row("words", str(packed.public.n_words))
    t.add_row("n_scored", str(packed.public.detector.expected_n_scored))
    t.add_row("expected_z", f"{packed.public.detector.expected_z:.4f}")
    t.add_row("g_digest", packed.public.detector.g_digest[:26] + "...")
    t.add_row("claims", str(len(packed.public.claims)))
    t.add_row("par", str(packed.public.par))
    console.print(t)
    for kind, path in written.items():
        _info(f"{kind}: {paths.rel(path)}")
    _ok(f"packed {passage_id}")
    _info(
        f"add a [[level]] block for {passage_id} to data/config/progression.toml, "
        "then `forge manifest --write`: asset_bundle_id and the blake3 record change with it"
    )


def _read_toml(path: Path) -> dict[str, Any]:
    import tomllib

    with path.open("rb") as fh:
        data: dict[str, Any] = tomllib.load(fh)
    return data


def _scoring_version(paths: Any) -> str:
    return str(_read_toml(paths.scoring_toml).get("scoring_version", "sc1"))


def _judge_prompt_id(paths: Any) -> str:
    return str(_read_toml(paths.judge_toml).get("prompt_id", "judge.observe.v3"))


def _asset_bundle_id(paths: Any) -> str:
    """From `data/MANIFEST.json` when it exists, else computed the same way.

    `forge manifest --write` owns the canonical value; packing before the
    manifest exists is normal (the manifest hashes the passages), so this
    computes the same thing rather than refusing.
    """
    if paths.manifest.exists():
        value = str(json.loads(paths.manifest.read_text(encoding="utf-8")).get("asset_bundle_id"))
        if value:
            return value
    from launder_forge.manifest import asset_bundle_id

    return asset_bundle_id(paths)


@app.command()
def claims(
    passage_id: Annotated[str, typer.Argument()],
    text_file: Annotated[
        Path | None, typer.Option(help="Passage text; defaults to the packed passage")
    ] = None,
    out: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Draft a claim list for human review."""
    from launder_forge.claims import draft_claims, write_draft

    paths = repo_paths()
    if text_file is not None:
        text = text_file.read_text(encoding="utf-8").strip()
    else:
        public = paths.passages / f"{passage_id}.public.json"
        if not public.exists():
            _bad(f"{public} does not exist; pass --text-file")
            raise typer.Exit(EXIT_UNAVAILABLE)
        text = json.loads(public.read_text(encoding="utf-8"))["text"]

    draft = draft_claims(passage_id, text)
    target = out or (paths.ensure(paths.passages) / f"{passage_id}.claims.draft.json")
    write_draft(target, draft)
    t = Table("id", "conf", "label", "text")
    for c in draft.claims:
        t.add_row(c.id, f"{c.confidence:.2f}", c.label, c.text[:60])
    console.print(t)
    _ok(f"wrote {target}")
    _info("reviewed: false — a human must read this before `forge pack` will accept it")


@app.command()
def vectors(
    check: Annotated[
        bool, typer.Option(help="Recompute and compare against the committed file")
    ] = True,
    write: Annotated[bool, typer.Option(help="Rewrite vectors.json (needs --force)")] = False,
    force: Annotated[
        bool, typer.Option(help="Allow overwriting the committed golden file")
    ] = False,
    checksum: Annotated[bool, typer.Option(help="Rewrite data/golden/CHECKSUM")] = False,
) -> None:
    """Emit / verify data/golden/*.

    Default mode is VERIFY. Regenerating goldens is always a reviewed diff and
    never a way to turn a red test green.
    """
    from launder_forge.vectors import emit_vectors, verify_vectors, write_checksum

    paths, cfg, table = _assets()
    ok = True
    if check and paths.vectors_json.exists():
        result = verify_vectors(paths, cfg, table)
        for m in result.mismatches:
            _bad(m)
        for s in result.skipped:
            _info(s)
        (_ok if result.ok else _bad)(f"{result.checked} golden assertions recomputed")
        ok = result.ok
    elif check:
        _bad(f"{paths.vectors_json} does not exist")
        ok = False

    if write:
        if paths.vectors_json.exists() and not force:
            _bad("vectors.json exists; --force is required to overwrite a committed golden file")
            raise typer.Exit(EXIT_FAIL)
        payload = emit_vectors(paths, cfg, table)

        # NO SILENT DOWNGRADES.
        #
        # `--force` used to be the only gate, and the emitter was quietly
        # thinner than the committed file: whole field groups the vitest parity
        # suite reads (`sampling_table`, `whitespace_class`, `insertion_case`,
        # the per-token `heat`/`masked` arrays) existed only because the file
        # had been built by hand. One `--write --force` deleted them, took the
        # golden file from 123 assertions to 71, and left `forge vectors` dying
        # on KeyError. The file was only recoverable from a Docker image built
        # before the run.
        #
        # A regenerator that cannot reproduce what it is overwriting has no
        # business overwriting it. Losing a key is now a refusal, not a diff.
        lost = _lost_golden_fields(paths.vectors_json, payload)
        if lost:
            _bad(
                "refusing to write: the emitter would DROP "
                + ", ".join(lost)
                + ". These are read by web/tests/parity.test.ts and web/tools/parity.mjs; "
                "regenerating without them silently weakens the one file that catches a "
                "Python/TS divergence. Teach emit_vectors to emit them, then re-run."
            )
            raise typer.Exit(EXIT_FAIL)

        paths.vectors_json.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        _ok(f"wrote {paths.vectors_json}")
    if checksum:
        _ok(f"wrote {write_checksum(paths)}")
    raise typer.Exit(EXIT_OK if ok else EXIT_FAIL)


@app.command()
def verify(
    node: Annotated[bool, typer.Option(help="Run the TS detector under node")] = True,
    json_out: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Recompute every checksum; cross-check the TS detector via node."""
    from launder_forge.verify import verify_all

    paths, cfg, _ = _assets()
    report = verify_all(paths, cfg, run_node=node)
    if json_out:
        console.print_json(json.dumps(report.as_dict()))
    else:
        t = Table("check", "ok", "detail")
        for name, ok, detail in report.checks:
            t.add_row(name, "yes" if ok else "NO", detail[:110])
        console.print(t)
    (_ok if report.ok else _bad)(
        "everything reproduces" if report.ok else f"{len(report.failures)} checks failed"
    )
    raise typer.Exit(EXIT_OK if report.ok else EXIT_FAIL)


@app.command("lint-copy")
def lint_copy_cmd(
    json_out: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """§10.7's two rules: every REGISTRY check has label/blurb/reject, and every
    {placeholder} resolves against declared params."""
    from launder_forge.lintcopy import lint_copy

    paths = repo_paths()
    report = lint_copy(paths)
    if json_out:
        console.print_json(json.dumps(report.as_dict()))
    else:
        for name in report.missing_blocks:
            _bad(f"rule 1: no [check.{name}] block in copy.toml")
        for key in report.missing_keys:
            _bad(f"rule 1: {key} is missing or empty")
        for item in report.unresolved:
            _bad(f"rule 2: {item}")
        for note in report.notes:
            _info(note)
        _info(f"check names from {report.registry_source}; params from {report.param_source}")
        (_ok if report.ok else _bad)(f"{report.checked_templates} templates checked")
    raise typer.Exit(EXIT_OK if report.ok else EXIT_FAIL)


@app.command("wm-config")
def wm_config_cmd(
    force: Annotated[bool, typer.Option(help="Overwrite an existing file")] = False,
) -> None:
    """Write data/assets/wm_config.v1.json — the exact config, canonical JSON.

    It is one of the four inputs to `asset_bundle_id`, in a fixed order, so its
    bytes are load-bearing.
    """
    from launder_core.schemas import canonical_json

    paths, cfg, _ = _assets()
    if paths.wm_config_asset.exists() and not force:
        _bad(f"{paths.wm_config_asset} exists; --force to replace (it changes asset_bundle_id)")
        raise typer.Exit(EXIT_FAIL)
    payload = {
        "context_history_size": cfg.context_history_size,
        "keys": list(cfg.keys),
        "ngram_len": cfg.ngram_len,
        "sampling_table_seed": cfg.sampling_table_seed,
        "sampling_table_size": cfg.sampling_table_size,
        "skip_first_ngram_calls": cfg.skip_first_ngram_calls,
        "wm_config_id": cfg.wm_config_id,
        "scheme": "sampling_table",
    }
    paths.ensure(paths.assets)
    paths.wm_config_asset.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    _ok(f"wrote {paths.wm_config_asset}  ({cfg.wm_config_id})")


@app.command()
def manifest(
    write: Annotated[bool, typer.Option(help="Write data/MANIFEST.json")] = False,
) -> None:
    """Recompute blake3 + bytes for every shipped artifact."""
    from launder_forge.doctor import doctor_report
    from launder_forge.manifest import build_manifest, write_manifest

    paths = repo_paths()
    payload = build_manifest(paths, env=doctor_report(matmul_size=256))
    console.print(f"{len(payload['files'])} files, asset_bundle_id = {payload['asset_bundle_id']}")
    if write:
        _ok(f"wrote {write_manifest(paths, payload)}")


def main() -> None:  # pragma: no cover - console entry point
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
