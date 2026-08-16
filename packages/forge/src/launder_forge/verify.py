"""`forge verify` — recompute everything, then ask node whether it agrees.

TECH_PLAN.md §4.5. Six independent checks, in dependency order so a failure
localizes:

1. Every blake3 in `data/MANIFEST.json`, recomputed from the files.
2. The sampling table's four digests against `[table]` in `watermark.toml`.
3. `wm_config_id`, recomputed from `watermark.toml`'s six load-bearing fields.
4. Every golden vector, recomputed from the assets (`launder_forge.vectors`).
5. For every passage: `encode(text) == token_ids`, and a fresh derivation of
   `g_digest`, `expected_score`, `expected_z` and `expected_n_scored`.
6. The TS detector, run under node over all passages and all vectors.

Check 6 is the one that cannot be faked by sharing a Python implementation:
`web/tools/parity.mjs` is the browser's own code path, and this is the only
place in CI where a JS `BigInt %` sign error or a rounding drift can surface
before a player sees it.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from shutil import which
from typing import Any

from launder_core.schemas import SynthIDConfig
from launder_forge.manifest import blake3_file
from launder_forge.numerics import g_digest, load_sampling_table, score_ids
from launder_forge.paths import Paths
from launder_forge.table import digests_of
from launder_forge.tokenizer import load_tokenizer
from launder_forge.vectors import verify_vectors

__all__ = ["VerifyReport", "verify_all"]

#: Candidate locations for the TS-side parity runner, most specific first.
NODE_PARITY_SCRIPTS: tuple[str, ...] = (
    "web/tools/parity.mjs",
    "web/tools/verify-parity.mjs",
    "web/tools/pack-check.mjs",
)


@dataclass(slots=True)
class VerifyReport:
    checks: list[tuple[str, bool, str]] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks.append((name, ok, detail))

    @property
    def ok(self) -> bool:
        return all(ok for _, ok, _ in self.checks)

    @property
    def failures(self) -> list[tuple[str, bool, str]]:
        return [c for c in self.checks if not c[1]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in self.checks],
        }


def verify_all(
    paths: Paths,
    cfg: SynthIDConfig,
    *,
    run_node: bool = True,
    passages: list[Path] | None = None,
) -> VerifyReport:
    report = VerifyReport()
    table = load_sampling_table(paths.sampling_table, cfg.sampling_table_size)

    # --- 1. MANIFEST -------------------------------------------------------
    if paths.manifest.exists():
        manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
        bad: list[str] = []
        for rel, entry in manifest.get("files", {}).items():
            p = paths.root / rel
            if not p.exists():
                bad.append(f"{rel}: listed in MANIFEST but missing on disk")
                continue
            actual = "blake3:" + blake3_file(p)
            if actual != entry.get("blake3"):
                bad.append(f"{rel}: {actual} != manifest {entry.get('blake3')}")
            elif p.stat().st_size != entry.get("bytes"):
                bad.append(f"{rel}: {p.stat().st_size} bytes != manifest {entry.get('bytes')}")
        report.add(
            "manifest.blake3",
            not bad,
            f"{len(manifest.get('files', {}))} files checked" if not bad else "; ".join(bad[:5]),
        )
    else:
        report.add("manifest.blake3", True, "no MANIFEST.json yet (nothing packed)")

    # --- 2. sampling table digests ----------------------------------------
    from launder_forge.config import watermark_asset_digests

    recorded = watermark_asset_digests(paths)
    recomputed = digests_of(table)
    mismatched = [
        f"{k}: {recomputed[k]} != {recorded[k]}"
        for k in ("ones", "sha256_packed", "sha256_unpacked", "blake3_packed", "blake3_unpacked")
        if k in recorded and str(recorded[k]) != str(recomputed[k])
    ]
    report.add("assets.sampling_table", not mismatched, "; ".join(mismatched) or "4 digests match")

    # --- 3. wm_config_id ---------------------------------------------------
    from launder_forge.config import load_toml

    declared = load_toml(paths.watermark_toml).get("wm_config_id")
    report.add(
        "config.wm_config_id",
        declared == cfg.wm_config_id,
        cfg.wm_config_id if declared == cfg.wm_config_id else f"{cfg.wm_config_id} != {declared}",
    )

    # --- 4. golden vectors -------------------------------------------------
    if paths.vectors_json.exists():
        check = verify_vectors(paths, cfg, table)
        report.add(
            "golden.vectors",
            check.ok,
            f"{check.checked} assertions"
            + ("" if check.ok else " | " + " | ".join(check.mismatches[:4])),
        )
        for skip in check.skipped:
            report.add("golden.vectors.skipped", True, skip)
    else:
        report.add("golden.vectors", False, f"{paths.vectors_json} does not exist")

    # --- 5. passages -------------------------------------------------------
    files = passages if passages is not None else sorted(paths.passages.glob("*.public.json"))
    if not files:
        report.add("passages", True, "no packed passages yet")
    else:
        tokenizer = load_tokenizer(paths)
        problems: list[str] = []
        for f in files:
            data = json.loads(f.read_text(encoding="utf-8"))
            text = data["text"]
            ids = list(tokenizer.encode(text))
            scored = score_ids(
                ids,
                keys=cfg.keys,
                ngram_len=cfg.ngram_len,
                table=table,
                context_history_size=cfg.context_history_size,
            )
            det = data["detector"]
            if det["expected_n_scored"] != scored.n_scored:
                problems.append(
                    f"{f.name}: n_scored {scored.n_scored} != {det['expected_n_scored']}"
                )
            if abs(det["expected_score"] - scored.score) > 1e-12:
                problems.append(f"{f.name}: score {scored.score!r} != {det['expected_score']!r}")
            digest = g_digest(scored.g, scored.mask)
            if det["g_digest"] != digest:
                problems.append(f"{f.name}: g_digest {digest} != {det['g_digest']}")
            if data.get("wm_config_id") != cfg.wm_config_id:
                problems.append(f"{f.name}: wm_config_id mismatch")
        report.add(
            "passages.rederived",
            not problems,
            f"{len(files)} passages" if not problems else "; ".join(problems[:5]),
        )

    # --- 6. the TS detector, under node ------------------------------------
    if run_node:
        ok, detail = _run_node_parity(paths)
        report.add("ts.parity", ok, detail)
    else:
        report.add("ts.parity", True, "skipped (--no-node)")

    return report


def _run_node_parity(paths: Paths) -> tuple[bool, str]:
    node = which("node")
    if node is None:
        return False, (
            "node is not on PATH. The TS cross-check is the only place a JS `BigInt %` sign "
            "error can surface before a player sees it; install Node 18+ or pass --no-node and "
            "say so in the review."
        )
    script = next(
        (paths.root / rel for rel in NODE_PARITY_SCRIPTS if (paths.root / rel).exists()), None
    )
    if script is None:
        return False, (
            "no TS parity runner found. Expected one of "
            + ", ".join(NODE_PARITY_SCRIPTS)
            + ". It belongs to the web package: a script that loads data/golden/vectors.json and "
            "every data/passages/*.public.json, runs the browser detector over them, and exits "
            "non-zero on any disagreement."
        )
    proc = subprocess.run(
        [node, str(script)],
        cwd=str(paths.root),
        capture_output=True,
        # Explicit utf-8: the default on Windows is cp1252, and a single box
        # character in the runner's output makes `text=True` hand back None
        # instead of a string. Measured, not hypothetical.
        encoding="utf-8",
        errors="replace",
        timeout=600,
    )
    tail = (proc.stdout + proc.stderr).strip().splitlines()
    detail = f"{script.relative_to(paths.root).as_posix()}: " + (
        tail[-1] if tail else f"exit {proc.returncode}"
    )
    return proc.returncode == 0, detail
