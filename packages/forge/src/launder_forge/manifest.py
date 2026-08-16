"""`data/MANIFEST.json` — blake3 + bytes for every artifact, plus provenance.

TECH_PLAN.md §6.6:

    asset_bundle_id = "ab1:" + blake3(sampling_table || wm_config || thresholds || tokenizer_blob)

The order is fixed and the concatenation is over raw file bytes. `forge verify`
recomputes every entry, which is what turns "somebody edited an asset and
forgot to regenerate" into a loud failure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from launder_forge.paths import Paths

__all__ = [
    "ASSET_BUNDLE_ORDER",
    "ManifestEntry",
    "asset_bundle_id",
    "blake3_file",
    "build_manifest",
    "load_manifest",
    "write_manifest",
]

#: The concatenation order for `asset_bundle_id`. Changing it changes the id
#: and therefore invalidates every passage; it is a constant, not a knob.
ASSET_BUNDLE_ORDER: tuple[str, ...] = (
    "data/assets/sampling_table.v1.bin",
    "data/assets/wm_config.v1.json",
    "data/assets/thresholds.v1.json",
    "data/assets/gemma3-tok.v1.bin.br",
)


def blake3_file(path: Path) -> str:
    import blake3

    h = blake3.blake3()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    path: str
    bytes: int
    blake3: str

    def as_dict(self) -> dict[str, Any]:
        return {"bytes": self.bytes, "blake3": f"blake3:{self.blake3}"}


def asset_bundle_id(paths: Paths) -> str:
    import blake3

    h = blake3.blake3()
    missing: list[str] = []
    for rel in ASSET_BUNDLE_ORDER:
        p = paths.root / rel
        if not p.exists():
            missing.append(rel)
            continue
        h.update(p.read_bytes())
    if missing:
        raise FileNotFoundError(
            "cannot compute asset_bundle_id; missing " + ", ".join(missing) + ". "
            "wm_config.v1.json comes from `forge tok pack`/`forge table build` bookkeeping and "
            "thresholds.v1.json from `forge calibrate`."
        )
    return "ab1:" + h.hexdigest()


def build_manifest(
    paths: Paths, *, env: dict[str, Any] | None = None, conformance: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Hash every shipped artifact under `data/`, excluding build inputs."""
    files: dict[str, Any] = {}
    roots = [paths.assets, paths.config, paths.passages, paths.regen, paths.golden]
    for root in roots:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            rel = paths.rel(p)
            files[rel] = ManifestEntry(rel, p.stat().st_size, blake3_file(p)).as_dict()
    try:
        bundle_id: str | None = asset_bundle_id(paths)
        bundle_note = ""
    except FileNotFoundError as exc:
        # Recorded rather than raised: a manifest of what DOES exist is useful
        # mid-pipeline, and `forge verify` reports the null id as its own line.
        bundle_id, bundle_note = None, str(exc)
    return {
        "schema": "launder.manifest/1",
        "asset_bundle_id": bundle_id,
        "asset_bundle_id_blocked_by": bundle_note,
        "files": files,
        "env": env or {},
        "conformance": conformance or {},
    }


def write_manifest(paths: Paths, manifest: dict[str, Any]) -> Path:
    paths.manifest.parent.mkdir(parents=True, exist_ok=True)
    paths.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return paths.manifest


def load_manifest(paths: Paths) -> dict[str, Any]:
    if not paths.manifest.exists():
        raise FileNotFoundError(
            f"{paths.manifest} does not exist. Run `forge pack` (which appends entries) or "
            "`forge verify --rebuild` to create it."
        )
    data: dict[str, Any] = json.loads(paths.manifest.read_text(encoding="utf-8"))
    return data
