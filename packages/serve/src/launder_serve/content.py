"""The read-only content layer: `data/config/*.toml`, passages, digests.

`data/` is THE CACHE (TECH_PLAN.md §3): written by forge, committed, read-only
in production. This module loads it once at boot, asserts the invariants that
make the numbers on screen mean something, and hands the result to the app as
an immutable `Content` object.

Three assertions are load-bearing and all three fail at boot:

1. **`wm_config_id`** computed from `watermark.toml` matches the id recorded in
   the same file, and every passage's declared `wm_config_id` matches it.
   A mismatch means the passages were generated under a different watermark and
   every number on screen would be fiction (§4.5 "runtime tripwire").
2. **The sampling table digest.** The table is key material (§4.3 #3); a table
   built on the wrong device gives g-values uncorrelated with the watermark —
   a needle that moves, looks fine, and measures nothing.
3. **`prompt_hash`** — asserted in `judge/prompt.py`, not here, because it
   needs the rendered prompt.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from launder_core.gates.checks.close_paraphrase import assert_tables_match_files
from launder_core.gates.feedback import CopyBook, load_copy
from launder_core.levels import load_levels
from launder_core.schemas import (
    DetectorExpectation,
    LevelConfig,
    LevelsFile,
    PassagePublic,
    PassageServer,
    ScoringConfig,
    SynthIDConfig,
    canonical_json,
)
from launder_core.watermark import assert_scoring_eos

__all__ = [
    "DEV_PASSAGE_ID",
    "Content",
    "DailySlotSpec",
    "JudgeConfig",
    "ScheduleFile",
    "WatermarkFile",
    "error_message",
    "load_content",
]

#: Keys of `scoring.toml` that `ScoringConfig` (extra="forbid") accepts. The
#: file also carries `[words]`, `[distance]` and `[lemma]` tables that belong to
#: core's normalizer and lemma table; feeding them to the model would raise.
_SCORING_SCALARS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "scoring_version",
        "nfc",
        "collapse_whitespace",
        "fold_smart_quotes",
        "fold_dashes",
        "fold_ellipsis",
        "nbsp_to_space",
        "lowercase",
        "strip_punctuation",
    }
)


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing. `data/` is committed and read-only in production "
            "(TECH_PLAN.md §3); a missing config file is a broken build, not a "
            "runtime condition."
        )
    with path.open("rb") as fh:
        data: dict[str, Any] = tomllib.load(fh)
    return data


# ---------------------------------------------------------------------------
# watermark.toml
# ---------------------------------------------------------------------------


class TableRecord(BaseModel):
    """The sampling table's recorded shape and digests."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    path: str
    format: str = "packbits_msb_first"
    packed_bytes: int = 8192
    values: int = 65536
    ones: int = 0
    sha256_packed: str = ""
    sha256_unpacked: str = ""
    blake3_packed: str = ""
    blake3_unpacked: str = ""


class WatermarkFile(BaseModel):
    """`data/config/watermark.toml`. CHANGING IT INVALIDATES EVERY PASSAGE."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    config: SynthIDConfig
    declared_wm_config_id: str
    scheme: str
    table: TableRecord
    tokenizer_blob: str = ""
    tokenizer_blob_blake3: str = ""
    #: `[scoring]` as written, kept so `assert_consistent` can check the eos
    #: policy the two runtimes have to agree on.
    scoring_block: dict[str, Any] = Field(default_factory=dict)
    source_path: str = ""

    @classmethod
    def load(cls, path: Path) -> WatermarkFile:
        raw = _read_toml(path)
        cfg = SynthIDConfig(
            ngram_len=int(raw["ngram_len"]),
            keys=tuple(int(k) for k in raw["keys"]),
            context_history_size=int(raw["context_history_size"]),
            sampling_table_size=int(raw["sampling_table_size"]),
            sampling_table_seed=int(raw["sampling_table_seed"]),
            skip_first_ngram_calls=bool(raw["skip_first_ngram_calls"]),
        )
        model = raw.get("model", {})
        return cls(
            config=cfg,
            declared_wm_config_id=str(raw["wm_config_id"]),
            scheme=str(raw.get("scheme", "sampling_table")),
            table=TableRecord.model_validate(raw.get("table", {})),
            tokenizer_blob=str(model.get("tokenizer_blob", "")),
            tokenizer_blob_blake3=str(model.get("tokenizer_blob_blake3", "")),
            scoring_block=dict(raw.get("scoring", {})),
            source_path=str(path),
        )

    @property
    def wm_config_id(self) -> str:
        return self.config.wm_config_id

    def assert_consistent(self, data_root: Path) -> None:
        """Boot assertion 1 and 2. Raises with the computed value in the message."""
        self.config.assert_id(self.declared_wm_config_id)
        # The eos policy the browser and the server BOTH have to pass explicitly.
        assert_scoring_eos(self.scoring_block, self.source_path or "watermark.toml")
        if self.scheme != "sampling_table":
            raise ValueError(
                f"watermark.toml declares scheme={self.scheme!r}. This project is the "
                "SAMPLING-TABLE SynthID variant, permanently: the DeepMind `main` "
                "iterated-LCG variant is bit-incompatible and cannot detect anything "
                "the HF processor generated (TECH_PLAN.md §4.1)."
            )
        table_path = (
            data_root.parent / self.table.path
            if not Path(self.table.path).is_absolute()
            else Path(self.table.path)
        )
        if not table_path.is_file():
            # `path` in the toml is repo-relative ("data/assets/...").
            table_path = data_root / "assets" / Path(self.table.path).name
        if not table_path.is_file():
            raise FileNotFoundError(
                f"sampling table {self.table.path} not found. It is KEY MATERIAL "
                "(TECH_PLAN.md §4.3): it is built once on CPU and committed, never "
                "regenerated by library code."
            )
        packed = table_path.read_bytes()
        if len(packed) != self.table.packed_bytes:
            raise ValueError(
                f"sampling table {table_path} is {len(packed)} bytes, "
                f"watermark.toml records {self.table.packed_bytes}."
            )
        if self.table.sha256_packed:
            actual = hashlib.sha256(packed).hexdigest()
            if actual != self.table.sha256_packed:
                raise ValueError(
                    f"sampling table digest mismatch: file is sha256 {actual}, "
                    f"watermark.toml records {self.table.sha256_packed}. The table is "
                    "key material; a different table means every g-value, every "
                    "threshold and every passage in data/ is invalid."
                )


# ---------------------------------------------------------------------------
# judge.toml
# ---------------------------------------------------------------------------


class JudgeSchemaOut(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    name: str = "launder_gate_observation"
    strict: bool = True
    additional_properties: bool = False
    required: tuple[str, ...] = ()
    unnatural_kinds: tuple[str, ...] = ()
    claim_how: tuple[str, ...] = ()
    max_added_claims_reported: int = 3


class JudgeVerdictPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    injection_fails_closed: bool = True
    unnatural_fails: bool = True
    use_verdict_opinion: bool = False


class JudgeCachePolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    key_fields: tuple[str, ...] = ()
    key_separator_codepoint: int = 31
    scope: str = "global"
    cache_failures: bool = True
    #: A provisional clear must NEVER become permanent.
    cache_errors: Literal[False] = False
    ttl_seconds: int = 0
    lru_entries: int = 2000

    @property
    def separator(self) -> str:
        return chr(self.key_separator_codepoint)


class JudgeLimits(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    daily_usd_cap: float = 2.0
    rate_limit_per_hour: int = 10
    rate_limit_burst: int = 3
    #: Railway's DOCUMENTED header. X-Forwarded-For is not in the documented set
    #: (§14.2 item 6).
    rate_limit_header: str = "X-Real-IP"
    #: A MISSING header means ONE SHARED BUCKET, never "unlimited".
    missing_header_policy: Literal["shared_bucket"] = "shared_bucket"
    pow_enabled: bool = False


class JudgeFailurePolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    on_provider_error: str = "fail_open_provisional"
    on_injection: str = "fail_closed"
    on_spend_cap: str = "fail_open_provisional"


class JudgeConfig(BaseModel):
    """`data/config/judge.toml`."""

    model_config = ConfigDict(frozen=True, extra="ignore", populate_by_name=True)

    schema_id: str = Field(default="launder.judge/1", alias="schema")
    judge_version: str = "g3"
    prompt_id: str = "judge.observe.v3"

    provider: str = "openai"
    failover_provider: str = "anthropic"
    model: str = ""
    failover_model: str = ""
    reasoning_effort: str = "none"
    temperature: float = 0.0
    max_output_tokens: int = 300
    timeout_s: float = 6.0
    max_retries: int = 1
    shadow_sample_rate: float = 0.0

    prompt_hash: str = ""
    prompt_hash_enforced: bool = True

    schema_out: JudgeSchemaOut = JudgeSchemaOut()
    verdict: JudgeVerdictPolicy = JudgeVerdictPolicy()
    cache: JudgeCachePolicy = JudgeCachePolicy()
    limits: JudgeLimits = JudgeLimits()
    failure_policy: JudgeFailurePolicy = JudgeFailurePolicy()

    @classmethod
    def load(cls, path: Path) -> JudgeConfig:
        return cls.model_validate(_read_toml(path))

    @property
    def prompt_hash_is_placeholder(self) -> bool:
        return (not self.prompt_hash) or self.prompt_hash.startswith("PLACEHOLDER")


# ---------------------------------------------------------------------------
# schedule.toml
# ---------------------------------------------------------------------------


class DailySlotSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    date: date
    passage_id: str
    level_id: str
    notes: str = ""


class IntroSlot(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    passage_id: str = "p_intro"
    level_id: str = "L1"


class ScheduleFile(BaseModel):
    """`data/config/schedule.toml`: date -> passage_id -> level_id.

    DAILY ROLLOVER IS UTC MIDNIGHT (§9.8). `strict = false` means a scheduled
    day whose passage file is missing is SKIPPED WITH A BOOT WARNING rather
    than crashing the server — the static game is playable without /api/daily.
    """

    model_config = ConfigDict(frozen=True, extra="ignore", populate_by_name=True)

    schema_id: str = Field(default="launder.schedule/1", alias="schema")
    epoch: date
    first_number: int = 1
    strict: bool = False
    intro: IntroSlot = IntroSlot()
    days: tuple[DailySlotSpec, ...] = Field(default=(), alias="day")

    @model_validator(mode="after")
    def _days_are_on_or_after_the_epoch(self) -> ScheduleFile:
        """A day before `epoch` yields `puzzle_number <= 0`.

        `DailyResponse.puzzle_number` is `ge=1`, so such a day is a 500 the
        first time somebody plays it — a boot-clean server that breaks at play
        time on a date nobody tested. This repo's rule is the other way round
        (§10.7): fail at boot, naming the file and the fix.
        """
        bad = [slot for slot in self.days if (slot.date - self.epoch).days + self.first_number < 1]
        if bad:
            raise ValueError(
                "schedule.toml schedules "
                + ", ".join(f"{s.date.isoformat()} ({s.passage_id})" for s in bad)
                + f" before its own epoch {self.epoch.isoformat()} (first_number="
                f"{self.first_number}), which gives a puzzle number below 1. Move the "
                "day, or move the epoch back — but note that moving the epoch "
                "renumbers every share string ever posted."
            )
        return self

    @classmethod
    def load(cls, path: Path) -> ScheduleFile:
        return cls.model_validate(_read_toml(path))

    def puzzle_number(self, day: date) -> int:
        """`(day - epoch).days + first_number` — a pure function of the date.

        Fixed so the share string never drifts if a day is inserted or removed.
        """
        return (day - self.epoch).days + self.first_number

    def for_date(self, day: date) -> DailySlotSpec | None:
        for slot in self.days:
            if slot.date == day:
                return slot
        return None


# ---------------------------------------------------------------------------
# copy.toml
# ---------------------------------------------------------------------------


def error_message(book: CopyBook, code: str, **params: object) -> str:
    """`[errors]` from copy.toml. Errors are specific and never apologize (§10.6).

    `CopyBook` owns `[check.*]`; `[errors]` is the HTTP layer's slice of the same
    file, so it is read through the same object rather than re-parsed. A template
    whose placeholders the caller did not supply renders raw rather than 500ing
    on a player who has done nothing wrong.
    """
    raw: Any = book.raw.get("errors", {})
    template = str(raw.get(code, "")) if isinstance(raw, Mapping) else ""
    if not template:
        return ""
    try:
        return template.format(**params)
    except (KeyError, IndexError, ValueError):
        return template


# ---------------------------------------------------------------------------
# passages
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PassageBundle:
    """A public passage plus the packed server sidecar (`claims`, `par`).

    The Docker image excludes `*.author.json` — it holds the answer key — so the
    claim list the judge sees comes from `<id>.server.json` when present and
    falls back to the public bundle's own `claims` otherwise (§11.3).
    """

    public: PassagePublic
    server: PassageServer | None = None

    @property
    def id(self) -> str:
        return self.public.id

    @property
    def claims(self) -> tuple[Any, ...]:
        if self.server is not None and self.server.claims:
            return self.server.claims
        return self.public.claims

    @property
    def par(self) -> int:
        return self.server.par if self.server is not None else self.public.par


#: The id the dev bundle is registered under. `web/index.html`'s checked-in
#: fixture names it, `schedule.toml`'s `[intro]` names it, and `boot.py` marks a
#: page built from it `dev: true`.
DEV_PASSAGE_ID: Final[str] = "p_dev"


def _dev_bundle(
    data_root: Path,
    *,
    wm_config_id: str,
    asset_bundle_id: str,
    scoring_version: str,
    judge_prompt_id: str,
    level_id: str,
) -> PassageBundle | None:
    """`data/dev/passage.txt` as a REAL passage, so a fresh clone is playable.

    `data/passages/` is empty in the committed tree and cannot be filled without
    a GPU and the gated Gemma-3 weights (§6.1, §6.2). Without this, `GET /` has
    nothing to render, `/api/detect` 404s on every keystroke and the needle never
    moves — which is exactly the state an audit found the shipped page in.

    What this is NOT: a daily. It is not scheduled, it is marked `dev: true` in
    the boot payload, and it is refused outright in production. The fixture is
    HUMAN PROSE reading z = 1.31 against a notch of 2.3263, i.e. it starts
    already under the line — a coherent thing to develop against and not a
    puzzle, which is why publishing it as a daily would be a lie rather than a
    shortcut.

    The four detector expectations are COMPUTED here by the same core the server
    scores with, never authored. `forge publish` is the supported path for a real
    passage.
    """
    fixture = data_root / "dev" / "passage.txt"
    if not fixture.is_file():
        return None
    # `.txt` files end with a newline because text files do; the PASSAGE is the
    # content without it. `web/tools/parity.mjs` and golden case 6 apply the
    # same rule, and a mismatch here would desync the browser on keystroke zero.
    text = fixture.read_text(encoding="utf-8").rstrip("\n")
    if not text:
        return None

    import blake3
    import numpy as np

    from launder_core.detect import detect_ids
    from launder_core.tokenizer import TokenizerUnavailable, encode_with_offsets
    from launder_core.watermark import compute_frame

    try:
        ids, _offsets = encode_with_offsets(text)
    except TokenizerUnavailable:
        return None

    frame = compute_frame(ids)
    reading = detect_ids(ids)
    digest = blake3.blake3()
    digest.update(np.ascontiguousarray(frame.g, dtype=np.uint8).tobytes())
    digest.update(b"|")
    digest.update(np.ascontiguousarray(frame.mask, dtype=np.uint8).tobytes())

    public = PassagePublic(
        id=DEV_PASSAGE_ID,
        level_id=level_id,
        wm_config_id=wm_config_id,
        asset_bundle_id=asset_bundle_id,
        scoring_version=scoring_version,
        text=text,
        n_words=len(text.split()),
        detector=DetectorExpectation(
            expected_n_scored=reading.result.n_scored,
            expected_score=reading.result.score,
            expected_z=reading.z,
            g_digest="blake3:" + digest.hexdigest(),
        ),
        claims=(),
        par=1,
        par_source="authored_reference",
        judge_prompt_id=judge_prompt_id,
    )
    return PassageBundle(public=public)


def _load_passages(
    passages_dir: Path, expected_wm: str
) -> tuple[dict[str, PassageBundle], list[str]]:
    bundles: dict[str, PassageBundle] = {}
    warnings: list[str] = []
    if not passages_dir.is_dir():
        return bundles, [f"{passages_dir} does not exist: no dailies will be served."]
    for public_path in sorted(passages_dir.glob("*.public.json")):
        raw = json.loads(public_path.read_text(encoding="utf-8"))
        public = PassagePublic.model_validate(raw)
        if public.wm_config_id != expected_wm:
            raise ValueError(
                f"{public_path.name} declares wm_config_id {public.wm_config_id} but "
                f"data/config/watermark.toml computes {expected_wm}. The passage was "
                "generated under a different watermark; every number the needle shows "
                "for it would be fiction (TECH_PLAN.md §4.5)."
            )
        sidecar_path = public_path.with_name(
            public_path.name.replace(".public.json", ".server.json")
        )
        sidecar: PassageServer | None = None
        if sidecar_path.is_file():
            sidecar = PassageServer.model_validate(
                json.loads(sidecar_path.read_text(encoding="utf-8"))
            )
        bundles[public.id] = PassageBundle(public=public, server=sidecar)
    return bundles, warnings


# ---------------------------------------------------------------------------
# asset_bundle_id
# ---------------------------------------------------------------------------


def _compute_asset_bundle_id(assets_dir: Path) -> str:
    """Provisional id over `data/assets/*`, used only when MANIFEST.json is absent.

    `forge pack` owns the canonical definition and writes it into
    `data/MANIFEST.json`. This fallback exists so a fresh clone boots; it is a
    boot FAILURE in production, because serving a made-up bundle id would make
    the browser's conformance assertion pass against the wrong assets.
    """
    digests: dict[str, str] = {}
    if assets_dir.is_dir():
        for path in sorted(assets_dir.iterdir()):
            if path.is_file():
                digests[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = canonical_json(digests)
    return "ab1:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _resolve_asset_bundle_id(
    data_root: Path, override: str, *, is_production: bool
) -> tuple[str, list[str]]:
    warnings: list[str] = []
    if override:
        return override, warnings
    manifest = data_root / "MANIFEST.json"
    if manifest.is_file():
        raw = json.loads(manifest.read_text(encoding="utf-8"))
        value = str(raw.get("asset_bundle_id", ""))
        if value:
            return value, warnings
    if is_production:
        raise RuntimeError(
            "asset_bundle_id is unavailable: data/MANIFEST.json is missing or has no "
            "`asset_bundle_id`, and ASSET_BUNDLE_ID is unset. `forge pack` writes the "
            "manifest (TECH_PLAN.md §3); serving a computed placeholder in production "
            "would let the browser's asset conformance check pass against assets it "
            "never verified."
        )
    computed = _compute_asset_bundle_id(data_root / "assets")
    warnings.append(
        f"data/MANIFEST.json is absent; asset_bundle_id computed provisionally as "
        f"{computed}. `forge pack` must produce the real manifest before deploy."
    )
    return computed, warnings


# ---------------------------------------------------------------------------
# the loaded whole
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Content:
    data_root: Path
    watermark: WatermarkFile
    #: Merged with `[defaults.*]` and validated against the REGISTRY by core's
    #: `load_levels`, which raises at BOOT on an unknown check name, a check out
    #: of phase order, or a param nobody reads.
    levels: dict[str, LevelConfig]
    levels_file: LevelsFile
    scoring: ScoringConfig
    judge: JudgeConfig
    schedule: ScheduleFile
    copy: CopyBook
    passages: dict[str, PassageBundle]
    asset_bundle_id: str
    warnings: tuple[str, ...] = ()

    @property
    def wm_config_id(self) -> str:
        return self.watermark.wm_config_id

    @property
    def scoring_version(self) -> str:
        return self.scoring.scoring_version

    @property
    def judge_version(self) -> str:
        return self.judge.judge_version

    def level(self, level_id: str) -> LevelConfig | None:
        return self.levels.get(level_id)

    def passage(self, passage_id: str) -> PassageBundle | None:
        return self.passages.get(passage_id)


def load_content(
    data_root: Path, *, asset_bundle_id: str = "", is_production: bool = False
) -> Content:
    """Load and validate everything under `data/`. Raises at BOOT, never at play."""
    cfg_dir = data_root / "config"

    watermark = WatermarkFile.load(cfg_dir / "watermark.toml")
    watermark.assert_consistent(data_root)

    levels_file = LevelsFile.model_validate(_read_toml(cfg_dir / "levels.toml"))
    # Core merges `[defaults.*]` under each level's own params and validates the
    # result against the registry. It raises at BOOT, never at play (§7.6).
    levels = load_levels(cfg_dir / "levels.toml")

    scoring_raw = _read_toml(cfg_dir / "scoring.toml")
    scoring_fields = {k: v for k, v in scoring_raw.items() if k in _SCORING_SCALARS}
    tie_break = scoring_raw.get("distance", {}).get("backtrace_tie_break")
    if tie_break is not None:
        scoring_fields["backtrace_tie_break"] = tuple(tie_break)
    scoring = ScoringConfig.model_validate(scoring_fields)

    # The lemma table decides L4 verdicts and exists in two places (the
    # committed files and close_paraphrase.py's no-data/ fallback). A split
    # value is two different fences, so it is a boot failure.
    assert_tables_match_files()

    judge = JudgeConfig.load(cfg_dir / "judge.toml")
    schedule = ScheduleFile.load(cfg_dir / "schedule.toml")
    copy = load_copy(cfg_dir / "copy.toml")

    passages, warnings = _load_passages(data_root / "passages", watermark.wm_config_id)

    if levels_file.judge_version != judge.judge_version:
        raise ValueError(
            f"levels.toml declares judge_version={levels_file.judge_version!r} but "
            f"judge.toml declares {judge.judge_version!r}. judge_version scopes the "
            "verdict cache; a split value serves stale verdicts under one of the two "
            "(TECH_PLAN.md §7.5)."
        )

    for slot in schedule.days:
        if slot.passage_id not in passages:
            msg = (
                f"schedule.toml maps {slot.date.isoformat()} -> {slot.passage_id}, which "
                f"has no data/passages/{slot.passage_id}.public.json."
            )
            if schedule.strict:
                raise ValueError(msg + " schedule.toml has strict = true.")
            warnings.append(msg + " Skipping that day (schedule.toml strict = false).")
        elif slot.level_id not in levels:
            raise ValueError(
                f"schedule.toml maps {slot.date.isoformat()} to level {slot.level_id}, "
                "which levels.toml does not define."
            )

    bundle_id, bundle_warnings = _resolve_asset_bundle_id(
        data_root, asset_bundle_id, is_production=is_production
    )
    warnings.extend(bundle_warnings)

    # The dev bundle, LAST, so a real passage of the same id always wins and so
    # it can carry the asset_bundle_id the rest of the tree resolved to.
    if not is_production and DEV_PASSAGE_ID not in passages:
        dev = _dev_bundle(
            data_root,
            wm_config_id=watermark.wm_config_id,
            asset_bundle_id=bundle_id,
            scoring_version=scoring.scoring_version,
            judge_prompt_id=judge.prompt_id,
            level_id=schedule.intro.level_id,
        )
        if dev is not None:
            passages[DEV_PASSAGE_ID] = dev
            warnings.append(
                f"no packed passage for today; serving the DEV fixture {DEV_PASSAGE_ID} from "
                "data/dev/passage.txt so the page is playable. It is human prose that already "
                "reads below the notch, so it is not a puzzle. `forge publish` is the "
                "supported way to add a real one (TECH_PLAN.md §6.4)."
            )

    return Content(
        data_root=data_root,
        watermark=watermark,
        levels=levels,
        levels_file=levels_file,
        scoring=scoring,
        judge=judge,
        schedule=schedule,
        copy=copy,
        passages=passages,
        asset_bundle_id=bundle_id,
        warnings=tuple(warnings),
    )
