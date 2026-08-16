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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from launder_core.detect.calibration import load_thresholds
from launder_core.gates.checks.close_paraphrase import assert_tables_match_files
from launder_core.gates.feedback import CopyBook, load_copy
from launder_core.levels import load_levels, validate_level
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
    "CampaignLevel",
    "Content",
    "JudgeConfig",
    "LevelSpec",
    "ProgressionFile",
    "WatermarkFile",
    "apply_level_overrides",
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

    #: Mirrors `Settings.judge_daily_usd_cap` / `rate_limit_judge_*`. The two
    #: must not disagree: the file is what a reader consults and the settings
    #: defaults are what an unset env var produces.
    daily_usd_cap: float = 500.0
    rate_limit_per_hour: int = 60
    rate_limit_burst: int = 10
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
    model: str = ""
    reasoning_effort: str = "none"
    temperature: float = 0.0
    max_output_tokens: int = 300
    timeout_s: float = 6.0
    #: Retries against THE provider. There is no second provider to fall through
    #: to: after the retry the judge raises and the submission clears
    #: provisionally.
    max_retries: int = 1

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
# progression.toml
# ---------------------------------------------------------------------------


class LevelSpec(BaseModel):
    """One `[[level]]` block: campaign position -> passage -> ruleset."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    n: int = Field(ge=1)
    passage_id: str
    #: An `L1..L6` id from `levels.toml`. The player never sees this string; it
    #: names the ordered CHECK LIST the level is played under.
    rules: str
    #: PER-LEVEL PARAM OVERRIDES, merged over the ruleset's own params:
    #: `overrides = { edit_budget = { max_word_distance = 6 } }`.
    #:
    #: This exists because a budget is only a real constraint when it is tuned
    #: to ITS passage. Budgets used to live only on the ruleset, so one number
    #: had to serve every level sharing it — and the number that fits the
    #: hardest passage is no constraint at all on the easiest. Measured: the
    #: shipped budgets of 12 and 18 were 2-6x what a greedy attack actually
    #: spent, so they never once decided an outcome.
    #:
    #: Validated at BOOT against the same registry rules as levels.toml: an
    #: override for a check the ruleset does not run, or a param that check
    #: does not read, is a startup failure rather than a rule nobody enforces.
    overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)


class ProgressionFile(BaseModel):
    """`data/config/progression.toml`: the campaign, in order, 1..N.

    The game is a linear campaign, not a daily: there is no date arithmetic
    here, nothing rolls over at UTC midnight, and the only ordering is `n`.

    `strict = true` means a level whose passage file is missing, or whose
    `rules` id `levels.toml` does not define, is a BOOT FAILURE. A hole in the
    campaign is a broken build — level 7 cannot "skip to tomorrow", it just
    dead-ends the player halfway through, which is worse than not booting.
    """

    model_config = ConfigDict(frozen=True, extra="ignore", populate_by_name=True)

    schema_id: str = Field(default="launder.progression/1", alias="schema")
    strict: bool = True
    levels: tuple[LevelSpec, ...] = Field(default=(), alias="level")

    @model_validator(mode="after")
    def _n_is_contiguous_from_one(self) -> ProgressionFile:
        """`n` must be exactly 1, 2, ... len(levels), in that order.

        `level_n` is the player's position AND the wire/DB key for progress, so
        a gap or a duplicate is not a cosmetic problem: "level 8 of 15" would
        name a level nobody can reach, and `unlocked = max(cleared) + 1` would
        point at a hole and strand the player there forever.
        """
        got = [spec.n for spec in self.levels]
        want = list(range(1, len(self.levels) + 1))
        if got != want:
            raise ValueError(
                f"progression.toml numbers its levels {got}, which is not the contiguous "
                f"run {want}. Renumber the `[[level]]` blocks so `n` runs 1..{len(want)} "
                "in file order: `level_n` is what the player sees, what the cookie "
                "carries and what the progress table is keyed on."
            )
        return self

    @model_validator(mode="after")
    def _one_level_per_passage(self) -> ProgressionFile:
        """No passage may be used by two levels.

        `Content.level_n_of` maps a passage BACK to its campaign position, and it
        is the only mapping there is: `/api/submit` takes `passage_id` from the
        request and refuses to take `level_n`, precisely so the client cannot
        assert where it is in the campaign. That inverse only exists if the
        forward map is injective. With a passage listed twice, `level_n_of`
        returns the FIRST match, so a player who cleared level 7 would have the
        clear recorded against level 3 and would unlock level 4 — which is the
        row "claiming the player cleared a level they never played" that
        `level_n_of`'s own docstring exists to prevent. Nothing checked it.
        """
        seen: dict[str, int] = {}
        for spec in self.levels:
            first = seen.setdefault(spec.passage_id, spec.n)
            if first != spec.n:
                raise ValueError(
                    f"progression.toml runs both level {first} and level {spec.n} on passage "
                    f"{spec.passage_id!r}. A passage is a level's identity: the submit request "
                    "carries the passage and the SERVER derives the campaign position from it, "
                    "so a passage on two levels means one of them can never be submitted to and "
                    "its clears are recorded against the other. Pack a second passage with "
                    "`forge pack`, or drop one of the levels."
                )
        return self

    @classmethod
    def load(cls, path: Path) -> ProgressionFile:
        return cls.model_validate(_read_toml(path))


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
#: fixture names it, and `boot.py` marks a page built from it `dev: true`.
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

    What this is NOT: a level of the campaign. It stands in for the WHOLE
    campaign — "level 1 of 1" — it is marked `dev: true` in the boot payload,
    and it is refused outright in production. The fixture is HUMAN PROSE
    reading z = 1.31 against a notch of 2.3263, i.e. it starts already under
    the line — a coherent thing to develop against and not a puzzle, which is
    why shipping it as a real level would be a lie rather than a shortcut.

    The four detector expectations are COMPUTED here by the same core the server
    scores with, never authored. A real level is `forge pack` plus a `[[level]]`
    block in `data/config/progression.toml`.
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
        return bundles, [f"{passages_dir} does not exist: the campaign has no passages."]
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
class CampaignLevel:
    """One playable position in the campaign, with its passage already resolved.

    `level_id` is the RULESET id (`L1..L6`) rather than the `LevelConfig`
    itself, because `main._drop_unserviceable_levels` rebuilds `Content.levels`
    after this tuple is built — a cached `LevelConfig` here would be a second
    copy of a level the app has since refused to serve.
    """

    n: int
    passage_id: str
    level_id: str
    #: See `LevelSpec.overrides`. Applied by `Content.resolve`, which is the one
    #: place a playable ruleset is produced, so boot rendering and the gate
    #: cannot disagree about what this level's rules actually are.
    overrides: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


def apply_level_overrides(
    level: LevelConfig, overrides: Mapping[str, Mapping[str, Any]], *, level_n: int
) -> LevelConfig:
    """Merge a campaign level's param overrides over its ruleset.

    Re-runs core's own `validate_level` on the result, so an override is held to
    exactly the standard `levels.toml` is: unknown check, unread param, missing
    required param and out-of-order phases all raise here rather than producing
    a level whose rules quietly differ from what the checklist advertises.
    """
    if not overrides:
        return level
    names = {spec.check for spec in level.checks}
    unknown = set(overrides) - names
    if unknown:
        raise ValueError(
            f"progression.toml level {level_n} overrides check(s) {sorted(unknown)}, which "
            f"ruleset {level.id} does not run (it runs {sorted(names)}). An override on a "
            "check that never executes is a constraint the player is never held to."
        )
    checks = tuple(
        spec.model_copy(update={"params": {**spec.params, **dict(overrides[spec.check])}})
        if spec.check in overrides
        else spec
        for spec in level.checks
    )
    merged = level.model_copy(update={"checks": checks})
    validate_level(merged)
    return merged


@dataclass(frozen=True)
class Content:
    data_root: Path
    watermark: WatermarkFile
    #: Merged with `[defaults.*]` and validated against the REGISTRY by core's
    #: `load_levels`, which raises at BOOT on an unknown check name, a check out
    #: of phase order, or a param nobody reads. Keyed by RULESET id (`L1..L6`).
    levels: dict[str, LevelConfig]
    levels_file: LevelsFile
    scoring: ScoringConfig
    judge: JudgeConfig
    progression: ProgressionFile
    #: The campaign as it can actually be played: `campaign[i].n == i + 1`, and
    #: every entry's passage is loaded. `progression.levels` is the FILE;
    #: this is the file after the passages were checked against it.
    campaign: tuple[CampaignLevel, ...]
    copy: CopyBook
    passages: dict[str, PassageBundle]
    asset_bundle_id: str
    #: True when `campaign` is the single-level dev fixture rather than the real
    #: campaign. `boot.py` puts it on the wire as `dev`.
    dev: bool = False
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

    @property
    def level_count(self) -> int:
        """The number the player is shown next to their level ("3 of 15")."""
        return len(self.campaign)

    def ruleset(self, level_id: str) -> LevelConfig | None:
        """The ordered check list `L1..L6` names. NOT a campaign position."""
        return self.levels.get(level_id)

    def passage(self, passage_id: str) -> PassageBundle | None:
        return self.passages.get(passage_id)

    def campaign_level(self, n: int) -> CampaignLevel | None:
        if 1 <= n <= len(self.campaign):
            return self.campaign[n - 1]
        return None

    def level_n_of(self, passage_id: str) -> int | None:
        """Which level a passage IS. `None` for a passage outside the campaign.

        The submit request carries `passage_id` and the ruleset id, never
        `level_n` — the client asserts nothing about where it is in the
        campaign, exactly as it asserts nothing about its own score.
        """
        for entry in self.campaign:
            if entry.passage_id == passage_id:
                return entry.n
        return None

    def resolve(self, n: int) -> tuple[PassageBundle, LevelConfig] | None:
        """`level_n` -> the passage to render and the ruleset to play it under.

        THE ONE PLACE per-level overrides are applied. Both the boot renderer
        and `/api/submit` come through here, so the rules the player is shown in
        the checklist are by construction the rules the gate runs.
        """
        entry = self.campaign_level(n)
        if entry is None:
            return None
        bundle = self.passage(entry.passage_id)
        ruleset = self.ruleset(entry.level_id)
        if bundle is None or ruleset is None:
            return None
        return bundle, apply_level_overrides(ruleset, entry.overrides, level_n=n)


def _assert_calibrations_exist(levels: Mapping[str, LevelConfig]) -> None:
    """Every `calibration` a level names must be a bucket set the file defines.

    THE BUG THIS EXISTS FOR: L5 shipped `detector_threshold.calibration = "code"`
    while `data/assets/thresholds.v1.json` defined only `default`.
    `load_levels` accepted it — `calibration` is a declared `config_param`, and
    core has no reason to know which bucket sets were measured — so nothing
    objected until `ServerDetector.read_tokens` called `load_thresholds(name=...)`
    at submit time and `parse_thresholds` raised `KeyError`. A 500 on the level's
    WIN CONDITION, at play, which ARCHITECTURE.md §8 forbids outright.

    It was invisible because L5 is out of the campaign AND is dropped at boot for
    a missing `unit_tests` dependency — masked by two accidents rather than by
    design, and one `[[level]]` block away from shipping. So this runs over every
    ruleset in the file, not only the ones the campaign currently plays.

    `forge verify` carries the same check for CI. This one is the boot half, and
    it is the half that matters: a config error has to stop the server starting,
    not wait for the first player to find it.
    """
    for level in levels.values():
        for spec in level.checks:
            name = spec.params.get("calibration")
            if name:
                # Raises KeyError naming the file and the sets it does define.
                load_thresholds(name=str(name))


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
    _assert_calibrations_exist(levels)

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
    progression = ProgressionFile.load(cfg_dir / "progression.toml")
    copy = load_copy(cfg_dir / "copy.toml")

    passages, warnings = _load_passages(data_root / "passages", watermark.wm_config_id)

    if levels_file.judge_version != judge.judge_version:
        raise ValueError(
            f"levels.toml declares judge_version={levels_file.judge_version!r} but "
            f"judge.toml declares {judge.judge_version!r}. judge_version scopes the "
            "verdict cache; a split value serves stale verdicts under one of the two "
            "(TECH_PLAN.md §7.5)."
        )

    # An unknown ruleset id is a config error rather than a packing state, so it
    # is checked BEFORE the dev fallback and regardless of `strict`: a campaign
    # that names a level `levels.toml` never defined has no check list to play
    # under and would 500 on the first submit of that level.
    for spec in progression.levels:
        if spec.rules not in levels:
            raise ValueError(
                f"progression.toml runs level {spec.n} under rules {spec.rules!r}, which "
                "data/config/levels.toml does not define. Fix the `rules` id in "
                "progression.toml, or add that level to levels.toml — the ids are "
                "`L1`..`L6` and they name the ordered check list, not the campaign "
                "position."
            )

    bundle_id, bundle_warnings = _resolve_asset_bundle_id(
        data_root, asset_bundle_id, is_production=is_production
    )
    warnings.extend(bundle_warnings)

    campaign, dev = _resolve_campaign(
        data_root,
        progression,
        passages,
        warnings,
        wm_config_id=watermark.wm_config_id,
        asset_bundle_id=bundle_id,
        scoring_version=scoring.scoring_version,
        judge_prompt_id=judge.prompt_id,
        is_production=is_production,
    )

    return Content(
        data_root=data_root,
        watermark=watermark,
        levels=levels,
        levels_file=levels_file,
        scoring=scoring,
        judge=judge,
        progression=progression,
        campaign=campaign,
        copy=copy,
        passages=passages,
        asset_bundle_id=bundle_id,
        dev=dev,
        warnings=tuple(warnings),
    )


def _resolve_campaign(
    data_root: Path,
    progression: ProgressionFile,
    passages: dict[str, PassageBundle],
    warnings: list[str],
    *,
    wm_config_id: str,
    asset_bundle_id: str,
    scoring_version: str,
    judge_prompt_id: str,
    is_production: bool,
) -> tuple[tuple[CampaignLevel, ...], bool]:
    """`progression.toml` + the loaded passages -> the campaign that can be played.

    Two distinct situations, and conflating them is what the two branches below
    exist to prevent:

    * **NO packed passages at all** is a fresh clone, not a broken build.
      `data/passages/` cannot be filled without a GPU and the gated Gemma-3
      weights (§6.1, §6.2), so the dev fixture stands in as the entire campaign
      — level 1 of 1, `dev: true`. Refused in production, where an empty
      `data/passages/` means the image was built wrong.
    * **SOME passages, one of them missing** is a hole in the campaign, and
      under `strict` it stops the process. There is no "skip the day" any more:
      the levels are ordered and the player walks through them, so a missing
      level 7 dead-ends everybody who clears level 6.
    """
    if not passages:
        if is_production:
            raise RuntimeError(
                "data/passages/ holds no passages, so there is no campaign to serve. "
                "`forge pack` writes them into the image (TECH_PLAN.md §3); a production "
                "boot with an empty passage set means the image was built without them."
            )
        first_rules = progression.levels[0].rules if progression.levels else "L1"
        dev_bundle = _dev_bundle(
            data_root,
            wm_config_id=wm_config_id,
            asset_bundle_id=asset_bundle_id,
            scoring_version=scoring_version,
            judge_prompt_id=judge_prompt_id,
            level_id=first_rules,
        )
        if dev_bundle is None:
            warnings.append(
                "data/passages/ is empty and data/dev/passage.txt could not be read, so "
                "there is nothing to render at all. Pack a passage with `forge pack` and "
                "give it a `[[level]]` block in data/config/progression.toml."
            )
            return (), False
        passages[DEV_PASSAGE_ID] = dev_bundle
        warnings.append(
            f"no packed passages; serving the DEV fixture {DEV_PASSAGE_ID} from "
            "data/dev/passage.txt as the whole campaign (level 1 of 1) so the page is "
            "playable. It is human prose that already reads below the notch, so it is "
            "not a puzzle. Pack a real one with `forge pack` and give it a `[[level]]` "
            "block in data/config/progression.toml."
        )
        return (CampaignLevel(n=1, passage_id=DEV_PASSAGE_ID, level_id=first_rules),), True

    resolved: list[CampaignLevel] = []
    for spec in progression.levels:
        if spec.passage_id not in passages:
            msg = (
                f"progression.toml runs level {spec.n} on {spec.passage_id}, which has no "
                f"data/passages/{spec.passage_id}.public.json. Pack that passage with "
                "`forge pack`, or remove the level and renumber the ones after it."
            )
            if progression.strict:
                raise ValueError(msg + " progression.toml has strict = true.")
            # Not strict: the campaign is the CONTIGUOUS PREFIX that resolves.
            # Dropping the hole and keeping level 8 would renumber nothing and
            # leave `unlocked` pointing at a level that cannot be reached.
            warnings.append(
                msg + f" progression.toml has strict = false, so the campaign stops at level "
                f"{spec.n - 1}."
            )
            break
        resolved.append(
            CampaignLevel(
                n=spec.n,
                passage_id=spec.passage_id,
                level_id=spec.rules,
                overrides=spec.overrides,
            )
        )
    return tuple(resolved), False
