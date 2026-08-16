"""Environment -> typed config, with the boot-time assertions (TECH_PLAN.md §11.5).

Everything here fails at BOOT, not at play. A misconfigured judge model, a
`wm_config_id` that does not match the passages in `data/`, a sampling table
whose bytes drifted — each of those produces a server that looks healthy and
serves fiction. The whole point of this module is that the process refuses to
start instead.

**API keys are never read, printed, logged, or compared here.** They are
`SecretStr`, and the only assertion made about them is "is it non-empty".
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from launder_serve.judge.gate import EST_USD_PER_CALL

__all__ = ["Settings", "get_settings", "repo_root"]

#: The judge is OpenAI-ONLY. There is no second paid provider and no failover
#: leg: the surviving policy is one provider, one retry, then fail open
#: provisional. A second vendor would mean a second SDK, a second key to seal, a
#: second prompt-shape to keep equivalent and a second set of verdicts in the
#: same cache namespace — for a gate whose hard cases are already deterministic.
JudgeProviderName = Literal["openai", "fake", "cassette"]

#: Providers that cost money and talk to the network. `prompt_hash` enforcement
#: and the API-key assertion are hard failures for these and warnings for the
#: offline two, because a PR environment deliberately runs `fake` with no keys
#: (§11.5: sealed variables do not propagate to PR environments).
PAID_PROVIDERS: frozenset[str] = frozenset({"openai"})


def repo_root() -> Path:
    """The workspace root, located from this file rather than from `cwd`.

    `packages/serve/src/launder_serve/settings.py` -> four parents up. Uvicorn,
    pytest and `alembic` all run with different working directories; deriving
    the data path from `cwd` is how `data/` goes missing in exactly one of them.
    """
    return Path(__file__).resolve().parents[4]


class Settings(BaseSettings):
    """`ENV` selects the environment; everything else has a working default.

    The defaults are the *development* defaults: `fake` judge, SQLite in a
    local file, no keys required. Production overrides them in Railway's
    variable panel, and `assert_keys_present()` refuses to boot if it did not.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    env: Literal["development", "test", "pr", "production"] = "development"

    #: `${{Postgres.DATABASE_URL}}` in production — a REFERENCE, never a pasted
    #: literal. Empty means "use the local SQLite dev file".
    database_url: str = ""

    #: Sealed in Railway. Never logged. Only ever checked for emptiness.
    openai_api_key: SecretStr = SecretStr("")

    judge_provider: JudgeProviderName = "fake"
    #: The model, overriding `data/config/judge.toml`'s pinned `model`. The env
    #: var wins so a model can be rolled back without a redeploy; a paid provider
    #: naming the sentinel `REPLACE_AT_BUILD_TIME` is a boot failure. Changing it
    #: changes `prompt_hash`, which is asserted at boot — that is deliberate, a
    #: verdict cache keyed on a prompt the model never saw is exactly the bug.
    judge_model: str = ""
    #: The authorised daily bill, enforced by the Postgres spend ledger. This is
    #: THE cost bound — it is atomic and shared, so it holds across every replica,
    #: unlike the per-IP bucket below, which is process-local. Sized for a Hacker
    #: News front page: the user expects the spike and has accepted the cost.
    judge_daily_usd_cap: float = Field(default=500.00, ge=0.0)
    #: What the ledger RESERVES per call, before the call. See `EST_USD_PER_CALL`
    #: for the derivation. Overridable without a code change because the ledger
    #: never reconciles: if the real price moves, this number is the only thing
    #: standing between `daily_usd_cap` and meaning something other than dollars.
    judge_est_usd_per_call: float = Field(default=EST_USD_PER_CALL, gt=0.0)
    judge_timeout_s: float = Field(default=6.0, gt=0.0)
    #: Replay cassette for CI. `cassette` RAISES on a miss, so CI can never
    #: silently start spending money.
    judge_cassette_path: Path | None = None
    #: Record every live judge call into `judge_cassette_path` as it happens.
    judge_record: bool = False

    #: The per-IP judge bucket, mirrored in `data/config/judge.toml`'s `[limits]`.
    #: Generous on purpose: a 15-level campaign is played in ONE SITTING, and
    #: corporate and mobile NAT put many players behind a single `X-Real-IP`, so a
    #: tight bucket throttles legitimate play long before it touches abuse.
    #: It is safe to be generous because this bucket is a FAIRNESS AND ABUSE brake,
    #: not the cost bound — that is `judge_daily_usd_cap`, which is atomic and
    #: lives in Postgres, so it holds across every replica while this bucket is
    #: process-local and each extra replica adds another full bucket
    #: (railway.json runs 3, so the real per-IP ceiling is ~3x these numbers).
    rate_limit_judge_per_hour: int = Field(default=60, ge=1)
    rate_limit_judge_burst: int = Field(default=10, ge=1)

    pow_enabled: bool = False
    parity_sample_rate: float = Field(default=0.005, ge=0.0, le=1.0)

    #: Deploy identity, surfaced by /healthz. Railway injects RAILWAY_GIT_COMMIT_SHA.
    git_sha: str = ""
    #: Set by `forge pack` into data/MANIFEST.json; the env var is the override.
    asset_bundle_id: str = ""

    port: int = 8000
    data_dir: Path | None = None
    web_dist_dir: Path | None = None

    @field_validator("judge_model", "git_sha", mode="before")
    @classmethod
    def _strip(cls, v: object) -> object:
        return v.strip() if isinstance(v, str) else v

    # -- derived paths -------------------------------------------------------

    @property
    def data_path(self) -> Path:
        return self.data_dir if self.data_dir is not None else repo_root() / "data"

    @property
    def web_dist_path(self) -> Path:
        return self.web_dist_dir if self.web_dist_dir is not None else repo_root() / "web" / "dist"

    @property
    def sha(self) -> str:
        for key in ("RAILWAY_GIT_COMMIT_SHA", "GIT_SHA", "SOURCE_COMMIT"):
            value = os.environ.get(key, "").strip()
            if value:
                return value[:7]
        return self.git_sha[:7] if self.git_sha else "unknown"

    @property
    def is_production(self) -> bool:
        return self.env == "production"

    @property
    def judge_is_paid(self) -> bool:
        return self.judge_provider in PAID_PROVIDERS

    def resolved_database_url(self) -> str:
        """The URL `make_engine()` receives.

        Production MUST provide one; development falls back to a WAL-mode
        SQLite file next to `data/` so `uvicorn` works from a fresh clone.
        """
        if self.database_url:
            return self.database_url
        if self.is_production:
            raise RuntimeError(
                "DATABASE_URL is empty in ENV=production. Set it to the Railway "
                "reference `${{Postgres.DATABASE_URL}}` — never a pasted literal "
                "(TECH_PLAN.md §11.5)."
            )
        return f"sqlite+aiosqlite:///{(repo_root() / 'launder-dev.sqlite3').as_posix()}"

    # -- boot assertions -----------------------------------------------------

    def assert_keys_present(self) -> list[str]:
        """Assert the judge is configured to actually judge. Returns warnings.

        Reads `SecretStr` only to ask whether it is empty. The value never
        leaves this method.
        """
        warnings: list[str] = []
        # PRODUCTION MUST NOT RUN AN OFFLINE JUDGE.
        #
        # This class's docstring has always promised that production "refuses to
        # boot" if it did not override the development defaults, and nothing
        # enforced it: `judge_provider` defaults to `fake`, so a Railway variable
        # panel missing `JUDGE_PROVIDER` produced a server that booted clean,
        # healthchecked green, and cleared the meaning arm of the gate for
        # everybody — `FakeJudge` reports `natural_prose=True` with every claim
        # present, so word salad and dropped claims both pass. `assert_keys_present`
        # only ever objected to `openai` with no key, i.e. to the one
        # misconfiguration that is loud.
        #
        # `pr` is deliberately NOT covered: sealed variables do not propagate to
        # PR environments (§11.5), so those run `fake` with no keys ON PURPOSE and
        # must still boot.
        if self.is_production and not self.judge_is_paid:
            raise RuntimeError(
                f"ENV=production with JUDGE_PROVIDER={self.judge_provider}. That provider "
                "never contacts a model, so `llm_gate` would clear every submission it "
                "reached and the meaning half of the gate would be off in production with "
                "nothing on screen to say so. Set JUDGE_PROVIDER to one of "
                f"{sorted(PAID_PROVIDERS)} and seal its key (TECH_PLAN.md §11.5), or set "
                "ENV to something other than production."
            )
        if self.judge_provider == "openai" and not self.openai_api_key.get_secret_value():
            raise RuntimeError(
                "JUDGE_PROVIDER=openai but OPENAI_API_KEY is empty. Set it as a "
                "SEALED variable on the app service (TECH_PLAN.md §11.5), or set "
                "JUDGE_PROVIDER=fake for an environment that must not spend money."
            )
        if self.judge_provider == "cassette" and self.judge_cassette_path is None:
            raise RuntimeError(
                "JUDGE_PROVIDER=cassette but JUDGE_CASSETTE_PATH is unset. The "
                "cassette provider raises on a miss by design; with no cassette "
                "file every submit would raise."
            )
        return warnings

    def assert_model_pinned(self, resolved_model: str) -> None:
        """A paid provider must name a real model, not the build-time sentinel.

        Takes the RESOLVED model — `JUDGE_MODEL or judge.toml's model` — rather
        than reading `self.judge_model`, because judge.toml is where the model is
        pinned and the env var is only the override. Asserting on the env var
        alone made `JUDGE_MODEL` mandatory in production, which contradicted the
        file that records the pin and the `prompt_hash` computed against it.
        """
        if not self.judge_is_paid:
            return
        if resolved_model in {"", "REPLACE_AT_BUILD_TIME"}:
            raise RuntimeError(
                f"JUDGE_PROVIDER={self.judge_provider} but the resolved judge model is "
                f"{resolved_model!r}. Pin it in data/config/judge.toml's `model` (and "
                "record the `prompt_hash` the next boot prints), or set JUDGE_MODEL to "
                "override it. It must be a NON-REASONING model (reasoning effort 'none' "
                "or equivalent) — reasoning tokens bill as output and push TTFT into tens "
                "of seconds, and a submit gate that stalls the player for 30 s is a "
                "broken game."
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings. Cached because `--workers 1` is load-bearing."""
    return Settings()
