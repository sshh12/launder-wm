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

__all__ = ["Settings", "get_settings", "repo_root"]

JudgeProviderName = Literal["openai", "anthropic", "fake", "cassette"]

#: Providers that cost money and talk to the network. `prompt_hash` enforcement
#: and the API-key assertion are hard failures for these and warnings for the
#: offline two, because a PR environment deliberately runs `fake` with no keys
#: (§11.5: sealed variables do not propagate to PR environments).
PAID_PROVIDERS: frozenset[str] = frozenset({"openai", "anthropic"})


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
    variable panel, and `assert_production_ready()` refuses to boot if it did
    not.
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
    anthropic_api_key: SecretStr = SecretStr("")

    judge_provider: JudgeProviderName = "fake"
    #: VERIFY (§7.5, §14.2 item 4): pin from the live pricing page at build time.
    #: `data/config/judge.toml` ships the sentinel `REPLACE_AT_BUILD_TIME`; the
    #: env var wins, and a paid provider with the sentinel still in place is a
    #: boot failure.
    judge_model: str = ""
    judge_failover_model: str = ""
    judge_daily_usd_cap: float = Field(default=2.00, ge=0.0)
    judge_timeout_s: float = Field(default=6.0, gt=0.0)
    #: Replay cassette for CI. `cassette` RAISES on a miss, so CI can never
    #: silently start spending money.
    judge_cassette_path: Path | None = None
    #: Record every live judge call into `judge_cassette_path` as it happens.
    judge_record: bool = False

    rate_limit_judge_per_hour: int = Field(default=10, ge=1)
    rate_limit_judge_burst: int = Field(default=3, ge=1)

    pow_enabled: bool = False
    parity_sample_rate: float = Field(default=0.005, ge=0.0, le=1.0)

    #: Deploy identity, surfaced by /healthz. Railway injects RAILWAY_GIT_COMMIT_SHA.
    git_sha: str = ""
    #: Set by `forge pack` into data/MANIFEST.json; the env var is the override.
    asset_bundle_id: str = ""

    port: int = 8000
    data_dir: Path | None = None
    web_dist_dir: Path | None = None

    @field_validator("judge_model", "judge_failover_model", "git_sha", mode="before")
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
        """Assert the selected provider has a key. Returns non-fatal warnings.

        Reads `SecretStr` only to ask whether it is empty. The value never
        leaves this method.
        """
        warnings: list[str] = []
        if self.judge_provider == "openai" and not self.openai_api_key.get_secret_value():
            raise RuntimeError(
                "JUDGE_PROVIDER=openai but OPENAI_API_KEY is empty. Set it as a "
                "SEALED variable on the app service (TECH_PLAN.md §11.5), or set "
                "JUDGE_PROVIDER=fake for an environment that must not spend money."
            )
        if self.judge_provider == "anthropic" and not self.anthropic_api_key.get_secret_value():
            raise RuntimeError(
                "JUDGE_PROVIDER=anthropic but ANTHROPIC_API_KEY is empty. Set it as "
                "a SEALED variable on the app service (TECH_PLAN.md §11.5)."
            )
        if self.judge_provider == "openai" and not self.anthropic_api_key.get_secret_value():
            warnings.append(
                "ANTHROPIC_API_KEY is empty: the §7.5 failover chain is disabled and a "
                "provider error will degrade straight to cleared+provisional."
            )
        if self.judge_provider == "cassette" and self.judge_cassette_path is None:
            raise RuntimeError(
                "JUDGE_PROVIDER=cassette but JUDGE_CASSETTE_PATH is unset. The "
                "cassette provider raises on a miss by design; with no cassette "
                "file every submit would raise."
            )
        return warnings

    def assert_model_pinned(self) -> None:
        """A paid provider must name a real model, not the judge.toml sentinel."""
        if not self.judge_is_paid:
            return
        bad = {"", "REPLACE_AT_BUILD_TIME"}
        if self.judge_model in bad:
            raise RuntimeError(
                f"JUDGE_PROVIDER={self.judge_provider} but JUDGE_MODEL is "
                f"{self.judge_model!r}. TECH_PLAN.md §7.5/§14.2 item 4 flags the model "
                "IDs in data/config/judge.toml as an unverified third-party scrape: "
                "pin JUDGE_MODEL from the live pricing page at build time. It must be a "
                "NON-REASONING model (reasoning effort 'none' or equivalent) — reasoning "
                "tokens bill as output and push TTFT into tens of seconds, and a submit "
                "gate that stalls the player for 30 s is a broken game."
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings. Cached because `--workers 1` is load-bearing."""
    return Settings()
