"""launder-forge — the local authoring pipeline (TECH_PLAN.md §6).

LOCAL ONLY. Never installed on Railway, never in the Docker image. This package
is allowed to depend on torch and transformers; `launder_core` is not.

Import policy
-------------
Nothing heavy is imported at package import time. `launder_forge.cli` imports
each command module lazily inside its command function, so `forge lint-copy`
and `forge tok pack` run in a checkout where torch is not installed, and
`forge doctor` can report *that torch is missing* instead of dying on the
import line.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
