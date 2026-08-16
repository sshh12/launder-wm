"""Every gate check, one file each — TECH_PLAN.md §7.1, §7.6.

Importing this package populates `launder_core.gates.registry.REGISTRY`. The
import list below is the ONLY place a new check has to be mentioned outside its
own file; keeping it explicit rather than globbing the directory means a typo in
a filename is an ImportError at boot instead of a check that silently never
registers.

The order of these imports does not matter — `data/config/levels.toml` decides
what runs and `GateCheck.phase` decides what order is legal.
"""

from __future__ import annotations

from launder_core.gates.checks.close_paraphrase import CloseParaphrase
from launder_core.gates.checks.detector_threshold import DetectorThreshold
from launder_core.gates.checks.edit_budget import EditBudget
from launder_core.gates.checks.llm_gate import LlmGate
from launder_core.gates.checks.locked_phrase import LockedPhrase
from launder_core.gates.checks.unicode_sanitation import UnicodeSanitation
from launder_core.gates.checks.unit_test import UnitTest
from launder_core.gates.checks.word_floor import WordFloor

__all__ = [
    "CloseParaphrase",
    "DetectorThreshold",
    "EditBudget",
    "LlmGate",
    "LockedPhrase",
    "UnicodeSanitation",
    "UnitTest",
    "WordFloor",
]
