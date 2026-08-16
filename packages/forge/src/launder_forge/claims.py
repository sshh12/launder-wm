"""`forge claims <id>` — draft a claim list for HUMAN REVIEW.

Claims drive the judge's meaning check, and `required` drives the tight fence
(TECH_PLAN.md §6.3, §7.4). A claim's `label` is the player-facing noun phrase —
"the cutting's own weather" — and it is **authored offline on purpose**: that is
what lets gate feedback name a dropped claim without ever quoting the player's
own text back at them, which is both an injection surface and bad UX.

So this command drafts; it does not decide. The output is written next to the
passage as `<id>.claims.draft.json` with `reviewed: false`, and `forge pack`
refuses a draft that still says false. A generated claim list that shipped
unreviewed would put model-written text on the rejection path, which is exactly
the thing §7.4 forbids.

An LLM drafter can be injected (`drafter=`), but none is wired here: the judge
providers live in `launder_serve.judge` and forge does not depend on serve. The
shipped drafter is deterministic, which also makes it testable.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["ClaimDraft", "ClaimDrafter", "draft_claims", "write_draft"]

_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'“])")

#: Words whose presence marks a sentence as scene-setting rather than a claim.
#: Deliberately small: over-filtering hides claims from the reviewer, which is
#: worse than showing one they will delete.
_HEDGES = frozenset({"perhaps", "maybe", "might", "could", "seems", "sometimes"})

_STOP = frozenset(
    [
        "a",
        "an",
        "the",
        "of",
        "to",
        "in",
        "on",
        "at",
        "by",
        "for",
        "with",
        "and",
        "or",
        "but",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "it",
        "its",
        "that",
        "this",
        "those",
        "these",
        "as",
        "from",
        "into",
        "over",
        "under",
        "after",
        "before",
        "while",
        "there",
        "here",
        "you",
        "your",
    ]
)


@dataclass(slots=True)
class ClaimDraft:
    id: str
    text: str
    label: str
    required: bool = False
    source_sentence: int = 0
    confidence: float = 0.0
    note: str = ""


ClaimDrafter = Callable[[str], list[ClaimDraft]]


def _label_for(sentence: str) -> str:
    """A short noun phrase a player would recognise, from the sentence's own
    content words. Never longer than five words — the label appears inside a
    rejection line and a long one reads as a quotation."""
    words = [w.strip(".,;:!?\"'()").lower() for w in sentence.split()]
    content = [w for w in words if w and w not in _STOP and len(w) > 2]
    return " ".join(content[:4]) if content else sentence.strip()[:40]


def heuristic_drafter(text: str) -> list[ClaimDraft]:
    """Sentence-per-claim with a confidence the reviewer can sort by.

    Confidence is *not* a probability. It is `content words / total words`,
    which ranks concrete sentences above connective ones and is stated here so
    nobody reads it as a model score.
    """
    sentences = [s.strip() for s in _SENTENCE.split(text.strip()) if s.strip()]
    drafts: list[ClaimDraft] = []
    for i, sentence in enumerate(sentences):
        words = [w.strip(".,;:!?\"'()").lower() for w in sentence.split()]
        if len(words) < 4:
            continue
        content = [w for w in words if w and w not in _STOP and len(w) > 2]
        confidence = len(content) / max(1, len(words))
        hedged = bool(_HEDGES.intersection(words))
        drafts.append(
            ClaimDraft(
                id=f"c{len(drafts) + 1}",
                text=sentence,
                label=_label_for(sentence),
                required=False,
                source_sentence=i,
                confidence=round(confidence, 3),
                note="hedged — probably not a claim" if hedged else "",
            )
        )
    return drafts


@dataclass(slots=True)
class ClaimDraftFile:
    passage_id: str
    reviewed: bool = False
    drafter: str = "heuristic"
    claims: list[ClaimDraft] = field(default_factory=list)
    instructions: str = (
        "Review every claim. Delete the ones that are not claims. Set `required: true` on the "
        "one or two a submission MUST preserve. Rewrite `label` into the noun phrase you would "
        "say out loud to a player who dropped it — it is rendered verbatim into a rejection "
        "line and must never quote the player's own text. Then set reviewed: true."
    )

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": "launder.claims.draft/1",
            "passage_id": self.passage_id,
            "reviewed": self.reviewed,
            "drafter": self.drafter,
            "instructions": self.instructions,
            "claims": [asdict(c) for c in self.claims],
        }


def draft_claims(
    passage_id: str,
    text: str,
    *,
    drafter: ClaimDrafter | None = None,
    drafter_name: str = "heuristic",
) -> ClaimDraftFile:
    fn = drafter or heuristic_drafter
    return ClaimDraftFile(passage_id=passage_id, drafter=drafter_name, claims=fn(text))


def write_draft(path: Path, draft: ClaimDraftFile) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(draft.to_json(), indent=2, ensure_ascii=False) + "\n", "utf-8")
    return path


def load_reviewed_claims(path: Path) -> list[dict[str, Any]]:
    """Read a reviewed draft, refusing one that is still marked unreviewed."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data.get("reviewed"):
        raise ValueError(
            f"{path} still has reviewed: false. A generated claim list drives the judge and its "
            "labels are rendered into rejection copy; it must be read by a human before it "
            "ships. Edit the file and set reviewed: true."
        )
    return [
        {
            "id": c["id"],
            "text": c["text"],
            "label": c["label"],
            "required": bool(c.get("required", False)),
        }
        for c in data.get("claims", [])
    ]
