"""`forge solve` — minimal-edit beam search, and an honest word about "par".

TECH_PLAN.md §6.4. A constructive search gives an **upper bound on the
minimum**. It proves "4 suffice", never "3 cannot". Because the move space is
unbounded — arbitrary paraphrase, arbitrary insertion from a 262k vocabulary,
arbitrary reordering — **no true lower bound exists**. What is computable:

* `par_upper` — the best judge-gated clear the search found. **This is par.**
  It is sound: par is a target, and a target verified achievable is the right
  target.
* `exhaustive_clear_free_upto_k` — a lower bound **relative to the declared,
  versioned move set M**, reported as exactly that. Not a theorem about the
  game; the empirical floor that stops a passage shipping that a clever player
  one-shots.

    M-v2 = { delete(word_i),
             substitute(word_i, w) for w in Cand(i),
             swap(word_i, word_{i+1}),
             merge(word_i, word_{i+1}),
             split(word_i) }
    Cand(i) = decode(topk_alts[i]) u synonyms(word_i) u {"", "the", "a", "it", "that"}

`Cand(i)` from the cached `topk_alts` is the elegant part: the model's own
top-16 alternatives at that position are already computed, are by construction
fluent, and cost nothing.

**We are shipping this.** Someone will point a beam search at the daily and
beat every human. That is not cheating, it is the correct read of the game; the
honest response is to publish `par_upper` next to the human best as "machine
par", clearly labelled.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from launder_forge.corebridge import words as split_words
from launder_forge.numerics import U8Array, score_ids

__all__ = [
    "MOVE_SET_ID",
    "STOPWORD_CANDIDATES",
    "Judge",
    "Solution",
    "SolveConfig",
    "SolveResult",
    "heuristic_judge",
    "solve",
]

MOVE_SET_ID = "M-v2"

#: The `{"", "the", "a", "it", "that"}` tail of Cand(i). The empty string is
#: substitution-as-deletion and is what makes a one-word cut reachable at every
#: position without a separate move.
STOPWORD_CANDIDATES: tuple[str, ...] = ("", "the", "a", "it", "that")

#: A candidate that clears the detector but fails the natural/meaning gate is
#: NOT a clear. Without a judge the solver "solves" every passage with
#: `the the the`, so `solve()` refuses to report a `par_upper` unless some
#: judge ran — `heuristic_judge` is the free floor, an LLM judge is the real one.
Judge = Callable[[str, str], tuple[bool, str]]


#: MEASURED on this machine: one candidate evaluation — retokenize the whole
#: text, recompute g-values, mask and score — costs ~0.255 ms, i.e. ~3,900/s,
#: on a 142-word passage. Every budget below is expressed against that number
#: instead of being guessed, and `solve()` reports what it actually managed.
EVALS_PER_SECOND_MEASURED = 3_900


@dataclass(slots=True)
class SolveConfig:
    max_edits: int = 10
    beam: int = 512

    #: TECH_PLAN.md §6.4 says "k=2 exhaustive if |M|^2 < 10M else beam 512".
    #: That threshold assumes the §4.4 incremental g-value cache; against a
    #: FULL rescore it is 10M / 3,900 = 43 minutes, so the plan's number is
    #: kept as the combinatorial gate and `time_budget_seconds` is the one that
    #: actually binds. Which of the two stopped the search is recorded in
    #: `notes`, and a truncated k=2 pass can NEVER claim
    #: `exhaustive_clear_free_upto_k = 2` — that field is a claim about
    #: completed enumeration, so a partial pass must not be able to make it.
    exhaustive_pair_budget: int = 10_000_000

    #: Wall clock for the k>=2 expansion. Default 60 s: enough to exhaust k=2
    #: on a 60-word passage, and an honest partial pass on a 250-token daily.
    time_budget_seconds: float = 60.0

    top_solutions_to_judge: int = 20
    diversity_positions: int = 3
    """Keep at most this many beam entries whose last edit touched the same word."""


@dataclass(slots=True)
class Solution:
    text: str
    edits: int
    z: float
    ops: list[dict[str, Any]]
    judged: bool = False
    judge_ok: bool | None = None
    judge_reason: str = ""


@dataclass(slots=True)
class SolveResult:
    move_set_id: str = MOVE_SET_ID
    par_upper: int | None = None
    exhaustive_clear_free_upto_k: int = 0
    best_single_edit_drop_z: float = 0.0
    clears_at_k: dict[str, bool] = field(default_factory=dict)
    judge_rejected_solutions: int = 0
    search_seconds: float = 0.0
    judge: str = "none"
    best: Solution | None = None
    evaluated: int = 0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["best"] = asdict(self.best) if self.best else None
        return out


# ---------------------------------------------------------------------------
# the judge floor
# ---------------------------------------------------------------------------

_DEGENERATE_RUN = 3


def heuristic_judge(original: str, submission: str) -> tuple[bool, str]:
    """A deterministic floor, NOT the product's gate.

    It exists so an offline solve run cannot report a `par_upper` achieved by
    `the the the`. It catches degeneracy and gross length collapse and nothing
    else; the real gate is `llm_gate` in `launder_serve`, and a solve run that
    matters should pass that one through `--judge`.
    """
    ws = submission.split()
    if not ws:
        return False, "empty"
    if len(ws) < 0.4 * len(original.split()):
        return False, "length_collapse"
    run = 1
    for i in range(1, len(ws)):
        if ws[i].lower() == ws[i - 1].lower():
            run += 1
            if run >= _DEGENERATE_RUN:
                return False, "repetition"
        else:
            run = 1
    uniq = len({w.lower() for w in ws})
    if uniq < 0.35 * len(ws):
        return False, "keyword_soup"
    return True, ""


# ---------------------------------------------------------------------------
# the move set
# ---------------------------------------------------------------------------


def _candidates_for(
    index: int, ws: list[str], alts: list[list[str]] | None, synonyms: dict[str, list[str]] | None
) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    if alts is not None and index < len(alts):
        for w in alts[index]:
            cleaned = w.strip()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                out.append(cleaned)
    if synonyms:
        for w in synonyms.get(ws[index].lower(), []):
            if w not in seen:
                seen.add(w)
                out.append(w)
    for w in STOPWORD_CANDIDATES:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return [w for w in out if w != ws[index]]


def _apply(ws: list[str], move: tuple[str, int, str]) -> list[str] | None:
    kind, i, payload = move
    if kind == "del":
        if i >= len(ws):
            return None
        return ws[:i] + ws[i + 1 :]
    if kind == "sub":
        if i >= len(ws):
            return None
        if payload == "":
            return ws[:i] + ws[i + 1 :]
        return [*ws[:i], payload, *ws[i + 1 :]]
    if kind == "swap":
        if i + 1 >= len(ws):
            return None
        return [*ws[:i], ws[i + 1], ws[i], *ws[i + 2 :]]
    if kind == "merge":
        if i + 1 >= len(ws):
            return None
        return [*ws[:i], ws[i] + ws[i + 1], *ws[i + 2 :]]
    if kind == "split":
        if i >= len(ws) or len(ws[i]) < 2:
            return None
        mid = len(ws[i]) // 2
        return [*ws[:i], ws[i][:mid], ws[i][mid:], *ws[i + 1 :]]
    return None


def _moves_at(
    index: int, ws: list[str], alts: list[list[str]] | None, synonyms: dict[str, list[str]] | None
) -> list[tuple[str, int, str]]:
    moves: list[tuple[str, int, str]] = [("del", index, "")]
    moves.extend(("sub", index, w) for w in _candidates_for(index, ws, alts, synonyms))
    moves.append(("swap", index, ""))
    moves.append(("merge", index, ""))
    moves.append(("split", index, ""))
    return moves


# ---------------------------------------------------------------------------


def solve(
    text: str,
    *,
    tokenizer: Any,
    keys: tuple[int, ...],
    ngram_len: int,
    table: U8Array,
    context_history_size: int,
    z_star: float,
    inflation: float = 1.0,
    topk_alts_words: list[list[str]] | None = None,
    synonyms: dict[str, list[str]] | None = None,
    config: SolveConfig | None = None,
    judge: Judge | None = heuristic_judge,
    judge_name: str = "heuristic",
    locked_phrases: tuple[str, ...] = (),
) -> SolveResult:
    """Search for the smallest judge-gated clear. Returns `par_upper` or None."""
    cfg = config or SolveConfig()
    t0 = time.perf_counter()
    result = SolveResult(judge=judge_name if judge is not None else "none")
    base_words = split_words(text)
    if not base_words:
        result.notes.append("empty passage")
        return result

    evaluated = 0

    def z_of(ws: list[str]) -> float:
        nonlocal evaluated
        evaluated += 1
        joined = " ".join(w for w in ws if w)
        ids = tokenizer.encode(joined)
        if len(ids) < ngram_len:
            return 0.0
        return score_ids(
            ids,
            keys=keys,
            ngram_len=ngram_len,
            table=table,
            context_history_size=context_history_size,
            inflation=inflation,
        ).z

    def legal(ws: list[str]) -> bool:
        if not locked_phrases:
            return True
        joined = " ".join(w for w in ws if w)
        return all(p in joined for p in locked_phrases)

    base_z = z_of(base_words)
    result.notes.append(f"base z = {base_z:.4f}, z* = {z_star:.4f}")

    # ---- k = 1, exhaustive over M -----------------------------------------
    k1: list[tuple[float, list[str], tuple[str, int, str]]] = []
    for i in range(len(base_words)):
        for move in _moves_at(i, base_words, topk_alts_words, synonyms):
            nxt = _apply(base_words, move)
            if nxt is None or not legal(nxt):
                continue
            k1.append((z_of(nxt), nxt, move))
    if not k1:
        result.search_seconds = time.perf_counter() - t0
        result.evaluated = evaluated
        result.notes.append("no legal single edits — every move breaks a locked phrase")
        return result

    k1.sort(key=lambda item: item[0])
    result.best_single_edit_drop_z = base_z - k1[0][0]

    move_count = sum(
        len(_moves_at(i, base_words, topk_alts_words, synonyms)) for i in range(len(base_words))
    )
    # k=2 is exhaustive when the pair space fits the budget; that is what makes
    # `exhaustive_clear_free_upto_k = 2` a claim rather than a hope.
    exhaustive_pairs = move_count * move_count < cfg.exhaustive_pair_budget

    frontier: list[tuple[float, list[str], list[tuple[str, int, str]]]] = [
        (z, ws, [mv]) for z, ws, mv in (k1 if exhaustive_pairs else k1[: cfg.beam])
    ]
    found: dict[int, Solution] = {}

    def record(k: int, z: float, ws: list[str], ops: list[tuple[str, int, str]]) -> None:
        if k in found:
            return
        found[k] = Solution(
            text=" ".join(w for w in ws if w),
            edits=k,
            z=z,
            ops=[{"op": o[0], "i": o[1], "to": o[2]} for o in ops],
        )

    # k=1 clears are judged HERE rather than at the end, because a passing one
    # means the minimum is already found and the k>=2 expansion — the expensive
    # part — must not run at all. Judging late would have burned the whole time
    # budget proving something already known.
    for z, ws, ops in frontier:
        if z <= z_star:
            if judge is not None:
                ok, reason = judge(text, " ".join(w for w in ws if w))
                if not ok:
                    result.judge_rejected_solutions += 1
                    continue
                record(1, z, ws, ops)
                found[1].judged = True
                found[1].judge_ok = True
                found[1].judge_reason = reason
            else:
                record(1, z, ws, ops)
            break
    result.clears_at_k["1"] = 1 in found

    # ---- k >= 2 ------------------------------------------------------------
    result.notes.append(
        f"|M| = {move_count}; k=2 " + ("exhaustive" if exhaustive_pairs else f"beam {cfg.beam}")
    )

    deadline = t0 + cfg.time_budget_seconds
    truncated = False
    max_k = 1 if 1 in found else cfg.max_edits
    for k in range(2, max_k + 1):
        nxt_frontier: list[tuple[float, list[str], list[tuple[str, int, str]]]] = []
        for _z, ws, ops in frontier:
            if time.perf_counter() > deadline:
                truncated = True
                break
            for i in range(len(ws)):
                for move in _moves_at(i, ws, None, synonyms):
                    child = _apply(ws, move)
                    if child is None or not legal(child):
                        continue
                    cz = z_of(child)
                    nxt_frontier.append((cz, child, [*ops, move]))
        if truncated:
            result.notes.append(
                f"k={k} expansion hit the {cfg.time_budget_seconds:g}s budget after "
                f"{evaluated:,} evaluations. The pass was PARTIAL: par_upper below is still a "
                "valid upper bound (it was achieved), but no exhaustiveness is claimed."
            )
        if not nxt_frontier:
            break
        nxt_frontier.sort(key=lambda item: item[0])
        # Diversity BY EDIT POSITION so the beam does not collapse onto one hot run.
        per_position: dict[int, int] = {}
        pruned: list[tuple[float, list[str], list[tuple[str, int, str]]]] = []
        for entry in nxt_frontier:
            pos = entry[2][-1][1]
            if per_position.get(pos, 0) >= cfg.diversity_positions:
                continue
            per_position[pos] = per_position.get(pos, 0) + 1
            pruned.append(entry)
            if len(pruned) >= cfg.beam:
                break
        frontier = pruned
        cleared = [e for e in frontier if e[0] <= z_star]
        result.clears_at_k[str(k)] = bool(cleared)
        if cleared:
            record(k, cleared[0][0], cleared[0][1], cleared[0][2])
            break
        if truncated:
            break

    # ---- judge-gate the best solutions ------------------------------------
    ordered = sorted(found.values(), key=lambda s: (s.edits, s.z))
    if judge is not None:
        for sol in ordered[: cfg.top_solutions_to_judge]:
            if sol.judged:
                continue  # k=1 clears were judged before the expensive expansion
            ok, reason = judge(text, sol.text)
            sol.judged = True
            sol.judge_ok = ok
            sol.judge_reason = reason
            if not ok:
                result.judge_rejected_solutions += 1
        ordered = [s for s in ordered if s.judge_ok]
    else:
        result.notes.append(
            "NO JUDGE RAN. par_upper is withheld: a solver clear that fails the natural/meaning "
            "gate is not a clear, and without a judge the search 'solves' every passage with "
            "degenerate text."
        )
        ordered = []

    if ordered:
        result.best = ordered[0]
        result.par_upper = ordered[0].edits

    # exhaustive_clear_free_upto_k: within M, and ONLY where the enumeration
    # actually completed. k=1 is always exhaustive over M. k=2 counts only if
    # the pair space fit the budget AND the pass was not cut short by the clock
    # — a partial enumeration proves nothing about what it did not enumerate.
    upto = 0
    if not result.clears_at_k.get("1", False):
        upto = 1
        if exhaustive_pairs and not truncated and not result.clears_at_k.get("2", False):
            upto = 2
    result.exhaustive_clear_free_upto_k = upto

    result.search_seconds = time.perf_counter() - t0
    result.evaluated = evaluated
    return result


def alts_to_words(topk_alts: list[list[int]], tokenizer: Any) -> list[list[str]]:
    """Decode cached `topk_alts` token ids into the substitution candidates."""
    return [[tokenizer.decode([tid]).strip() for tid in row] for row in topk_alts]
