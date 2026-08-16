"""`forge analyze` — teacher-forced optionality metrics (TECH_PLAN.md §6.2, §6.3).

The watermark only bites where the model had near-equal choices: per-token
signal is `(1 - C_t)/4` where `C_t = sum_i p_i^2` is the collision probability
of the **post-warper** distribution. At `C = 1` the signal is exactly zero.
Low-entropy text is unwatermarkable, unlaunderable, and an unplayable level —
so these numbers are the triage input that decides which candidates survive.

Everything computed here is `*.author.json` only. `PassagePublic` forbids all
six fields by name (§4.6 mechanism 2): the browser cannot recompute them for
edited text, so shipping them would tempt a heat mirror coloured by
generation-time entropy — stale after keystroke one, and it teaches exactly the
heuristic CONCEPT.md forbids.

Memory: the naive teacher-forced pass materialises `[B, T, 262144]`, which is
2 GB for a single 200-token sequence in fp32. The forward pass therefore stops
at the decoder and the LM head is applied in `chunk` position-slices (§6.1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from launder_forge.numerics import (
    LCG_INCREMENT,
    LCG_MULTIPLIER,
    F64Array,
    U8Array,
)

__all__ = ["Optionality", "analyze_ids", "g_mass_for_distribution"]


@dataclass(slots=True)
class Optionality:
    """Per-position generation-time metrics. Author-side only, always.

    Indexing: entry `i` describes the distribution the model held **before**
    emitting `token_ids[i]`, so entry 0 is the distribution over the first
    token given only the prompt.
    """

    entropy_nats: list[float]
    eff_choices: list[float]
    collision: list[float]
    signal: list[float]
    top1_prob: list[float]
    g_mass: list[float]
    wm_boost: list[float]
    topk_alts: list[list[int]]
    topk_probs: list[list[float]]

    @property
    def median_eff_choices(self) -> float:
        return float(np.median(self.eff_choices)) if self.eff_choices else 0.0


def _u64(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(x, dtype=np.int64).view(np.uint64)


def g_mass_for_distribution(
    context_ids: np.ndarray,
    candidate_ids: np.ndarray,
    probs: F64Array,
    *,
    keys: tuple[int, ...],
    table: U8Array,
) -> tuple[float, F64Array]:
    """`(g_mass, per_depth_g_mass)` for one position.

    `g_mass_L = sum_v p_v * g_L(context || v)` — the probability mass the
    watermark would reward at depth L. The mean over depths is the scalar the
    author bundle stores. Computed over the **surviving support** of the
    post-warper distribution, which after `top_p = 0.95` is tens of tokens, not
    262,144: the truncated tail carries no probability, so it cannot carry
    g-mass either.
    """
    m = np.uint64(LCG_MULTIPLIER)
    inc = np.uint64(LCG_INCREMENT)
    h = np.uint64(1)
    for tok in _u64(context_ids):
        h = (h + tok) * m + inc
    with np.errstate(over="ignore"):
        h_cand = (h + _u64(candidate_ids)) * m + inc  # (V,)
        keys_u = _u64(np.asarray(keys, dtype=np.int64))
        hl = (h_cand[:, None] + keys_u[None, :]) * m + inc  # (V, depth)
    size = table.shape[0]
    g = table[(hl & np.uint64(size - 1)).astype(np.intp)]  # (V, depth)
    per_depth = probs @ g.astype(np.float64)
    return float(per_depth.mean()), per_depth


def analyze_ids(
    bundle: Any,
    token_ids: list[int],
    *,
    keys: tuple[int, ...],
    ngram_len: int,
    table: U8Array,
    temperature: float = 0.95,
    top_p: float = 0.95,
    top_k: int = 16,
    chunk: int = 32,
    prompt_ids: list[int] | None = None,
) -> Optionality:
    """One teacher-forced pass over a cached passage.

    `prompt_ids` is prepended for the forward pass only. It changes the
    distribution the first few tokens were drawn from (the prompt-boundary
    effect), but never the stored `token_ids`, and never the scoring unit.
    """
    torch = _require_torch()
    model = bundle.model
    device = bundle.device

    full = list(prompt_ids or []) + list(token_ids)
    offset = len(prompt_ids or [])
    ids = torch.tensor([full], dtype=torch.long, device=device)

    decoder = model.get_decoder() if hasattr(model, "get_decoder") else model.model
    head = model.get_output_embeddings()
    if head is None:  # pragma: no cover - architecture guard
        raise RuntimeError(
            f"{type(model).__name__} exposes no output embedding; cannot run the LM head "
            "separately, which is what keeps the [B, T, 262144] logits tensor off the GPU."
        )

    with torch.inference_mode():
        hidden = decoder(input_ids=ids).last_hidden_state  # [1, T, H]

    entropy: list[float] = []
    eff: list[float] = []
    collision: list[float] = []
    signal: list[float] = []
    top1: list[float] = []
    gmass: list[float] = []
    boost: list[float] = []
    alts: list[list[int]] = []
    alt_probs: list[list[float]] = []

    n_positions = len(full)
    for start in range(0, n_positions, chunk):
        end = min(start + chunk, n_positions)
        with torch.inference_mode():
            logits = head(hidden[:, start:end, :]).float()[0]  # [chunk, V]
            logits = logits / temperature
            probs_t = torch.softmax(logits, dim=-1)
            if 0.0 < top_p < 1.0:
                srt, idx = torch.sort(probs_t, descending=True, dim=-1)
                cum = torch.cumsum(srt, dim=-1)
                keep = cum - srt < top_p
                keep[..., 0] = True
                srt = srt * keep
                srt = srt / srt.sum(dim=-1, keepdim=True)
                probs_t = torch.zeros_like(probs_t).scatter_(-1, idx, srt)
            probs_np = probs_t.cpu().numpy().astype(np.float64)

        for r in range(end - start):
            pos = start + r
            # Position `pos` predicts token `pos + 1`; only positions that
            # predict a stored token are of interest.
            target_index = pos + 1
            if target_index < offset or target_index >= n_positions:
                continue
            p = probs_np[r]
            support = np.nonzero(p > 0)[0]
            ps = p[support]
            ps = ps / ps.sum()
            h = float(-(ps * np.log(np.clip(ps, 1e-300, None))).sum())
            c = float((ps**2).sum())
            entropy.append(h)
            eff.append(float(np.exp(h)))
            collision.append(c)
            signal.append((1.0 - c) / 4.0)
            top1.append(float(ps.max()))

            order = np.argsort(-ps)[:top_k]
            alts.append([int(support[i]) for i in order])
            alt_probs.append([float(ps[i]) for i in order])

            ctx_start = max(0, target_index - (ngram_len - 1))
            context = np.asarray(full[ctx_start:target_index], dtype=np.int64)
            gm, _per_depth = g_mass_for_distribution(
                context, support.astype(np.int64), ps, keys=keys, table=table
            )
            gmass.append(gm)

            # wm_boost: how much the watermark ACTUALLY bit at this position —
            # the realised mean g-value of the token that was emitted, minus
            # the g-mass the distribution offered. Positive means the
            # tournament steered towards a g=1 token; ~0 means it could not.
            emitted = np.asarray([full[target_index]], dtype=np.int64)
            realised, _ = g_mass_for_distribution(
                context, emitted, np.asarray([1.0]), keys=keys, table=table
            )
            boost.append(realised - gm)

    return Optionality(
        entropy_nats=entropy,
        eff_choices=eff,
        collision=collision,
        signal=signal,
        top1_prob=top1,
        g_mass=gmass,
        wm_boost=boost,
        topk_alts=alts,
        topk_probs=alt_probs,
    )


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "`forge analyze` needs torch and the Gemma-3 weights: the optionality metrics are "
            "properties of the model's post-warper distribution and cannot be recovered from "
            "token ids. Run `uv sync --extra cuda`, then `forge doctor`."
        ) from exc
    return torch
