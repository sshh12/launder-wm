"""`forge gen` — batched watermarked generation (TECH_PLAN.md §6.2).

Requires CUDA and the gated `google/gemma-3-4b-it` weights. Neither is stubbed:
every function is written to run, and the ones that cannot run here fail with a
message naming the missing piece and the command that fixes it.

Three decisions are enforced in code rather than left to the caller:

1. **The sampling table is overwritten with the committed CPU bytes.** §4.3
   failure #3 — HF builds the table with `torch.Generator(device).manual_seed(0)`
   and CUDA (Philox) need not agree with CPU (MT19937). Generating with a
   device-built table and detecting with the committed one yields g-values
   uncorrelated with the watermark: a needle that moves and measures nothing.
   `forge check-table` reports whether they actually differ; the override ships
   regardless, because the guarantee is free and the failure is silent.
2. **The warper settings of §6.2, including `top_k = 0`.** A small top_k
   truncates the very optionality the game is built on, and `repetition_penalty`
   distorts the distribution the cached `g_mass` describes.
3. **The scoring unit is the completion text alone**, re-tokenized with
   `add_special_tokens=False`. Never prompt+completion (§6.2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from launder_core.schemas import SynthIDConfig

__all__ = [
    "GEN_KWARGS",
    "Candidate",
    "ModelBundle",
    "generate_batch",
    "load_model",
    "make_watermark_processor",
]

#: §6.2, verbatim. `do_sample=True` is mandatory — watermarking is a NO-OP on
#: greedy decoding, and a greedy run would produce passages with no watermark
#: and no error message.
GEN_KWARGS: dict[str, Any] = {
    "do_sample": True,
    "temperature": 0.95,
    "top_p": 0.95,
    "top_k": 0,
    "repetition_penalty": 1.0,
    "min_new_tokens": 110,
    "max_new_tokens": 380,
}


@dataclass(slots=True)
class ModelBundle:
    model: Any
    tokenizer: Any
    model_id: str
    revision: str
    dtype: str
    device: str


@dataclass(slots=True)
class Candidate:
    """One generated passage, before analysis."""

    candidate_id: str
    text: str
    token_ids: list[int]
    prompt_rendered: str
    gen_params: dict[str, Any]
    seed: int
    status: str = "generated"
    reject_reasons: list[str] = field(default_factory=list)


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "generation needs torch. The forge pins it behind an extra so the workspace "
            "resolves on CPU-only machines: `uv sync --extra cuda` on the 5090 box."
        ) from exc
    return torch


def _require_cuda() -> Any:
    torch = _require_torch()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "generation requires CUDA and none is available. Run `forge doctor` for the full "
            "report. Nothing downstream of generation needs a GPU: score, solve, triage, "
            "calibrate, vectors, pack and verify all run from cached token ids on CPU."
        )
    return torch


def load_model(model_id: str = "google/gemma-3-4b-it", revision: str | None = None) -> ModelBundle:
    """bf16, no quantization, `attn_implementation="sdpa"`.

    Quantization perturbs the logits, and the watermark's premise is that the
    EXACT post-warper distribution determines both the sampled token and the
    cached `g_mass` (§6.1). flash-attn wheels on Windows are a trap; sdpa is
    the supported path.
    """
    torch = _require_cuda()
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("transformers is not installed; run `uv sync`") from exc

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation="sdpa",
        )
    except Exception as exc:
        raise RuntimeError(
            f"could not load {model_id}. It is gated (`gated: manual`): accept the Gemma Terms "
            "of Use on the model page, put HF_TOKEN in .env, and set HF_HOME to a drive with "
            "9 GB free. The architecture is Gemma3ForConditionalGeneration; the 1B text-only "
            f"variant is a different repo. Underlying error: {type(exc).__name__}: {exc}"
        ) from exc

    model.eval()  # type: ignore[no-untyped-call]
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for batched decode
    return ModelBundle(
        model=model,
        tokenizer=tokenizer,
        model_id=model_id,
        revision=str(getattr(model.config, "_commit_hash", "") or revision or ""),
        dtype="bfloat16",
        device="cuda",
    )


def _watermarking_config_cls() -> Any:
    """`SynthIDTextWatermarkingConfig` moved between transformers majors."""
    errors = []
    for module in (
        "transformers.generation.configuration_utils",
        "transformers.generation.watermarking",
        "transformers",
    ):
        try:
            mod = __import__(module, fromlist=["SynthIDTextWatermarkingConfig"])
            return mod.SynthIDTextWatermarkingConfig
        except (ImportError, AttributeError) as exc:
            errors.append(f"{module}: {exc}")
    raise RuntimeError(
        "could not locate SynthIDTextWatermarkingConfig in this transformers install. "
        "Tried:\n  " + "\n  ".join(errors)
    )


def make_watermark_processor(cfg: SynthIDConfig, table_path: Path, device: str = "cuda") -> Any:
    """Build the HF processor and OVERWRITE its sampling table with ours.

    This is the single most important line in the generation path. See §4.3 #3.
    """
    torch = _require_torch()
    from transformers.generation.logits_process import SynthIDTextWatermarkLogitsProcessor

    from launder_forge.numerics import load_sampling_table

    proc = SynthIDTextWatermarkLogitsProcessor(
        ngram_len=cfg.ngram_len,
        keys=list(cfg.keys),
        sampling_table_size=cfg.sampling_table_size,
        sampling_table_seed=cfg.sampling_table_seed,
        context_history_size=cfg.context_history_size,
        device=torch.device(device),
        skip_first_ngram_calls=cfg.skip_first_ngram_calls,
    )
    committed = load_sampling_table(table_path, cfg.sampling_table_size)
    override = torch.from_numpy(committed.astype(np.int64)).to(device)
    if override.shape != proc.sampling_table.shape:
        raise RuntimeError(
            f"committed sampling table has shape {tuple(override.shape)} but the processor "
            f"expects {tuple(proc.sampling_table.shape)}"
        )
    identical = bool(torch.equal(override, proc.sampling_table.to(override.dtype)))
    proc.sampling_table = override
    # Provenance for the run manifest. WHETHER the override changed anything is
    # itself the TECH_PLAN.md §14.2 item 1 measurement, taken on the machine that
    # actually generates, which is the only place it can be taken.
    proc.launder_table_overridden = True  # type: ignore[attr-defined]
    proc.launder_table_was_identical = identical  # type: ignore[attr-defined]
    return proc


def generate_batch(
    bundle: ModelBundle,
    prompts: list[str],
    *,
    max_new_tokens: int = 380,
    min_new_tokens: int = 110,
    temperature: float = 0.95,
    seed: int = 0,
    watermark: SynthIDConfig | None,
    table_path: Path | None = None,
    system_prompt: str | None = None,
) -> list[Candidate]:
    """Generate one batch. `watermark=None` produces NEGATIVES for `calibrate`."""
    torch = _require_cuda()
    tok = bundle.tokenizer

    rendered: list[str] = []
    for prompt in prompts:
        if hasattr(tok, "apply_chat_template") and tok.chat_template:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            rendered.append(
                tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            )
        else:
            rendered.append(prompt)

    enc = tok(rendered, return_tensors="pt", padding=True, add_special_tokens=False)
    enc = {k: v.to(bundle.device) for k, v in enc.items()}

    kwargs = dict(GEN_KWARGS)
    kwargs.update(
        temperature=temperature, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens
    )
    if watermark is not None:
        if table_path is None:
            raise ValueError("watermarked generation needs the committed sampling table path")
        cls = _watermarking_config_cls()
        kwargs["watermarking_config"] = cls(
            ngram_len=watermark.ngram_len,
            keys=list(watermark.keys),
            context_history_size=watermark.context_history_size,
            sampling_table_seed=watermark.sampling_table_seed,
            sampling_table_size=watermark.sampling_table_size,
            skip_first_ngram_calls=watermark.skip_first_ngram_calls,
        )
        # Belt and braces: HF constructs its own processor from the config
        # above, so we also pass an explicit processor whose table we control.
        from transformers import LogitsProcessorList

        kwargs["logits_processor"] = LogitsProcessorList(
            [make_watermark_processor(watermark, table_path, bundle.device)]
        )
        kwargs.pop("watermarking_config")

    torch.manual_seed(seed)
    with torch.inference_mode():
        out = bundle.model.generate(**enc, **kwargs)

    prompt_len = int(enc["input_ids"].shape[1])
    completions = out[:, prompt_len:]
    results: list[Candidate] = []
    for i in range(completions.shape[0]):
        ids = [int(x) for x in completions[i].tolist()]
        if tok.eos_token_id is not None and tok.eos_token_id in ids:
            ids = ids[: ids.index(tok.eos_token_id)]
        if tok.pad_token_id is not None:
            ids = [x for x in ids if x != tok.pad_token_id]
        text = tok.decode(ids, skip_special_tokens=True).strip()
        # THE CANONICAL SCORING UNIT: re-tokenize the passage text ALONE.
        # The generated ids include chat scaffolding artefacts and the decode
        # may normalise whitespace; only `encode(text)` is what the browser
        # will see, so that is what we store.
        canonical = [int(x) for x in tok(text, add_special_tokens=False)["input_ids"]]
        results.append(
            Candidate(
                candidate_id=f"c_{seed}_{i:05d}",
                text=text,
                token_ids=canonical,
                prompt_rendered=rendered[i],
                gen_params={
                    k: v for k, v in kwargs.items() if isinstance(v, (int, float, bool, str))
                },
                seed=seed + i,
            )
        )
    return results
