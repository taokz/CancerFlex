"""Real 4-bit (bitsandbytes NF4) quant config — env-gated, no-op unless EVAL_QUANT=nf4.

Contrast with `ptq_hook.py` (FAKE-quant: RTN quant->dequant, stays bf16, measures precision
only). This produces a `BitsAndBytesConfig` that makes `from_pretrained` store the LM's linear
weights **packed in 4-bit** (real VRAM reduction), keeping the multimodal vision path + lm_head
in bf16. See docs/methodology/int4_ptq_vs_bf16.md and the design spec
docs/archive/designs/2026-07-15-real-nf4-quant-eval-design.md.

Usage in a loader:
    from bnb_quant import maybe_bnb_config_from_env
    qcfg = maybe_bnb_config_from_env()   # None unless EVAL_QUANT=nf4
    model = AutoModelForImageTextToText.from_pretrained(
        src, dtype=torch.bfloat16, quantization_config=qcfg, device_map="auto")

Skip-list (Codex-verified for Gemma3ForConditionalGeneration in transformers 5.3): vision+projector
nest under `self.model`, lm_head is top-level → names MUST be prefixed. `llm_int8_skip_modules`
is honored for 4-bit (quantizer_bnb_4bit.py).
"""
import os

# top-level module-name prefixes kept in bf16 (NOT quantized)
SKIP_MODULES = ["model.vision_tower", "model.multi_modal_projector", "lm_head"]


def maybe_bnb_config_from_env():
    """Return a BitsAndBytesConfig when EVAL_QUANT=nf4, else None (no-op).

    Refuses to coexist with the fake-quant hook (EVAL_PTQ_BITS) — the two are mutually exclusive
    quant paths and combining them would double-quantize / mislabel results.
    """
    mode = os.environ.get("EVAL_QUANT", "").strip().lower()
    if not mode:
        return None
    if mode != "nf4":
        raise ValueError(f"EVAL_QUANT={mode!r} unsupported (only 'nf4' implemented)")
    if os.environ.get("EVAL_PTQ_BITS"):
        raise RuntimeError(
            "EVAL_QUANT=nf4 (real bnb 4-bit) and EVAL_PTQ_BITS (fake RTN quant) are mutually "
            "exclusive — set only one.")
    import torch
    from transformers import BitsAndBytesConfig
    compute = os.environ.get("EVAL_QUANT_COMPUTE_DTYPE", "bfloat16")
    cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=getattr(torch, compute),
        llm_int8_skip_modules=list(SKIP_MODULES),
    )
    return cfg


def assert_real_4bit(model, *, require_frac=0.5):
    """Fail loudly if the model did NOT actually load in 4-bit (guards a silent all-bf16 load
    that would fake the memory win). Returns a dict summary. Asserts Linear4bit appears under
    the LM layers but NOT under the skipped vision/projector/lm_head modules."""
    import bitsandbytes as bnb
    n_lin4 = n_linear = 0
    in_lm = in_skip = 0
    for name, mod in model.named_modules():
        is4 = isinstance(mod, bnb.nn.Linear4bit)
        is_lin = is4 or mod.__class__.__name__ in ("Linear",)
        if is_lin:
            n_linear += 1
        if is4:
            n_lin4 += 1
            if ".language_model." in name or name.startswith("model.language_model"):
                in_lm += 1
            if any(s in name for s in ("vision_tower", "multi_modal_projector")) or name == "lm_head":
                in_skip += 1
    frac = n_lin4 / max(1, n_linear)
    summary = {"n_linear4bit": n_lin4, "n_linear_total": n_linear, "frac_4bit": round(frac, 3),
               "lin4_in_lm": in_lm, "lin4_in_skipped": in_skip}
    if n_lin4 == 0 or frac < require_frac:
        raise RuntimeError(f"real-4bit assertion FAILED — {summary} (expected majority of LM "
                           f"linears as Linear4bit; a ~0% load means weights stayed bf16)")
    if in_skip > 0:
        raise RuntimeError(f"real-4bit assertion FAILED — {in_skip} Linear4bit under SKIPPED "
                           f"vision/projector/lm_head modules; skip-list did not apply: {summary}")
    return summary
