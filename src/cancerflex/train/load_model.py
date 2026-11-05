"""Model + processor loading for iter_02 (medgemma / gemma-3n).

Full-parameter (no peft / no LoRA) per owner directive. attn_implementation
follows HF / TRL VLM SFT recipe:
  - Gemma3 (medgemma)    : "eager" — Gemma3 FA2 path has known correctness
                           bugs (TRL docs + HF transformers#34532); eager is
                           the recommended fix on B200.
  - Gemma-3n             : "flash_attention_2" — Gemma-3n text path is FA2-clean
                           in the training environment (flash_attn 4.0.0b4).
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
from transformers.models.gemma3.modeling_gemma3 import Gemma3ForConditionalGeneration
from transformers.models.gemma3.processing_gemma3 import Gemma3Processor

MEDGEMMA_BASE = os.environ.get("CANCERFLEX_BASE_MODEL", "google/medgemma-1.5-4b-it")
GEMMA3N_BASE  = "google/gemma-3n-E4B"


def _find_vision_tower(model):
    """Locate the vision tower across transformers versions. In 5.x Gemma3
    the tower is `model.model.vision_tower`; older layouts use
    `model.vision_tower` directly. Returns the submodule or None.
    """
    for attr_path in (("model", "vision_tower"), ("vision_tower",)):
        cur = model
        ok = True
        for a in attr_path:
            if not hasattr(cur, a):
                ok = False
                break
            cur = getattr(cur, a)
        if ok:
            return cur
    return None


def load_medgemma(vision_freeze: bool = True,
                  attn_impl: str = "eager",
                  base_path: str | None = None):
    """medgemma 1.5 4B-it for direct SFT / matformer / teacher inference.

    `base_path` loads a LOCAL fine-tuned checkpoint's weights instead of the stock
    MEDGEMMA_BASE. Needed because importance ordering must be computed on the model it will
    actually be applied to: all three reference works sort BEFORE elastic training, and our
    existing permutation was scored on the stock base while every router arm runs on the
    width-sampled a2 checkpoint. The processor still comes from MEDGEMMA_BASE — fine-tuned
    checkpoints here do not ship a processor_config."""
    processor = Gemma3Processor.from_pretrained(MEDGEMMA_BASE)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        base_path or MEDGEMMA_BASE,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    )
    if vision_freeze:
        vt = _find_vision_tower(model)
        if vt is None:
            raise RuntimeError("vision_freeze=True but no vision_tower found on model")
        for p in vt.parameters():
            p.requires_grad = False
    return model, processor


def load_matformer_pure(vision_freeze: bool = True,
                        attn_impl: "str | dict" = "eager",   # str, or per-submodule {"text":..,"vision":..}
                        factor: int = 8,
                        base_path: str | None = None,
                        widths: "list[int] | None" = None):
    """Multimodal MatFormer medgemma built on the *pure* implementation at
    preliminary/Med-Matformer-pure — paper-faithful nested prefix-slice
    SwiGLU FFN STRUCTURE, NO neuron-ranking / activation-selection surgery
    (unlike an earlier, unreleased variant). The vision tower is left
    untouched. NOTE: this loader fixes only the structure; the training
    OBJECTIVE (paper-aligned sampled vs OLMo-style joint) is the trainer's
    `sampling` knob — see matformer_pure_main.py.

    Returns (model, processor, controller). `controller` is the shared
    GranularityController; mutate `controller.set(width)` to switch the active
    subnet across all nested MLPs.
    """
    from cancerflex.matformer.model import load_matformer_medgemma  # type: ignore

    model, controller = load_matformer_medgemma(
        path=base_path or MEDGEMMA_BASE,   # ①b: neuron-ranked permuted base via base_model_override
        factor=factor,
        dtype=torch.bfloat16,
        text_only=False,                      # multimodal Gemma3ForConditionalGeneration
        attn_implementation=attn_impl,
    )
    # OPTIONAL denser/custom width grid. `factor` only ever yields the paper's
    # exponential set {D/8, D/4, D/2, D}; measuring the COST of adding a width the
    # supernet has never been at (elastic_router_lessons.md §3.8: untrained widths
    # collapse *below both neighbours*) requires putting that width in the grid so
    # `sampling`/`granularity_weights` can select it. Prefix slicing is valid at any
    # d, so this is a training-schedule knob, not a structural change.
    if widths is not None:
        widths = [int(w) for w in widths]
        if sorted(widths) != widths or len(set(widths)) != len(widths):
            raise ValueError(f"widths must be sorted ascending and unique, got {widths}")
        if widths[-1] != controller.widths[-1]:
            # widths[-1] must stay the base intermediate_size, else the largest
            # granularity is no longer bit-identical to the base model.
            raise ValueError(f"widths[-1] must equal full width {controller.widths[-1]}, got {widths[-1]}")
        if widths[0] < 1:
            raise ValueError(f"widths must be positive, got {widths}")
        controller.widths = widths            # shared by reference across every NestedGemma3MLP
        controller.current_width = widths[-1]
        print(f"[load_matformer_pure] width grid overridden: {widths}", flush=True)
    processor = Gemma3Processor.from_pretrained(MEDGEMMA_BASE)
    if vision_freeze:
        vt = _find_vision_tower(model)
        if vt is None:
            raise RuntimeError("vision_freeze=True but no vision_tower found on matformer model")
        for p in vt.parameters():
            p.requires_grad = False
    return model, processor, controller


def load_gemma3n(vision_freeze: bool = True,
                 attn_impl: str = "flash_attention_2"):
    """gemma-3n-E4B as student for distillation."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(GEMMA3N_BASE)
    model = AutoModelForImageTextToText.from_pretrained(
        GEMMA3N_BASE,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    )
    if vision_freeze:
        vt = _find_vision_tower(model)
        if vt is not None:
            for p in vt.parameters():
                p.requires_grad = False
    # gemma-3n also ships an AUDIO tower (+ audio embeddings). Our VQA is
    # image+text only, so those params are never used in the loss. Under real
    # DDP (the SeqKD SFT path) unused trainable params trigger "Expected to
    # have finished reduction…" / find_unused_parameters errors. Freezing them
    # is correct (we never train audio) and also trims the option-KD path.
    n_audio = 0
    for name, p in model.named_parameters():
        if "audio" in name.lower() and p.requires_grad:
            p.requires_grad = False
            n_audio += 1
    if n_audio:
        print(f"[load_gemma3n] froze {n_audio} audio-tower params (unused for VQA)", flush=True)
    return model, processor


def load_teacher(ckpt_dir: str | Path):
    """Load a finished medgemma checkpoint (full-param) in eval mode for
    distillation teacher use. Returns (model, processor, info).

    Supports two on-disk formats:
      (a) Legacy iter_01 single-file: <dir>/model.pt (torch state-dict).
      (b) iter_02 TRL/HF format: <dir>/model.safetensors  +  config.json
          (what `SFTTrainer.save_model(...)` and per-epoch `checkpoint-N/`
          write). Loaded via `AutoModelForImageTextToText.from_pretrained`
          straight from the ckpt dir — uses native HF safetensors loading
          (strict, complete).

    `strict=True` (or HF native load) is enforced — silently accepting a
    partial-load teacher would let distillation train against a corrupted
    teacher signal. Caller still receives missing/unexpected dicts (empty
    when format (b) is used) for logging.
    """
    ckpt_dir = Path(ckpt_dir)
    if (ckpt_dir / "model.safetensors").exists() or (ckpt_dir / "config.json").exists():
        # (b) HF format — load directly from the ckpt dir.
        processor = Gemma3Processor.from_pretrained(MEDGEMMA_BASE)
        model = Gemma3ForConditionalGeneration.from_pretrained(
            str(ckpt_dir), torch_dtype=torch.bfloat16, attn_implementation="eager",
        )
        info = {"missing": [], "unexpected": []}
    else:
        # (a) Legacy model.pt (iter_01).
        model, processor = load_medgemma(vision_freeze=True, attn_impl="eager")
        state = torch.load(ckpt_dir / "model.pt", map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=True)
        info = {"missing": missing, "unexpected": unexpected}
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, processor, info
