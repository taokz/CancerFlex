"""Helpers to construct a Matformer-ized MedGemma model.

Two entry points:
- `wrap_medgemma_mlps(model, controller)` — in-place swap of every Gemma3MLP
  in a loaded HF model with a NestedGemma3MLP. Preserves weights.
- `load_matformer_medgemma(path, controller=..., factor=..., dtype=...)` —
  high-level: load MedGemma-1.5-4B-IT (text-only), wrap, return (model, ctrl).
"""

from __future__ import annotations

from typing import Optional

import os
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM

from .granularity import GranularityController
from .nested_mlp import NestedGemma3MLP

# Base checkpoint: local path or Hub id, configurable through CANCERFLEX_BASE_MODEL.
DEFAULT_MEDGEMMA_PATH = os.environ.get("CANCERFLEX_BASE_MODEL", "google/medgemma-1.5-4b-it")


def _replace_mlps(module: nn.Module, controller: GranularityController, text_config) -> int:
    """Recurse over `module`, replacing every Gemma3MLP child with NestedGemma3MLP.
    Returns the number of replacements made.
    """
    from transformers.models.gemma3.modeling_gemma3 import Gemma3MLP

    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, Gemma3MLP):
            nested = NestedGemma3MLP(text_config, controller)
            nested.gate_proj.weight.data.copy_(child.gate_proj.weight.data)
            nested.up_proj.weight.data.copy_(child.up_proj.weight.data)
            nested.down_proj.weight.data.copy_(child.down_proj.weight.data)
            nested.to(device=child.gate_proj.weight.device, dtype=child.gate_proj.weight.dtype)
            setattr(module, name, nested)
            n += 1
        else:
            n += _replace_mlps(child, controller, text_config)
    return n


def wrap_medgemma_mlps(model: nn.Module, controller: GranularityController) -> int:
    """In-place replacement of Gemma3MLP with NestedGemma3MLP. Returns count."""
    cfg = model.config
    # Text-only `Gemma3ForCausalLM` has `cfg.intermediate_size` directly.
    # Multimodal `Gemma3ForConditionalGeneration` has `cfg.text_config.intermediate_size`.
    text_config = cfg.text_config if hasattr(cfg, "text_config") else cfg
    if controller.widths[-1] != text_config.intermediate_size:
        raise ValueError(
            f"controller's largest width {controller.widths[-1]} must equal "
            f"text_config.intermediate_size {text_config.intermediate_size}"
        )
    return _replace_mlps(model, controller, text_config)


def load_matformer_medgemma(
    path: str = DEFAULT_MEDGEMMA_PATH,
    factor: int = 8,
    controller: Optional[GranularityController] = None,
    dtype: torch.dtype = torch.bfloat16,
    text_only: bool = True,
    attn_implementation: str = "sdpa",
):
    """Load MedGemma-1.5 (text-only by default) and Matformer-ize its MLPs.

    Returns:
        (model, controller)
    """
    cfg = AutoConfig.from_pretrained(path)
    text_cfg = cfg.text_config if hasattr(cfg, "text_config") else cfg

    if controller is None:
        controller = GranularityController.from_exponential_factor(
            full_width=text_cfg.intermediate_size, factor=factor
        )

    if text_only:
        # Pull the text-only weights via the AutoModelForCausalLM path.
        # HF will instantiate `Gemma3ForCausalLM` from the text sub-config.
        model = AutoModelForCausalLM.from_pretrained(
            path,
            dtype=dtype,
            attn_implementation=attn_implementation,
        )
    else:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            path,
            dtype=dtype,
            attn_implementation=attn_implementation,
        )

    n_replaced = wrap_medgemma_mlps(model, controller)
    if n_replaced == 0:
        raise RuntimeError(
            "No Gemma3MLP modules were found to replace — check transformers version "
            "and that the loaded model is actually Gemma3-based."
        )
    return model, controller
