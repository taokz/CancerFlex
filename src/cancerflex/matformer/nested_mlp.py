"""Nested-width SwiGLU MLP for Gemma3.

Drop-in replacement for `transformers.models.gemma3.modeling_gemma3.Gemma3MLP`.
Holds full-size projection weights (same shape as base Gemma3) and selects a
prefix slice at forward time based on a shared `GranularityController`.

At the largest width (== `intermediate_size`), the forward is bit-identical
to base Gemma3MLP — this is enforced by `tests/test_xl_equivalence.py`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers.activations import ACT2FN

from .granularity import GranularityController


class NestedGemma3MLP(nn.Module):
    """SwiGLU with prefix-slice nesting on the intermediate dim.

    Params:
        gate_proj.weight : (intermediate_size, hidden_size)
        up_proj.weight   : (intermediate_size, hidden_size)
        down_proj.weight : (hidden_size, intermediate_size)

    Forward at current width `d` uses:
        gate_proj.weight[:d, :], up_proj.weight[:d, :], down_proj.weight[:, :d]
    """

    def __init__(self, config, controller: GranularityController):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_activation]
        self.controller = controller
        if controller.widths[-1] != self.intermediate_size:
            raise ValueError(
                f"controller's largest width {controller.widths[-1]} must equal "
                f"intermediate_size {self.intermediate_size}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── ROUTED (Gumbel straight-through) mode ──────────────────────────────
        # When `route_gate` is set (a per-neuron gate in [0,1] over the FULL intermediate dim,
        # typically a straight-through hard config: hard forward, soft backward), compute the FULL
        # intermediate and multiply by the gate. This makes the forward DIFFERENTIABLE w.r.t. the
        # gate → the router (Nemotron-Elastic style), at the cost of full-width FFN compute. Default
        # None → the original hard-slice behaviour, so eval and non-routed training are unchanged.
        g = getattr(self, "route_gate", None)
        if g is not None:
            gate = self.gate_proj(x)
            up = self.up_proj(x)
            h = self.act_fn(gate) * up                 # (..., intermediate_size)
            return self.down_proj(h * g)               # g broadcasts over (..., intermediate_size)
        d = self.controller.current_width
        if d == self.intermediate_size:
            # Fast path: full-width forward is bit-identical to base Gemma3MLP.
            gate = self.gate_proj(x)
            up = self.up_proj(x)
            return self.down_proj(self.act_fn(gate) * up)
        # Sliced path
        gate_w = self.gate_proj.weight[:d, :]
        up_w = self.up_proj.weight[:d, :]
        down_w = self.down_proj.weight[:, :d]
        gate = torch.nn.functional.linear(x, gate_w)
        up = torch.nn.functional.linear(x, up_w)
        return torch.nn.functional.linear(self.act_fn(gate) * up, down_w)

    def extra_repr(self) -> str:
        return f"hidden={self.hidden_size}, intermediate={self.intermediate_size}, widths={self.controller.widths}"
