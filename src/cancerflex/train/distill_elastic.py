"""MatFormer-style elastic-width wrapper for the gemma-3n student.

Why this exists
---------------
gemma-3n is a MatFormer (nested) model: E4B contains E2B (and a continuum of
Mix-n-Match configs) as *prefix slices* of each feed-forward block. If we
distill medgemma -> gemma-3n by only ever training the **full** E4B forward,
the shared weights drift under the full-width path and the nested sub-models
(the on-device E2B we actually want at the edge) silently degrade.

So we make the student elastic during distillation: at each step we sample a
width fraction, forward the student through only that slice, and distill the
(width-independent) teacher into it. Over training every sampled sub-model is
pulled toward the teacher, so the whole nested family stays good — the proper
"distill logits into each subnet" reading of MatFormer KD.

Mechanism
---------
`Gemma3nTextMLP` is a SwiGLU block:
    gate_proj : Linear(hidden, intermediate[layer])
    up_proj   : Linear(hidden, intermediate[layer])
    down_proj : Linear(intermediate[layer], hidden)
    (+ optional activation_sparsity top-k over the intermediate dim)

A MatFormer sub-model of fraction f uses the first d = round(f * intermediate)
neurons:  gate_proj.weight[:d], up_proj.weight[:d], down_proj.weight[:, :d].
The `_gaussian_topk` sparsity recomputes mean/std over the *active* d neurons,
so sparsity adapts to the sub-model automatically — exactly the MatFormer
behaviour we want.

We do NOT swap the module class (that would re-register params and disturb
gradient-checkpointing / optimizer state). Instead we monkeypatch each
`Gemma3nTextMLP.forward` to a bound elastic forward that reads a shared
`ElasticWidthController`. The full-width path (f == 1.0) is bit-identical to
the stock forward (same ops, no slicing), so width 1.0 == base gemma-3n.

Per-layer fraction
------------------
gemma-3n's `intermediate_size` is per-layer (a Sequence), so the controller
holds a *fraction* (not an absolute width) and each MLP resolves its own d.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

# gemma-3n's MLP class name (matched by name so we don't need the import here).
_GEMMA3N_MLP_CLSNAME = "Gemma3nTextMLP"


@dataclass
class ElasticWidthController:
    """Shared, mutable width selector for every elastic MLP in a model.

    `fracs` is sorted ascending and its last entry MUST be 1.0 (the full
    model). A single assignment to `current` switches the whole model's width
    because every MLP holds this same controller by reference.
    """

    fracs: list[float]
    current: float = field(init=False)

    def __post_init__(self) -> None:
        if not self.fracs:
            raise ValueError("fracs must be non-empty")
        self.fracs = sorted(set(float(f) for f in self.fracs))
        if self.fracs[-1] != 1.0:
            raise ValueError(f"largest frac must be 1.0 (full model), got {self.fracs}")
        if self.fracs[0] <= 0.0:
            raise ValueError(f"fracs must be > 0, got {self.fracs}")
        self.current = 1.0

    def set(self, frac: float) -> None:
        self.current = float(frac)

    def set_largest(self) -> None:
        self.current = 1.0

    def sample(self, rng) -> float:
        """Pick a width fraction (uniform over `fracs`) and select it.

        `rng` is a random.Random / np.random.Generator-like with .choice or a
        plain `random` module; we only need an index draw, so accept anything
        exposing `randrange`-equivalent via `int(rng.random()*n)` to stay
        dependency-light and deterministic with the caller's seeded stream.
        """
        n = len(self.fracs)
        idx = int(rng.random() * n)
        if idx >= n:
            idx = n - 1
        self.current = self.fracs[idx]
        return self.current


def _elastic_mlp_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """Bound replacement for Gemma3nTextMLP.forward with prefix-slice nesting."""
    frac = self._elastic_ctrl.current
    full = self.intermediate_size
    d = full if frac >= 1.0 else max(1, int(round(frac * full)))
    if d == full:
        # Fast path == stock gemma-3n forward (bit-identical at full width).
        gate_proj = self.gate_proj(hidden_states)
        if self.activation_sparsity > 0.0:
            gate_proj = self._gaussian_topk(gate_proj)
        activations = self.act_fn(gate_proj)
        up_proj = self.up_proj(hidden_states)
        return self.down_proj(activations * up_proj)
    # Sliced (sub-model) path.
    gate_proj = F.linear(hidden_states, self.gate_proj.weight[:d, :])
    if self.activation_sparsity > 0.0:
        gate_proj = self._gaussian_topk(gate_proj)   # mean/std over active d
    activations = self.act_fn(gate_proj)
    up_proj = F.linear(hidden_states, self.up_proj.weight[:d, :])
    return F.linear(activations * up_proj, self.down_proj.weight[:, :d])


def install_elastic_mlps(model: nn.Module, controller: ElasticWidthController) -> int:
    """Monkeypatch every Gemma3nTextMLP in `model` to the elastic forward and
    attach the shared controller. Returns the number of MLPs wrapped.

    Idempotent: re-installing on an already-elastic model just refreshes the
    controller reference.
    """
    n = 0
    for module in model.modules():
        if type(module).__name__ == _GEMMA3N_MLP_CLSNAME:
            module._elastic_ctrl = controller
            # Bind the function as this instance's forward (shadows the class
            # method for this object only).
            module.forward = _elastic_mlp_forward.__get__(module, type(module))
            n += 1
    if n == 0:
        raise RuntimeError(
            f"install_elastic_mlps: found no {_GEMMA3N_MLP_CLSNAME} modules — "
            "is this a gemma-3n student?"
        )
    return n


def build_controller(cfg: dict) -> ElasticWidthController | None:
    """Construct an ElasticWidthController from a config dict, or None if
    elastic distillation is disabled.

    Config keys:
      matformer_distill : bool   (default False) — master switch
      matformer_fracs   : list[float] (default [0.125, 0.25, 0.5, 1.0]) — widths to train
    """
    if not cfg.get("matformer_distill", False):
        return None
    fracs = cfg.get("matformer_fracs", [0.125, 0.25, 0.5, 1.0])
    return ElasticWidthController(fracs=list(fracs))
