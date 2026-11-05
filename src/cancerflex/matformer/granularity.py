"""Granularity controller for nested-width Matformer FFNs.

A single controller object is shared across every NestedGemma3MLP in a model.
Training loops mutate `controller.current_width` between submodel forwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class GranularityController:
    """Holds the canonical list of nested intermediate-dim widths and the
    currently-selected width. Shared by reference across all NestedGemma3MLP
    modules so a single assignment switches the whole model.

    `widths` is sorted ascending. The last entry MUST equal the base
    `intermediate_size` of the underlying Gemma3 model — this is what makes
    the largest granularity bit-identical to the base model.
    """

    widths: List[int]
    current_width: int = field(init=False)

    def __post_init__(self) -> None:
        if not self.widths:
            raise ValueError("widths must be non-empty")
        if sorted(self.widths) != list(self.widths):
            raise ValueError(f"widths must be sorted ascending, got {self.widths}")
        if len(set(self.widths)) != len(self.widths):
            raise ValueError(f"widths must be unique, got {self.widths}")
        self.current_width = self.widths[-1]

    @classmethod
    def from_factor(cls, full_width: int, num_granularities: int) -> "GranularityController":
        """Exponential schedule: widths = full_width * {1/K, 2/K, ..., 1}.

        With K=4 and full_width=10240 this gives {2560, 5120, 7680, 10240}.
        With K=4 and the "factor-8" convention from the Matformer paper, use
        `from_exponential_factor(full_width, factor=8)` instead.
        """
        if num_granularities < 1:
            raise ValueError("num_granularities >= 1")
        widths = [full_width * i // num_granularities for i in range(1, num_granularities + 1)]
        return cls(widths=widths)

    @classmethod
    def from_exponential_factor(cls, full_width: int, factor: int = 8) -> "GranularityController":
        """Paper-style exponential schedule. `factor=8` => widths {D/8, D/4, D/2, D}.

        Matches devvrit/matformer + RAIVNLab/MatFormer-OLMo conventions.
        """
        if factor < 2 or (factor & (factor - 1)) != 0:
            raise ValueError(f"factor must be a power of 2, got {factor}")
        if full_width % factor != 0:
            raise ValueError(f"full_width {full_width} not divisible by factor {factor}")
        smallest = full_width // factor
        widths = []
        d = full_width
        while d >= smallest:
            widths.append(d)
            d //= 2
        widths = sorted(set(widths))
        return cls(widths=widths)

    def set(self, width: int) -> None:
        # A sub-width forward is a pure PREFIX slice (`nested_mlp.py`: `gate_proj.weight[:d, :]`,
        # `down_proj.weight[:, :d]`), so ANY width in (0, max] is structurally valid — the canonical
        # `widths` list records which widths were TRAINED, not which are representable. Restricting
        # `set()` to the trained list therefore blocked a legitimate measurement: evaluating a uniform
        # slice at an untrained width, which is the only way to tell whether the accuracy-vs-width
        # frontier is dense or whether untrained widths fall off it.
        # Out-of-range is still a hard error. A non-canonical width is allowed but warned about ONCE,
        # so a typo in a training config cannot pass silently.
        if not isinstance(width, int) or width <= 0 or width > self.widths[-1]:
            raise ValueError(f"width {width} out of range (0, {self.widths[-1]}]")
        if width not in self.widths and not getattr(self, "_warned_noncanonical", False):
            import warnings
            warnings.warn(
                f"GranularityController: width {width} is NOT one of the trained widths {self.widths}. "
                "This is a valid prefix slice but the neurons in it were only ever trained at wider "
                "settings, so treat the result as a measurement of an UNTRAINED width.",
                stacklevel=2)
            self._warned_noncanonical = True
        self.current_width = width

    def set_largest(self) -> None:
        self.current_width = self.widths[-1]

    def __repr__(self) -> str:
        return f"GranularityController(widths={self.widths}, current={self.current_width})"
