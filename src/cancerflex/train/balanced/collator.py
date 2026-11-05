"""BalancedCollator — VLMSFTCollator + a per-row `cell_id` tensor.

The parent returns only processor tensors (input_ids/labels/pixel_values); it
drops rows whose image fails to load. compute_loss needs each surviving row's
(task,pillar) cell for the A4 reweight, so we attach `cell_id` aligned with the
rows the parent KEPT. Fast path (no drops) costs nothing; the rare drop path
re-checks loadability (cache-cheap) to preserve alignment. `cell_id` is popped
in BalancedSFTTrainer.compute_loss before the model forward.
"""
from __future__ import annotations

from pathlib import Path

import torch

from cancerflex.train.collator import VLMSFTCollator
from .weights import cell_key

from cancerflex.paths import ROOT  # noqa: E402  (set CANCERFLEX_ROOT)


class BalancedCollator(VLMSFTCollator):
    def __init__(self, processor, cell_index: dict[str, int], **kw) -> None:
        super().__init__(processor, **kw)
        self.cell_index = cell_index

    def _row_cell(self, rec: dict) -> int:
        return self.cell_index.get(cell_key(rec.get("task", "?"), rec.get("pillar", "?")), -1)

    def _loadable(self, rec: dict) -> bool:
        try:
            import canonical_image  # collator already put core_code on sys.path
            canonical_image.load_cached(rec["image_path"], root=ROOT)
            return True
        except Exception:
            return False

    def __call__(self, batch):
        out = super().__call__(batch)
        b = int(out["input_ids"].shape[0])
        if b == 0:
            return out
        if b == len(batch):                       # fast path: nothing dropped
            cells = [self._row_cell(r) for r in batch]
        else:                                     # rare: realign to kept rows
            cells = [self._row_cell(r) for r in batch if self._loadable(r)][:b]
            if len(cells) < b:                    # safety: never misalign sizes
                cells += [-1] * (b - len(cells))
        out["cell_id"] = torch.tensor(cells, dtype=torch.long)
        return out
