"""BalancedSFTTrainer — SFTTrainer with seq-len-normalized loss + deferred reweight.

Two changes vs the default SFT loss (which sums token CE and divides by total
tokens in the batch → long box-JSON answers dominate short closed answers):

  1. seq-len-norm (shared default-ON): per-row loss = MEAN CE over that row's
     answer tokens, then averaged across rows → every row weighs equally
     regardless of answer length (NODE C; proxy_protocol §2).
  2. deferred per-cell reweight (A4 half of A2+A4): multiply each row's loss by
     its cell weight, but only AFTER `defer_step` (DRW, Cao NeurIPS'19). Before
     that, weight = 1. `reweight_enabled=False` (the A2 arm) skips it entirely.

The sampler half of A2 is realized upstream as a physical resample (weights.py),
so this trainer does NOT override `_get_train_sampler` → DDP sharding is the
stock, proven path.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from trl import SFTTrainer


class BalancedSFTTrainer(SFTTrainer):
    def __init__(self, *args, cell_weight=None, defer_step: int = 0,
                 reweight_enabled: bool = False, **kw) -> None:
        super().__init__(*args, **kw)
        # per-cell-index weight table (mean≈1 over the resampled mix), or None
        self.cell_weight = (None if cell_weight is None
                            else torch.as_tensor(cell_weight, dtype=torch.float32))
        self.defer_step = int(defer_step)
        self.reweight_enabled = bool(reweight_enabled)
        self._reweight_logged = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        cell_id = inputs.pop("cell_id", None)        # pop BEFORE forward (not a model kwarg)
        # Pop labels too: we compute CE ourselves from logits, so the model must NOT also
        # run its internal causal-LM loss (it would materialize the shifted-logits tensor a
        # second time — wasteful under grad-checkpointing). [codex review #1]
        labels = inputs.pop("labels", None)
        if labels is None or labels.numel() == 0:
            # Unreachable in practice: the canonical image cache means loads don't fail, and
            # Phase-1 ran 22593 steps on this collator without an empty batch. Fail FAST rather
            # than return a graph-disconnected zero, which would desync DDP all-reduce. [codex #3]
            raise RuntimeError("BalancedSFTTrainer: empty batch (all images failed to load)")

        outputs = model(**inputs)
        logits = outputs.logits                      # [B, T, V]
        V = logits.size(-1)
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        per_tok = F.cross_entropy(
            shift_logits.reshape(-1, V), shift_labels.reshape(-1),
            ignore_index=-100, reduction="none",
        ).view(shift_labels.shape)                   # [B, T-1]
        mask = (shift_labels != -100).to(per_tok.dtype)
        row_tok = mask.sum(dim=1).clamp(min=1.0)
        row_loss = (per_tok * mask).sum(dim=1) / row_tok        # seq-len-norm: per-row mean CE

        w = torch.ones_like(row_loss)
        if (self.reweight_enabled and self.cell_weight is not None
                and cell_id is not None and self.state.global_step >= self.defer_step):
            cw = self.cell_weight.to(row_loss.device)
            w = torch.where(cell_id >= 0, cw[cell_id.clamp(min=0)], torch.ones_like(row_loss))
            if not self._reweight_logged:
                print(f"[BalancedSFTTrainer] DRW reweight ACTIVE at step "
                      f"{self.state.global_step} (defer_step={self.defer_step})", flush=True)
                self._reweight_logged = True

        # Divide by the FIXED per-rank row count (.mean()), NOT the variable per-rank w.sum():
        # cell_weight is pre-normalized to mean≈1 over the corpus, so (row_loss*w).mean() is an
        # unbiased weighted mean whose denominator is identical across DDP ranks → DDP gradient
        # averaging recovers the exact global weighted mean (no all-reduce needed). [codex #2]
        loss = (row_loss * w).mean()
        return (loss, outputs) if return_outputs else loss
