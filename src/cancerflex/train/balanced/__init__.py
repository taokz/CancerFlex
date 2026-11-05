"""Phase-2 imbalance-sampling arms (NODE C primitives) for the multi-task SFT.

Independent, opt-in modules layered on the validated train_v2 SFT stack — they
do NOT modify `train_v2/dataset.py`, `collator.py`, or `sft_main.py`:

  - weights.py  : two-level (task x pillar) physical resample (A2 sampler,
                  realized the same DDP-safe way as Phase-1's pillar-temp) +
                  per-cell loss-reweight tables (A4: inverse-sqrt / effective-number).
  - collator.py : BalancedCollator(VLMSFTCollator) — appends per-row `cell_id`.
  - trainer.py  : BalancedSFTTrainer(SFTTrainer) — seq-len-normalized loss
                  (shared default-on) + deferred per-cell reweight (DRW).

Arms (see docs/methodology/phase2_proxy_protocol.md):
  A2     = two-level resample (tau=0.5) + seq-len-norm.
  A2+A4  = mild two-level resample (tau=0.75) + seq-len-norm + DEFERRED inverse-sqrt
           reweight (off until defer_step, then on) — DRW (Cao NeurIPS'19).
"""
