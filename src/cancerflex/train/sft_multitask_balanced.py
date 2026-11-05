"""Phase-2 balanced multi-task SFT worker (torchrun target, like train_v2/sft_main.py).

Trains ONE imbalance-sampling arm on the de-leaked U0-U6 corpus. Mirrors
sft_main's model/data/TRL wiring; the only deltas are (a) arm balancing
(two-level resample + optional deferred per-cell reweight), (b) seq-len-norm
loss via BalancedSFTTrainer, (c) max_steps support for the short proxy runs.

Arms (config `arm`):
  a2     : two-level resample (sampler_tau, default 0.5) + seq-len-norm.
  a2a4   : mild two-level resample (sampler_tau, default 0.75) + seq-len-norm
           + DEFERRED inverse-sqrt per-cell reweight (defer_frac of max_steps).

Launch (4-GPU DDP):
  torchrun --standalone --nproc_per_node=4 \
    -m cancerflex.train.sft_multitask_balanced \
    --config .../configs/multitask_a2a4.yaml [--resume]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

from cancerflex.paths import ROOT  # noqa: E402  (set CANCERFLEX_ROOT)
sys.path.insert(0, str(ROOT))


def _validate_corpus(corpus_dir: str, split: str = "train") -> None:
    """Hard pre-flight (SPEC): no LM row may have answer=None or pixel_mask."""
    base = Path(corpus_dir) if Path(corpus_dir).is_absolute() else ROOT / corpus_dir
    if (base / f"{split}.jsonl").exists():
        raise SystemExit(f"[phase2] root {base/f'{split}.jsonl'} would shadow shards.")
    files = sorted((base / "shards" / split).rglob("*.jsonl"))
    if not files:
        raise SystemExit(f"[phase2] no shards under {base/'shards'/split}")


def _resolve_resume(output_dir: Path, want: bool):
    if not want or not output_dir.exists():
        return None
    return True if any(p.name.startswith("checkpoint-") for p in output_dir.iterdir()) else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    arm = cfg.get("arm", "a2")
    corpus = cfg.get("corpus_dir", "Experiment/datasets/v1_final_deleaked")
    _validate_corpus(corpus, "train")

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    os.environ["WANDB_MODE"] = "offline"
    os.environ.setdefault("WANDB_PROJECT", cfg.get("wandb_project", "cancerflex"))
    os.environ.setdefault("WANDB_DIR", str(output_dir))

    import numpy as np
    from trl import SFTConfig
    from cancerflex.train import dataset as _ds
    from cancerflex.train import load_model as _lm
    from cancerflex.train.balanced import weights as _w
    from cancerflex.train.balanced.collator import BalancedCollator
    from cancerflex.train.balanced.trainer import BalancedSFTTrainer

    # ── model ─────────────────────────────────────────────────────────────
    # Load the model: a prior fine-tuned ckpt (DRW continuation) OR the medgemma base.
    # MUST use from_pretrained (NOT load_state_dict overlay) — save_pretrained writes
    # canonical keys that are remapped vs the in-memory module names, so a raw
    # load_state_dict matched 0/883 tensors and silently kept BASE weights.
    init_from = cfg.get("init_from")
    if init_from:
        import torch as _torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from cancerflex.train.load_model import (
            MEDGEMMA_BASE as _BASE, _find_vision_tower)
        p = Path(init_from) if Path(init_from).is_absolute() else ROOT / init_from
        if not (p / "config.json").exists():
            raise SystemExit(f"[phase2] init_from missing config.json: {p}")
        processor = AutoProcessor.from_pretrained(_BASE)        # base processor (A2 didn't change it)
        model = AutoModelForImageTextToText.from_pretrained(
            str(p), torch_dtype=_torch.bfloat16,
            attn_implementation=cfg.get("attn_impl", "eager"))
        if cfg.get("vision_freeze", True):
            vt = _find_vision_tower(model)
            if vt is not None:
                for prm in vt.parameters():
                    prm.requires_grad = False
        print(f"[phase2] init_from {p}: loaded fine-tuned model via from_pretrained "
              f"(vision_freeze={cfg.get('vision_freeze', True)})", flush=True)
    else:
        model, processor = _lm.load_medgemma(
            vision_freeze=cfg.get("vision_freeze", True),
            attn_impl=cfg.get("attn_impl", "eager"),
        )
    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()

    # ── data: natural → cell stats → arm resample ─────────────────────────
    seed = int(cfg.get("seed", 0))
    natural = _ds.load_v1_split(
        "train",
        corpus_dir=corpus,
        limit=cfg.get("train_subsample_n"),
        u6_image_path_allowlist=cfg.get("u6_image_path_allowlist"),
    )
    cell_index = _w.build_cell_index(natural)
    natural_counts = _w.cell_counts(natural, cell_index)

    sampler_tau = float(cfg.get("sampler_tau", 0.5 if arm == "a2" else 0.75))
    train_ds = _w.two_level_balance(natural, tau=sampler_tau, seed=seed)
    print(f"[phase2] arm={arm} sampler_tau={sampler_tau} natural={len(natural):,} "
          f"resampled={len(train_ds):,} cells={len(cell_index)}", flush=True)

    # Optional: oversample a freshly-added data slice for a warm-start continue-train
    # (must run BEFORE the A4 reweight table so cell_ids_for sees the final mix).
    over_ds = cfg.get("oversample_datasets")
    if over_ds:
        n0 = len(train_ds)
        train_ds = _w.oversample_datasets(
            train_ds, over_ds, float(cfg.get("oversample_frac", 0.4)), seed=seed)
        print(f"[phase2] oversample {over_ds} -> frac={cfg.get('oversample_frac', 0.4)}: "
              f"{n0:,} -> {len(train_ds):,} rows", flush=True)

    # ── A4 deferred reweight table (a2a4 only) ────────────────────────────
    max_steps = int(cfg.get("max_steps", -1))
    reweight_enabled = (cfg.get("reweight", "none") not in (None, "none", "off"))
    cell_weight = None
    defer_step = 0
    if reweight_enabled:
        kind = cfg.get("reweight", "inv_sqrt")
        raw = (_w.inverse_sqrt_weights(natural_counts) if kind == "inv_sqrt"
               else _w.effective_number_weights(natural_counts, float(cfg.get("reweight_beta", 0.99999))))
        resampled_cell_ids = _w.cell_ids_for(train_ds, cell_index)
        cell_weight = _w.normalize_reweight(raw, resampled_cell_ids, cap=cfg.get("reweight_cap", 8.0))
        defer_step = int(round(float(cfg.get("defer_frac", 0.5)) * max(max_steps, 1)))
        order = np.argsort(natural_counts)
        inv = {v: k for k, v in cell_index.items()}
        print(f"[phase2] reweight={kind} defer_step={defer_step} cap={cfg.get('reweight_cap',8.0)}", flush=True)
        print("[phase2]   smallest cells w:", {inv[i]: round(float(cell_weight[i]), 2) for i in order[:4]}, flush=True)
        print("[phase2]   largest cells  w:", {inv[i]: round(float(cell_weight[i]), 2) for i in order[-4:]}, flush=True)

    # ── TRL config (mirror sft_main defaults) ─────────────────────────────
    sft_cfg = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=float(cfg.get("num_train_epochs", 1)),
        max_steps=max_steps,
        save_strategy=cfg.get("save_strategy", "steps"),
        save_steps=int(cfg.get("save_steps", 1000)),
        save_total_limit=int(cfg.get("save_total_limit", 5)),
        save_only_model=bool(cfg.get("save_only_model", False)),   # weights only, no optimizer state (default off = unchanged)
        bf16=True,
        per_device_train_batch_size=int(cfg.get("batch_size", 16)),
        gradient_accumulation_steps=int(cfg.get("grad_accum", 1)),
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=cfg.get("optim", "adamw_torch_fused"),
        learning_rate=float(cfg.get("lr", 2e-5)),
        warmup_ratio=float(cfg.get("warmup_ratio", 0.03)),
        lr_scheduler_type=cfg.get("lr_scheduler", "cosine"),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        max_grad_norm=float(cfg.get("max_grad_norm", 1.0)),
        logging_steps=int(cfg.get("logging_steps", 50)),
        report_to=["wandb"] if cfg.get("use_wandb", True) else [],
        run_name=cfg.get("run_name", output_dir.name),
        dataloader_num_workers=int(cfg.get("dataloader_num_workers", 8)),
        dataloader_prefetch_factor=int(cfg.get("dataloader_prefetch_factor", 4)),
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        eval_strategy="no",
        seed=seed,
        data_seed=seed,
        ddp_find_unused_parameters=bool(cfg.get("ddp_find_unused_parameters", False)),
        push_to_hub=False,
        disable_tqdm=False,
    )

    # resolution augmentation: OFF unless resolution_aug_prob > 0 (every existing config unchanged);
    # resolution_aug_pillars restricts it to those pillars (train_v2/collator.py). Added 2026-09-26.
    collator = BalancedCollator(processor, cell_index, max_length=cfg.get("max_seq_length", 1024),
                                resolution_aug_prob=float(cfg.get("resolution_aug_prob", 0.0)),
                                resolution_aug_min_scale=float(cfg.get("resolution_aug_min_scale", 0.143)),
                                resolution_aug_seed=int(cfg.get("seed", 0)),
                                resolution_aug_pillars=cfg.get("resolution_aug_pillars"))
    trainer = BalancedSFTTrainer(
        model=model, args=sft_cfg, train_dataset=train_ds, data_collator=collator,
        processing_class=processor.tokenizer,
        cell_weight=cell_weight, defer_step=defer_step, reweight_enabled=reweight_enabled,
    )

    resume = _resolve_resume(output_dir, args.resume)
    print(f"[phase2] resume={resume} max_steps={max_steps}", flush=True)
    result = trainer.train(resume_from_checkpoint=resume)

    trainer.save_model(str(output_dir / "final"))
    (output_dir / "train_metrics.json").write_text(json.dumps(result.metrics, indent=2, default=str))
    print(f"[phase2] DONE arm={arm} -> {output_dir/'final'}  metrics={result.metrics}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
