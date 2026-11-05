#!/usr/bin/env python3
"""Does the exported narrow model behave like the runtime-sliced one, and what does it cost?

WHY THIS EXISTS. Exporting a prefix slice is only useful if it is EQUIVALENT to the runtime
slicing every published width number was produced with. If it is not, the exported model is a
different model and none of those numbers transfer to it. This script answers that directly
instead of assuming it, and in the same pass measures the thing the export was built for:
the deployment footprint, which runtime slicing does not reduce at all.

WHAT IT COMPARES. On the SAME real evaluation rows, the same prompts and the same images:

  A  parent checkpoint + NestedGemma3MLP wrapper, controller.set(width)   <- what run_closed does
  B  exported checkpoint, loaded as a plain Gemma3, no MatFormer code

Two different questions are reported separately because they have different tolerances.
Logit max-abs-difference is expected to be small but NOT necessarily zero: A multiplies by a
strided view (`down_proj.weight[:, :d]`) while B multiplies by a contiguous tensor, and cuBLAS
may select a different kernel for each. What must hold exactly is ARGMAX AGREEMENT, because
that is what every closed-set eval number is computed from. Agreement below 100% means the
export is not a drop-in replacement and the published width numbers do NOT transfer.

WHAT IT MEASURES. Weights on disk, resident VRAM after load (torch allocator and NVML device
total, which differ and are both reported), and single-stream forward latency.

Usage:
  python scripts/export/verify_exported_width.py --parent <dir> --exported <dir> \
      --width 1280 [--pillar skin --task U2 --n 24] [--out report.json]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from cancerflex.paths import ROOT  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from run_closed import score_options_build  # noqa: E402 — the canonical input builder
BASE_PROCESSOR = os.environ.get("CANCERFLEX_BASE_MODEL", "google/medgemma-1.5-4b-it")


def nvml_used_mib() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
        return int(out[0])
    except Exception:
        return None


def dir_weight_bytes(d: Path) -> int:
    return sum(f.stat().st_size for f in d.glob("*.safetensors"))


def load_rows(pillar: str, task: str, n: int):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
    from evaluation.eval_iterator import iter_eval_rows
    rows = []
    for row, img, prompt, gold, meta in iter_eval_rows(pillar, task=task, split="test_id", limit=n):
        rows.append((img, prompt, ["Yes", "No"]))
    return rows


@torch.no_grad()
def run_model(model, processor, rows, device, warmup=2):
    """Last-position logits per row, plus median single-row forward latency.

    Inputs are built by run_closed's OWN `score_options_build`, not by a local
    apply_chat_template call. That function documents why: the template must be applied to a
    LIST of conversations or Gemma3's pan-and-scan crop-to-token alignment comes out wrong.
    Reusing it means this compares the models on exactly the tensors the evaluation feeds them.
    """
    logits, lat = [], []
    for i, rec in enumerate(rows):
        inp = score_options_build(processor, [rec])
        inp = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inp.items()}
        torch.cuda.synchronize(); t0 = time.time()
        out = model(**inp)
        torch.cuda.synchronize()
        if i >= warmup:                       # first passes pay kernel autotune, not model cost
            lat.append((time.time() - t0) * 1000.0)
        logits.append(out.logits[0, -1, :].float().cpu())
    return torch.stack(logits), (sorted(lat)[len(lat) // 2] if lat else float("nan"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", required=True)
    ap.add_argument("--exported", required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--pillar", default="skin")
    ap.add_argument("--task", default="U2")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = "cuda"
    parent, exported = Path(args.parent), Path(args.exported)
    processor = AutoProcessor.from_pretrained(BASE_PROCESSOR)
    rows = load_rows(args.pillar, args.task, args.n)
    print(f"[verify] {len(rows)} real rows from {args.pillar} {args.task} test_id")
    rep = {"parent": str(parent), "exported": str(exported), "width": args.width,
           "pillar": args.pillar, "task": args.task, "n_rows": len(rows)}

    # ── A: parent + runtime slice (exactly the run_closed path) ─────────────────
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    m = AutoModelForImageTextToText.from_pretrained(str(parent), torch_dtype=torch.bfloat16,
                                                    attn_implementation="eager")
    from cancerflex.matformer.model import wrap_medgemma_mlps
    from cancerflex.matformer.granularity import GranularityController
    gran = parent / "matformer_granularities.json"
    if not gran.exists():
        gran = parent.parent / "matformer_granularities.json"
    ctrl = GranularityController(widths=[int(w) for w in json.loads(gran.read_text())["widths"]])
    n_wrapped = wrap_medgemma_mlps(m, ctrl)
    assert n_wrapped > 0, "wrap replaced 0 MLPs"
    ctrl.set(args.width)
    m = m.to(device).eval()
    rep["A_runtime_sliced"] = {
        "n_mlps_wrapped": n_wrapped,
        "weights_on_disk_gib": dir_weight_bytes(parent) / 2**30,
        "torch_allocated_gib": torch.cuda.memory_allocated() / 2**30,
        "nvml_device_mib": nvml_used_mib(),
        "n_params": sum(p.numel() for p in m.parameters()),
    }
    logits_a, lat_a = run_model(m, processor, rows, device)
    rep["A_runtime_sliced"]["median_forward_ms"] = lat_a
    rep["A_runtime_sliced"]["peak_torch_gib"] = torch.cuda.max_memory_allocated() / 2**30
    del m; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

    # ── B: exported, plain Gemma3, no MatFormer in the path ────────────────────
    m = AutoModelForImageTextToText.from_pretrained(str(exported), torch_dtype=torch.bfloat16,
                                                    attn_implementation="eager").to(device).eval()
    assert not any(type(mod).__name__ == "NestedGemma3MLP" for mod in m.modules()), \
        "exported model still contains NestedGemma3MLP — it is not standalone"
    rep["B_exported"] = {
        "weights_on_disk_gib": dir_weight_bytes(exported) / 2**30,
        "torch_allocated_gib": torch.cuda.memory_allocated() / 2**30,
        "nvml_device_mib": nvml_used_mib(),
        "n_params": sum(p.numel() for p in m.parameters()),
    }
    logits_b, lat_b = run_model(m, processor, rows, device)
    rep["B_exported"]["median_forward_ms"] = lat_b
    rep["B_exported"]["peak_torch_gib"] = torch.cuda.max_memory_allocated() / 2**30
    del m; torch.cuda.empty_cache()

    # ── Equivalence ────────────────────────────────────────────────────────────
    diff = (logits_a - logits_b).abs()
    agree = (logits_a.argmax(-1) == logits_b.argmax(-1)).float().mean().item()
    rep["equivalence"] = {
        "argmax_agreement": agree,
        "logit_max_abs_diff": diff.max().item(),
        "logit_mean_abs_diff": diff.mean().item(),
        "logit_a_scale_mean_abs": logits_a.abs().mean().item(),
    }
    a = rep["A_runtime_sliced"]; b = rep["B_exported"]
    rep["savings"] = {
        "disk_gib_A": a["weights_on_disk_gib"], "disk_gib_B": b["weights_on_disk_gib"],
        "disk_reduction_x": a["weights_on_disk_gib"] / b["weights_on_disk_gib"],
        "vram_torch_reduction_x": a["torch_allocated_gib"] / b["torch_allocated_gib"],
        "param_reduction_x": a["n_params"] / b["n_params"],
        "latency_ratio_B_over_A": b["median_forward_ms"] / a["median_forward_ms"],
    }

    print(json.dumps(rep, indent=2))
    print()
    print(f"argmax agreement      {agree*100:.2f}%   (must be 100% for the published width "
          f"numbers to transfer)")
    print(f"logit max abs diff    {diff.max().item():.4e}   on logits of mean abs "
          f"{logits_a.abs().mean().item():.3f}")
    print(f"weights on disk       {a['weights_on_disk_gib']:.2f} -> {b['weights_on_disk_gib']:.2f} GiB "
          f"({rep['savings']['disk_reduction_x']:.2f}x)")
    print(f"VRAM (torch alloc)    {a['torch_allocated_gib']:.2f} -> {b['torch_allocated_gib']:.2f} GiB "
          f"({rep['savings']['vram_torch_reduction_x']:.2f}x)")
    print(f"median forward        {a['median_forward_ms']:.1f} -> {b['median_forward_ms']:.1f} ms")

    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=2))
        print(f"\nwrote {args.out}")
    return 0 if agree == 1.0 else 2


if __name__ == "__main__":
    sys.exit(main())
