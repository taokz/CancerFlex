#!/usr/bin/env python3
"""Materialise a nested-width MatFormer/elastic checkpoint into a STANDALONE narrow model.

WHY THIS EXISTS. Every width result we have so far was produced by RUNTIME slicing:
`run_closed.py` wraps each `Gemma3MLP` in a `NestedGemma3MLP` that keeps the FULL-width
weights resident and takes a prefix slice inside `forward`. That genuinely reduces FLOPs,
but it does NOT reduce the memory footprint by a single byte, because the discarded columns
are still allocated. A width claim about DEPLOYMENT SIZE therefore has no evidence behind it
unless the narrow model is actually written out. This script writes it out.

WHAT IT DOES. For every text-decoder MLP it keeps the first `width` intermediate units:

    gate_proj.weight[:width, :]      up_proj.weight[:width, :]      down_proj.weight[:, :width]

and rewrites `text_config.intermediate_size`, so the result loads as an ORDINARY Gemma3
checkpoint with no MatFormer code in the path.

WHY A PREFIX IS THE RIGHT SLICE. It is only correct because the parent was trained nested:
importance permutation reorders each layer's intermediate units by h = act(gate) * up so that
any prefix is the most-important subset, and the elastic stage then trains those prefixes as
real subnetworks. Applying this to a checkpoint that was never nested-trained produces a model
that loads and generates garbage, so the granularity schedule is REQUIRED and `--width` must
be one of its trained widths.

WHAT IS DELIBERATELY NOT TOUCHED. The SigLIP vision tower's MLPs are `fc1`/`fc2`; they were
never wrapped, were never width-sampled, and are frozen. Only keys under `language_model.`
containing `.mlp.` are sliced. Attention, embeddings, norms and the multimodal projector are
copied verbatim.

Usage:
  python scripts/export/export_matformer_width.py --ckpt <dir> --width 1280 --out <dir>
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# Files a standalone checkpoint needs that are not weights. Missing ones are skipped:
# these checkpoints inherit the processor from the base model at eval time.
SIDECAR = ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
           "generation_config.json", "special_tokens_map.json",
           "preprocessor_config.json", "processor_config.json"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-untrained-width", action="store_true",
                    help="skip the granularity-schedule check (produces an UNVALIDATED model)")
    args = ap.parse_args()

    ckpt, out = Path(args.ckpt), Path(args.out)
    d = args.width

    # ── The nested-training gate. A prefix slice is meaningless without it. ──────
    gran = ckpt / "matformer_granularities.json"
    if not gran.exists():
        gran = ckpt.parent / "matformer_granularities.json"
    if gran.exists():
        widths = [int(w) for w in json.loads(gran.read_text())["widths"]]
        if d not in widths and not args.allow_untrained_width:
            print(f"FATAL: width {d} is not a trained width {widths} (from {gran})", file=sys.stderr)
            return 1
        print(f"[export] granularity schedule {widths}  (from {gran})")
    elif not args.allow_untrained_width:
        print(f"FATAL: no matformer_granularities.json near {ckpt}; this checkpoint was not "
              f"nested-trained, so a prefix slice is not a subnetwork. Use "
              f"--allow-untrained-width only to build a deliberate negative control.", file=sys.stderr)
        return 1

    src = ckpt / "model.safetensors"
    if not src.exists():
        print(f"FATAL: {src} not found", file=sys.stderr)
        return 1

    out.mkdir(parents=True, exist_ok=True)
    tensors, sliced, copied, kept_full = {}, 0, 0, []
    with safe_open(src, framework="pt") as h:
        meta = h.metadata() or {}
        for k in h.keys():
            t = h.get_tensor(k)
            # ONLY the text decoder's SwiGLU. Vision MLPs are fc1/fc2 and never match.
            if k.startswith("language_model.") and ".mlp." in k:
                if k.endswith("gate_proj.weight") or k.endswith("up_proj.weight"):
                    t = t[:d, :].contiguous(); sliced += 1
                elif k.endswith("down_proj.weight"):
                    t = t[:, :d].contiguous(); sliced += 1
                else:
                    kept_full.append(k)
            tensors[k] = t
            copied += 1

    if sliced != 3 * 34 and not args.allow_untrained_width:
        print(f"WARNING: sliced {sliced} MLP tensors, expected 3 per decoder layer", file=sys.stderr)
    if kept_full:
        print(f"[export] NOTE: {len(kept_full)} language-model .mlp. tensors did not match a known "
              f"projection name and were copied whole: {kept_full[:4]}")

    save_file(tensors, str(out / "model.safetensors"), metadata={**meta, "format": "pt"})

    # ── Rewrite the width so the result is an ordinary Gemma3 checkpoint ─────────
    cfg = json.loads((ckpt / "config.json").read_text())
    tcfg = cfg["text_config"] if "text_config" in cfg else cfg
    old_w = tcfg.get("intermediate_size")
    tcfg["intermediate_size"] = d
    cfg["_exported_from"] = str(ckpt)
    cfg["_exported_width"] = d
    cfg["_exported_note"] = (
        "Materialised prefix slice of a nested-width MatFormer checkpoint. Text-decoder FFN "
        f"intermediate_size {old_w} -> {d}; vision tower, attention and embeddings unchanged. "
        "Loads as a standard Gemma3 model with no MatFormer code in the path.")
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    for name in SIDECAR:
        s = ckpt / name
        if s.exists():
            shutil.copy2(s, out / name)

    n_out = sum(t.numel() for t in tensors.values())
    print(f"[export] {ckpt}  width {old_w} -> {d}")
    print(f"[export] tensors {copied} total, {sliced} sliced")
    print(f"[export] params {n_out:,}   weights {(out/'model.safetensors').stat().st_size/2**30:.2f} GiB")
    print(f"[export] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
