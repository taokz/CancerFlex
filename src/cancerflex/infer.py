"""Single inference entry point for a CancerFlex checkpoint.

Loads the checkpoint (never bundled with this code), wraps the language decoder's feed-forward
blocks so a nested width can be selected, and answers one prompt about one image.

Example (synthetic input, no real data):

    cancerflex-infer --checkpoint /path/to/weights --synthetic --width 1280

Example (real image):

    cancerflex-infer --checkpoint /path/to/weights --image scan.png \
        --prompt "Is a lesion visible in this image? Answer Yes or No."
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from cancerflex.matformer.granularity import GranularityController
from cancerflex.matformer.model import wrap_medgemma_mlps
from cancerflex.synthetic import DEFAULT_PROMPT, make_synthetic_image

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def load_widths(checkpoint: Path, full_width: int) -> list[int]:
    """Read the trained nested widths from ``matformer_granularities.json`` or fall back to {D/8, D/4, D/2, D}."""
    for cand in (checkpoint / "matformer_granularities.json", checkpoint.parent / "matformer_granularities.json"):
        if cand.exists():
            data = json.loads(cand.read_text())
            widths = data["widths"] if isinstance(data, dict) else data
            return sorted(int(w) for w in widths)
    return GranularityController.from_exponential_factor(full_width, factor=8).widths


def load_model(checkpoint: str | Path, device: str = "cpu", dtype: str = "fp32", attn: str = "eager"):
    """Load processor + model from ``checkpoint`` and wrap its MLPs. Returns (model, processor, controller)."""
    ckpt = Path(checkpoint)
    has_processor = any((ckpt / f).exists() for f in
                        ("preprocessor_config.json", "processor_config.json", "tokenizer_config.json"))
    processor_src = ckpt if has_processor else os.environ.get("CANCERFLEX_BASE_MODEL", "google/medgemma-1.5-4b-it")
    processor = AutoProcessor.from_pretrained(processor_src)
    model = AutoModelForImageTextToText.from_pretrained(ckpt, dtype=DTYPES[dtype], attn_implementation=attn)
    text_cfg = model.config.text_config if hasattr(model.config, "text_config") else model.config
    controller = GranularityController(widths=load_widths(ckpt, text_cfg.intermediate_size))
    n = wrap_medgemma_mlps(model, controller)
    if n == 0:
        raise RuntimeError("no Gemma3MLP modules found to wrap; is this a Gemma-3 based checkpoint?")
    model.to(device).eval()
    return model, processor, controller


@torch.no_grad()
def answer(model, processor, image: Image.Image, prompt: str, max_new_tokens: int = 16) -> str:
    """Run one image + prompt through the chat template and return the decoded continuation."""
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, images=[image], return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen = out[0, inputs["input_ids"].shape[1]:]
    return processor.decode(gen, skip_special_tokens=True).strip()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="directory with the model weights (and processor files)")
    ap.add_argument("--image", help="path to an RGB image")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--synthetic", action="store_true", help="use a procedurally generated image instead of --image")
    ap.add_argument("--width", type=int, default=None, help="nested feed-forward width (default: full width)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="bf16" if torch.cuda.is_available() else "fp32")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--json", action="store_true", help="print a JSON record instead of plain text")
    args = ap.parse_args(argv)

    if not args.synthetic and not args.image:
        ap.error("pass --image PATH or --synthetic")
    image = make_synthetic_image() if args.synthetic else Image.open(args.image).convert("RGB")

    model, processor, controller = load_model(args.checkpoint, device=args.device, dtype=args.dtype)
    width = args.width or controller.widths[-1]
    controller.set(width)
    text = answer(model, processor, image, args.prompt, max_new_tokens=args.max_new_tokens)
    rec = {"width": width, "trained_widths": controller.widths, "prompt": args.prompt, "answer": text,
           "image": "synthetic" if args.synthetic else args.image}
    print(json.dumps(rec, indent=2) if args.json else text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
