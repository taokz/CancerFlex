"""Synthetic sample input so the inference entry point runs end to end without real data.

The image is procedurally generated (noise background plus one darker ellipse); it carries no
medical content and exists only to exercise the pipeline.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

DEFAULT_PROMPT = "Is a lesion visible in this image? Answer Yes or No."


def make_synthetic_image(size: int = 896, seed: int = 0) -> Image.Image:
    """Return an RGB image of ``size`` x ``size`` pixels with a smooth noise background and one ellipse."""
    rng = np.random.default_rng(seed)
    base = rng.normal(loc=190, scale=12, size=(size, size, 3)).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(base, mode="RGB")
    draw = ImageDraw.Draw(img)
    c = size // 2
    r = size // 8
    draw.ellipse([c - r, c - int(r * 0.7), c + r, c + int(r * 0.7)], fill=(120, 80, 70))
    return img


def write_sample(out_dir: str | Path, size: int = 896, seed: int = 0) -> dict:
    """Write ``sample.png`` and ``sample.json`` (prompt + metadata) into ``out_dir``; return the record."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    img_path = out / "sample.png"
    make_synthetic_image(size=size, seed=seed).save(img_path)
    record = {"image_path": str(img_path), "prompt": DEFAULT_PROMPT, "synthetic": True, "seed": seed}
    (out / "sample.json").write_text(json.dumps(record, indent=2))
    return record


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="examples/synthetic")
    ap.add_argument("--size", type=int, default=896)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rec = write_sample(args.out_dir, size=args.size, seed=args.seed)
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
