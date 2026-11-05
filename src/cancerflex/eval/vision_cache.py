"""Frozen-vision feature cache (Track B) — the CONTRACT shared by the offline
builder and the eval-time integration.

WHY: closed eval is compute-bound on the SigLIP vision tower (~85% of per-row cost,
py-spy-confirmed, PERFORMANCE_AND_RESOURCE_NOTES §9/§10). `vision_freeze: true` in all
multitask SFT configs and the vision_tower weights are bit-identical across checkpoints
(verified phase1 vs a2_full), so the vision encoding of a given image is
CHECKPOINT-INDEPENDENT. Compute it ONCE per image, cache it, and every checkpoint's eval
skips the vision tower.

WHAT IS CACHED — the post-avg-pool, PRE-norm tensor `[256, 1152]` (FP32, ~1.18 MB/img):
  Gemma3 `get_image_features` = vision_tower(px).last_hidden_state [B,4096,1152]
      -> projector.forward: reshape -> avg_pool(4096->256, PARAM-FREE) -> [B,256,1152]
      -> mm_soft_emb_norm (RMSNorm, PER-CKPT) -> matmul(mm_input_projection_weight, PER-CKPT)
      -> image_features [B,256,2560]
  The cache boundary is right AFTER avg_pool: everything before it (vision_tower + avg_pool)
  is frozen + param-free => checkpoint-independent + small. The per-ckpt part (norm + proj)
  is cheap and rerun at eval via `project_pooled`.

DTYPE — FP32, not bf16 (changed 2026-06-30): the closed-eval forward runs the vision tower
  under autocast(bf16), which leaves avg_pool + the mm_soft_emb_norm RMSNorm in FP32, so the
  model keeps this PRE-norm pooled in fp32 internally. Storing it as bf16 rounds it (~0.2%
  rel), which propagates through the 34-layer LM to a ~0.7-logit / ~0.1-0.5-logprob score
  drift — i.e. NOT bit-identical with the live pixel_values path. fp32 storage is bit-exact
  (verified 0.0 via run_closed.py's EVAL_VISION_CACHE gate). The earlier bf16 cache
  (`v1_siglip_pooled_bf16`) is WRONG — delete + rebuild under the new version.

CORRECTNESS: `project_pooled(model, pooled_vision(model, px))` MUST equal
`model.get_image_features(px).pooler_output` bit-for-bit (this module's self-test asserts the
DECOMPOSITION). NOTE the self-test does NOT exercise the on-disk bf16/fp32 round-trip nor the
autocast the eval forward uses — that end-to-end bit-identity is gated separately by
run_closed.py's EVAL_VISION_CACHE path (tmp/verify_vision_cache.py). Caching is keyed by
manifest-relative image path; bump CACHE_VERSION on any change.
"""
from __future__ import annotations
import os
from pathlib import Path
import torch

from cancerflex.paths import ROOT  # noqa: E402  (set CANCERFLEX_ROOT)
# bump on any vision_tower / pooling / DTYPE change (v1 bf16 was too lossy -> not bit-exact).
CACHE_VERSION = "v2_siglip_pooled_fp32"
CACHE_ROOT = ROOT / "Experiment" / "datasets" / "_vision_cache" / CACHE_VERSION


def vfeat_path(rel_path: str) -> Path:
    """Cache file for a source image (mirror rel path, '.vfeat.pt')."""
    return CACHE_ROOT / (rel_path + ".vfeat.pt")


def _vision_parts(model):
    """Locate (vision_tower, multi_modal_projector, get_image_features) regardless of
    whether they sit on `model` or `model.model` (ForConditionalGeneration wrapper)."""
    for owner in (getattr(model, "model", None), model):
        if owner is not None and hasattr(owner, "vision_tower") and hasattr(owner, "multi_modal_projector"):
            return owner.vision_tower, owner.multi_modal_projector, owner.get_image_features
    raise RuntimeError("no vision_tower/multi_modal_projector found on model")


@torch.no_grad()
def pooled_vision(model, pixel_values: torch.Tensor) -> torch.Tensor:
    """Frozen, checkpoint-INDEPENDENT half: vision_tower + the projector's param-free
    avg-pool. Returns [B, mm_tokens, vision_hidden] (e.g. [B,256,1152]). EXACT replica of
    the pre-norm steps in Gemma3MultiModalProjector.forward."""
    vt, proj, _ = _vision_parts(model)
    vo = vt(pixel_values=pixel_values, return_dict=True).last_hidden_state  # [B, P*P, H]
    B, _, H = vo.shape
    x = vo.transpose(1, 2).reshape(B, H, proj.patches_per_image, proj.patches_per_image).contiguous()
    x = proj.avg_pool(x)                 # [B, H, tps, tps]  (param-free)
    x = x.flatten(2).transpose(1, 2)     # [B, mm_tokens, H]
    return x


@torch.no_grad()
def project_pooled(model, pooled: torch.Tensor) -> torch.Tensor:
    """Per-CHECKPOINT half: mm_soft_emb_norm + projection. Takes `pooled` from
    `pooled_vision` (or the cache) -> image_features [B, mm_tokens, text_hidden]. EXACT
    replica of the post-pool steps in Gemma3MultiModalProjector.forward."""
    _, proj, _ = _vision_parts(model)
    normed = proj.mm_soft_emb_norm(pooled)
    return torch.matmul(normed, proj.mm_input_projection_weight).type_as(pooled)


def build_entry(model, rel_path: str, pixel_values: torch.Tensor, overwrite: bool = False) -> str:
    """Compute + cache pooled vision for one image's pixel_values [1,...]. Atomic write.

    BIT-IDENTITY REQUIREMENTS — must mirror exactly how the closed-eval forward
    (scripts/eval/run_closed.py score_options) runs the vision tower, else cached scores
    drift from the live pixel_values path (gated by EVAL_VISION_CACHE; tmp/verify_vision_cache.py):
      * pass FLOAT32 `pixel_values` (the processor's native dtype) — do NOT pre-cast to bf16;
        eval feeds float32 px and lets autocast down-cast inside the tower.
      * run the tower under autocast(bf16) — eval's forward is wrapped in autocast(bf16),
        which keeps avg_pool / RMSNorm in fp32 while matmul/conv run bf16. A pure-bf16 run
        (bf16 px, no autocast) diverges.
      * store the avg-pooled tensor in FP32 (see module header) — bf16 storage is NOT
        bit-identical (~0.7-logit drift); fp32 is bit-exact (0.0)."""
    out = vfeat_path(rel_path)
    if out.exists() and not overwrite:
        return "skipped"
    use_amp = bool(getattr(pixel_values, "is_cuda", False))
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
        pooled = pooled_vision(model, pixel_values)        # [1, mm_tokens, H]
    pooled = pooled.float().cpu()                          # FP32 (bit-exact with eval)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".pt.tmp")
    torch.save(pooled, tmp)
    os.replace(tmp, out)
    return "built"


def load_entry(rel_path: str) -> torch.Tensor | None:
    """Load cached pooled vision ([1, mm_tokens, H] FP32) or None on miss."""
    p = vfeat_path(rel_path)
    if not p.exists():
        return None
    return torch.load(p, map_location="cpu")


def self_test(model, pixel_values: torch.Tensor) -> dict:
    """Bit-identical gate: project_pooled(pooled_vision(px)) vs model.get_image_features(px)
    .pooler_output. Returns max abs diff + allclose. MUST be ~0 (bf16-negligible) to trust."""
    vt, proj, get_if = _vision_parts(model)
    ref = get_if(pixel_values, return_dict=True).pooler_output            # model's own path
    ours = project_pooled(model, pooled_vision(model, pixel_values))      # cache path
    d = (ref.float() - ours.float()).abs()
    return {"shape_ref": tuple(ref.shape), "shape_ours": tuple(ours.shape),
            "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()),
            "allclose_exact": bool(torch.equal(ref, ours)),
            "allclose_1e-3": bool(torch.allclose(ref.float(), ours.float(), atol=1e-3))}
