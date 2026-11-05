"""Multimodal SFT collator for medgemma (Gemma3) / gemma-3n.

Follows the TRL VLM SFT recipe (training_vlm_sft.md) + Google MedGemma fine-tune
notebook conventions:
  - Use `processor.apply_chat_template(conv, tokenize=False)` to build text;
    re-process with `processor(text=..., images=...)` to get inputs.
  - Mask `pad_token_id`, `boi_token` (255999), `eoi_token`, and the image
    soft-token (`262144`) to -100 in labels so the LM never tries to predict
    image-placeholder tokens.
  - Mask the user / system turns to -100 too — only the assistant turn
    contributes to loss. We achieve this by tokenizing the prompt-only
    conversation and zeroing those positions.

`<image>` literal in our v1 corpus prompts is replaced with the *one* image
placeholder the processor expects (single image / record); Gemma3 expands it
to `mm_tokens_per_image` (=256) soft tokens internally.
"""
from __future__ import annotations

import math as _math
import os as _os
import random as _random
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from cancerflex.paths import ROOT  # noqa: E402  (set CANCERFLEX_ROOT)

# Shared canonical image loader (single source of truth for decode + cache).
import cancerflex.eval  # noqa: F401,E402  (puts the evaluation modules on sys.path)
import canonical_image  # noqa: E402

# IDs in medgemma-1.5-4b-it (Gemma3) — verified in config.json:
GEMMA3_IMAGE_SOFT_TOKEN_ID = 262144  # the soft tokens that hold image features
GEMMA3_BOI_ID = 255999               # <start_of_image>
# eoi_token_index also exists; we look it up on the processor when available.


def _load_image_rgb(rel_path: str) -> Image.Image:
    """Load a manifest-relative image as RGB via the shared canonical loader.

    Delegates to `canonical_image.load_cached`: returns the cached 896x896
    preprocessing output when present (cheap subset built offline), otherwise
    decodes the original (DICOM VOI LUT + MONOCHROME1 / PIL). This is the SSOT
    shared with `eval_iterator`, so train/eval rendering can no longer drift.
    """
    return canonical_image.load_cached(rel_path, root=ROOT)


class VLMSFTCollator:
    """Collate function for TRL SFTTrainer on (prompt, image, answer) records.

    Args:
        processor: HF AutoProcessor (Gemma3 / Gemma3n).
        image_token: the placeholder string to inject into the user message
            (Gemma3 = "<image>"; Gemma-3n native has its own — auto-detect).
        max_length: hard token cap; medgemma prompts are short (≤ 256 text
            tokens) and one image expands to 256 soft tokens, so 1024 is
            comfortable. None = no cap.
    """

    # pillar / task -> id maps for the GOAL-ROUTER's "modality" and "task" guides (emitted only
    # when emit_router_ids=True). Kept here so the collator is the single source of the id encoding.
    ROUTER_PILLAR2ID = {"skin": 0, "oral": 1, "blood": 2, "cervix": 3, "breast": 4, "thyroid": 5}
    ROUTER_TASK2ID = {f"U{i}": i for i in range(7)}

    def __init__(self, processor, image_token: str | None = None,
                 max_length: int | None = 1024, emit_router_ids: bool = False,
                 resolution_aug_prob: float = 0.0,
                 resolution_aug_min_scale: float = 0.143,
                 resolution_aug_seed: int = 0,
                 resolution_aug_pillars: list[str] | None = None) -> None:
        self.processor = processor
        # ── resolution augmentation, OFF unless resolution_aug_prob > 0 ──────────────────────
        # WHY IT EXISTS. In this corpus acquisition resolution is confounded with the class label,
        # and the model uses it. oral U4 training has `oral cancer` at a 0.07 MP median and 62% of
        # its rows at or below 0.2 MP, against OSMF at 12.00 MP -- a 171x span between two classes
        # of the same option list -- because the two lowest-resolution sources are almost entirely
        # `oral cancer`. Degrading ONLY the resolution of internal images at evaluation moves
        # predictions exactly the way that confound predicts: oral cancer +117, OSMF -33,
        # leukoplakia 7 -> 0, the same zero the model produces on the external set
        # (eval_results/ORAL_DEGRADATION_PROBE_20260920.json). Blur has been learned as a class
        # cue. The cross-pillar census says this is not an oral defect: it tracks how many sources
        # contribute a class, and the two pillars with NO single-source class (skin 8.4 sources per
        # class, breast 2.8) show a 2.2-3.9x span against cervix's 455x
        # (eval_results/RESOLUTION_CLASS_CONFOUND_20260920.json).
        #
        # WHAT IT DOES. With probability `resolution_aug_prob`, downsample the image by a factor
        # drawn LOG-uniformly from [resolution_aug_min_scale, 1.0] and restore it to its original
        # size. The image keeps its pixel dimensions and loses real detail, which is what the
        # confound is made of.
        #
        # IT ONLY REMOVES DETAIL, AND THAT ASYMMETRY IS CORRECT, not a limitation. Detail cannot be
        # fabricated, so a class that is already low-resolution is untouched, while a
        # high-resolution class is sometimes presented at low resolution. The correlation is broken
        # from the side it can be broken from: "blurry" stops implying "oral cancer" because OSMF
        # now appears blurry too.
        #
        # JUSTIFIED INTERNALLY. The case for this rests entirely on the training corpus -- oral
        # cancer at 0.07 MP against OSMF at 12.00 MP -- and would stand if the external set did not
        # exist. It is NOT tuned to the external set's resolution, which matters because that set
        # is the only external population oral has.
        #
        # DEFAULT SCALE FLOOR. 0.143 ~= 128/896: the collator receives the cached 896x896 image, so
        # this spans 128x128 up to full size. The external set's 510x270 carries the same pixel
        # count as a 371x371 square, which sits inside that range rather than at its edge.
        #
        # Deterministic given `resolution_aug_seed`; uses its own RNG so it cannot perturb any
        # other random stream in the run.
        self.res_aug_prob = float(resolution_aug_prob)
        self.res_aug_min_scale = float(resolution_aug_min_scale)
        self._res_rng = _random.Random(resolution_aug_seed)
        # PILLAR FILTER (2026-09-26, dense combined repair). None = every row, the original behaviour
        # (byte-identical: the check runs before the RNG is touched). A list restricts the augmentation
        # to rows of those pillars, so a mixed corpus can degrade oral images without blurring cervix or
        # thyroid rows that the elastic oral-only repair never blurred.
        self.res_aug_pillars = None if resolution_aug_pillars is None else set(resolution_aug_pillars)
        if self.res_aug_prob > 0 and _os.environ.get("RANK", "0") == "0":
            print(f"[collator] resolution augmentation ON: p={self.res_aug_prob}, "
                  f"scale log-uniform in [{self.res_aug_min_scale}, 1.0], "
                  f"seed={resolution_aug_seed}, pillars={sorted(self.res_aug_pillars) if self.res_aug_pillars else 'all'}",
                  flush=True)
        self.emit_router_ids = emit_router_ids
        self.tokenizer = processor.tokenizer
        # Gemma3 uses "<image>" inside chat template; Gemma3n exposes
        # processor.image_token (e.g. "<image_soft_token>"). Detect.
        self.image_token = (
            image_token
            or getattr(processor, "image_token", None)
            or "<image>"
        )
        self.max_length = max_length
        # Padding side MUST be right for causal LMs (left-pad breaks the
        # shifted-labels math in TRL's compute_loss).
        self.tokenizer.padding_side = "right"
        # gemma-3n base ships WITHOUT a chat_template (only medgemma-it does).
        # Without a template, processor.apply_chat_template would raise.
        # `has_ct` switches the per-record builder to a native fallback:
        # prepend `image_token` + render plain text user/assistant turns.
        self.has_ct = bool(self.tokenizer.chat_template)

        # Build the set of label-mask ids (image / boi / eoi / pad).
        self.label_mask_ids: set[int] = set()
        for tok_id in (
            GEMMA3_IMAGE_SOFT_TOKEN_ID,
            GEMMA3_BOI_ID,
            getattr(processor, "image_token_id", None),
            getattr(processor, "boi_token_id", None),
            getattr(processor, "eoi_token_id", None),
            self.tokenizer.pad_token_id,
        ):
            if isinstance(tok_id, int) and tok_id >= 0:
                self.label_mask_ids.add(tok_id)

    def _build_conversation(self, prompt: str, answer: str) -> tuple[list, list]:
        """Return (full_conv, user_only_conv). The renderer-produced prompt
        contains a literal `<image>` — split on it so the image is in the
        right position relative to surrounding text."""
        if "<image>" in prompt:
            parts = prompt.split("<image>", 1)
            content_user = [
                {"type": "text", "text": parts[0]},
                {"type": "image"},
                {"type": "text", "text": parts[1]},
            ]
        else:
            content_user = [{"type": "image"}, {"type": "text", "text": prompt}]
        user_only = [{"role": "user", "content": content_user}]
        full = user_only + [
            {"role": "assistant", "content": [{"type": "text", "text": str(answer)}]},
        ]
        return full, user_only

    def _augment_resolution(self, img: "Image.Image", rec: dict | None = None) -> "Image.Image":
        """Randomly remove detail. Identity when the feature is off, which is the default."""
        if self.res_aug_pillars is not None and (rec or {}).get("pillar") not in self.res_aug_pillars:
            return img
        if self.res_aug_prob <= 0 or self._res_rng.random() >= self.res_aug_prob:
            return img
        lo = max(self.res_aug_min_scale, 1e-3)
        f = _math.exp(self._res_rng.uniform(_math.log(lo), 0.0))   # log-uniform in [lo, 1.0]
        w, h = img.size
        nw, nh = max(8, int(round(w * f))), max(8, int(round(h * f)))
        if nw >= w and nh >= h:
            return img
        return img.resize((nw, nh), Image.Resampling.BILINEAR).resize((w, h), Image.Resampling.BILINEAR)

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        full_texts: list[str] = []
        user_texts: list[str] = []
        images: list[list[Image.Image]] = []
        kept: list[dict] = []

        for rec in batch:
            if rec.get("multi_image"):
                # ── multi-image record (2026-09-23, Hou multi-image repair) ─────────────────────
                # ALL frames of one patient in ONE user turn, rendered exactly like the multi-image
                # eval (scripts/eval/hou_multiimage_20260923.py): content = [image_1..image_N,
                # text(prompt with "<image>" removed)]. Opt-in: only rows carrying multi_image=True
                # take this path, so every existing corpus is byte-identical in behaviour. A row whose
                # frames cannot all be read is skipped, like an unreadable single image.
                try:
                    imgs = [self._augment_resolution(_load_image_rgb(p), rec) for p in rec["image_paths"]]
                except Exception:
                    continue
                if not self.has_ct:
                    raise ValueError("multi_image rows need a chat template (Gemma3 / medgemma path)")
                content_user = [{"type": "image"} for _ in imgs]
                content_user.append({"type": "text", "text": rec["prompt"].replace("<image>", "")})
                user_conv = [{"role": "user", "content": content_user}]
                full_conv = user_conv + [
                    {"role": "assistant", "content": [{"type": "text", "text": str(rec["answer"])}]}]
                full_texts.append(self.processor.apply_chat_template(
                    full_conv, tokenize=False, add_generation_prompt=False))
                user_texts.append(self.processor.apply_chat_template(
                    user_conv, tokenize=False, add_generation_prompt=True))
                images.append(imgs)
                kept.append(rec)
                continue
            try:
                img = _load_image_rgb(rec["image_path"])
            except Exception:
                continue                              # skip unreadable
            img = self._augment_resolution(img, rec)
            if self.has_ct:
                # medgemma-it path — use the model's chat template.
                full_conv, user_conv = self._build_conversation(rec["prompt"], rec["answer"])
                full_t = self.processor.apply_chat_template(
                    full_conv, tokenize=False, add_generation_prompt=False,
                )
                user_t = self.processor.apply_chat_template(
                    user_conv, tokenize=False, add_generation_prompt=True,
                )
            else:
                # gemma-3n base path — no chat template, so render as plain
                # `<image_soft_token><prompt><answer>`. The processor still
                # expands the image_token internally. The "<image>" placeholder
                # in our v1 prompts is replaced with the model's image_token,
                # not left as a literal string.
                prompt_text = rec["prompt"].replace("<image>", self.image_token, 1)
                if self.image_token not in prompt_text:
                    prompt_text = self.image_token + prompt_text
                user_t = prompt_text                          # "USER" side
                full_t = prompt_text + str(rec["answer"])     # "USER + ASSIST"
            full_texts.append(full_t)
            user_texts.append(user_t)
            images.append([img])
            kept.append(rec)

        if not kept:
            # All samples failed image-load → return an empty 0-sized batch
            # the Trainer will skip (loss=0 step).
            return {"input_ids": torch.empty(0, 0, dtype=torch.long)}

        full_inputs = self.processor(
            text=full_texts, images=images, padding=True,
            truncation=True, max_length=self.max_length, return_tensors="pt",
        )
        # Tokenize the user-only side separately *with the same image set* so
        # token alignment matches: positions 0..L_user are the prompt + image
        # soft tokens; we mask all of these to -100 in labels.
        user_inputs = self.processor(
            text=user_texts, images=images, padding=True,
            truncation=True, max_length=self.max_length, return_tensors="pt",
        )
        full_ids = full_inputs["input_ids"]                # (B, T_full)
        # Multi-image rows are long (up to ~31 frames x 256 soft tokens); truncation at max_length
        # would cut image tokens and silently misalign the answer. Fail loudly instead.
        for i, rec in enumerate(kept):
            if rec.get("multi_image"):
                n_soft = int((full_ids[i] == GEMMA3_IMAGE_SOFT_TOKEN_ID).sum())
                if n_soft != 256 * len(images[i]):
                    raise ValueError(
                        f"multi_image row {rec.get('record_id')}: {n_soft} image soft tokens for "
                        f"{len(images[i])} frames -- truncated by max_length={self.max_length}?")
        user_lens = (user_inputs["input_ids"] != self.tokenizer.pad_token_id).sum(dim=1)

        labels = full_ids.clone()
        # Mask everything up to the assistant turn start (per-row L_user).
        for i, L in enumerate(user_lens.tolist()):
            labels[i, :L] = -100
        # Mask image soft tokens / boi / eoi / pad anywhere in the sequence.
        if self.label_mask_ids:
            mask_ids = torch.tensor(sorted(self.label_mask_ids), device=labels.device)
            labels[torch.isin(labels, mask_ids)] = -100

        full_inputs["labels"] = labels
        if self.emit_router_ids:
            # aligns with `kept` (rows that survived image-load), same order as full_inputs rows.
            full_inputs["router_pillar"] = torch.tensor(
                [self.ROUTER_PILLAR2ID.get(r.get("pillar"), 0) for r in kept], dtype=torch.long)
            full_inputs["router_task"] = torch.tensor(
                [self.ROUTER_TASK2ID.get(r.get("task"), 0) for r in kept], dtype=torch.long)
        return full_inputs
