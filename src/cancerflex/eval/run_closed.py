#!/usr/bin/env python3
"""MedGemma closed-option eval runner for active-current and reusable subsets.

Dataset boundary:
- For active-current eval, use current v1_final shards through
  `eval_iterator.iter_eval_rows`. Official reports should include only
  task/pillar pairs with `test_ood` rows; `scripts/eval/active_current.py`
  prints the current command set.
- For legacy-to-active difference eval, set `EVAL_ROW_LOCK_DIR` to the
  generated row-lock directory so only rows with `action=run_model_inference`
  are scored. Reused legacy predictions must be merged separately.
- U0 has a dedicated runner, `scripts/eval/run_u0.py`, because its active
  official boundary is conditional on OOD availability.

Architecture:
- Iterates 6 pillars x {U1, U2, U3, U4} x {test_id, test_ood}.
- For a MatFormer checkpoint, scalar EVAL_MATFORMER_WIDTH keeps the historical
  one-width behavior. A comma-list loads/wraps once, switches the shared controller
  in-place, and requires EVAL_MATFORMER_OUT_DIRS JSON so every width retains its own
  raw CSVs and resume boundary.
- Per row: scores each candidate class with sum-log-prob over the answer's
  token sequence. The vision encoder runs once per row (prefix KV cache), then
  K extension passes (one per class). Argmax gives the prediction.
- For U1/U2/U3 (binary): also record yes_score - no_score for AUROC.
- For U4 (multiclass): record per-option scores; softmax → AUROC. Heterogeneous-
  option pillars (U4_GEN_PILLARS, default skin/oral) instead GENERATE the label
  (closed-set mean-logprob mode-collapses on their multi-token class names) and
  take AUROC from the generation's first-token step — see U4_GEN_PILLARS.
- Persist per-row results to CSV as rows finish, so an interrupted run can
  resume by skipping completed row ids.
- At the end of each (pillar, task) pair, write metrics_<pillar>_<task>.json
  with AUROC/accuracy/macro-F1/ECE and per-OOD-axis breakdown where applicable.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import queue
import re
import signal
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from cancerflex.paths import ROOT  # noqa: E402  (set CANCERFLEX_ROOT)
# the evaluation modules import each other by flat name; make this directory importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_iterator import iter_eval_rows, load_eval_manifest  # noqa: E402
from prompt_templates import PILLAR_CONFIG  # noqa: E402

OUT_DIR = Path(os.environ.get("EVAL_OUT_DIR", str(ROOT / "eval_results" / "medgemma_baseline")))
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUT_DIR / "run_log.jsonl"

# Populated only by the MatFormer-pure loader.  Keeping this state module-local
# preserves load_model()'s long-standing three-value return contract (run_u0.py
# imports it) while main() can re-select another trained uniform width in-place.
_MATFORMER_CONTROLLER = None
_MATFORMER_REQUESTED_WIDTHS: list[int] = []

MODEL_PATH = os.environ.get("CANCERFLEX_BASE_MODEL", "google/medgemma-1.5-4b-it")

# Active v1 pillars per handoff §2.
DEFAULT_PILLARS = ["skin", "breast", "cervix", "oral", "thyroid", "blood"]
DEFAULT_TASKS = ["U2", "U3", "U4"]
U0_DEFAULT_SOURCE = os.environ.get("EVAL_U0_SOURCE", "sidecar")

# ── U4 heterogeneous-option pillars: GENERATE the label + first-token softmax for
# AUROC, instead of length-normalized closed-set scoring. ─────────────────────
# WHY: closed-set `score_options` ranks each U4 class by length-normalized mean
# token-logprob. On heterogeneous multi-token class NAMES (skin "seborrheic
# keratosis"; oral "oral potentially malignant disorder (OPMD)") this mode-
# collapses to whichever name is longest/most-fluent, ~independent of the image:
# the diagnostic penalty sits on the first token and is divided by the token
# count, so mean ≈ ℓ₁/n → argmin(ℓ₁/n) ≈ "longest name wins" (measured: skin/oral
# support-weighted macro-F1 collapse to 0.03/0.06). The model DID learn U4 — free
# generation is image-grounded (skin 0.79 / oral 0.96 macro-F1, full test_id). So
# for these pillars we score U4 by GENERATION (greedy-decode → map to nearest
# candidate class → drives accuracy/macro-F1, the U4 floor metric) and take the
# threshold-free AUROC from the generation's FIRST decoding step (per-candidate
# first-token logprob → softmax) — a strictly better probability than the
# collapsed mean, at no extra forward. breast (BIRADS, uniform single-digit token)
# and binary U1–U3 (Yes/No) do NOT collapse → stay closed-set.
# UNIFICATION (2026-07-08): U4 is now ONE method for every graded pillar (skin,
# oral, blood, cervix, breast) — all GENERATIVE. Motivation: closed-set length-norm
# argmax collapses on multi-word class names (measured: blood/cervix concentrate
# ~89%/95% on a single long name; weighted-F1 → 0.07/0.21), so per-pillar metric
# choices ("skin/oral gen-F1, blood/cervix AUROC-only, breast closed-F1") read as
# metric-shopping. A single greedy generation pass (score_generate_hybrid) removes
# the collapse AND emits BOTH reported numbers at once:
#   • decoded class  → support-weighted macro-F1  = the UNIFORM U4 HEADLINE
#   • first-token option distribution → macro-OvO AUROC = the UNIFORM SECONDARY
#     (threshold-free, does not collapse; same first-token source for all pillars,
#     so the AUROC is finally computed identically everywhere).
# Matching is STRICT: earliest shown-option substring, NO alias/synonym table. The
# prompt lists the exact harmonized option strings (identical to the training
# prompt — verified byte-for-byte), so a model that does not emit the shown string
# (e.g. "neutrophil" when the option is "neutrophil lineage") is scored WRONG. That
# is a real capability signal, not a scoring artifact; for pillars whose SSOT names
# carry harmonization suffixes ("neutrophil lineage", "platelet/thrombocyte") the
# suffix-independent AUROC secondary is the fair companion read (see
# U4_NOTE_BY_PILLAR + eval_metric_calculation.md). thyroid U4 is N/A — NOT a disabled
# cell but a genuine data gap: no in-distribution fine-grained U4 labels exist
# (only OOD TI-RADS in DDTI/Stanford). Override with EVAL_U4_GEN_PILLARS=...
# (empty = all closed-set, legacy). Full analysis: docs/archive/u4_mode_collapse.md.
U4_GEN_PILLARS = {
    p.strip().lower() for p in
    os.environ.get("EVAL_U4_GEN_PILLARS", "skin,oral,blood,cervix,breast").split(",") if p.strip()
}
U4_GEN_MAX_NEW = int(os.environ.get("EVAL_U4_GEN_MAX_NEW", "24"))

# ── Cells that must NOT be evaluated (skipped here; both yield a clean cell, not a
#    misleading number). Two DISTINCT reasons — the reported STATUS differs even
#    though the skip mechanism is the same (see aggregate_overall_scores.py
#    DISABLED_PAIRS vs NA_PAIRS, which surface the distinction to readers):
#      • DISABLED (task undefined for the pillar): blood/cervix/oral/thyroid U1 —
#        no clinically-meaningful binary target-present label; eval gold is
#        single-class (all-positive) → accuracy uninterpretable, AUROC undefined.
#      • N/A (task defined, no in-distribution data): thyroid U4 — the train/test_id
#        pool is irreducibly binary (benign vs malignant); the only fine-grained
#        thyroid U4 is OOD TI-RADS (DDTI/Stanford), so no in-distribution headline
#        can be computed.
# MIRRORS the SSOT registry Experiment/core_code/data_build/build_task_views.py:33
# (DISABLED_TASKS). run_closed reads U1–U4 from the eval MANIFEST (iter_eval_rows),
# which still carries these rows, so unlike the shard-built corpus it does NOT
# auto-exclude them → we skip them explicitly here.
# KEEP IN SYNC with build_task_views.DISABLED_TASKS if that registry changes.
CLOSED_DISABLED_PAIRS = {
    ("blood", "U1"),
    ("cervix", "U1"),
    ("oral", "U1"),
    ("thyroid", "U1"),
    ("thyroid", "U4"),
}

# Rows/pairs DROPPED by inference-level failures (OOM, inference error, pair crash).
# Nonzero => results are PARTIAL; the run exits nonzero so run_multitask's rc-gate
# marks the eval invalid instead of silently publishing partial metrics.
_RUN_DROPS = 0


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────


def log_event(category: str, **fields) -> None:
    """Append one JSON event to run_log.jsonl."""
    rec = {"ts": dt.datetime.utcnow().isoformat() + "Z", "category": category, **fields}
    try:
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
    except Exception:
        # Never let logging take down the run.
        pass


def _parse_matformer_widths() -> list[int]:
    """Parse the backward-compatible scalar-or-list width environment knob.

    A scalar EVAL_MATFORMER_WIDTH behaves exactly as before.  A comma-list is
    the load-once mode and requires EVAL_MATFORMER_OUT_DIRS in main() so raw
    rows for one width can never trigger resume-skips for another width.
    """
    raw = os.environ.get("EVAL_MATFORMER_WIDTH")
    if not raw:
        return []
    try:
        widths = [int(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(
            f"EVAL_MATFORMER_WIDTH must be an integer or comma-list, got {raw!r}"
        ) from exc
    if not widths:
        raise ValueError("EVAL_MATFORMER_WIDTH contains no widths")
    if len(set(widths)) != len(widths):
        raise ValueError(f"EVAL_MATFORMER_WIDTH contains duplicate widths: {widths}")
    return widths


def _activate_out_dir(path: Path) -> None:
    """Route every subsequent raw CSV, metric, and log write to ``path``."""
    global OUT_DIR, LOG_PATH
    OUT_DIR = path
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH = OUT_DIR / "run_log.jsonl"


def _multi_width_out_dirs(widths: list[int]) -> dict[int, Path]:
    """Validate the explicit width-to-output mapping for load-once evaluation.

    The JSON environment form is intentionally explicit, for example
    ``EVAL_MATFORMER_OUT_DIRS='{"1280":".../w1280","10240":".../w10240"}'``.
    Guessing directory names would risk mixing/resuming raw rows across widths.
    """
    raw = os.environ.get("EVAL_MATFORMER_OUT_DIRS")
    if not raw:
        raise ValueError(
            "comma-list EVAL_MATFORMER_WIDTH requires EVAL_MATFORMER_OUT_DIRS="
            "<JSON object mapping every width to its EVAL_OUT_DIR>"
        )
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid EVAL_MATFORMER_OUT_DIRS JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("EVAL_MATFORMER_OUT_DIRS must be a JSON object")
    if any(not isinstance(path, str) or not path.strip() for path in parsed.values()):
        raise ValueError("EVAL_MATFORMER_OUT_DIRS values must be non-empty path strings")
    try:
        out_dirs = {int(width): Path(path) for width, path in parsed.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "EVAL_MATFORMER_OUT_DIRS keys must be integer widths and values must be paths"
        ) from exc
    if set(out_dirs) != set(widths):
        raise ValueError(
            f"EVAL_MATFORMER_OUT_DIRS keys {sorted(out_dirs)} do not exactly match "
            f"requested widths {widths}"
        )
    if len(set(out_dirs.values())) != len(out_dirs):
        raise ValueError("EVAL_MATFORMER_OUT_DIRS must use a distinct directory per width")
    return out_dirs


# ─────────────────────────────────────────────────────────────────────────────
# Class-list resolution per (pillar, task)
# ─────────────────────────────────────────────────────────────────────────────


# U4 option-set overrides per pillar — used when PILLAR_CONFIG's `u4_classes`
# doesn't align with the manifest gold vocabulary. Add a pillar here only when
# the prompt-template axis is fundamentally wrong (not just a vocab mismatch
# that gold harmonization can fix).
U4_OPTION_OVERRIDES = {
    # thyroid PILLAR_CONFIG lists US features (composition / echogenicity / …)
    # but manifest gold is TI-RADS scores + binary B/M. Use TI-RADS axis.
    "thyroid": ["benign", "TI-RADS 2", "TI-RADS 3", "TI-RADS 4a", "TI-RADS 4b",
                "TI-RADS 4c", "TI-RADS 5", "malignant"],
}


def _candidate_classes(pillar: str, task: str) -> list[str]:
    """Return the per-row scoring candidate set.

    Binary tasks (U2/U3): ["Yes", "No"].
    U4: pillar-specific class list from PILLAR_CONFIG (parsed from the
    rendered 'u4_classes' string), unless overridden in U4_OPTION_OVERRIDES.
    """
    if task in ("U0", "U1", "U2", "U3"):
        return ["Yes", "No"]
    if task == "U4":
        if pillar in U4_OPTION_OVERRIDES:
            return list(U4_OPTION_OVERRIDES[pillar])
        # PILLAR_CONFIG keys can be "skin", or "breast_mammo"/"breast_us",
        # or "cervix_cytology"/"cervix_via" (modality-split). Match exact OR
        # `<pillar>_*` prefix, and union the u4_classes across matches.
        # u4_classes may use '/' (most pillars), ',' separators (thyroid),
        # and parenthetical sub-options (thyroid: "composition (a/b/c)").
        out: list[str] = []
        seen: set[str] = set()
        plower = pillar.lower()
        for key, cfg in PILLAR_CONFIG.items():
            head = key[0] if isinstance(key, tuple) else str(key)
            hlower = head.lower()
            if not (hlower == plower or hlower.startswith(plower + "_")):
                continue
            raw = cfg.get("u4_classes", "")
            # Pull parenthetical contents (these are leaf options).
            paren_inner = re.findall(r"\(([^)]*)\)", raw)
            outside = re.sub(r"\([^)]*\)", "", raw)
            for chunk in [outside] + paren_inner:
                for p in re.split(r"[/,]", chunk):
                    p = p.strip(" .,")
                    if p and len(p) <= 60 and p not in seen:
                        seen.add(p)
                        out.append(p)
        return out
    raise ValueError(f"task {task!r} not in scoring scope")


# ─────────────────────────────────────────────────────────────────────────────
# Scoring primitive — vision-prefix KV cache + per-option extension passes
# ─────────────────────────────────────────────────────────────────────────────


def _has_chat_template(processor) -> bool:
    """True iff the processor exposes a chat template. medgemma-IT does; the
    gemma-3n base processor does NOT — so its only correct scorer is the
    per-record `score_options` native path. Used to skip the batched scorer
    (which always calls apply_chat_template) for chat-template-less models."""
    return bool(getattr(processor, "chat_template", None)
                or getattr(getattr(processor, "tokenizer", None), "chat_template", None))


def _answer_ids(processor, opt: str) -> list[int]:
    """Token ids of an option AS IT APPEARS IN CONTEXT, i.e. after "Answer: "
    with a leading space. SentencePiece merges that space into the first piece
    ("Yes"→[10784] but " Yes"→[8438]); the model at the answer position predicts
    the space-prefixed token, so options must be tokenized space-prefixed or the
    scored logprob is read off the wrong token id (corrupts every score).

    ONLY for the CLOSED path, where a space really does precede the option. The
    GENERATION path must use _model_turn_answer_ids instead; see its docstring."""
    return processor.tokenizer(" " + opt, add_special_tokens=False)["input_ids"]


_CT_TAIL_LEN_CACHE: dict[int, int] = {}


def _ct_tail_len(processor) -> int:
    """Number of tokens the chat template appends AFTER the user text (add_generation_prompt=False).

    For MedGemma this is TWO tokens, `<end_of_turn>` and `\\n` (chat_template.jinja). Until
    2026-09-24 the closed scorer assumed ONE, so the scored position was the anchor option token
    itself (the distribution AFTER "Answer: Yes"), not "Answer:" — see docs/WEAK_CELLS_TRACKER.md.
    Measured from the template, not assumed: render "Answer: Yes" and count what follows " Yes"."""
    key = id(processor)
    if key not in _CT_TAIL_LEN_CACHE:
        ids = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": "Answer: Yes"}]}],
            add_generation_prompt=False, tokenize=True, return_dict=True)["input_ids"]
        ids = list(ids[0]) if hasattr(ids[0], "__len__") else list(ids)
        yes = _answer_ids(processor, "Yes")
        assert len(yes) == 1, yes
        pos = max(i for i, t in enumerate(ids) if int(t) == yes[0])
        _CT_TAIL_LEN_CACHE[key] = len(ids) - 1 - pos
    return _CT_TAIL_LEN_CACHE[key]


def _ct_strip(processor) -> int:
    """Template-tail tokens to strip in the closed scorer.

    DEFAULT = the measured tail (_ct_tail_len; 2 for MedGemma: <end_of_turn> + "\n") -> the logits
    are read at "Answer:", as intended. Until 2026-09-24 this was hard-coded 1 (off by one: scored
    at the anchor-option position); every STORED closed-set result (paper matrix, comparators through
    this scorer) was produced that way and, by user decision 2026-09-24, is NOT re-scored (Hou check:
    0.394 -> 0.373 internal, 0.564 -> 0.554 external). To reproduce or extend a stored cell on the SAME
    method, set EVAL_CLOSED_POS_LEGACY=1."""
    if os.environ.get("EVAL_CLOSED_POS_LEGACY", "0") == "1":
        return 1
    return _ct_tail_len(processor)


_MODEL_TURN_PREFIX_CACHE: dict[int, str] = {}


def _model_turn_prefix(processor) -> str:
    """The text the chat template emits immediately before the model's own first token."""
    key = id(processor)
    if key not in _MODEL_TURN_PREFIX_CACHE:
        _MODEL_TURN_PREFIX_CACHE[key] = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": ""}]}],
            add_generation_prompt=True, tokenize=False)
    return _MODEL_TURN_PREFIX_CACHE[key]


def _model_turn_answer_ids(processor, opt: str) -> list[int]:
    """Token ids of an option as the model EMITS it, at the start of its own turn.

    FIXED 2026-09-19. score_generate_hybrid used _answer_ids here, i.e. it tokenised
    " " + option. That is right after "Answer: " and wrong after "<start_of_turn>model\n",
    where nothing precedes the option: `negative` is 27851 but ` negative` is 5676, `LSIL`
    begins `LS`=15843 but ` LSIL` begins ` LS`=46450. The scored logprob was therefore read
    at ids the model never emits. Evidence from the persisted CSVs, needing no model: over
    851 external cervix U4 rows the best-scored option's logprob had median -10.449 and
    maximum -6.575 (probability ~3e-5) with ZERO rows above -1, which is impossible for a
    greedily emitted token since it is the argmax over ~262k ids; and agreement between the
    generated label and the score argmax was 0.601 where it is 1.0 by construction when the
    ids are right. Everything read from `predicted_parsed` was unaffected, as was the closed
    path's `binary_score`, which agrees in sign with `predicted_parsed` on 0.994-1.000 of rows.

    Derived from the chat template rather than hard-coded, so it follows the processor.
    """
    tok = processor.tokenizer
    prefix = _model_turn_prefix(processor)
    pre = tok(prefix, add_special_tokens=False)["input_ids"]
    full = tok(prefix + opt, add_special_tokens=False)["input_ids"]
    if full[:len(pre)] != pre:
        # The option merged into the prefix's last piece; fall back to bare tokenisation,
        # which is still closer to the emitted form than the space-prefixed one.
        return tok(opt, add_special_tokens=False)["input_ids"]
    return full[len(pre):]


def _inputs_embeds_with_cached_vision(model, input_ids, cached_pooled, device):
    """Build gemma3 `inputs_embeds` from a CACHED frozen-vision pooled tensor instead of
    running the SigLIP vision tower from pixel_values (the vision tower is frozen across
    checkpoints, so its pooled feature for an image is checkpoint-independent; contract in
    Experiment/core_code/shared/vision_cache.py). Replicates Gemma3Model.forward's
    text-embed + image-feature masked_scatter EXACTLY — OOV guard, project_pooled (per-ckpt
    norm+proj), cast to inputs_embeds dtype, and get_placeholder_mask's
    mask.unsqueeze(-1).expand_as(...) — so the resulting inputs_embeds (hence every
    downstream logit and KV entry) is bit-identical to the pixel_values path."""
    import vision_cache  # noqa: E402 (lazy: only on cache hit)
    cfg = model.config
    image_token_id = cfg.image_token_id  # attribute_map alias of image_token_index
    text_vocab = cfg.get_text_config().vocab_size
    # OOV guard (gemma3): zero image-token positions before embedding when the image token
    # id is outside the text vocab (those positions are overwritten by the scatter anyway).
    if image_token_id >= text_vocab:
        llm_input_ids = input_ids.clone()
        llm_input_ids[input_ids == image_token_id] = 0
    else:
        llm_input_ids = input_ids
    inputs_embeds = model.get_input_embeddings()(llm_input_ids)
    image_features = vision_cache.project_pooled(model, cached_pooled.to(device))
    image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
    special_image_mask = (input_ids == image_token_id).unsqueeze(-1).expand_as(
        inputs_embeds).to(inputs_embeds.device)
    return inputs_embeds.masked_scatter(special_image_mask, image_features)


@torch.no_grad()
def score_options(
    model, processor, image, prompt_text: str, options: list[str], device,
    rel_path: str | None = None,
) -> list[float]:
    """Return mean_token_logprob per option (length-normalised).

    Vision encoder runs ONCE per record. Each option is then scored by
    extending from the cached prefix with its tokens.
    """
    if not options:
        return []

    # Build prompt with the first option appended as anchor; we then strip
    # the option tokens to obtain the shared prefix.
    opt0 = options[0]
    has_ct = _has_chat_template(processor)
    opt0_ids = _answer_ids(processor, opt0)  # in-context (space-prefixed) form
    # Tokens to strip from the end to land the prefix right before the answer:
    # the option tokens, plus the chat template's tail — see _ct_strip (default = measured tail,
    # correct; EVAL_CLOSED_POS_LEGACY=1 = the old off-by-one used by stored results). The gemma-3n
    # native path appends no turn-end token.
    n_strip = len(opt0_ids) + (_ct_strip(processor) if has_ct else 0)

    text = prompt_text.replace("<image>", "") + "\nAnswer: " + opt0
    if has_ct:  # medgemma-IT path (chat template)
        content = ([{"type": "image", "image": image}] if image is not None else [])
        content.append({"type": "text", "text": text})
        full_inp = processor.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=False,
            tokenize=True, return_tensors="pt", return_dict=True,
        )
    else:  # gemma-3n native path (no chat template; processor expands image token)
        itok = getattr(processor, "image_token", None) or getattr(processor, "boi_token", "<image_soft_token>")
        ntext = (itok + text) if image is not None else text
        full_inp = processor(text=[ntext],
                             images=([[image]] if image is not None else None),
                             return_tensors="pt")
    full_len = full_inp["input_ids"].shape[1]
    prefix_len = full_len - n_strip

    seq_keys = [k for k in full_inp.keys()
                if full_inp[k].ndim >= 2 and full_inp[k].shape[-1] == full_len]
    prefix_inp = {k: full_inp[k][:, :prefix_len] for k in seq_keys}
    if "pixel_values" in full_inp:
        prefix_inp["pixel_values"] = full_inp["pixel_values"]
    prefix_inp = {k: v.to(device) for k, v in prefix_inp.items()}

    # Optional frozen-vision cache fast path (opt-in via EVAL_VISION_CACHE=1; default OFF =
    # byte-for-byte the original pixel_values path below). The SigLIP vision tower is ~85%
    # of per-row cost and is FROZEN across checkpoints, so its pooled output per image is
    # cached once (shared/vision_cache.py). On a cache HIT we skip the tower: build
    # inputs_embeds ourselves (embed input_ids + scatter the cached, per-ckpt-projected
    # image features into the image-token slots, EXACTLY as gemma3's forward does) and
    # forward with inputs_embeds and NO pixel_values. token_type_ids stays in `prefix_inp`
    # so the bidirectional image attention mask is built identically; because
    # past_key_values is None here, that mask's `is_first_iteration` is True regardless of
    # pixel_values, so dropping pixel_values is bit-identical. Any miss / cache-off /
    # image-less / no-path row falls back to the unchanged pixel_values path.
    cached_pooled = None
    if (os.environ.get("EVAL_VISION_CACHE") == "1" and rel_path is not None
            and "pixel_values" in prefix_inp):
        import vision_cache  # noqa: E402 (lazy: only when cache enabled)
        cached_pooled = vision_cache.load_entry(rel_path)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        if cached_pooled is not None:
            inputs_embeds = _inputs_embeds_with_cached_vision(
                model, prefix_inp["input_ids"], cached_pooled, device)
            kw = {k: v for k, v in prefix_inp.items()
                  if k not in ("input_ids", "pixel_values")}
            prefix_out = model(inputs_embeds=inputs_embeds, use_cache=True, **kw)
        else:
            prefix_out = model(**prefix_inp, use_cache=True)
    kv = prefix_out.past_key_values
    if hasattr(kv, "get_seq_length"):
        kv_len = kv.get_seq_length()
    else:
        kv_len = kv[0][0].shape[2]
    first_logits = prefix_out.logits[0, -1, :]

    scores: list[float] = []
    for opt in options:
        opt_ids = _answer_ids(processor, opt)  # in-context (space-prefixed) form
        n_opt = len(opt_ids)
        if n_opt == 0:
            scores.append(-math.inf)
            continue
        lp_sum = torch.log_softmax(first_logits.float(), dim=-1)[opt_ids[0]].item()
        if n_opt > 1:
            ext_ids = torch.tensor([opt_ids[:-1]], dtype=torch.long, device=device)
            pos_ids = torch.arange(kv_len, kv_len + n_opt - 1, device=device).unsqueeze(0)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                ext_out = model(input_ids=ext_ids, past_key_values=kv,
                                position_ids=pos_ids, use_cache=False)
            for j in range(1, n_opt):
                lp_sum += torch.log_softmax(ext_out.logits[0, j - 1].float(), dim=-1)[opt_ids[j]].item()
        scores.append(lp_sum / n_opt)
    return scores


def _match_class(text: str, options: list[str],
                 synonyms: dict[str, str] | None = None) -> str:
    """Map free-generated text to the nearest candidate class: the class whose
    name appears EARLIEST as a case-insensitive substring. Returns '<unparsed>'
    if none matches — that is a WRONG prediction, NOT a dropped row (it stays in
    the metric denominator). Mirrors tmp/u4_gen_rescore.py's verified mapper.

    `synonyms` ({normalised surface form -> canonical option}, from
    `u4_axis.eval_surface_forms`) is supplied ONLY for eval units the gold side
    coarsens with a clinical synonym table. It exists because the two sides used
    different vocabularies: gold folded HSIL / ASC-H / SCC into
    `high_grade_or_worse`, while this matcher demanded that underscore-coded
    bucket name verbatim, so a clinically correct "HSIL" scored `<unparsed>`.
    cervix was the only non-zero out-of-vocabulary cell in the benchmark
    (253/851 = 29.7% external) for exactly this reason.

    When `synonyms` is empty or None the original literal path runs UNCHANGED, so
    every non-coarsened unit is bit-identical to before. When it is supplied,
    matching happens on the `_norm`-ed text (underscores/hyphens -> spaces) so a
    generated "ASC-H" meets the gold table's "asc h"; the map already contains the
    canonical bucket names, so echoing the prompt verbatim still matches."""
    t = (text or "").lower()
    if synonyms:
        tn = re.sub(r"\s+", " ", re.sub(r"[_\-]+", " ", t)).strip()
        hits = [(tn.find(k), v) for k, v in synonyms.items() if k in tn]
        return min(hits)[1] if hits else "<unparsed>"
    hits = [(t.find(c.lower()), c) for c in options if c.lower() in t]
    return min(hits)[1] if hits else "<unparsed>"


@torch.no_grad()
def score_generate_hybrid(model, processor, image, prompt_text: str,
                          options: list[str], device,
                          synonyms: dict[str, str] | None = None,
                          rendered_options: list[str] | None = None):
    """Hybrid U4 scorer for heterogeneous-option pillars (see U4_GEN_PILLARS).
    ONE greedy generation pass yields BOTH outputs the metrics need:
      • pred_class  — decode the model's answer and map it to the nearest
        candidate class (drives accuracy / support-weighted macro-F1, the U4
        floor metric). Avoids the closed-set length-norm collapse entirely.
      • first_scores — per-option log P(option's first token) read off the
        generation's FIRST decoding step (out.scores[0]). softmax(first_scores)
        is the class-probability used for the threshold-free macro-OvO AUROC;
        this is the natural model-turn first-token distribution, which does not
        collapse. Persisted to scores_json aligned to `options` (== classes_json),
        so _aggregate_pair's existing U4 AUROC path consumes it unchanged.
    `rendered_options` (prompt-stem version >= 2 only; see u4_axis.surface_options):
    the surface forms the PROMPT showed, aligned 1:1 with `options`. The first-token
    id for the OvO companion is read from the rendered form (what the model will
    actually generate), while `options` stays canonical for classes_json and
    _match_class. None (the default, version 1) → identical to before.

    Returns (pred_class, first_scores, raw_text). No extra forward beyond generate
    (the first-token logits are a by-product of the same decode).
    """
    content = ([{"type": "image", "image": image}] if image is not None else [])
    content.append({"type": "text", "text": prompt_text.replace("<image>", "")})
    enc = processor.apply_chat_template(
        [{"role": "user", "content": content}], add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt").to(device)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model.generate(**enc, max_new_tokens=U4_GEN_MAX_NEW, do_sample=False,
                             output_scores=True, return_dict_in_generate=True)
    gen_ids = out.sequences[0, enc["input_ids"].shape[1]:]
    raw_text = processor.decode(gen_ids, skip_special_tokens=True)
    pred_class = _match_class(raw_text, options, synonyms)
    # out.scores[0] is [batch, vocab] logits for the first generated token == the model-turn
    # first-token distribution, so each option's id must be its MODEL-TURN form, with no leading
    # space. Using the space-prefixed _answer_ids here read ids the model never emits and made
    # every value in scores_json meaningless; fixed 2026-09-19, see _model_turn_answer_ids.
    lsm0 = torch.log_softmax(out.scores[0][0].float(), dim=-1)
    _first_src = rendered_options if (rendered_options and len(rendered_options) == len(options)) else options
    first_scores = [lsm0[_model_turn_answer_ids(processor, opt)[0]].item() for opt in _first_src]
    return pred_class, first_scores, raw_text


def score_options_build(processor, records):
    """CPU-only half of the batched scorer: build the padded processor inputs
    (apply_chat_template on a LIST → correct Gemma3 pan-and-scan crop↔token alignment).
    Returns CPU tensors (NOT moved to device) so it can run in a prefetch worker thread
    while the GPU is busy with the previous batch. records: list of (image, prompt, options).
    """
    msgs = []
    for image, prompt, options in records:
        content = []
        if image is not None:
            content.append({"type": "image", "image": image})
        content.append({"type": "text",
                        "text": prompt.replace("<image>", "") + "\nAnswer: " + options[0]})
        msgs.append([{"role": "user", "content": content}])
    return processor.apply_chat_template(
        msgs, add_generation_prompt=False, tokenize=True,
        return_tensors="pt", return_dict=True, padding=True,
    )


@torch.no_grad()
def score_options_forward(model, processor, inputs, records, device, rel_paths=None):
    """GPU half: forward the prebuilt `inputs` and gather SINGLE-TOKEN option logprobs per
    record (bit-exact-construction with score_options; bf16 batched matmul still differs at
    the ~bf16 level from bz=1). Returns per-record score lists; None for any record with a
    multi-token option, so the caller falls back to per-record score_options (e.g. U4).

    Frozen-vision cache (EVAL_VISION_CACHE=1 + rel_paths given): if EVERY row in the batch
    has a cached pooled tensor, skip the SigLIP tower — build inputs_embeds from the cached,
    per-ckpt-projected features (same _inputs_embeds_with_cached_vision as the per-record
    path, which handles multi-crop and is bit-verified) and forward with NO pixel_values.
    A crop-count guard (cached image-token count must equal the built input's image-token
    count) plus any miss falls back to the unchanged pixel_values forward — never silently
    wrong. Result is bit-identical to the pixel_values batched forward (same frozen features)."""
    tok = processor.tokenizer
    has_ct = _has_chat_template(processor)
    inputs = inputs.to(device)
    cached_cat = None
    # OPT-IN (EVAL_VISION_CACHE_BATCH=1) until the bit-exact gate below is run and passes,
    # so an already-running matrix job (which sets only EVAL_VISION_CACHE=1) keeps the old
    # pixel_values forward and is never touched by this unverified path. Flip the default to
    # "1" (track with EVAL_VISION_CACHE) once cached-batched == plain-batched is verified ~0.
    if (os.environ.get("EVAL_VISION_CACHE") == "1"
            and os.environ.get("EVAL_VISION_CACHE_BATCH", "0") == "1"
            and rel_paths and "pixel_values" in inputs and "input_ids" in inputs):
        import vision_cache  # noqa: E402 (lazy: only when cache enabled)
        loaded = [vision_cache.load_entry(rp) if rp else None for rp in rel_paths]
        if len(loaded) == inputs["input_ids"].shape[0] and all(x is not None for x in loaded):
            cat = torch.cat([x for x in loaded], dim=0)   # [sum_crops, mm_tokens, H]
            img_id = model.config.image_token_id
            n_img_tok = int((inputs["input_ids"] == img_id).sum())
            if cat.shape[0] * cat.shape[1] == n_img_tok:   # crop-count guard
                cached_cat = cat
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        if cached_cat is not None:
            inputs_embeds = _inputs_embeds_with_cached_vision(
                model, inputs["input_ids"], cached_cat, device)
            kw = {k: v for k, v in inputs.items()
                  if k not in ("input_ids", "pixel_values")}
            out = model(inputs_embeds=inputs_embeds, use_cache=False, **kw)
        else:
            out = model(**inputs, use_cache=False)
    attn = inputs["attention_mask"]
    ar = torch.arange(attn.shape[1], device=attn.device)
    last_real = (attn * ar).argmax(1)  # last real-token index per row (padding-side agnostic)
    res = []
    for b, (image, prompt, options) in enumerate(records):
        # position whose logits predict the first answer token == the "Answer: " token,
        # i.e. last_real - (len(opt0_tokens) + template tail); == score_options' logits[0,-1].
        # Options tokenized space-prefixed (in-context form) — see _answer_ids. The chat
        # template (medgemma) appends <end_of_turn> + "\n" after the answer (2 tokens); _ct_strip
        # uses the measured 2 by default, and the legacy +1 (stored results) only under
        # EVAL_CLOSED_POS_LEGACY=1; the gemma-3n native build appends no
        # turn-end token (+0) — mirror score_options' n_strip.
        n_strip = len(_answer_ids(processor, options[0])) + (_ct_strip(processor) if has_ct else 0)
        lsm = torch.log_softmax(out.logits[b, int(last_real[b]) - n_strip].float(), dim=-1)
        sc, single = [], True
        for opt in options:
            oid = _answer_ids(processor, opt)
            if len(oid) == 1:
                sc.append(float(lsm[oid[0]]))
            else:
                single = False
                break
        res.append(sc if single else None)
    return res


@torch.no_grad()
def score_options_batch(model, processor, records, device, rel_paths=None):
    """Batched scorer (build + forward in one call). See score_options_build / _forward.
    Single-token (binary) fast path; returns None per record for multi-token options.
    rel_paths (per-record manifest image paths) enables the frozen-vision cache fast path."""
    return score_options_forward(model, processor,
                                 score_options_build(processor, records), records, device,
                                 rel_paths=rel_paths)


@torch.no_grad()
def score_options_batch_g3n(model, processor, records, device):
    """Batched binary scorer for gemma-3n (no chat template).

    CRITICAL: unlike medgemma, gemma-3n's logits at a given position DEPEND on
    later tokens (its sliding-window / bidirectional-image attention mask is
    built from the full sequence length), so the medgemma trick of appending
    the answer and reading back N positions DIVERGES from the verified bz=1
    scorer (empirically Δmargin up to ~18). The verified score_options instead
    reads logits of the PREFIX (answer stripped). So we batch the PREFIXES:
    build each record's full (image-token + prompt + "Answer: "+opt0), slice off
    the opt0 tokens to land on the shared prefix, RIGHT-pad the prefixes to a
    common length (real tokens stay at positions 0..L-1, == per-record), forward
    once, and read logits at each row's last real token. gemma-3n's vision tower
    is fixed-size (256 soft tokens/image) so cross-image batching is alignment-
    safe. Single-token (binary) options only; returns None per record for any
    multi-token option (caller falls back to per-record score_options, e.g. U4).
    Verified to match score_options within bf16 noise (see tmp/verify_g3n_batch.py).
    """
    itok = getattr(processor, "image_token", None) or \
        getattr(processor, "boi_token", "<image_soft_token>")
    pad_id = processor.tokenizer.pad_token_id or 0
    builts, maxlen = [], 0
    for image, prompt, options in records:
        text = prompt.replace("<image>", "") + "\nAnswer: " + options[0]
        ntext = (itok + text) if image is not None else text
        fi = processor(text=[ntext],
                       images=([[image]] if image is not None else None),
                       return_tensors="pt")
        n_strip = len(_answer_ids(processor, options[0]))  # g3n: no <end_of_turn>
        plen = fi["input_ids"].shape[1] - n_strip
        builts.append((fi, plen))
        maxlen = max(maxlen, plen)
    B = len(records)
    has_tt = "token_type_ids" in builts[0][0]
    input_ids = torch.full((B, maxlen), pad_id, dtype=torch.long)
    attn = torch.zeros((B, maxlen), dtype=torch.long)
    tt = torch.zeros((B, maxlen), dtype=torch.long) if has_tt else None
    pix = []
    for b, (fi, plen) in enumerate(builts):
        input_ids[b, :plen] = fi["input_ids"][0, :plen]
        attn[b, :plen] = 1
        if has_tt:
            tt[b, :plen] = fi["token_type_ids"][0, :plen]
        if "pixel_values" in fi:
            pix.append(fi["pixel_values"])
    inp = {"input_ids": input_ids.to(device), "attention_mask": attn.to(device)}
    if has_tt:
        inp["token_type_ids"] = tt.to(device)
    if pix:
        inp["pixel_values"] = torch.cat(pix, 0).to(device)
    # use_cache=True (NOT False): gemma-3n constructs its sliding-window / image
    # bidirectional attention mask differently on the cached vs non-cached path,
    # so the SAME prefix yields different logits (Δmargin ~4). The verified bz=1
    # score_options uses use_cache=True; match it exactly or scores diverge.
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(**inp, use_cache=True)
    ar = torch.arange(maxlen, device=device)
    last_real = (attn.to(device) * ar).argmax(1)  # = plen-1 (right-padded)
    res = []
    for b, (image, prompt, options) in enumerate(records):
        lsm = torch.log_softmax(out.logits[b, int(last_real[b])].float(), dim=-1)
        sc, single = [], True
        for opt in options:
            oid = _answer_ids(processor, opt)
            if len(oid) == 1:
                sc.append(float(lsm[oid[0]]))
            else:
                single = False
                break
        res.append(sc if single else None)
    return res


# ─────────────────────────────────────────────────────────────────────────────
# CSV resume + write
# ─────────────────────────────────────────────────────────────────────────────


def _csv_path(pillar: str, task: str, shard: tuple[int, int] | None = None) -> Path:
    """Per-shard CSV path. Each shard writes its own file to avoid races;
    ``_aggregate_pair`` concats all shards' files. ``shard=None`` returns
    the canonical single-process path."""
    if shard is None:
        return OUT_DIR / f"{pillar}_{task}.csv"
    return OUT_DIR / f"{pillar}_{task}_s{shard[0]}of{shard[1]}.csv"


def _all_shard_csvs(pillar: str, task: str) -> list[Path]:
    """All per-shard CSVs for a (pillar, task), used at aggregation time.
    Returns the canonical single-file path if it exists, else any
    ``<pillar>_<task>_sIofN.csv`` shards on disk."""
    canonical = OUT_DIR / f"{pillar}_{task}.csv"
    if canonical.exists():
        return [canonical]
    return sorted(OUT_DIR.glob(f"{pillar}_{task}_s*of*.csv"))


def _completed_paths(csv_path: Path) -> set[str]:
    """Return set of image_paths already in the CSV (skip on resume)."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()
    try:
        df = pd.read_csv(csv_path, usecols=["image_path"], low_memory=False)
        return set(df["image_path"].astype(str).tolist())
    except Exception as exc:
        log_event("csv_read_error", path=str(csv_path), error=str(exc))
        return set()


def _open_csv_writer(csv_path: Path, fieldnames: list[str]):
    is_new = (not csv_path.exists()) or csv_path.stat().st_size == 0
    f = open(csv_path, "a", newline="", buffering=1)  # line-buffered
    w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    if is_new:
        w.writeheader()
    return f, w


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate metrics (per pair, at end-of-pair)
# ─────────────────────────────────────────────────────────────────────────────


def _weighted_f1(golds, preds) -> float | None:
    """Support-weighted macro-F1 (PanDerm-style multiclass primary metric)."""
    from collections import Counter
    n = len(golds)
    if n == 0:
        return None
    cg = Counter(golds)
    tot = 0.0
    for c in set(golds):
        tp = sum(1 for g, p in zip(golds, preds) if g == c and p == c)
        fp = sum(1 for g, p in zip(golds, preds) if g != c and p == c)
        fn = sum(1 for g, p in zip(golds, preds) if g == c and p != c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        tot += (cg[c] / n) * f1
    return float(tot)


def _aggregate_pair(pillar: str, task: str) -> dict | None:
    paths = _all_shard_csvs(pillar, task)
    if not paths:
        # No shard CSVs (dir whose shards were already consolidated): fall back to
        # the canonical per-pair CSV so --aggregate_only can recompute metrics
        # (e.g. a CI-method refresh) from the persisted rows, no re-inference.
        canonical = _csv_path(pillar, task)
        paths = [canonical] if canonical.exists() else []
    if not paths:
        return None
    parts = []
    for p in paths:
        try:
            parts.append(pd.read_csv(p, low_memory=False))
        except Exception as exc:
            log_event("aggregate_read_error", path=str(p), error=str(exc))
    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True)
    # If sharding produced row duplication across resumes, dedup by image_path.
    if "image_path" in df.columns:
        df = df.drop_duplicates(subset=["image_path"], keep="last")
    if len(df) == 0:
        return None
    sys.path.insert(0, str(ROOT / "preliminary" / "eval"))
    from eval_metrics import (  # noqa: E402
        auroc_binary, auroc_ovo_macro, auroc_ovr_per_class, macro_f1,
        softmax_from_scores, delong_ci, stratified_bootstrap_ci, bootstrap_auroc_ci,
    )

    out = {"pillar": pillar, "task": task, "n_total": int(len(df))}
    df = df[df["parse_ok"] == True]  # noqa: E712
    out["n_parsed"] = int(len(df))
    out["parse_failure_rate"] = 1.0 - out["n_parsed"] / max(1, out["n_total"])

    # U4 gold is pre-normalized by u4_axis primary-eval policy
    # (gold_label = primary option label; u4_eval_unit = scoring group).
    # Fine/native u4_unit may still be present for audit, but is not the default
    # metric axis.
    for split in ("test_id", "test_ood"):
        sub = df[df["split"] == split]
        if len(sub) == 0:
            continue
        block: dict = {"n": int(len(sub))}

        # Pick the "gold" column based on task.
        gold_col = "gold_label"
        pred_col = "predicted_parsed" if task in ("U1", "U2", "U3") else "predicted_class"

        # Accuracy & macro-F1
        if gold_col in sub.columns and pred_col in sub.columns:
            block["accuracy"] = float((sub[gold_col] == sub[pred_col]).mean())
            # Canonicalise BEFORE stringifying. `accuracy` above compares the raw columns, where
            # pandas coerces across dtypes (0.0 == 0 -> True). macro_f1 compares STRINGS, and when a
            # gold column is stored float64 while the prediction is int64 the string forms are "0.0"
            # and "0" — no class ever matches and macro_f1 comes out exactly 0.0 for a cell whose
            # accuracy is 0.87. That silently zeroed 9 (pillar, task) cells across every eval
            # directory (blood/breast/cervix/skin U1-U3; skin/U1 escaped only because its gold column
            # happened to be int64), and one of those zeros was once read as an accuracy and reported
            # as a model collapse. Integral numerics collapse to a plain int string; everything else
            # falls back to str().
            def _canon(v):
                if pd.isna(v):
                    return "_"
                try:
                    f = float(v)
                except (TypeError, ValueError):
                    return str(v)
                return str(int(f)) if f.is_integer() else repr(f)
            golds = [_canon(v) for v in sub[gold_col]]
            preds = [_canon(v) for v in sub[pred_col]]
            labels = sorted({g for g in golds} | {p for p in preds})
            block["macro_f1"] = float(macro_f1(golds, preds, labels))

        # Binary sensitivity / specificity (PanDerm: screening reports sensitivity)
        if task in ("U1", "U2", "U3"):
            _y = pd.to_numeric(sub["gold_label"], errors="coerce")
            _p = pd.to_numeric(sub["predicted_parsed"], errors="coerce")
            _m = _y.notna() & _p.notna()
            _yv, _pv = _y[_m].astype(int), _p[_m].astype(int)
            block["n_pos"], block["n_neg"] = int((_yv == 1).sum()), int((_yv == 0).sum())
            if (_yv == 1).any():
                block["sensitivity"] = float((_pv[_yv == 1] == 1).mean())
            if (_yv == 0).any():
                block["specificity"] = float((_pv[_yv == 0] == 0).mean())

        # AUROC — binary uses binary_score; multiclass uses scores_json + softmax.
        if task in ("U1", "U2", "U3") and "binary_score" in sub.columns:
            y = pd.to_numeric(sub["gold_label"], errors="coerce").fillna(-1).astype(int)
            s = pd.to_numeric(sub["binary_score"], errors="coerce")
            mask = (y >= 0) & s.notna()
            if mask.sum() > 10 and y[mask].nunique() == 2:
                block["auroc"] = auroc_binary(y[mask].tolist(), s[mask].tolist())
                # CI: patient-clustered bootstrap — unifies binary AUROC (U0-U3)
                # on one bootstrap method (replaces DeLong; U0 already bootstraps
                # by source_image in run_u0, U4 bootstraps its F1). Group by the
                # natural sampling unit patient_id (fall back image_path, else
                # per-row). The point AUROC above is unchanged; only the CI differs.
                _gcol = ("patient_id" if "patient_id" in sub.columns
                         else "image_path" if "image_path" in sub.columns else None)
                _grp = (sub.loc[mask, _gcol].fillna("_na").astype(str).tolist()
                        if _gcol else list(range(int(mask.sum()))))
                _boot = bootstrap_auroc_ci(y[mask].tolist(), s[mask].tolist(), _grp)
                block["auroc_ci95_bootstrap_grouped"] = {
                    "low": _boot["low"], "high": _boot["high"],
                    "n_boot": _boot["n_boot"], "group_by": _gcol or "row",
                }
                # TODO(imbalance-metrics, 2026-05-19): wire in MCC / PR-AUC /
                # balanced accuracy from preliminary/.../eval_metrics.py once
                # we decide to refresh metric JSONs. Pulling these in needs
                # no re-inference — scores_json + gold_label are persisted.
                # Sketch:
                #   from eval_metrics import (mcc_binary, average_precision_binary,
                #                             balanced_accuracy_binary)
                #   block["mcc"]             = mcc_binary(y[mask].tolist(),
                #                                         (s[mask] > 0).astype(int).tolist())
                #   block["ap_positive"]     = average_precision_binary(
                #                                  y[mask].tolist(), s[mask].tolist(),
                #                                  positive_class=1)
                #   block["ap_negative"]     = average_precision_binary(
                #                                  y[mask].tolist(), s[mask].tolist(),
                #                                  positive_class=0)
                #   block["balanced_acc"]    = balanced_accuracy_binary(
                #                                  y[mask].tolist(),
                #                                  (s[mask] > 0).astype(int).tolist())
                # 暂时不测，等 paper writing 决定后一并刷出。
        elif task == "U4" and ("u4_eval_unit" in sub.columns or "u4_unit" in sub.columns):
            # Per primary eval unit: support-weighted F1 + accuracy.
            # Fine/native u4_unit is only a compatibility fallback.
            import json as _json
            import u4_axis
            per_unit: dict = {}
            unit_col = "u4_eval_unit" if "u4_eval_unit" in sub.columns else "u4_unit"
            for unit, ud in sub.groupby(sub[unit_col].fillna("_none")):
                if unit == "_none" or len(ud) == 0:
                    continue
                g = ud["gold_label"].astype(str).tolist()
                p = ud["predicted_class"].astype(str).tolist()
                ub: dict = {
                    "n": int(len(ud)),
                    "accuracy": float((ud["gold_label"].astype(str) == ud["predicted_class"].astype(str)).mean()),
                    "weighted_f1": _weighted_f1(g, p),
                    "macro_f1": float(macro_f1(g, p, sorted(set(g) | set(p)))),
                }
                # macro-OvO AUROC (Hand & Till 2001): probabilistic, threshold-free,
                # uses the full per-class score vector (softmax of scores_json, aligned
                # to this unit's option set) rather than the argmax. Complements
                # weighted-F1 (which is argmax-based). None if <2 gold classes present.
                cls_u = []
                if "classes_json" in ud.columns:
                    try:
                        cls_u = _json.loads(next(v for v in ud["classes_json"].dropna().astype(str) if v))
                    except Exception:
                        cls_u = []
                if not cls_u:
                    cls_u = u4_axis.primary_options(str(unit)) or u4_axis.options(str(unit))
                if cls_u and "scores_json" in ud.columns:
                    try:
                        S = np.array([_json.loads(s) for s in ud["scores_json"]], dtype=float)
                        if S.ndim == 2 and S.shape[1] == len(cls_u):
                            ovo = auroc_ovo_macro(g, softmax_from_scores(S), [str(c) for c in cls_u])
                            if ovo is not None:
                                ub["auroc_ovo_macro"] = float(ovo)
                    except Exception:
                        pass
                try:
                    ci = stratified_bootstrap_ci(
                        lambda pl: _weighted_f1(list(pl["g"]), list(pl["p"])),
                        payload={"g": np.array(g), "p": np.array(p)},
                        strata=g, B=1000, alpha=0.05, seed=20260527,
                    )
                    ub["weighted_f1_ci95_bootstrap"] = {
                        "low": ci.get("low"), "high": ci.get("high"), "n_boot": ci.get("n_boot")}
                except Exception:
                    pass
                per_unit[str(unit)] = ub
            block["per_unit"] = per_unit
            _wf = [v["weighted_f1"] for v in per_unit.values() if v.get("weighted_f1") is not None]
            if _wf:
                block["macro_weighted_f1_over_units"] = float(np.mean(_wf))
            _ovo = [v["auroc_ovo_macro"] for v in per_unit.values() if v.get("auroc_ovo_macro") is not None]
            if _ovo:
                block["macro_ovo_auroc_over_units"] = float(np.mean(_ovo))

        # Per-OOD-axis (test_ood only)
        if split == "test_ood" and "ood_axis" in sub.columns:
            ood_block = {}
            for ax, ax_df in sub.groupby("ood_axis"):
                if pd.isna(ax) or len(ax_df) < 5:
                    continue
                ax_block: dict = {"n": int(len(ax_df)),
                                  "accuracy": float((ax_df[gold_col] == ax_df[pred_col]).mean())}
                if task in ("U1", "U2", "U3") and "binary_score" in ax_df.columns:
                    y = pd.to_numeric(ax_df["gold_label"], errors="coerce").fillna(-1).astype(int)
                    s = pd.to_numeric(ax_df["binary_score"], errors="coerce")
                    m = (y >= 0) & s.notna()
                    if m.sum() > 5 and y[m].nunique() == 2:
                        ax_block["auroc"] = auroc_binary(y[m].tolist(), s[m].tolist())
                ood_block[str(ax)] = ax_block
            block["per_ood_axis"] = ood_block
        out[split] = block
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Prefetch — overlap producer-side image decode with consumer-side GPU forward
# ─────────────────────────────────────────────────────────────────────────────


def _prefetch(iterable, maxsize: int = 64):
    """Run the (decoding) producer generator on a background daemon thread feeding
    a bounded FIFO queue, so producer CPU work — image decode (load_cached →
    webp/DICOM), prompt render — overlaps with the consumer's processor-build +
    GPU forward. While the GPU scores batch N, the producer decodes batch N+1.
    This closes the documented gap below (decode previously ran inline on the
    consumer thread, leaving the B200 idle between batches).

    OUTPUT-IDENTICAL: one producer thread + a FIFO queue yields items in the EXACT
    order of the wrapped generator; only the *timing* of decode changes, never
    which rows are produced or their contents — the scoring / sharding / CSV path
    is untouched. Producer exceptions are re-raised in the consumer (same failure
    semantics as the inline loop). The bounded queue caps resident decoded images.
    On consumer early-exit (e.g. --limit break) the producer is signalled to stop
    and the queue drained so no thread hangs. Opt-out: EVAL_PREFETCH=0.
    """
    if os.environ.get("EVAL_PREFETCH", "1") == "0":
        yield from iterable
        return
    q: "queue.Queue" = queue.Queue(maxsize=maxsize)
    stop = threading.Event()

    def _put(payload) -> bool:
        # Block until enqueued, or the consumer signalled stop (early-break/error).
        # Used for BOTH data and the end/exception sentinels, so a momentarily full
        # queue can never DROP a sentinel — a dropped end/exception marker would
        # leave the consumer blocked on q.get() forever (codex review BLOCKER).
        while not stop.is_set():
            try:
                q.put(payload, timeout=0.5)
                return True
            except queue.Full:
                continue
        return False

    def _producer():
        try:
            for item in iterable:
                if stop.is_set() or not _put((False, item)):
                    return
        except BaseException as exc:  # surface producer failure to the consumer
            _put((True, exc))
            return
        _put((True, None))

    t = threading.Thread(target=_producer, daemon=True, name="eval-prefetch")
    t.start()
    try:
        while True:
            is_end, payload = q.get()
            if is_end:
                if payload is not None:
                    raise payload
                return
            yield payload
    finally:
        stop.set()
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass
        t.join(timeout=2.0)   # deterministic cleanup before the next run_pair


# ─────────────────────────────────────────────────────────────────────────────
# Per-pair run loop
# ─────────────────────────────────────────────────────────────────────────────


def run_pair(model, processor, device, pillar: str, task: str,
             limit: int | None = None,
             shard: tuple[int, int] | None = None,
             batch_size: int = 32) -> None:
    import u4_axis
    classes = _candidate_classes(pillar, task)   # U4: per-row primary option set used instead
    # Heterogeneous-option U4 pillars are scored by generation (label) + first-token
    # softmax (AUROC) instead of the collapse-prone closed-set mean — see U4_GEN_PILLARS.
    use_gen = (task == "U4" and pillar.lower() in U4_GEN_PILLARS)
    # HARD fail-fast guard: these U4 pillars have heterogeneous multi-token class
    # names and MODE-COLLAPSE under closed-set mean-logprob scoring (the length-norm
    # "longest-name-wins" failure documented at U4_GEN_PILLARS). They MUST be scored
    # by generation. A narrowed / empty EVAL_U4_GEN_PILLARS (e.g. exported as "" or
    # overridden to a subset by a driver) would silently route them to closed-set and
    # emit plausible-but-INVALID numbers. Refuse instead of producing them.
    # (Set EVAL_U4_ALLOW_CLOSED=1 only for a deliberate closed-vs-gen method study.)
    U4_MUST_GENERATE = {"skin", "oral", "blood", "cervix", "breast"}
    if (task == "U4" and pillar.lower() in U4_MUST_GENERATE and not use_gen
            and os.environ.get("EVAL_U4_ALLOW_CLOSED", "0") != "1"):
        raise RuntimeError(
            f"[U4-guard] pillar='{pillar}' would be scored CLOSED-SET mean-logprob, which "
            f"mode-collapses on its heterogeneous class names -> INVALID U4 numbers. "
            f"EVAL_U4_GEN_PILLARS='{os.environ.get('EVAL_U4_GEN_PILLARS', '<unset>')}' "
            f"resolved to gen-set={sorted(U4_GEN_PILLARS)}. Fix it to include "
            f"'{pillar.lower()}' (default is all five), or set EVAL_U4_ALLOW_CLOSED=1 "
            f"for a deliberate method-comparison study.")
    if task != "U4" and not classes:
        log_event("pair_skipped", pillar=pillar, task=task, reason="no candidate classes")
        return
    csv_path = _csv_path(pillar, task, shard=shard)
    completed = _completed_paths(csv_path)
    log_event("pair_start", pillar=pillar, task=task,
              n_classes=("per-u4_eval_unit" if task == "U4" else len(classes)),
              classes=("per-u4_eval_unit" if task == "U4" else classes), csv=str(csv_path),
              scorer=("generate+firsttoken" if use_gen else "closed-set"),
              shard=f"{shard[0]}/{shard[1]}" if shard else None,
              already_done=len(completed))

    fieldnames = [
        "image_path", "patient_id", "dataset_id", "split", "subsplit", "ood_axis",
        "u4_unit", "u4_label", "u4_eval_unit", "u4_eval_label", "u4_eval_tier",
        "gold_label", "raw_text", "predicted_parsed", "predicted_class", "parse_ok",
        "binary_score", "scores_json", "classes_json", "raw_argmax_class",
        "elapsed_ms",
    ]
    f, w = _open_csv_writer(csv_path, fieldnames)

    yielded = 0
    skipped_done = 0
    errors = 0
    t_pair_start = time.time()
    last_progress = time.time()

    def _emit(meta, label, cur_classes, scores, elapsed_ms, gen_pred=None, raw_text=None):
        nonlocal yielded
        argmax_idx = int(np.argmax(scores)) if scores else -1
        argmax_class = cur_classes[argmax_idx] if argmax_idx >= 0 else ""
        parse_ok = bool(scores) and all(math.isfinite(s) for s in scores)
        binary_score = ""
        predicted_parsed = ""
        # For the hybrid U4 gen path, the LABEL is generation's (gen_pred), while
        # `scores` are first-token logprobs kept only for AUROC — so predicted_class
        # must be gen_pred, NOT argmax(scores). argmax_class is still persisted as
        # raw_argmax_class (first-token argmax) for audit.
        pred_class = argmax_class
        if task in ("U1", "U2", "U3"):
            # Yes vs No: binary_score = logp(Yes) - logp(No)
            try:
                yes_i = cur_classes.index("Yes")
                no_i = cur_classes.index("No")
                binary_score = float(scores[yes_i] - scores[no_i])
                predicted_parsed = 1 if argmax_idx == yes_i else 0
            except (ValueError, IndexError):
                parse_ok = False
        elif task == "U4":
            if gen_pred is not None:
                pred_class = gen_pred
            predicted_parsed = pred_class
        w.writerow({
            "image_path": meta["image_path"],
            "patient_id": meta.get("patient_id"),
            "dataset_id": meta.get("dataset_id"),
            "split": meta.get("split"),
            "subsplit": meta.get("subsplit"),
            "ood_axis": meta.get("ood_axis"),
            "u4_unit": meta.get("u4_unit"),
            "u4_label": meta.get("u4_label"),
            "u4_eval_unit": meta.get("u4_eval_unit"),
            "u4_eval_label": meta.get("u4_eval_label"),
            "u4_eval_tier": meta.get("u4_eval_tier"),
            "gold_label": label,
            "raw_text": raw_text if raw_text is not None else "",
            "predicted_parsed": predicted_parsed,
            "predicted_class": pred_class,
            "raw_argmax_class": argmax_class,
            "parse_ok": parse_ok,
            "binary_score": binary_score,
            "scores_json": json.dumps([float(s) if math.isfinite(s) else None for s in scores]),
            "classes_json": json.dumps([str(c) for c in cur_classes], ensure_ascii=False),
            "elapsed_ms": round(elapsed_ms, 1),
        })
        yielded += 1

    # ── Batched scorer ──────────────────────────────────────────────────────────────
    # Buffer rows, then score each batch in one forward (effective batch size = batch_size)
    # so the B200 isn't idle (bz=1 left it ~0% util). Binary tasks (U1/U2/U3 — single-token
    # Yes/No options) use the batched fast path; U4 (multi-token options) is scored
    # per-record (batching returns None for every multi-token row, so we skip the wasted
    # batched forward — codex review caught this double-cost).
    # NOTE: producer-side image decode is now overlapped with the GPU forward via
    # _prefetch (background thread + bounded queue; the wrapped iter_eval_rows runs
    # ahead while this _flush scores the current batch). The webp/DICOM decode in
    # load_cached releases the GIL, so a single producer thread gives real overlap.
    # Still inline here (cheap, GIL-bound): the processor build (resize/tokenize)
    # inside score_options_batch — a further win would move that to the producer too.
    buffer: list[dict] = []

    def _flush():
        nonlocal errors, last_progress
        if not buffer:
            return
        t0 = time.time()
        batched: list = [None] * len(buffer)
        # Binary tasks (single-token Yes/No) use a batched fast path so the B200
        # isn't idle at bz=1. medgemma-IT uses the chat-template build; gemma-3n
        # (no chat template, FIXED-crop vision) uses the native batched build
        # (score_options_batch_g3n). U4 (multi-token options) stays per-record.
        # gemma-3n batched binary scorer is OPT-IN (EVAL_G3N_BATCH=1): it is
        # order-invariant and metric-accurate (Δacc 0, ΔAUROC ~0.003) but its
        # bf16 batched-matmul margin drift (mean ~0.24, max ~1.5) exceeds the
        # verified-reference tolerance, so official soundness runs leave it OFF
        # (per-record exact; fill the GPU via parallel --shard worker processes).
        _g3n_batch_on = bool(os.environ.get("EVAL_G3N_BATCH"))
        _use_batch = _has_chat_template(processor) or _g3n_batch_on
        if task != "U4" and _use_batch:
            recs = [(b["image"], b["prompt"], b["cur_classes"]) for b in buffer]
            try:
                if _has_chat_template(processor):   # medgemma: frozen-vision cache fast path
                    rel_paths = [b["meta"].get("image_path") for b in buffer]
                    batched = score_options_batch(model, processor, recs, device,
                                                  rel_paths=rel_paths)
                else:                               # gemma-3n native (no cache path yet)
                    batched = score_options_batch_g3n(model, processor, recs, device)
            except torch.cuda.OutOfMemoryError as exc:
                log_event("oom_batch", pillar=pillar, task=task, n=len(buffer), error=str(exc))
                torch.cuda.empty_cache()
                batched = [None] * len(buffer)
            except Exception as exc:
                log_event("batch_error", pillar=pillar, task=task, error=str(exc),
                          traceback=traceback.format_exc(limit=3))
                batched = [None] * len(buffer)
        per_ms = (time.time() - t0) * 1000.0 / max(1, len(buffer))
        last_image = buffer[-1]["meta"]["image_path"]
        for b, sc in zip(buffer, batched):
            gen_pred = None
            gen_raw = None
            try:
                if use_gen and sc is None:
                    # Hybrid: generation drives the label; scores = first-token
                    # logprobs (for AUROC). One decode, no extra forward.
                    # Coarsened eval units (cervix Bethesda) need the gold side's
                    # own synonym vocabulary on the prediction side too; every other
                    # unit gets {} and takes the untouched literal path.
                    # Prompt-stem version (default 1) rides along in meta from
                    # eval_iterator; under version >= 2 the synonym map also accepts
                    # the version-2 surface forms and the first-token ids come from
                    # the rendered option strings.
                    _stem_v = b["meta"].get("prompt_stem_version")
                    _syn = u4_axis.eval_surface_forms(
                        pillar, b["meta"].get("u4_eval_unit") or b["meta"].get("u4_unit"),
                        version=_stem_v,
                    ) if task == "U4" else {}
                    _rendered = (b["meta"].get("u4_eval_options_rendered")
                                 if (task == "U4" and _stem_v and int(_stem_v) >= 2) else None)
                    gen_pred, scores, gen_raw = score_generate_hybrid(
                        model, processor, b["image"], b["prompt"], b["cur_classes"], device,
                        synonyms=_syn, rendered_options=_rendered)
                else:
                    scores = sc if sc is not None else score_options(
                        model, processor, b["image"], b["prompt"], b["cur_classes"], device,
                        rel_path=b["meta"].get("image_path"))
            except torch.cuda.OutOfMemoryError as exc:
                log_event("oom", pillar=pillar, task=task,
                          image_path=b["meta"]["image_path"], error=str(exc))
                torch.cuda.empty_cache()
                errors += 1
                continue
            except Exception as exc:
                log_event("inference_error", pillar=pillar, task=task,
                          image_path=b["meta"]["image_path"], error=str(exc),
                          traceback=traceback.format_exc(limit=3))
                errors += 1
                continue
            _emit(b["meta"], b["label"], b["cur_classes"], scores, per_ms, gen_pred=gen_pred,
                  raw_text=gen_raw)
        buffer.clear()
        if time.time() - last_progress > 60.0:
            f.flush()
            rate = yielded / max(1.0, time.time() - t_pair_start)
            log_event("pair_progress", pillar=pillar, task=task,
                      done_this_session=yielded, errors=errors,
                      rate_per_sec=round(rate, 2), batch_size=batch_size,
                      last_image=last_image)
            last_progress = time.time()

    # notes §8: optionally shard BEFORE decode (inside iter_eval_rows) so each
    # shard decodes only its 1/N of images instead of all N shards decoding all.
    # Opt-in via EVAL_SHARD_BEFORE_DECODE (default off → identical post-yield shard).
    _shard_predecode = bool(os.environ.get("EVAL_SHARD_BEFORE_DECODE")) and shard is not None
    seen_rows = 0
    # Prefetch the decoding generator on a background thread so image decode
    # overlaps the GPU forward (output-identical; see _prefetch). maxsize ~1 batch
    # ahead is enough to hide forward latency without inflating RAM across shards.
    _rows = iter_eval_rows(
        pillar, task, split=os.environ.get("EVAL_SPLIT", "both"), template_idx=0, verbose=False,
        shard=(shard if _shard_predecode else None))
    try:
        for row, image, prompt, label, meta in _prefetch(_rows, maxsize=max(16, batch_size)):
            if (not _shard_predecode) and shard is not None and seen_rows % shard[1] != shard[0]:
                seen_rows += 1
                continue
            seen_rows += 1
            if limit is not None and yielded + len(buffer) >= limit:
                break
            if meta["image_path"] in completed:
                skipped_done += 1
                continue
            if image is None:
                log_event("image_load_failure", pillar=pillar, task=task,
                          image_path=meta["image_path"])
                errors += 1
                continue
            if task == "U4":
                cur_classes = (
                    meta.get("u4_eval_options")
                    or u4_axis.primary_options(meta.get("u4_eval_unit") or "")
                    or u4_axis.options(meta.get("u4_unit") or "")
                )
                if not cur_classes:
                    continue
            else:
                cur_classes = classes
            buffer.append({"image": image, "prompt": prompt, "cur_classes": cur_classes,
                           "meta": meta, "label": label})
            if len(buffer) >= batch_size:
                _flush()
        _flush()
    finally:
        f.close()

    rate = yielded / max(1.0, time.time() - t_pair_start)
    log_event("pair_done", pillar=pillar, task=task, processed_this_session=yielded,
              errors=errors, skipped_already_done=skipped_done,
              rate_per_sec=round(rate, 2),
              elapsed_sec=round(time.time() - t_pair_start, 1))
    if errors:
        global _RUN_DROPS
        _RUN_DROPS += errors   # surface dropped rows -> nonzero exit at run end

    # End-of-pair metrics — single-shard runs only. When sharded, each shard
    # has just its own slice; the canonical metrics get written by a separate
    # ``--aggregate_only`` pass after all shards finish.
    if shard is None:
        try:
            metrics = _aggregate_pair(pillar, task)
            if metrics is not None:
                (OUT_DIR / f"metrics_{pillar}_{task}.json").write_text(
                    json.dumps(metrics, indent=2, default=str)
                )
                log_event("pair_metrics_written", pillar=pillar, task=task,
                          summary={k: metrics.get(k) for k in ("n_total", "n_parsed", "parse_failure_rate")})
        except Exception as exc:
            log_event("metrics_error", pillar=pillar, task=task, error=str(exc),
                      traceback=traceback.format_exc(limit=3))


# ─────────────────────────────────────────────────────────────────────────────
# Model load + main
# ─────────────────────────────────────────────────────────────────────────────


# Real bnb 4-bit (NF4) support for the CLOSED path. Set by the medgemma-HF load
# block when a quantization_config is actually handed to from_pretrained; checked
# by the load_model() wrapper below.
_QUANT_APPLIED = False


def load_model():
    """Enforce EVAL_QUANT support, then delegate to the real loader.

    Why this wrapper exists: `EVAL_QUANT=nf4` (real bitsandbytes 4-bit) was
    accepted silently on EVERY closed load path and applied on NONE of them, so a
    run could emit bf16 numbers labelled NF4 — the same class of silent
    wrong-measurement as the U4 closed-set episode (PERF §14). It now (a) rejects
    the combination with the MatFormer width wrapper, since bitsandbytes packs the
    FFN weights and a nested prefix slice cannot be taken from packed data, and
    (b) fails loudly if the mode was requested but no load path consumed it.
    A no-op when EVAL_QUANT is unset.
    """
    global _QUANT_APPLIED
    mode = os.environ.get("EVAL_QUANT", "").strip().lower()
    if mode and os.environ.get("EVAL_MATFORMER_WIDTH"):
        raise ValueError(
            f"EVAL_QUANT={mode!r} is incompatible with EVAL_MATFORMER_WIDTH: bitsandbytes "
            "packs the FFN weights, so the nested prefix slice cannot be taken. Run real "
            "int4 at FULL width only (plain load, no EVAL_MATFORMER_WIDTH)."
        )
    _QUANT_APPLIED = False
    out = _load_model_inner()
    if mode and not _QUANT_APPLIED:
        raise RuntimeError(
            f"EVAL_QUANT={mode!r} was requested but no load path applied it — refusing to "
            "report unquantized numbers as quantized. Real 4-bit on the closed path is "
            "supported only for an HF-format medgemma/matformer checkpoint (EVAL_CKPT=<dir "
            "with config.json>, EVAL_ARCH unset or 'medgemma', no EVAL_MATFORMER_WIDTH)."
        )
    return out


def _load_model_inner():
    global _MATFORMER_CONTROLLER, _MATFORMER_REQUESTED_WIDTHS
    _MATFORMER_CONTROLLER = None
    _MATFORMER_REQUESTED_WIDTHS = _parse_matformer_widths()
    from transformers import AutoProcessor, AutoModelForImageTextToText
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Guard: width selection only has meaning for a matformer checkpoint.
    if os.environ.get("EVAL_MATFORMER_WIDTH") and not os.environ.get("EVAL_CKPT"):
        raise ValueError("EVAL_MATFORMER_WIDTH requires EVAL_CKPT=<matformer final dir>")

    # Aligned eval of a TRAINED full-param checkpoint: set EVAL_CKPT=<dir with
    # model.pt>. We rebuild the same matformer-medgemma (apply_lora=False) and load
    # the saved state_dict, then reuse this script's exact eval pipeline
    # (render_prompt + u4_axis options + score_options) — so the trained model is
    # scored identically to the baseline and to what the v1 corpus was built for.
    ckpt = os.environ.get("EVAL_CKPT")
    if ckpt:
        arch = os.environ.get("EVAL_ARCH", "medgemma")  # medgemma | gemma3n | matformer
        ckpt_dir = Path(ckpt)

        # --- gemma-3n elastic-width (MatFormer distill) Mix'n'Match eval -----
        # The current DFT/GKD/MiniLLM distill students co-train nested width
        # fracs (e.g. [0.125, 0.25, 0.5, 1.0]) via
        # distill_elastic.install_elastic_mlps — a
        # forward-only monkeypatch that prefix-slices the SwiGLU FFNs and adds
        # NO new params (so the saved state-dict keys == stock gemma-3n). To
        # score a sub-network we rebuild the SAME student (train_v2 loader),
        # install the elastic MLPs, overlay the trained state-dict, and select
        # the active frac. EVAL_ELASTIC_FRAC=<f> picks it; frac 1.0 == full E4B
        # (bit-identical to the stock forward). EVAL_ELASTIC_FRACS lists the
        # trained set (default "0.125,0.25,0.5,1.0", must include 1.0).
        efrac = os.environ.get("EVAL_ELASTIC_FRAC")
        if efrac is not None and arch == "gemma3n":
            efrac = float(efrac)
            fracs = [float(x) for x in
                     os.environ.get("EVAL_ELASTIC_FRACS", "0.125,0.25,0.5,1.0").split(",")]
            from cancerflex.train import load_model as _lm
            from cancerflex.train.distill_elastic import (
                ElasticWidthController, install_elastic_mlps)
            log_event("model_load_start", path=ckpt, device=device,
                      mode="elastic_frac", arch=arch, frac=efrac, fracs=fracs,
                      cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"))
            model, processor = _lm.load_gemma3n(
                vision_freeze=True, attn_impl=os.environ.get("EVAL_ATTN", "eager"))
            ctrl = ElasticWidthController(fracs=fracs)
            if efrac not in ctrl.fracs:
                raise ValueError(
                    f"EVAL_ELASTIC_FRAC={efrac} not in trained fracs {ctrl.fracs}")
            n_el = install_elastic_mlps(model, ctrl)
            state = torch.load(ckpt_dir / "model.pt", map_location="cpu")
            res = model.load_state_dict(state, strict=False)
            missing, unexpected = list(res.missing_keys), list(res.unexpected_keys)
            # The trained save is a full stock-keyed state-dict and the elastic
            # wrapper registers no new params, so ANY missing/unexpected key is
            # an arch mismatch that would silently corrupt the eval -> fail loud.
            if missing or unexpected:
                raise RuntimeError(
                    f"elastic load key mismatch: {len(missing)} missing "
                    f"(e.g. {missing[:3]}), {len(unexpected)} unexpected "
                    f"(e.g. {unexpected[:3]})")
            ctrl.set(efrac)
            model = model.to(device).eval()
            log_event("ckpt_loaded", ckpt=ckpt, arch=arch, fmt="pt",
                      elastic_frac=efrac, n_mlps_elastic=n_el,
                      n_params=sum(p.numel() for p in model.parameters()))
            return model, processor, device

        # --- Matformer-pure UNIFORM nested-width slice eval (NOT Mix'n'Match) -
        # controller.set(w) = one global FFN width across ALL layers; no per-layer
        # mixing is implemented. See docs/methodology/matformer_width_eval.md.
        # iter_02 matformer_pure trains nested-prefix-slice SwiGLU FFNs; the
        # saved `final/` is the full-width (xl) state-dict in standard Gemma3
        # layout. To score a sub-network, rebuild the wrapped (NestedGemma3MLP)
        # model, overlay the trained xl weights (identical param names), and set
        # the active intermediate width via the shared GranularityController.
        # EVAL_MATFORMER_WIDTH=<int>[,<int>...] selects one or more granularities
        # (e.g. 1280,10240); 10240 == full and is bit-identical to the plain HF
        # load. Multiple widths reuse this one wrapped model in main().
        if _MATFORMER_REQUESTED_WIDTHS:
            import sys
            mwidth = _MATFORMER_REQUESTED_WIDTHS[0]
            factor = int(os.environ.get("EVAL_MATFORMER_FACTOR", "8"))
            load_fields = dict(path=ckpt, device=device,
                               mode="matformer_pure_width", width=mwidth, factor=factor,
                               cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"))
            if len(_MATFORMER_REQUESTED_WIDTHS) > 1:
                load_fields["requested_widths"] = _MATFORMER_REQUESTED_WIDTHS
            log_event("model_load_start", **load_fields)
            # Load the trained xl checkpoint via from_pretrained — it is saved in
            # standard medgemma on-disk key layout, so HF performs the correct
            # key remapping (a raw load_state_dict onto a runtime-built model
            # mismatches `language_model.model.*` vs `model.language_model.*`).
            model = AutoModelForImageTextToText.from_pretrained(
                str(ckpt_dir),
                torch_dtype=torch.bfloat16,
                attn_implementation="eager",
            )
            # Now wrap each Gemma3MLP with a NestedGemma3MLP (copies the trained
            # weights in-place) and select the active intermediate width.
            PURE = str(ROOT / "preliminary" / "Med-Matformer-pure")
            if PURE not in sys.path:
                sys.path.insert(0, PURE)
            from cancerflex.matformer.model import wrap_medgemma_mlps
            from cancerflex.matformer.granularity import GranularityController
            text_cfg = model.config.text_config if hasattr(model.config, "text_config") else model.config
            # Source of truth = the granularity schedule training saved next to
            # the checkpoint; recompute from factor only if that file is absent
            # (avoids silently evaluating an untrained width on config drift).
            gran_path = ckpt_dir / "matformer_granularities.json"
            if not gran_path.exists():
                gran_path = ckpt_dir.parent / "matformer_granularities.json"
            if gran_path.exists():
                meta = json.loads(gran_path.read_text())
                controller = GranularityController(widths=[int(w) for w in meta["widths"]])
                gran_src = str(gran_path)
            else:
                controller = GranularityController.from_exponential_factor(
                    full_width=text_cfg.intermediate_size, factor=factor)
                gran_src = f"recomputed(factor={factor})"
            if controller.widths[-1] != text_cfg.intermediate_size:
                raise ValueError(
                    f"granularity full width {controller.widths[-1]} != model "
                    f"intermediate_size {text_cfg.intermediate_size}")
            invalid_widths = [w for w in _MATFORMER_REQUESTED_WIDTHS
                              if w not in controller.widths]
            if invalid_widths:
                raise ValueError(
                    f"width(s) {invalid_widths} not in {controller.widths} (from {gran_src})")
            n_wrapped = wrap_medgemma_mlps(model, controller)
            if n_wrapped == 0:
                raise RuntimeError("wrap_medgemma_mlps replaced 0 MLPs — arch mismatch")
            controller.set(mwidth)
            _MATFORMER_CONTROLLER = controller
            processor = AutoProcessor.from_pretrained(MODEL_PATH)
            model = model.to(device).eval()
            log_event("ckpt_loaded", ckpt=ckpt, arch="matformer_pure", width=mwidth,
                      widths=list(controller.widths), gran_src=gran_src,
                      n_mlps_wrapped=n_wrapped,
                      n_params=sum(p.numel() for p in model.parameters()))
            return model, processor, device

        # iter_02 (TRL SFTTrainer) saves `model.safetensors` + `config.json`
        # (HF-format); iter_01 saved `model.pt` (torch state-dict). Detect.
        is_hf_fmt = (ckpt_dir / "model.safetensors").exists() or \
                    (ckpt_dir / "model.safetensors.index.json").exists()
        log_event("model_load_start", path=ckpt, device=device, mode="ckpt", arch=arch,
                  fmt=("hf" if is_hf_fmt else "pt"),
                  cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"))
        if is_hf_fmt:
            # iter_02 path: load the trained model directly via HF.
            # We do not wrap with matformer — iter_02 direct-SFT trains a
            # vanilla Gemma3ForConditionalGeneration; matformer training
            # (subnet-sampled) saves a model that ALSO loads here as Gemma3
            # since matformer-medgemma extends it; the matmng flag stays at
            # default (full / xl).
            if arch == "gemma3n":
                # gemma-3n student ckpt (iter_02 distill stages).
                base = os.environ.get("EVAL_BASE", "google/gemma-3n-E4B")
                processor = AutoProcessor.from_pretrained(base)
            else:
                processor = AutoProcessor.from_pretrained(MODEL_PATH)
            # Real bnb 4-bit (NF4) quantize-on-load — NO-OP unless EVAL_QUANT=nf4
            # (scripts/eval/bnb_quant.py). bitsandbytes requires a device_map, so the
            # usual "load on CPU then .to(device)" is replaced by device_map="auto"
            # when quantizing. Mutually exclusive with EVAL_PTQ_BITS (fake RTN) — the
            # config builder raises on that combination — and with the MatFormer width
            # wrapper (rejected by the load_model() wrapper above).
            from bnb_quant import maybe_bnb_config_from_env, assert_real_4bit
            _qcfg = maybe_bnb_config_from_env()
            model = AutoModelForImageTextToText.from_pretrained(
                str(ckpt_dir),
                torch_dtype=torch.bfloat16,
                attn_implementation="eager",
                device_map=("auto" if _qcfg is not None else None),
                quantization_config=_qcfg,
            )
            if _qcfg is None:
                # from_pretrained without device_map leaves the model on CPU; the
                # scorer puts inputs on `device`, so move weights there too.
                model = model.to(device)
            else:
                global _QUANT_APPLIED
                _QUANT_APPLIED = True
                # Fail loudly if the load silently came back unquantized.
                log_event("closed_real_4bit", quant="nf4",
                          **assert_real_4bit(model))
        else:
            # iter_01 path: rebuild architecture, then overlay state_dict.
            from cancerflex.legacy_unreleased.common.checkpoint import load_checkpoint
            if arch == "gemma3n":
                from cancerflex.legacy_unreleased.strategy_a import load_gemma3n
                base = os.environ.get("EVAL_BASE", "google/gemma-3n-E4B")
                model, processor = load_gemma3n(base, vision_freeze=True, apply_lora=False)
                load_checkpoint(model, ckpt, strict=False)
            else:
                from cancerflex.legacy_unreleased.strategy_b import load_matformer_medgemma
                model, processor = load_matformer_medgemma(MODEL_PATH, apply_lora=False)
                load_checkpoint(model, ckpt, strict=False)
                cur = model
                if hasattr(cur, "base_model") and hasattr(cur.base_model, "model"):
                    cur = cur.base_model.model
                if hasattr(cur, "matmng"):
                    cur.matmng.set_flag(os.environ.get("EVAL_SUBNET", "xl"))
        model.eval()
        log_event("ckpt_loaded", ckpt=ckpt, arch=arch, fmt=("hf" if is_hf_fmt else "pt"),
                  n_params=sum(p.numel() for p in model.parameters()))
        return model, processor, device

    # Baseline (no checkpoint): pick the architecture via EVAL_ARCH env var.
    arch = os.environ.get("EVAL_ARCH", "medgemma")
    if arch == "gemma3n":
        # Gemma-3n native (full E4B by default; E2B is a separate HF ckpt → set EVAL_BASE).
        # Mix-and-Match arbitrary sub-networks require the Google matformer-lab tool — TODO.
        from cancerflex.legacy_unreleased.strategy_a import load_gemma3n
        base = os.environ.get("EVAL_BASE", "google/gemma-3n-E4B")
        log_event("model_load_start", path=base, device=device, mode="baseline", arch="gemma3n",
                  cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"))
        model, processor = load_gemma3n(base, vision_freeze=True, apply_lora=False)
        # Per-layer Mix'n'Match hook — NO-OP unless EVAL_GEMMA3N_PERLAYER is set.
        # Installs heterogeneous FFN-width prefix-slicing on the native MatFormer
        # (E4B) so run_closed exercises the per-layer path. See
        # scripts/eval/gemma3n_mnm.py + docs/methodology/gemma3n_matformer_control.md.
        import sys as _sys
        _evdir = str(ROOT / "scripts" / "eval")
        if _evdir not in _sys.path:
            _sys.path.insert(0, _evdir)
        from gemma3n_mnm import apply_perlayer_from_env as _apply_mnm
        _mnm_ctrl = _apply_mnm(model)
        if _mnm_ctrl is not None:
            log_event("gemma3n_perlayer_mnm",
                      spec=os.environ.get("EVAL_GEMMA3N_PERLAYER"),
                      summary=_mnm_ctrl.summary())
        model.eval()
        log_event("model_loaded", arch="gemma3n", base=base,
                  n_params=sum(p.numel() for p in model.parameters()))
        return model, processor, device
    if arch == "matformer_medgemma":
        # Med-Matformer Mix'n'Match baseline (medgemma wrapped with matformer adapters); pick the
        # sub-network width via EVAL_SUBNET ∈ {"s","m","l","xl"} (xl = full). Used to show
        # Mix-and-Match-style sub-network eval (the "matformer original" scheme) on medgemma.
        from cancerflex.legacy_unreleased.strategy_b import load_matformer_medgemma
        subnet = os.environ.get("EVAL_SUBNET", "xl")
        log_event("model_load_start", path=MODEL_PATH, device=device, mode="baseline",
                  arch="matformer_medgemma", subnet=subnet,
                  cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"))
        model, processor = load_matformer_medgemma(MODEL_PATH, apply_lora=False)
        cur = model
        if hasattr(cur, "base_model") and hasattr(cur.base_model, "model"):
            cur = cur.base_model.model
        if hasattr(cur, "matmng"):
            cur.matmng.set_flag(subnet)
        model.eval()
        log_event("model_loaded", arch="matformer_medgemma", subnet=subnet,
                  n_params=sum(p.numel() for p in model.parameters()))
        return model, processor, device

    log_event("model_load_start", path=MODEL_PATH, device=device,
              cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"))
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    try:
        from flash_attn import flash_attn_func  # noqa: F401
        attn_impl = "flash_attention_2"
    except (ImportError, OSError):
        attn_impl = "eager"
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, attn_implementation=attn_impl,
        device_map="auto",
    )
    model.eval()
    log_event("model_loaded", attn_impl=attn_impl, n_params=sum(p.numel() for p in model.parameters()))
    return model, processor, device


def main():
    global _RUN_DROPS
    ap = argparse.ArgumentParser()
    ap.add_argument("--pillars", default=",".join(DEFAULT_PILLARS))
    ap.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    ap.add_argument("--source", choices=["active", "sidecar"], default=U0_DEFAULT_SOURCE,
                    help="U0 data source when --tasks includes U0. 'active' reads v1_final JSONL; "
                         "'sidecar' preserves the legacy U0 QC sidecar behavior.")
    ap.add_argument("--limit_per_pair", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=32,
                    help="effective batch size for the closed-task scorer (processor-list "
                         "batched forward; binary single-token fast path, U4 falls back per-row). "
                         "Larger = more B200 utilization; the SigLIP vision tower runs eager "
                         "attention so very large batches (>~48) may OOM on the vision encoder.")
    ap.add_argument("--shard", default=None,
                    help="i/N — process only rows where row_idx%N==i (0-indexed)")
    ap.add_argument("--aggregate_only", action="store_true",
                    help="skip model loading; concat all per-shard CSVs into "
                         "the canonical <pillar>_<task>.csv and re-write "
                         "metrics_<pillar>_<task>.json from the merged data")
    args = ap.parse_args()
    # The frozen-vision embedding cache is keyed by image path, so it would silently score the
    # UNcropped image under EDGE_IMAGE_CROP_JSON (canonical_image._crop_box). Refuse the combo.
    if os.environ.get("EDGE_IMAGE_CROP_JSON") and os.environ.get("EVAL_VISION_CACHE") == "1":
        raise ValueError("EDGE_IMAGE_CROP_JSON is incompatible with EVAL_VISION_CACHE=1 "
                         "(path-keyed embedding cache bypasses the crop); unset the cache")

    pillars = [p.strip() for p in args.pillars.split(",") if p.strip()]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    requested_widths = _parse_matformer_widths()

    if len(requested_widths) > 1:
        if args.aggregate_only:
            raise ValueError(
                "multi-width --aggregate_only is unsupported; aggregate each width's "
                "persisted EVAL_OUT_DIR separately"
            )
        if "U0" in tasks:
            raise ValueError(
                "multi-width run_closed does not support U0 because run_u0 owns a separate "
                "output-dir global; launch U0 once per width instead"
            )
        width_out_dirs = _multi_width_out_dirs(requested_widths)
        _activate_out_dir(width_out_dirs[requested_widths[0]])
    else:
        width_out_dirs = {requested_widths[0]: OUT_DIR} if requested_widths else {}

    if args.aggregate_only:
        for task in tasks:
            for pillar in pillars:
                if task == "U0":
                    from scripts.eval import run_u0
                    metrics = run_u0._aggregate(pillar)
                    print(json.dumps(metrics, default=str))
                    continue
                paths = _all_shard_csvs(pillar, task)
                canonical = _csv_path(pillar, task)
                if not paths and not canonical.exists():
                    log_event("aggregate_only_no_shards", pillar=pillar, task=task)
                    continue
                # If only shard files exist, merge into canonical so resuming
                # / re-aggregation works against a single file going forward.
                if not canonical.exists():
                    parts = [pd.read_csv(p, low_memory=False) for p in paths]
                    merged = pd.concat(parts, ignore_index=True)
                    if "image_path" in merged.columns:
                        merged = merged.drop_duplicates(
                            subset=["image_path"], keep="last")
                    merged.to_csv(canonical, index=False)
                    log_event("aggregate_only_merged",
                              pillar=pillar, task=task,
                              n_shards=len(paths), n_rows=len(merged),
                              canonical=str(canonical))
                metrics = _aggregate_pair(pillar, task)
                if metrics is not None:
                    (OUT_DIR / f"metrics_{pillar}_{task}.json").write_text(
                        json.dumps(metrics, indent=2, default=str)
                    )
                    log_event("aggregate_only_metrics_written",
                              pillar=pillar, task=task)
        return

    shard = None
    if args.shard:
        i, N = args.shard.split("/")
        shard = (int(i), int(N))

    # Keep the historical run_start-before-load event (including its exact
    # fields) for no-width and scalar-width callers. Multi-width adds the width
    # identity because subsequent starts live in different log files.
    start_fields = dict(pillars=pillars, tasks=tasks,
                        limit_per_pair=args.limit_per_pair,
                        shard=f"{shard[0]}/{shard[1]}" if shard else None,
                        pid=os.getpid(),
                        cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"))
    if len(requested_widths) > 1:
        start_fields.update(matformer_width=requested_widths[0],
                            requested_widths=requested_widths)
    log_event("run_start", **start_fields)

    model, processor, device = load_model()
    # INT4 PTQ quantize-on-load hook (NO-OP unless EVAL_PTQ_BITS is set); see scripts/eval/ptq_hook.py
    from ptq_hook import maybe_quantize_from_env
    maybe_quantize_from_env(model)
    # Vision-tower input-resolution hook (NO-OP unless EVAL_VISION_RES is set); see
    # scripts/eval/vision_res_hook.py (P0 probe, docs/VISION_ENCODER_COMPRESSION_20261008.md).
    from vision_res_hook import maybe_set_vision_resolution_from_env
    maybe_set_vision_resolution_from_env(model, processor, log_event)

    # Graceful shutdown on SIGTERM.
    def _on_sigterm(signum, frame):
        log_event("sigterm_received", signum=signum)
        # Retain incrementally flushed per-pair CSVs for resume, but make the
        # scheduler/driver see termination as failure rather than completion.
        sys.exit(128 + signum)
    signal.signal(signal.SIGTERM, _on_sigterm)

    # One ordinary run when no width is requested; one in-place controller
    # selection per requested MatFormer width otherwise.  Each width keeps its
    # own raw CSV directory, so resume/dedup semantics remain width-local.
    width_runs = requested_widths or [None]
    for run_index, width in enumerate(width_runs):
        if width is not None:
            if _MATFORMER_CONTROLLER is None:
                raise RuntimeError(
                    "EVAL_MATFORMER_WIDTH did not load the MatFormer-pure controller")
            if len(width_runs) > 1:
                _activate_out_dir(width_out_dirs[width])
            if run_index > 0 and torch.cuda.is_available():
                torch.cuda.synchronize()
            _MATFORMER_CONTROLLER.set(width)

        drops_before_width = _RUN_DROPS
        if run_index > 0:
            log_event("run_start", **dict(start_fields, matformer_width=width))

        for task in tasks:
            for pillar in pillars:
                if (pillar, task) in CLOSED_DISABLED_PAIRS:
                    # By-design N/A (single-class / OOD-only) — do not materialize a cell.
                    log_event("pair_disabled", pillar=pillar, task=task,
                              reason="CLOSED_DISABLED_PAIRS (mirrors build_task_views.DISABLED_TASKS)")
                    continue
                try:
                    if task == "U0":
                        # U0 scoring is per-row independent, so it row-shards exactly
                        # like U1–U4: pass shard through so each shard scores its 1/N of
                        # rows into `<pillar>_U0_s{i}of{N}.csv` (canonical metrics come
                        # from a later --aggregate_only pass). shard=None keeps the
                        # single-process canonical path.
                        from scripts.eval import run_u0
                        print(json.dumps(run_u0.run_pillar_source(
                            model, processor, device, pillar, source=args.source,
                            limit=args.limit_per_pair, shard=shard), default=str))
                    else:
                        run_pair(model, processor, device, pillar, task,
                                 limit=args.limit_per_pair, shard=shard,
                                 batch_size=args.batch_size)
                except Exception as exc:
                    _RUN_DROPS += 1
                    log_event("pair_crash", pillar=pillar, task=task, error=str(exc),
                              traceback=traceback.format_exc(limit=5))

        if len(width_runs) > 1:
            log_event("run_done", matformer_width=width,
                      width_run_drops=_RUN_DROPS - drops_before_width,
                      run_drops=_RUN_DROPS)
        else:
            log_event("run_done", run_drops=_RUN_DROPS)

    if _RUN_DROPS > 0 and not args.aggregate_only:
        print(f"[run_closed] *** {_RUN_DROPS} row(s)/pair(s) DROPPED by inference "
              f"failures -> results PARTIAL; exiting nonzero ***", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
