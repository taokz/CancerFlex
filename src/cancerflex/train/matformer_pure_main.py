"""Multimodal MatFormer SFT on the *pure* implementation.

Model: preliminary/Med-Matformer-pure — nested prefix-slice SwiGLU FFNs over
MedGemma-1.5-4B-IT's language decoder. Vision tower untouched. NO neuron-rank
/ activation-selection surgery (that lives only in an earlier, unreleased
Med-Matformer and is deliberately excluded here). "Pure" refers to this
paper-faithful nested STRUCTURE; whether the training OBJECTIVE matches the
paper is a separate axis, set by `sampling` below.

Training objective (config `sampling`) — see
docs/methodology/matformer_training_alignment.md for the verified evidence:
  - "stochastic" = PAPER-ALIGNED MatFormer (arXiv 2310.07707 §3.2, Eq. 2, and
    the official devvrit/matformer repro): each OPTIMIZER step samples ONE
    granularity from {p_i} (`granularity_weights`, default uniform) and trains
    only that submodel. The width is derived deterministically from
    (seed, global_step), so it is constant across grad-accum microbatches
    (one width per gradient update, as Eq. 2 requires), identical on every
    DDP rank, and exactly reproducible across checkpoint resume.
  - "joint" (default) = MatFormer-OLMo-style, DynaBERT-family: every step
    forwards ALL granularities on the SAME batch, each contributing a
    weighted CE loss (`granularity_weights` = loss weights here); grads
    accumulate, then ONE optimizer step. This is what the
    RAIVNLab/MatFormer-OLMo release actually implements, but it is NOT the
    paper's stated objective — the paper (§2, App. B.4) attributes exactly
    this scheme to DynaBERT and credits sampling with "four times more
    gradient updates for the same amount of data and compute".
  ALL matformer runs through 2026-07-14 (incl. the definitive
  matformer_vanilla_a2sampler_deleaked) used "joint". Label results
  accordingly: joint-trained ≠ paper's MatFormer training.

DDP correctness:
  - joint: widths are visited ascending, so the LAST backward is at full width
    (xl) where every FFN param is in the graph. The first K-1 backwards run
    under `accelerator.no_sync(model)` to defer all-reduce; only the final xl
    backward syncs the fully-accumulated grad once. → correct + 1× comm, and
    `ddp_find_unused_parameters` can stay False.
  - stochastic: a single sub-xl backward leaves the unused FFN tail without
    grad → requires ddp_find_unused_parameters=True (set automatically).

Reuses train_v2 collator + dataset + step-based checkpointing (resume keeps
optimizer/scheduler/RNG/dataloader-cursor via TRL/HF Trainer).

Usage:
    torchrun --standalone --nproc_per_node=4 \
      -m cancerflex.train.matformer_pure_main \
      --config .../configs_v2/iter_02_matformer_pure_4gpu.yaml [--resume]
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import sys
from functools import lru_cache
from pathlib import Path

import yaml

from cancerflex.paths import ROOT  # noqa: E402  (set CANCERFLEX_ROOT)
sys.path.insert(0, str(ROOT))


def _resolve_resume(output_dir: Path, want_resume: bool):
    if not want_resume or not output_dir.exists():
        return None
    has_ckpt = any(p.name.startswith("checkpoint-") for p in output_dir.iterdir())
    return True if has_ckpt else None


def _constant_lr_lambda(_current_step: int) -> float:
    """Flat multiplier for the Router param group under `router_lr_scheduler: constant`.

    A module-level def rather than an inline lambda: it captures no closure state, so it cannot
    carry a stale config value into a resumed run, and it is identifiable by name when inspecting a
    scheduler. It is NOT serialised — torch 2.10 `LambdaLR.state_dict()` stores `None` for anything
    that is a `types.FunctionType` and only `fn.__dict__.copy()` for callable objects (which is why
    the audited run's `scheduler.pt` shows `lr_lambdas = [{}, {}]`: HF's cosine is a
    `functools.partial`). The flat schedule therefore survives a resume only because
    `create_scheduler` rebuilds it from `router_lr_scheduler` in the config — the same mechanism the
    backbone's own schedule relies on.
    """
    return 1.0


# ── Mix'n'Match-fix arms: pure helpers (CPU-testable, no torch) ────────────────
# These support the three ablation arms that CONTINUE-train ckpt-94568 to cover
# the deploy configs (narrow uniform widths + per-layer heterogeneous mixing) the
# stochastic (one-uniform-width-per-step) checkpoint never learned. See
# docs/methodology/mnm_fix_ablation.md.
DEFAULT_MIX_KIND_WEIGHTS = {
    # per-optimizer-step CONFIG-KIND probabilities (Arm A / C sampler). Chosen to
    # COVER every deploy config the stochastic ckpt collapses on while anchoring
    # the endpoints:
    #   uniform      – a single trained width on every layer (keeps narrow-uniform
    #                  1280/2560 robust; this is what stochastic already trained)
    #   full         – 10240 everywhere (anchor; guards against full-width regression)
    #   mixed_iid    – each layer i.i.d. over the 4 trained widths (max per-layer
    #                  heterogeneity coverage — the axis that collapsed at t*=0)
    #   mixed_banded – adjacent-band mix (lo,hi) with HI biased to later layers
    #                  (the Gemma-3n MatFormer-Lab prior: deeper layers want capacity)
    "uniform": 0.25, "full": 0.15, "mixed_iid": 0.30, "mixed_banded": 0.30,
}


def _sample_perlayer_config(seed, global_step, widths, weights, n_layers,
                            kind_weights=None, high_late_bias=True, salt=0):
    """Deterministic per-layer width config for the mixing-aware arms.

    Returns (kind, width_list) with len(width_list)==n_layers, every entry a
    TRAINED width. Derived from (seed, global_step, salt) as a PURE FUNCTION — NOT
    a stateful RNG — so, exactly like the stochastic width draw:
      * the config is constant across the grad-accum microbatches of one optimizer
        step (all share global_step) — a mixing config per gradient update;
      * identical on every DDP rank (global_step is), so all ranks build the SAME
        compute graph (mismatched per-rank widths would corrupt all-reduce);
      * reproducible across checkpoint resume (global_step is checkpointed).
    `salt` lets one step draw several independent configs (sandwich students).
    """
    kw = dict(kind_weights or DEFAULT_MIX_KIND_WEIGHTS)
    rng = random.Random((int(seed) << 28) ^ (0x9E3779B1 * int(global_step) + 0x2545 * int(salt) + 1))
    kinds = list(kw.keys())
    kvals = [kw[k] for k in kinds]
    kind = rng.choices(kinds, weights=kvals, k=1)[0]
    if kind == "full":
        return kind, [widths[-1]] * n_layers
    if kind == "uniform":
        return kind, [rng.choices(list(widths), weights=list(weights), k=1)[0]] * n_layers
    if kind == "mixed_iid":
        return kind, [rng.choices(list(widths), weights=list(weights), k=1)[0]
                      for _ in range(n_layers)]
    if kind == "mixed_banded":
        bi = rng.randrange(len(widths) - 1)      # adjacent granularity band
        lo, hi = widths[bi], widths[bi + 1]
        wl = []
        for i in range(n_layers):
            frac = i / max(1, n_layers - 1)
            p_hi = (0.15 + 0.70 * frac) if high_late_bias else 0.5
            wl.append(hi if rng.random() < p_hi else lo)
        return kind, wl
    raise ValueError(f"unknown mix kind {kind!r}")


def _contiguous_group_sizes(n_layers, n_groups):
    base, rem = divmod(n_layers, n_groups)
    return tuple(base + (1 if g >= n_groups - rem else 0) for g in range(n_groups))


@lru_cache(maxsize=None)
def _budget_group_pool(widths, n_layers, n_groups, budget, min_utilization, banded):
    """All heterogeneous group configs whose integer cost lies in one budget band.

    SCOPE — READ FIRST, and note this note was itself WRONG until 2026-08-26.

    This function is reachable only under `if per_layer_mixed: if budget_levels:`. With
    `budget_levels` unset the per-layer draw goes to `_sample_perlayer_config`, whose `uniform`
    kind emits an all-layers-equal config over all trained widths.

    THE DEPLOYED MODELS DO SET IT. `the released elastic checkpoint/
    resolved_config.yaml:27,33,34` sets `sandwich_budget_levels`,
    `sandwich_budget_min_utilization` and `sandwich_budget_groups`, and that run's `train.log:22`
    prints the `budget-stratified students:` banner, which is only reachable inside the branch.
    `elastic_backbone_v2corrected` sets the same keys. This note previously asserted the opposite
    on the strength of `eval_results/WIDTH_SAMPLING_MEASURED.json`, whose generator read
    `budget_levels` instead of `sandwich_budget_levels`, got None, and measured the inert branch.

    CONSEQUENCE, replayed seed-exactly over the deployed run's 23,642 steps and verified against
    both the audit JSON's schedule cells and all 1,183 logged `matformer/mean_width` values:
    of 47,284 intermediate draws, uniform 2,560 was drawn 0 times and uniform 5,120 0 times,
    because the pool builder below drops every all-groups-equal configuration. The only uniform
    configurations trained are 1,280 via `always_smallest` and 10,240 via the full-capacity
    anchor, both at every update. The two intermediate widths are still the most-trained widths
    at the layer level, 28.3% and 37.2% of all layer-group slots.

    Cached once per process: the production geometry is only 4^8 = 65,536 configs. `banded=True`
    restricts a config to one adjacent width pair and requires at least as much average index in
    the later half of the network as the earlier half.

    Two properties of this pool are easy to misread and both matter when interpreting results:

    * The band is [min_utilization * budget, budget], NOT the exact cost. At the production
      geometry and budget 0.375 the band holds 8,657 configs while only 874 cost exactly
      34 * 3840; an evaluation that pins exact cost is therefore scoring a 10% subset of what
      training sampled.
    * UNIFORM configs are excluded here (`len(set(idx)) == 1`). Combined with
      `always_smallest` supplying widths[0] and the teacher supplying widths[-1], that means a
      run using budget levels trains exactly TWO uniform configurations -- the narrowest and
      the full one. Every intermediate uniform width is reachable by prefix slicing but was
      never trained as a uniform configuration, so it is off-policy at evaluation time even
      when its width is one of the four the sampler used as a component.
    """
    widths = tuple(int(w) for w in widths)
    sizes = _contiguous_group_sizes(int(n_layers), int(n_groups))
    denom = sum(sizes) * widths[-1]
    lo = math.ceil(float(min_utilization) * float(budget) * denom - 1e-9)
    hi = math.floor(float(budget) * denom + 1e-9)
    pool = []
    for idx in itertools.product(range(len(widths)), repeat=n_groups):
        if len(set(idx)) == 1:
            continue
        cost = sum(sizes[g] * widths[k] for g, k in enumerate(idx))
        if not lo <= cost <= hi:
            continue
        if banded:
            if max(idx) - min(idx) != 1:
                continue
            mid = n_groups // 2
            if sum(idx[mid:]) / len(idx[mid:]) < sum(idx[:mid]) / len(idx[:mid]):
                continue
        pool.append(idx)
    return tuple(pool)


def _sample_budget_stratified_config(seed, global_step, student_index, n_random,
                                     widths, n_layers, n_groups, budget_levels,
                                     min_utilization):
    """One deterministic, legal heterogeneous config from a uniformly cycled budget level.

    Draw ids cycle through every budget exactly. Successive complete cycles alternate between the
    full feasible group-config pool and the adjacent-band/high-late pool, so with two students per
    step every five steps train each of five levels once from each pool.
    """
    draw_id = int(global_step) * int(n_random) + int(student_index)
    level_idx = draw_id % len(budget_levels)
    cycle = draw_id // len(budget_levels)
    banded = bool(cycle % 2)
    pool = _budget_group_pool(
        tuple(widths), int(n_layers), int(n_groups), float(budget_levels[level_idx]),
        float(min_utilization), banded)
    if not pool:
        raise ValueError(
            f"no {'banded ' if banded else ''}heterogeneous configs for budget level "
            f"{level_idx}={budget_levels[level_idx]} in utilization band "
            f"[{min_utilization}, 1.0]")
    rng = random.Random(
        (int(seed) << 28) ^ (0x9E3779B1 * draw_id) ^ (0x85EBCA6B * cycle) ^ 0xC2B2AE35)
    group_idx = pool[rng.randrange(len(pool))]
    sizes = _contiguous_group_sizes(int(n_layers), int(n_groups))
    width_list = []
    for size, idx in zip(sizes, group_idx):
        width_list.extend([int(widths[idx])] * size)
    return level_idx, ("budget_banded" if banded else "budget_iid"), width_list


def _sandwich_student_configs(seed, global_step, widths, weights, n_layers,
                              per_layer_mixed, n_random, always_smallest,
                              kind_weights=None, high_late_bias=True,
                              budget_levels=None, budget_min_utilization=0.85,
                              budget_n_groups=8):
    """Build the sub-full STUDENT configs for a sandwich step (Arm B/C).

    The 'sandwich rule' (Universally-Slimmable Nets, Yu & Huang 2019): each step
    train the largest (=teacher, added separately as the LAST forward), the
    SMALLEST, and a few random widths — the smallest + random get inplace
    distillation from the largest. Here:
      * Arm B (sandwich):        random students are UNIFORM sub-full widths.
      * Arm C (mixed_sandwich):  random students are PER-LAYER MIXED configs.
    Full width is NOT returned here (it is the teacher + the final synced student).

    PRECEDENCE, because it is not visible from a config file: when `budget_levels` is supplied
    it REPLACES the `kind_weights` sampler for the per-layer-mixed draws. `kind_weights` is then
    never read, whatever a config sets it to. See the launch-time note beside the
    `sandwich_budget_levels` validation.
    """
    cfgs = []
    if always_smallest:
        cfgs.append([widths[0]] * n_layers)                 # the collapse-prone 1/8 uniform
    # RANDOM-MID pool = sub-full widths EXCLUDING the full AND (when the smallest is
    # already added explicitly above) the smallest — so a random draw is a genuine MID
    # width, not a second 1280 (codex P1: the old pool {1280,2560,5120} could re-draw
    # 1280, double-weighting the smallest and starving the mids the sandwich needs).
    sub = [w for w in widths if w != widths[-1]]
    if always_smallest:
        sub = [w for w in sub if w != widths[0]]
    sub = sub or [widths[0]]
    for j in range(n_random):
        if per_layer_mixed:
            if budget_levels:
                _, _, wl = _sample_budget_stratified_config(
                    seed, global_step, j, n_random, widths, n_layers, budget_n_groups,
                    budget_levels, budget_min_utilization)
            else:
                kw = {k: v for k, v in (kind_weights or DEFAULT_MIX_KIND_WEIGHTS).items()
                      if k != "full"}
                _, wl = _sample_perlayer_config(
                    seed, global_step, widths, weights, n_layers, kind_weights=kw,
                    high_late_bias=high_late_bias, salt=j + 1)
            cfgs.append(wl)
        else:
            rng = random.Random((int(seed) << 30) ^ (0x27D4EB2F * int(global_step) + j + 3))
            w = rng.choice(sub)
            cfgs.append([w] * n_layers)
    return cfgs


def _install_perlayer_controllers(model, widths):
    """Give each NestedGemma3MLP its OWN GranularityController (in decoder-layer
    order) so a heterogeneous PER-LAYER width config can be set — mirrors the eval
    driver scripts/eval/gate_widthsampled.py (lines ~260-265) and gemma3n_mnm.py.
    (Training today shares ONE controller across all MLPs → uniform-only.)"""
    import re
    from cancerflex.matformer.nested_mlp import NestedGemma3MLP
    from cancerflex.matformer.granularity import GranularityController
    mlps = [(n, m) for n, m in model.named_modules() if isinstance(m, NestedGemma3MLP)]
    mlps.sort(key=lambda kv: int(re.search(r"layers\.(\d+)\.", kv[0]).group(1)))
    ctrls = []
    for _, m in mlps:
        c = GranularityController(widths=list(widths))
        m.controller = c
        ctrls.append(c)
    if not ctrls:
        raise RuntimeError("_install_perlayer_controllers: no NestedGemma3MLP found")
    return ctrls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    os.environ["WANDB_MODE"] = "offline"
    os.environ.setdefault("WANDB_PROJECT", cfg.get("wandb_project", "cancerflex"))
    os.environ.setdefault("WANDB_DIR", str(output_dir))

    import contextlib
    import torch
    import torch.distributed as dist
    # DataLoader workers must use SPAWN, not fork: forking AFTER CUDA init (model.to(cuda) below)
    # deadlocks multi-worker loaders (hung at step 0, futex, GPU idle — hit 2026-07-24). spawn-safe
    # because this module has an `if __name__=="__main__"` guard so workers don't re-run main().
    import torch.multiprocessing as _tmp
    try:
        _tmp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    from transformers.modeling_outputs import BaseModelOutputWithPooling
    # tf32 matmul (low-risk throughput on the many
    # fp32-accumulate matmuls; bf16 params unaffected.
    torch.set_float32_matmul_precision("high")
    from trl import SFTConfig, SFTTrainer
    from cancerflex.train import dataset as _ds
    from cancerflex.train import load_model as _lm
    from cancerflex.train.collator import VLMSFTCollator

    sampling = cfg.get("sampling", "joint")
    # joint/stochastic = the ORIGINAL uniform-width training schemes (unchanged).
    # mixed/sandwich/mixed_sandwich = the Mix'n'Match-FIX arms (per-layer capable):
    #   mixed          – Arm A: mixing-aware hard-label SFT (sample one per-layer
    #                    config/step, normal CE loss).
    #   sandwich       – Arm B: inplace self-distillation (full-width teacher →
    #                    UNIFORM sub-width students match soft logits + hard labels).
    #   mixed_sandwich – Arm C: A+B (teacher → PER-LAYER-MIXED students, KD + CE).
    _MODES = ("joint", "stochastic", "mixed", "sandwich", "mixed_sandwich")
    if sampling not in _MODES:
        raise ValueError(f"sampling must be one of {_MODES}, got {sampling}")
    is_mixarm = sampling in ("mixed", "sandwich", "mixed_sandwich")

    # ── model (pure matformer, multimodal) ────────────────────────────────
    model, processor, controller = _lm.load_matformer_pure(
        vision_freeze=cfg.get("vision_freeze", True),
        attn_impl=cfg.get("attn_impl", "eager"),
        factor=int(cfg.get("matformer_factor", 8)),
        base_path=cfg.get("base_model_override"),   # ①b: neuron-ranked permuted MedGemma (else plain base)
        widths=cfg.get("matformer_widths"),         # optional denser grid; see load_model.py
    )
    # Guardrail vs PERF §17: any future direct model(...) call in this file that
    # forgets use_cache=False falls back to config — make the fallback safe too.
    model.config.use_cache = False
    widths = list(controller.widths)                       # ascending; widths[-1] == full
    n_g = len(widths)
    weights = cfg.get("granularity_weights") or [1.0 / n_g] * n_g
    if len(weights) != n_g:
        raise ValueError(f"granularity_weights len {len(weights)} != #widths {n_g}")
    # persist the granularity schedule for eval-time reconstruction; `sampling`
    # stamps the TRAINING SCHEME into the run dir so downstream eval results
    # can be labeled joint-trained (OLMo-style) vs stochastic (paper Eq. 2)
    (output_dir / "matformer_granularities.json").write_text(
        json.dumps({"widths": widths, "weights": weights,
                    "factor": int(cfg.get("matformer_factor", 8)),
                    # `factor` alone no longer reconstructs `widths` when the grid is
                    # overridden — record that explicitly so a reader who computes
                    # {D/8,D/4,D/2,D} from factor and gets a different list knows why.
                    "widths_overridden": cfg.get("matformer_widths") is not None,
                    "sampling": sampling}, indent=2))

    # ── train scope (memory knob) ──────────────────────────────────────────
    # 'ffn_only' freezes every LM param except the nested SwiGLU FFNs (gate/up/
    # down) — the ONLY weights the matformer slices — so a 4B model + optimizer
    # fits a 24 GB L4. 'all' (default, matches how ckpt-94568 was trained) trains
    # the full LM and needs a B200. Vision is already frozen (vision_freeze).
    train_scope = cfg.get("train_scope", "all")
    if train_scope not in ("all", "ffn_only", "vision_only"):
        raise ValueError(f"train_scope must be 'all', 'ffn_only' or 'vision_only', got {train_scope}")
    if train_scope == "vision_only":
        # vision-elastic arm with the LM FROZEN: only the SigLIP tower + projector train (the LM's weights
        # stay bit-identical to the init checkpoint). Requires vision_freeze: false + vision_elastic.
        if bool(cfg.get("vision_freeze", True)) or not cfg.get("vision_elastic"):
            raise ValueError("train_scope=vision_only requires vision_freeze: false and a vision_elastic block")
        n_v = 0
        for n, p in model.named_parameters():
            p.requires_grad = ("vision_tower" in n) or ("multi_modal_projector" in n)
            n_v += int(p.requires_grad)
        if os.environ.get("RANK", "0") == "0":
            n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"[matformer_pure] train_scope=vision_only: {n_v} tower/projector tensors trainable "
                  f"({n_tr/1e6:.1f}M params); LM frozen", flush=True)
    if train_scope == "ffn_only":
        from cancerflex.matformer.nested_mlp import NestedGemma3MLP
        for p in model.parameters():
            p.requires_grad = False
        n_ffn = 0
        for m in model.modules():
            if isinstance(m, NestedGemma3MLP):
                for p in m.parameters():
                    p.requires_grad = True
                    n_ffn += 1
        if os.environ.get("RANK", "0") == "0":
            n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"[matformer_pure] train_scope=ffn_only: {n_ffn} FFN tensors trainable "
                  f"({n_tr/1e9:.2f}B params)", flush=True)

    # ── Mix'n'Match-fix arm setup (per-layer controllers + sampler/KD knobs) ──
    layer_ctrls = None
    n_layers = 0
    route_mlps = []                      # ordered NestedGemma3MLPs, for the Gumbel router's per-layer gate
    if is_mixarm:
        layer_ctrls = _install_perlayer_controllers(model, widths)
        n_layers = len(layer_ctrls)
        import re as _re
        from cancerflex.matformer.nested_mlp import NestedGemma3MLP as _NMLP
        _ml = [(n, m) for n, m in model.named_modules() if isinstance(m, _NMLP)]
        _ml.sort(key=lambda kv: int(_re.search(r"layers\.(\d+)\.", kv[0]).group(1)))
        route_mlps = [m for _, m in _ml]
    # sampler (Arm A/C) + distillation (Arm B/C) hyper-params (defaults in header)
    mix_kind_weights = cfg.get("mix_kind_weights") or DEFAULT_MIX_KIND_WEIGHTS
    mix_high_late = bool(cfg.get("mix_high_late_bias", True))
    kd_alpha = float(cfg.get("kd_alpha", 0.5))            # sub-width loss = alpha*CE + (1-alpha)*KD
    kd_temp = float(cfg.get("kd_temp", 2.0))
    # "reject sampling" in this project = the inline per-token correctness gate (see _kd_kl).
    # Default OFF -> loss identical to before, so this is a pure opt-in control arm.
    kd_correctness_gate = bool(cfg.get("kd_correctness_gate", False))
    sw_n_random = int(cfg.get("sandwich_n_random", 2))    # random students / step (+ smallest + full)
    sw_smallest = bool(cfg.get("sandwich_always_smallest", True))
    sw_budget_levels = cfg.get("sandwich_budget_levels")
    sw_budget_min_util = float(cfg.get("sandwich_budget_min_utilization", 0.85))
    sw_budget_groups = int(cfg.get("sandwich_budget_groups", 8))
    if sw_budget_levels:
        if sampling != "mixed_sandwich":
            raise ValueError("sandwich_budget_levels requires sampling: mixed_sandwich")
        if sw_n_random <= 0:
            raise ValueError("sandwich_budget_levels requires sandwich_n_random > 0")
        if not 0.0 <= sw_budget_min_util <= 1.0:
            raise ValueError("sandwich_budget_min_utilization must be in [0, 1]")
        # `sandwich_budget_levels` REPLACES the mix_kind_weights sampler rather than tuning
        # it: _sandwich_student_configs takes the budget-stratified branch and never reads
        # kind_weights. A config that sets both looks like it is asking for a 25/15/30/30
        # mixture and silently gets budget-stratified draws instead. That misreading survived
        # into several config headers and into a downstream analysis before it was caught, so
        # say it out loud at launch rather than leave the knob quietly inert.
        if cfg.get("mix_kind_weights"):
            print("[matformer_pure] NOTE: mix_kind_weights is INERT because "
                  "sandwich_budget_levels is set — every random student is a budget-stratified "
                  "heterogeneous draw (uniform configs are excluded from those pools), and the "
                  f"requested mixture {cfg['mix_kind_weights']} is not used.", flush=True)
    # FULL-width loss weight (codex P0 #3): the teacher forward is no_grad (zero
    # gradient), so WITHOUT this the trained mass is L_1280 + L_mid + L_full unweighted
    # → full gets only 1/3 while 2 sub-widths pull the SHARED params (attn/embed/LM-head
    # under train_scope=all, not just FFN). Weighting the full CE ×2 → 50/25/25 (Nemotron-
    # style skew toward full). kd_alpha>=1.0 => CE-only CONTROL arm (KD off, teacher fwd
    # skipped) so any 1280 gain can be attributed to KD vs plain sandwich SFT.
    full_loss_weight = float(cfg.get("full_loss_weight", 2.0))
    ce_only_control = kd_alpha >= 1.0
    # ── vision-elastic (2026-10-08; docs/VISION_ENCODER_COMPRESSION_20261008.md §5–7, options A+B) ──
    # The frozen SigLIP tower becomes a STUDENT: per step one (input resolution, encoder depth) pair is
    # sampled; the student tower runs the batch at that resolution (bilinear+antialias downsample of the
    # 896-px pixel_values, interpolated position grid) with only the first `depth` encoder blocks, and its
    # projector pools to the same 256 soft tokens. The frozen `sandwich_teacher_model` (frozen_initial)
    # supplies the targets at 896 px / full depth: dense token cosine (teacher 64x64 grid average-pooled to
    # the student grid) + cosine on the 256 projected soft tokens, weighted w_dense / w_soft and added to
    # the full-width student's CE. Narrow LM students consume the student tower output DETACHED (they
    # train the LM only); the tower's gradient comes from the two cosine terms + the full-width CE, in
    # the one synced backward. Saved checkpoints keep the full 27-block tower and native projector.
    # 1-GPU only for now: with depth < 27 the unused blocks would trip DDP's unused-parameter check.
    vision_elastic = cfg.get("vision_elastic") or None
    ve_res = ve_res_w = ve_depths = ve_depth_w = None
    ve_w_dense = ve_w_soft = 0.0
    ve_vision_lr = None
    ve_narrow_live = False
    if vision_elastic:
        if sampling not in ("sandwich", "mixed_sandwich"):
            raise ValueError("vision_elastic requires sampling: sandwich|mixed_sandwich")
        if bool(cfg.get("vision_freeze", True)):
            raise ValueError("vision_elastic requires vision_freeze: false (the tower is the student)")
        ve_res = [int(r) for r in vision_elastic.get("resolutions", [448, 672, 896])]
        ve_res_w = [float(x) for x in (vision_elastic.get("resolution_weights") or [1.0] * len(ve_res))]
        ve_depths = [int(d) for d in vision_elastic.get("depths", [27])]
        ve_depth_w = [float(x) for x in (vision_elastic.get("depth_weights") or [1.0] * len(ve_depths))]
        if len(ve_res_w) != len(ve_res) or len(ve_depth_w) != len(ve_depths):
            raise ValueError("vision_elastic: weights must match resolutions / depths in length")
        ve_w_dense = float(vision_elastic.get("w_dense", 0.5))
        ve_w_soft = float(vision_elastic.get("w_soft", 0.5))
        ve_vision_lr = vision_elastic.get("vision_lr")
        ve_vision_lr = float(ve_vision_lr) if ve_vision_lr is not None else None
        # narrow_students_train_tower: when true the narrow LM students read the LIVE student tower output
        # (gradient reaches the tower from every width; the shared tower graph is kept with retain_graph
        # until the full-width backward). Default false = narrow students see it detached (doc §9 P1).
        ve_narrow_live = bool(vision_elastic.get("narrow_students_train_tower", False))
    # Elastic-backbone conversion can distil from an immutable dense parent instead of
    # letting the trainable full-width student move its own target. Historical configs
    # keep the self-teacher default; the corrected dense-parent lineage opts in.
    sandwich_teacher_mode = str(cfg.get("sandwich_teacher_mode", "self"))
    if sandwich_teacher_mode not in ("self", "frozen_initial"):
        raise ValueError(
            "sandwich_teacher_mode must be self|frozen_initial, "
            f"got {sandwich_teacher_mode!r}"
        )
    if sandwich_teacher_mode == "frozen_initial" and (
        sampling not in ("sandwich", "mixed_sandwich") or ce_only_control
    ):
        raise ValueError(
            "sandwich_teacher_mode=frozen_initial requires sandwich KD "
            "(sampling=sandwich|mixed_sandwich and kd_alpha < 1)"
        )
    sandwich_teacher_model = None
    if vision_elastic and sandwich_teacher_mode != "frozen_initial":
        raise ValueError("vision_elastic requires sandwich_teacher_mode: frozen_initial "
                         "(the frozen copy is the 896-px / full-depth tower teacher)")
    # ── GOAL-ROUTER (opt-in `router_enable: true`) — budget-first, allow-missing ──────────
    # Attached as a submodule of `model` so HF's default optimizer trains it (no create_optimizer
    # override) and it rides in the checkpoint. See width_router.py + docs/methodology/
    # elastic_router_lessons.md §2b. Router decides 8 GROUP widths from (budget[,pillar][,task]); a SURROGATE
    # predicts each config's (CE - CE_full) so the router is trained without differentiating the hard
    # FFN slice. Staged via dropout on the pillar/task guides: stage-1 = 1.0 (budget-only).
    router_enable = bool(cfg.get("router_enable", False))
    # "surrogate" = Flextron-style frozen-backbone + loss-predictor (deprecated: surrogate didn't
    # converge). "gumbel" = Nemotron-Elastic: co-train backbone+router, full-width SELF-teacher KD
    # through a Gumbel straight-through gated FFN, NO surrogate. Default gumbel.
    router_mode = str(cfg.get("router_mode", "gumbel"))
    router_teacher_mode = str(cfg.get("router_teacher_mode", "self"))
    if router_teacher_mode not in ("self", "frozen_initial"):
        raise ValueError(
            "router_teacher_mode must be self|frozen_initial, "
            f"got {router_teacher_mode!r}"
        )
    if router_teacher_mode == "frozen_initial" and (
        not router_enable or router_mode != "gumbel"
    ):
        raise ValueError(
            "router_teacher_mode=frozen_initial requires the Gumbel router path"
        )
    router_teacher_model = None
    router_head_lr = float(cfg.get("router_head_lr", 1.0e-3))  # separate lr for the router head vs
    #   the (co-trained) backbone `lr` — the head is tiny and freshly-init, the backbone must move
    #   gently (codex: separate param groups so head grads don't fight the backbone's clip/lr).
    # ── the Router head's OWN lr schedule ────────────────────────────────────────────────────────
    # `None` (default) = inherit the backbone's `lr_scheduler`, i.e. every historical run reproduces
    # bit-for-bit. `constant` = hold `router_head_lr` flat for the whole run, with no warmup.
    #
    # WHY THIS KNOB EXISTS [MEASURED, router_training_forensic_audit.md §3.1]. Two param groups were
    # never two schedules: HF builds ONE LambdaLR and multiplies both groups by the same factor.
    # `elastic_augdata_router_1000/checkpoint-1000/scheduler.pt` records
    # `base_lrs = [5e-06, 0.001]` but `_last_lr = [0.0, 0.0]` — the Router head was annealed to
    # EXACTLY zero by the backbone's cosine. 81.1% of that run's lr-integral was spent by step 500,
    # which is precisely when its decoded configuration "stabilised". The decisive tell is that the
    # table's movement per unit lr-integral is CONSTANT (2.09-2.36e-3) across every window while the
    # raw movement falls 12x: it stopped because the optimizer stopped, not because the objective
    # did. A tiny, freshly-initialised 160-parameter table has no reason to share a 4B-parameter
    # backbone's decay envelope.
    # Continuation source for the router HEAD (see width_router.load_router_state); both are hoisted here so
    # the final artifact can record them unconditionally, router enabled or not.
    router_init_from = cfg.get("router_init_from")
    router_init_from_report = None
    router_lr_scheduler = cfg.get("router_lr_scheduler")
    if router_lr_scheduler is not None:
        router_lr_scheduler = str(router_lr_scheduler)
        if router_lr_scheduler != "constant":
            raise ValueError(
                "router_lr_scheduler must be null (inherit the backbone schedule) or "
                f"'constant', got {router_lr_scheduler!r}")
        # Fail loudly rather than no-op: with no router param group there is no second lambda to
        # swap, and a config asserting a router lr policy while running no router is a config error.
        if not router_enable:
            raise ValueError(
                "router_lr_scheduler is set but router_enable is false — there is no router "
                "param group to give a separate schedule to")
    if router_init_from is not None and not router_enable:
        raise ValueError(
            "router_init_from is set but router_enable is false — there is no router head to "
            "continue")
    router_hidden = int(cfg.get("router_hidden", 256))
    router_groups = int(cfg.get("router_groups", 8))
    router_w = float(cfg.get("router_loss_weight", 1.0))         # weight on the router's own loss
    surrogate_w = float(cfg.get("surrogate_loss_weight", 1.0))   # weight on surrogate fitting
    router_budget_w = float(cfg.get("router_budget_weight", 1.0))
    # NEMOTRON discrete budget levels (iteration 3): the router's budget input is a ONE-HOT over these
    # target mean-width fractions, and the cost-matching penalty is L2 to each level's target. A
    # continuous scalar budget collapsed to a single config for all budgets (router ignored it).
    router_budget_levels = [float(b) for b in cfg.get("router_budget_levels", [0.2, 0.4, 0.6, 0.8])]
    # how many budget LEVELS to forward per step (anti-collapse). "one" = a single random level (iter 3).
    # "minmax" = forward the narrowest AND widest level in the SAME step so the router cannot collapse to
    # one config (sandwich-style — a narrow and a wide target are penalized together). "all" = every level.
    router_multi_budget = str(cfg.get("router_multi_budget", "one"))
    if router_multi_budget not in ("one", "stratified", "minmax", "all"):
        raise ValueError(
            "router_multi_budget must be one|stratified|minmax|all, "
            f"got {router_multi_budget!r}"
        )
    # SANDWICH RULE (Universally-Slimmable Nets): every step should train the LARGEST and the SMALLEST
    # width, plus middles. The gumbel/router branch trains the largest (teacher + full-width CE anchor)
    # and whatever widths the router picks, but the SMALLEST uniform width gets no gradient unless the
    # router happens to request it — and it never will, because the budget levels are interior. Left
    # unanchored the backbone can drift at the narrowest width over a long run, which also biases the
    # gain eval in our favour (uni_1280 is one of its baselines). Enable this to restore the rule.
    # Default False so an in-flight run that resumes keeps its original behaviour.
    router_train_smallest = bool(cfg.get("router_train_smallest", False))
    router_smallest_weight = float(cfg.get("router_smallest_weight", 1.0))
    router_tau = float(cfg.get("router_tau", 1.0))
    # Reference-faithful Gumbel relaxation schedule. A legacy scalar remains a
    # constant schedule so every existing config/checkpoint is reproducible.
    router_tau0 = float(cfg.get("router_tau_start", router_tau))
    router_tau1 = float(cfg.get("router_tau_end", router_tau))
    # ── anti-collapse knobs (added 2026-07-27 after router_cotrain_scale collapsed) ───────────
    # Defaults reproduce the OLD behaviour exactly, so resuming any earlier arm is unaffected.
    # `arch=table` gives each budget level its own logit row (no shared trunk to outgrow the
    # conditioning); `logit_bound` caps |logit| so Gumbel exploration cannot be extinguished;
    # `t_sample` is the EXPLORATION temperature (tau is only backward sharpness — it divides out
    # of the argmax); `budget_all_levels` supervises every level every step for ~free.
    # Full derivation + the measurements behind each: width_router.py LevelTableRouter docstring.
    router_arch = str(cfg.get("router_arch", "mlp"))
    router_logit_bound = float(cfg.get("router_logit_bound", 3.0))
    router_t_sample0 = float(cfg.get("router_t_sample_start", 1.0))
    router_t_sample1 = float(cfg.get("router_t_sample_end", 1.0))
    # Structured perturb-and-MAP variant: every hard routed forward is solved at the exact
    # registered compute, while the soft ST backward still carries task gradient. Default off so
    # every prior checkpoint/resume retains independent per-group Gumbel sampling.
    router_exact_compute_gumbel = bool(
        cfg.get("router_exact_compute_gumbel", False)
    )
    if router_exact_compute_gumbel and router_mode != "gumbel":
        raise ValueError(
            "router_exact_compute_gumbel requires router_mode='gumbel'"
        )
    router_budget_all = bool(cfg.get("router_budget_all_levels", False))
    router_budget_t = float(cfg.get("router_budget_t", 1.0))
    router_budget_all_w = float(cfg.get("router_budget_all_weight", 0.0))
    # Penalise the DEPLOYED (argmax) fraction, not the soft one — the soft form leaves any
    # level whose target is near 0.469 unpenalised from step 0. See budget_loss_all_levels.
    router_budget_hard = bool(cfg.get("router_budget_hard", False))
    # CEILING semantics (decided 2026-07-27, see elastic_router_lessons.md §3.11): a budget is an
    # upper bound, not a value to match. `router_budget_mode: ceiling` swaps the symmetric squared
    # term for the EXACT expected hinge E[max(F - budget, 0)], computed by an integer dynamic
    # program over the 188 achievable fractions. Unlike LLaMaFlex's max(E[F] - b, 0) it cannot
    # report zero for a distribution whose samples exceed the ceiling, and unlike the hard
    # straight-through form it is linear in the probabilities, so it has no argmax and no kink —
    # which also removes the boundary chattering diagnosed in §3.10. Default `target` keeps the
    # old behaviour so earlier arms remain reproducible.
    router_budget_mode = str(cfg.get("router_budget_mode", "target"))
    if router_budget_mode not in ("target", "ceiling"):
        raise ValueError(f"router_budget_mode must be target|ceiling, got {router_budget_mode}")
    router_readout_every = int(cfg.get("router_readout_every", 0))   # 0 = off
    # Deployment/readout policy only; it never changes the stochastic training route. The exact
    # constrained mode maximizes router score under each hard ceiling while requiring the selected
    # subnet to use at least this fraction of its budget. Default argmax preserves every old run.
    router_readout_mode = str(cfg.get("router_readout_mode", "argmax"))
    if router_readout_mode not in ("argmax", "constrained"):
        raise ValueError(
            f"router_readout_mode must be argmax|constrained, got {router_readout_mode}")
    router_readout_min_util = float(cfg.get("router_readout_min_utilization", 0.85))
    # Hoisted here (not read from `cfg` inside training_step) to match how every other router knob
    # is passed into the nested Trainer: plain closure locals, no dict lookup in the hot loop.
    router_seed_base = int(cfg.get("seed", 0))
    router_total_steps = float(cfg.get("max_steps", 1) or 1)
    router_pillar_drop = float(cfg.get("router_pillar_dropout", 1.0))  # 1.0 => never show pillar (stage 1)
    router_task_drop = float(cfg.get("router_task_dropout", 1.0))      # 1.0 => never show task (stage 1)
    # FLEXTRON recipe (recommended on a base that already runs per-layer mixes, e.g. widthsampled):
    # FREEZE the backbone, train ONLY the router+surrogate heads. This makes the surrogate target
    # STATIONARY (no co-training drift) and stops the fresh-head grads from polluting the backbone's
    # global grad-clip — the two instabilities seen in the co-train run. router_warmup_steps holds the
    # ROUTER loss off (surrogate-only) until the surrogate has settled.
    router_freeze_backbone = bool(cfg.get("router_freeze_backbone", False))
    router_warmup_steps = int(cfg.get("router_warmup_steps", 0))
    # Optional causal successor to the frozen-direct objective. For each routed
    # batch, compare its per-example CE with the exact same-compute uniform
    # prefix and hinge only positive budget x task x pillar mean regret. The
    # Router remains budget-only; cell ids affect the training objective, not
    # the deployed policy or its inputs.
    router_uniform_regret_w = float(cfg.get("router_uniform_regret_weight", 0.0))
    router_uniform_regret_margin = float(cfg.get("router_uniform_regret_margin", 0.0))
    if router_uniform_regret_w < 0.0 or not math.isfinite(router_uniform_regret_w):
        raise ValueError("router_uniform_regret_weight must be finite and nonnegative")
    if router_uniform_regret_margin < 0.0 or not math.isfinite(router_uniform_regret_margin):
        raise ValueError("router_uniform_regret_margin must be finite and nonnegative")
    if router_uniform_regret_w > 0.0 and not (
        router_enable
        and router_mode == "gumbel"
        and router_exact_compute_gumbel
        and router_freeze_backbone
    ):
        raise ValueError(
            "router_uniform_regret_weight > 0 requires the frozen exact-compute "
            "Gumbel Router path"
        )
    if router_freeze_backbone and router_mode == "gumbel" and router_train_smallest:
        raise ValueError(
            "router_train_smallest has no trainable path when the Gumbel Router "
            "backbone is frozen; disable it for Router-only direct-loss training"
        )
    PILLAR2ID = {"skin": 0, "oral": 1, "blood": 2, "cervix": 3, "breast": 4, "thyroid": 5}
    TASK2ID = {f"U{i}": i for i in range(7)}
    router_uniform_widths = []
    if router_uniform_regret_w > 0.0:
        for budget in router_budget_levels:
            exact_width = float(budget) * float(widths[-1])
            uniform_width = int(round(exact_width))
            if abs(exact_width - uniform_width) > 1e-6:
                raise ValueError(
                    f"budget {budget} has no integer same-compute uniform width"
                )
            if not int(widths[0]) <= uniform_width <= int(widths[-1]):
                raise ValueError(
                    f"same-compute uniform width {uniform_width} is outside the elastic range"
                )
            router_uniform_widths.append(uniform_width)
    if router_enable:
        if sampling not in ("sandwich", "mixed_sandwich"):
            raise SystemExit(f"router_enable needs sampling=sandwich|mixed_sandwich, got '{sampling}'")
        from cancerflex.train.width_router import (
            RouterBundle,
            exponential_router_temperature,
            groupwise_uniform_regret_hinge,
            load_router_state,
            stratified_budget_level,
        )
        _dev = next(model.parameters()).device
        model.width_router = RouterBundle(
            widths, n_layers=n_layers, n_groups=router_groups,
            n_pillars=len(PILLAR2ID), n_tasks=len(TASK2ID), hidden=router_hidden,
            budget_levels=router_budget_levels,
            router_arch=router_arch,
            logit_bound=router_logit_bound).to(device=_dev, dtype=torch.float32)
        # ── CONTINUATION: load a TRAINED router head out of an existing checkpoint ───────────────
        # `base_model_override` cannot do this. It feeds `from_pretrained`, which runs BEFORE the
        # RouterBundle is constructed a few lines above, so a checkpoint's `width_router.*` tensors
        # are simply unexpected keys and are dropped — the table would silently restart at its zero
        # init and a run advertised as "continue from the endpoint" would in fact be a fresh start.
        # Discovered while implementing the audit's control experiment (§7); it is exactly the class
        # of silent-restart error that would have made that control unfalsifiable.
        if router_init_from is not None:
            if cfg.get("router_init_group_widths") is not None:
                raise ValueError(
                    "router_init_from and router_init_group_widths both set — a continuation "
                    "cannot also be a fresh warm start; choose one")
            router_init_from_report = load_router_state(
                model.width_router, router_init_from)
            print(
                f"[matformer_pure] router head CONTINUED from {router_init_from}: "
                f"{router_init_from_report['n_tensors']} tensors, "
                f"max|table|={router_init_from_report['table_max_abs']:.6f}, "
                f"table_sha256={router_init_from_report['table_sha256'][:16]}",
                flush=True,
            )
            if router_init_from_report["table_max_abs"] == 0.0:
                raise ValueError(
                    f"router_init_from={router_init_from} carries an all-zero table; that is a "
                    "fresh init, not a continuation — refusing to start a run whose premise is "
                    "already false")
        router_init_configs = cfg.get("router_init_group_widths")
        if router_init_configs is not None:
            initialized = model.width_router.initialize_table_from_group_widths(
                router_init_configs,
                chosen_logit=float(cfg.get("router_init_chosen_logit", 0.5)),
            )
            print(
                "[matformer_pure] router table warm-started from "
                f"{len(initialized)} measured group configs "
                f"(chosen_logit={float(cfg.get('router_init_chosen_logit', 0.5))})",
                flush=True,
            )
        print(f"[matformer_pure] RouterBundle attached: groups={router_groups} K={len(widths)} "
              f"params={sum(p.numel() for p in model.width_router.parameters())/1e6:.3f}M "
              f"router-space {len(widths)}^{router_groups} | guides pillar_drop={router_pillar_drop} "
              f"task_drop={router_task_drop} w_router={router_w} w_surr={surrogate_w} "
              f"w_budget={router_budget_w}", flush=True)
        if router_freeze_backbone:
            for _n, _p in model.named_parameters():
                _p.requires_grad_("width_router" in _n)
            _nt = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"[matformer_pure] FROZEN backbone: only RouterBundle trainable "
                  f"({_nt/1e6:.3f}M params); mode={router_mode}; "
                  f"surrogate_warmup={router_warmup_steps}",
                  flush=True)
        if router_uniform_regret_w > 0.0:
            print(
                "[matformer_pure] cellwise matched-uniform regret: "
                f"weight={router_uniform_regret_w} "
                f"margin={router_uniform_regret_margin} "
                f"uniform_widths={router_uniform_widths}",
                flush=True,
            )

    def _set_uniform(w):
        for c in layer_ctrls:
            c.set(w)

    def _set_perlayer(wl):
        for c, w in zip(layer_ctrls, wl):
            c.set(w)

    def _set_gates(gate_Lfull):          # gate_Lfull: (n_layers, full_intermediate) straight-through
        for i, m in enumerate(route_mlps):
            m.route_gate = gate_Lfull[i]

    def _clear_gates():
        for m in route_mlps:
            m.route_gate = None

    if cfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()

    # ── data ──────────────────────────────────────────────────────────────
    train_ds = _ds.load_v1_split(
        "train",
        corpus_dir=cfg.get("corpus_dir", "Experiment/datasets/v1_final_deleaked"),
        limit=cfg.get("train_subsample_n"),
        u6_image_path_allowlist=cfg.get("u6_image_path_allowlist"),
    )
    # ── task-imbalance resampling (two MUTUALLY EXCLUSIVE options) ────────
    #   sampler_tau (A2 / NODE-C): two-level task×pillar resample. Protects the
    #     thin GENERATION tail (U5 detection, U6) that pillar-only balancing
    #     STARVES — U1/skin≈36% of rows drowns U5 → the model answers a detection
    #     prompt with the closed "Yes/No" (parse-fail 100%, mIoU 0) while closed
    #     U1–U4 train fine. This is the recipe that gave a2_full U5 mIoU 0.848 on
    #     this de-leaked corpus. (tmp/handoff/matformer_tail_collapse_2026-07-07.md)
    #   balance_temperature (Phase-1 legacy): pillar-ONLY temperature resample —
    #     balances pillars, does nothing about task imbalance → the collapse above.
    # Do NOT set both: the second would resample an already-resampled set.
    #   label_balance_tau (2026-09-20): class-level temperature resample WITHIN a cell, over the
    #     field named by label_balance_field (default u4_label). The two options above balance
    #     task and pillar and say nothing about the class mix inside a cell; on a single-cell
    #     continue-training corpus sampler_tau is a no-op. oral U4 is 75.6% `healthy`, so uniform
    #     over rows spends three quarters of the gradient on the majority class while the tail
    #     classes a run exists to repair get 5.1% and 2.2% — and the measured external failure is
    #     that the model already over-predicts `healthy`, whose external share is 16.9%.
    _n_sampler = sum(cfg.get(k) is not None
                     for k in ("sampler_tau", "balance_temperature", "label_balance_tau"))
    if _n_sampler > 1:
        raise ValueError(
            "set at most ONE of sampler_tau / balance_temperature / label_balance_tau — each is "
            "a PHYSICAL resample, so two would stack and the second would resample an "
            "already-resampled set, giving a mix neither config describes.")
    if cfg.get("label_balance_tau") is not None:
        from cancerflex.train.balanced.weights import (
            label_balance, label_balance_report,
        )
        _lbf = cfg.get("label_balance_field", "u4_label")
        if os.environ.get("RANK", "0") == "0":
            print(f"[matformer_pure] label_balance on {_lbf!r}, tau={cfg['label_balance_tau']}\n"
                  + label_balance_report(train_ds, _lbf, float(cfg["label_balance_tau"])),
                  flush=True)
        train_ds = label_balance(train_ds, field=_lbf,
                                 tau=float(cfg["label_balance_tau"]),
                                 seed=cfg.get("seed", 0))
    elif cfg.get("sampler_tau") is not None:
        if cfg.get("balance_temperature") is not None:
            raise ValueError(
                "set sampler_tau OR balance_temperature, not both — two "
                "physical resamples would stack (a2 uses sampler_tau alone).")
        from cancerflex.train.balanced.weights import (
            two_level_balance,
        )
        train_ds = two_level_balance(
            train_ds, tau=float(cfg["sampler_tau"]), seed=cfg.get("seed", 0))
    elif cfg.get("balance_temperature") is not None:
        train_ds = _ds.temperature_balance(
            train_ds, by="pillar", tau=float(cfg["balance_temperature"]),
            seed=cfg.get("seed", 0),
        )
    rank0 = (os.environ.get("RANK", "0") == "0")
    if rank0:
        _samp = (f"two_level(task×pillar) tau={cfg['sampler_tau']}"
                 if cfg.get("sampler_tau") is not None
                 else (f"pillar-only temp={cfg['balance_temperature']}"
                       if cfg.get("balance_temperature") is not None
                       else (f"label({cfg.get('label_balance_field', 'u4_label')}) "
                             f"tau={cfg['label_balance_tau']}"
                             if cfg.get("label_balance_tau") is not None else "NONE")))
        _mode = {
            "stochastic": "paper-aligned Eq.2 (one sampled UNIFORM width / optimizer step)",
            "joint": "MatFormer-OLMo-style joint (DynaBERT-family; all widths / step)",
            "mixed": "Arm A: mixing-aware SFT (one PER-LAYER config / step, hard-label CE)",
            "sandwich": "Arm B: inplace self-distill (full-width teacher → uniform sub-width students, KD+CE)",
            "mixed_sandwich": "Arm C: A+B (full-width teacher → per-layer-MIXED students, KD+CE)",
        }.get(sampling, sampling)
        print(f"[matformer_pure] sampling={sampling} = {_mode}", flush=True)
        print(f"[matformer_pure] widths={widths} weights={weights}", flush=True)
        if is_mixarm:
            print(f"[matformer_pure] n_layers={n_layers} kind_weights={mix_kind_weights} "
                  f"high_late={mix_high_late}", flush=True)
            if sampling in ("sandwich", "mixed_sandwich"):
                print(f"[matformer_pure] sandwich: kd_alpha={kd_alpha} kd_temp={kd_temp} "
                      f"n_random={sw_n_random} always_smallest={sw_smallest}", flush=True)
                if sw_budget_levels:
                    print(f"[matformer_pure] budget-stratified students: levels={sw_budget_levels} "
                          f"min_util={sw_budget_min_util} groups={sw_budget_groups}", flush=True)
        print(f"[matformer_pure] resample={_samp}  train rows: {len(train_ds):,}", flush=True)

    # ── TRL config ──────────────────────────────────────────────────────
    sft_cfg = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=float(cfg.get("num_train_epochs", 1)),
        max_steps=int(cfg.get("max_steps", -1)),
        # step-based ckpt (NOT per-epoch) so a sub-epoch crash is resumable
        save_strategy=cfg.get("save_strategy", "steps"),
        save_steps=int(cfg.get("save_steps", 2000)),
        # `null` = keep EVERY checkpoint (HF's own meaning for None). Needed by runs whose
        # deliverable is the trajectory rather than the endpoint — e.g. addwidth_cost_3840,
        # where the early checkpoints ARE the measurement. int(None) used to crash here.
        save_total_limit=(None if cfg.get("save_total_limit", 2) is None
                          else int(cfg.get("save_total_limit", 2))),
        # joint: full-width backward syncs all params → no unused-param search.
        # stochastic / mixed (Arm A): a single sub-xl backward → must search.
        # sandwich / mixed_sandwich (Arm B/C): the full-width student runs LAST as
        #   the single synced backward (covers every FFN param, exactly like joint),
        #   so no search — and find_unused=False is REQUIRED for the no_sync +
        #   accumulate-then-sync multi-backward pattern to reduce grads correctly.
        # gumbel router co-train: the final synced backward is the FULL-WIDTH one, which covers every
        #   FFN param but NOT `width_router.*` — the router's grads come only from the no_sync'd level
        #   backwards. With find_unused=False DDP asserts every grad-requiring param took part in that
        #   final backward and errors out, so this path needs the search. (Its grads are additionally
        #   all-reduced by hand after the full-width backward; see training_step.) Discovered
        #   2026-07-26 — this branch had only ever been run on a single GPU, where DDP is absent.
        ddp_find_unused_parameters=(sampling in ("stochastic", "mixed")
                                    or (bool(cfg.get("router_enable", False))
                                        and str(cfg.get("router_mode", "")) == "gumbel"
                                        and not bool(cfg.get("router_freeze_backbone", False)))),
        bf16=True,
        per_device_train_batch_size=int(cfg.get("batch_size", 16)),
        gradient_accumulation_steps=int(cfg.get("grad_accum", 1)),
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", False)),
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
        # prefetch & persistent_workers REQUIRE num_workers>0 (torch/HF raise otherwise) — auto-off at 0.
        dataloader_prefetch_factor=(int(cfg["dataloader_prefetch_factor"])
                                    if int(cfg.get("dataloader_num_workers", 8)) > 0
                                    and cfg.get("dataloader_prefetch_factor") is not None else None),
        dataloader_pin_memory=True,
        dataloader_persistent_workers=int(cfg.get("dataloader_num_workers", 8)) > 0,
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        eval_strategy="no",
        seed=int(cfg.get("seed", 0)),
        data_seed=int(cfg.get("seed", 0)),
        push_to_hub=False,
    )

    # resolution_aug_prob defaults to 0.0, i.e. every existing config keeps byte-identical
    # behaviour. See the block in collator.py for why this exists and why it only removes detail.
    collator = VLMSFTCollator(processor, max_length=cfg.get("max_seq_length", 512),
                              emit_router_ids=router_enable,
                              resolution_aug_prob=float(cfg.get("resolution_aug_prob", 0.0)),
                              resolution_aug_min_scale=float(cfg.get("resolution_aug_min_scale", 0.143)),
                              resolution_aug_seed=int(cfg.get("seed", 0)),
                              resolution_aug_pillars=cfg.get("resolution_aug_pillars"))

    # ── joint / stochastic matformer trainer ─────────────────────────────
    class MatformerTrainer(SFTTrainer):
        def _mf_width_idx(self, global_step):
            """Stochastic-mode width draw as a PURE FUNCTION of the update
            index: a throwaway RNG seeded from (run seed, global_step). All
            microbatches of one optimizer step share global_step → one width
            per gradient update (paper Eq. 2); all ranks agree; resume replays
            the identical width sequence."""
            rng = random.Random((self._mf_seed << 20) ^ int(global_step))
            return rng.choices(range(n_g), weights=weights, k=1)[0]

        def create_optimizer(self):
            # Two param groups so the fresh router HEAD (lr=router_head_lr) and the co-trained
            # BACKBONE (lr=args.learning_rate, gentle) don't share one lr — the head's large early
            # grads otherwise dominate. Only used when a width_router is attached; else defers to HF.
            if self.optimizer is not None:
                return self.optimizer
            rp = [p for n, p in self.model.named_parameters() if p.requires_grad and "width_router" in n]
            # vision-elastic: tower + projector get their own lr (vision_lr) when asked for.
            _is_vis = (lambda n: ("vision_tower" in n or "multi_modal_projector" in n))
            vp = ([p for n, p in self.model.named_parameters() if p.requires_grad and _is_vis(n)]
                  if (vision_elastic and ve_vision_lr is not None) else [])
            if not rp and not vp:
                return super().create_optimizer()
            bp = [p for n, p in self.model.named_parameters()
                  if p.requires_grad and "width_router" not in n and not (vp and _is_vis(n))]
            groups = []
            if bp:
                groups.append({"params": bp, "lr": self.args.learning_rate})
            if vp:
                groups.append({"params": vp, "lr": ve_vision_lr})
                print(f"[matformer_pure] optimizer: vision group lr={ve_vision_lr} "
                      f"({sum(p.numel() for p in vp)/1e6:.1f}M params)", flush=True)
            if rp:
                groups.append({"params": rp, "lr": router_head_lr})
            # Index of the ROUTER group, for the per-group lr schedule below. Two param groups were
            # never two schedules until create_scheduler was overridden too (audit §3.1).
            self._mf_router_group_index = (len(groups) - 1) if rp else None
            self.optimizer = torch.optim.AdamW(
                groups,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon, weight_decay=self.args.weight_decay, fused=True)
            print(f"[matformer_pure] optimizer: backbone lr={self.args.learning_rate} "
                  f"({sum(p.numel() for p in bp)/1e6:.1f}M) | router lr={router_head_lr} "
                  f"({sum(p.numel() for p in rp)/1e6:.3f}M)", flush=True)
            return self.optimizer

        def create_scheduler(self, num_training_steps, optimizer=None):
            """Give the ROUTER param group its own lr lambda when asked to.

            HF builds one `LambdaLR` and, when handed a single callable, replicates it across every
            param group — so the Router head inherited the backbone's cosine and was annealed to
            exactly 0.0 (`scheduler.pt: _last_lr = [0.0, 0.0]`, audit §3.1). Rather than
            re-implement HF's schedules, take the scheduler HF just built and swap ONLY the Router
            group's entry in `lr_lambdas`. The backbone's envelope is therefore byte-identical to
            what it would have been, and `router_lr_scheduler: null` is a strict no-op.
            """
            scheduler = super().create_scheduler(num_training_steps, optimizer)
            group_index = getattr(self, "_mf_router_group_index", None)
            if router_lr_scheduler is None or group_index is None:
                return scheduler
            lambdas = getattr(scheduler, "lr_lambdas", None)
            if lambdas is None:
                raise RuntimeError(
                    f"router_lr_scheduler={router_lr_scheduler!r} needs a LambdaLR-style scheduler, "
                    f"but lr_scheduler={self.args.lr_scheduler_type} produced "
                    f"{type(scheduler).__name__}, which has no per-group lr_lambdas")
            if not 0 <= group_index < len(lambdas):
                raise RuntimeError(
                    f"router param group {group_index} outside the scheduler's "
                    f"{len(lambdas)} lambdas")
            lambdas[group_index] = _constant_lr_lambda      # flat, no warmup, for the whole run
            # LambdaLR set this group's lr from the ORIGINAL lambda in its constructor (at step 0
            # that is the warmup value, i.e. 0.0). Re-derive it now so step 0 is already flat;
            # last_epoch is deliberately untouched.
            target = self.optimizer.param_groups[group_index]
            target["lr"] = scheduler.base_lrs[group_index] * _constant_lr_lambda(
                scheduler.last_epoch)
            print(f"[matformer_pure] router lr schedule: '{router_lr_scheduler}' "
                  f"(flat at {target['lr']:.3e}); backbone keeps "
                  f"'{self.args.lr_scheduler_type}'", flush=True)
            return scheduler

        def log(self, logs, *args, **kwargs):
            # surface the sampled width in wandb so the {p_i} draw is auditable
            if sampling == "stochastic" and getattr(self, "_mf_last_width", None):
                logs = {**logs, "matformer/width": float(self._mf_last_width)}
            if is_mixarm and getattr(self, "_mf_last_meanw", None) is not None:
                logs = {**logs, "matformer/mean_width": float(self._mf_last_meanw)}
            # sandwich diagnostics (codex P1 #6): log full-CE / smallest-CE / smallest-KD
            # SEPARATELY, not just the summed loss, so a null/regressing run is explainable
            # (e.g. smallest-KD not falling => teacher too far; full-CE rising => full drift).
            if sampling in ("sandwich", "mixed_sandwich"):
                for k, attr in (("matformer/full_ce", "_mf_last_full_ce"),
                                ("matformer/smallest_ce", "_mf_last_smallest_ce"),
                                ("matformer/smallest_kd", "_mf_last_smallest_kd"),
                                # kd_correctness_gate: share of answer tokens where the teacher's top-1 == gold (the
                                # KL is kept there, dropped elsewhere); last KD call of the step. Unset when the gate is off.
                                ("matformer/kd_gate_keep", "_mf_last_kd_gate_keep"),
                                # router path without a dedicated smallest-width student: this is the
                                # LAST budget level's CE/KD, not the narrowest width's.
                                ("matformer/level_ce", "_mf_last_level_ce"),
                                ("matformer/level_kd", "_mf_last_level_kd")):
                    v = getattr(self, attr, None)
                    if v is not None:
                        logs = {**logs, k: float(v)}
            if vision_elastic:
                for k, attr in (("vision/res", "_mf_last_ve_res"), ("vision/depth", "_mf_last_ve_depth"),
                                ("vision/dense_cos_loss", "_mf_last_ve_dense"),
                                ("vision/soft_cos_loss", "_mf_last_ve_soft")):
                    v = getattr(self, attr, None)
                    if v is not None:
                        logs = {**logs, k: float(v)}
            # GOAL-ROUTER diagnostics: surrogate fit loss (should fall), router loss, and the
            # router's chosen mean width (should track the sampled budgets, ~mid).
            for k, attr in (("router/surrogate_loss", "_mf_last_surr_loss"),
                            # F11 / audit §3.10: share of target variance explained by CONFIG
                            # IDENTITY. This, not surrogate_loss, decides whether a config-only
                            # predictor can work at all (surrogate_loss mixes in the irreducible
                            # within-config, per-example spread).
                            ("router/surr_var_between_configs", "_mf_surr_var_between"),
                            ("router/surr_target_std", "_mf_surr_target_std"),
                            ("router/surr_n_configs", "_mf_surr_n_configs"),
                            ("router/surr_var_between_null", "_mf_surr_var_between_null"),
                            ("router/surr_var_example_block", "_mf_surr_var_example_block"),
                            ("router/surr_n_examples", "_mf_surr_n_examples"),
                            # THE criterion: config main effect vs the config x example interaction
                            # in a blocked two-way layout. F ~ 1 => no config effect.
                            ("router/surr_F", "_mf_surr_F"),
                            # THE number: unbiased variance explained by config identity.
                            ("router/surr_omega_sq", "_mf_surr_omega_sq"),
                            ("router/router_loss", "_mf_last_router_loss"),
                            ("router/uniform_regret", "_mf_last_uniform_regret"),
                            ("router/uniform_regret_max", "_mf_last_uniform_regret_max"),
                            ("router/uniform_regret_active_cells", "_mf_last_uniform_regret_active"),
                            ("router/uniform_regret_present_cells", "_mf_last_uniform_regret_present"),
                            ("router/uniform_ce", "_mf_last_uniform_ce"),
                            ("router/mean_width", "_mf_last_router_meanw"),
                            # NOT a collapse check — length-1 in `one` mode, so min==max always.
                            ("router/mean_width_min", "_mf_router_meanw_min"),
                            ("router/mean_width_max", "_mf_router_meanw_max"),
                            # THE collapse check: deterministic argmax over ALL budget levels.
                            # n_distinct == 1 -> collapsed; == n_levels -> fully separated.
                            ("router/n_distinct", "_mf_router_ndistinct"),
                            ("router/target_err", "_mf_router_target_err"),
                            ("router/min_margin", "_mf_router_min_margin"),
                            ("router/max_abs_logit", "_mf_router_max_logit"),
                            ("router/budget_all", "_mf_router_budget_all"),
                            # ceiling-semantics diagnostics — n_over is the one that decides
                            # feasibility; target_err is retained only for continuity with §3.10.
                            ("router/n_over_budget", "_mf_router_n_over"),
                            ("router/min_margin_raw", "_mf_router_min_margin_raw"),
                            # THE decision margin of a constrained decode (best minus second-best
                            # FEASIBLE configuration). Unlike n_distinct/target_err/frac_L*, this
                            # is not constant by construction — audit §3.4/§3.8.
                            ("router/min_decision_gap", "_mf_router_min_decision_gap"),
                            ("router/max_abs_logit_raw", "_mf_router_max_logit_raw")):
                v = getattr(self, attr, None)
                if v is not None:
                    logs = {**logs, k: float(v)}
            _gaps = getattr(self, "_mf_router_decision_gaps", None)
            if _gaps is not None:
                logs = {**logs,
                        **{f"router/decision_gap_L{i}": float(x) for i, x in enumerate(_gaps)}}
            for k, attr in (("router/tau", "_mf_router_tau"),
                            ("router/kappa", "_mf_router_kappa")):
                v = getattr(self, attr, None)
                if v is not None:
                    logs = {**logs, k: float(v)}
            # per-level realized fraction from the argmax — the trajectory that shows WHEN levels
            # separate (or when they merge). Logged individually so a flat line is visible.
            _fr = getattr(self, "_mf_router_argmax_fracs", None)
            if _fr is not None:
                logs = {**logs, **{f"router/frac_L{i}": float(x) for i, x in enumerate(_fr)}}
            return super().log(logs, *args, **kwargs)

        def _fwd_loss(self, model, inputs):
            """Loss-only forward: the model computes the masked-CE LM loss
            directly (collator supplies `labels`). Avoids TRL compute_loss's
            per-call entropy / token-accuracy / distributed-gather work, which
            would otherwise run 4× per joint step (codex rank-2).

            use_cache=False: bypassing TRL compute_loss also bypasses its
            inputs["use_cache"]=False, and with gradient_checkpointing off
            nothing else disables it — every forward would build and discard
            a KV cache (PERF §17)."""
            with self.compute_loss_context_manager():
                out = model(**inputs, use_cache=False)
            return out.loss

        # ── vision-feature cache (codex round-2 rank-1, option B) ──────────
        # The frozen SigLIP tower produces identical features across the 4
        # granularity forwards (FFN nesting only touches the language decoder).
        # Compute it once per step under no_grad, then temporarily swap
        # vision_tower.forward to a stub returning the cached hidden state so
        # the normal Gemma3 forward re-runs only the (trainable) projector +
        # token-merge each granularity. Removes 3/4 SigLIP forwards (~15-35%).
        def _gemma_core(self, model):
            return self.accelerator.unwrap_model(model).model

        def _cached_vision_hidden(self, model, inputs):
            pixel_values = inputs.get("pixel_values")
            if pixel_values is None:
                return None
            core = self._gemma_core(model)
            # The no_grad + stub-swap below silently drops vision gradients, so it
            # is only sound with a frozen tower — fail loudly otherwise (codex 07-21 #7).
            if not getattr(self, "_vision_frozen_ok", False):
                if any(p.requires_grad for p in core.vision_tower.parameters()):
                    raise RuntimeError(
                        "vision-hidden cache requires a frozen vision tower "
                        "(set vision_freeze: true or disable the cache)")
                self._vision_frozen_ok = True
            with self.compute_loss_context_manager(), torch.no_grad():
                out = core.vision_tower(pixel_values=pixel_values, return_dict=True)
            return out.last_hidden_state.detach()

        @contextlib.contextmanager
        def _use_cached_vision_hidden(self, model, hidden):
            if hidden is None:
                yield
                return
            core = self._gemma_core(model)
            orig_forward = core.vision_tower.forward

            def cached_forward(*args, **kwargs):
                return BaseModelOutputWithPooling(
                    last_hidden_state=hidden, pooler_output=None)

            core.vision_tower.forward = cached_forward
            try:
                yield
            finally:
                core.vision_tower.forward = orig_forward

        def _fwd_loss_cached_vision(self, model, inputs, hidden):
            with self._use_cached_vision_hidden(model, hidden):
                return self._fwd_loss(model, inputs)

        # ── self-distillation (Arm B/C) forward + KD loss ─────────────────
        def _fwd_out(self, model, inputs):
            """Logits-only forward for the sandwich arms. All callers consume
            .logits and compute CE/KD themselves (_ce_per_example / _kd_kl with
            the labels held by the caller) — so `labels` is dropped here: passing
            it would make HF compute a throwaway model-native CE per forward,
            which upcasts the full [B,S,262k-vocab] logits to fp32 (~GBs of
            transient, 4×/microbatch) for a loss nobody reads (codex 07-21 #1).

            use_cache=False for the same reason as _fwd_loss (PERF §17)."""
            inp = {k: v for k, v in inputs.items() if k != "labels"}
            with self.compute_loss_context_manager():
                return model(**inp, use_cache=False)

        def _fwd_out_cached_vision(self, model, inputs, hidden):
            with self._use_cached_vision_hidden(model, hidden):
                return self._fwd_out(model, inputs)

        # ── vision-elastic helpers (options A+B; see the config block near `vision_elastic`) ──
        def _ve_sample(self):
            rng = random.Random((self._mf_seed << 24) ^ 0x5A5A00 ^ int(self.state.global_step))
            return (rng.choices(ve_res, weights=ve_res_w, k=1)[0],
                    rng.choices(ve_depths, weights=ve_depth_w, k=1)[0])

        @contextlib.contextmanager
        def _ve_student_config(self, core, res, depth):
            """Put the STUDENT tower + projector into (res, depth) for one step: projector pool kernel
            for `res`, encoder truncated to its first `depth` blocks (post-layernorm then acts on block
            `depth`'s output). Restored on exit, so checkpoints always hold the full tower."""
            vt, proj = core.vision_tower, core.multi_modal_projector
            enc = vt.vision_model.encoder
            patch = int(vt.config.patch_size)
            grid = res // patch
            tps = int(proj.tokens_per_side)
            if res % patch or grid % tps:
                raise ValueError(f"vision_elastic: resolution {res} incompatible with patch {patch} / "
                                 f"tokens_per_side {tps}")
            full_layers = enc.layers
            if depth < 1 or depth > len(full_layers):
                raise ValueError(f"vision_elastic: depth {depth} outside 1..{len(full_layers)}")
            saved = (proj.patches_per_image, proj.kernel_size, proj.avg_pool)
            try:
                proj.patches_per_image = grid
                proj.kernel_size = grid // tps
                proj.avg_pool = torch.nn.AvgPool2d(kernel_size=grid // tps, stride=grid // tps)
                if depth < len(full_layers):
                    enc.layers = torch.nn.ModuleList(list(full_layers)[:depth])
                yield grid
            finally:
                proj.patches_per_image, proj.kernel_size, proj.avg_pool = saved
                enc.layers = full_layers

        def _ve_student_vision(self, core, pixel_values, res):
            native = int(core.vision_tower.config.image_size)
            pv = pixel_values
            if res != native:
                pv = torch.nn.functional.interpolate(
                    pixel_values.float(), size=(res, res), mode="bilinear",
                    antialias=True, align_corners=False).to(pixel_values.dtype)
            with self.compute_loss_context_manager():
                s_vis = core.vision_tower(pixel_values=pv, interpolate_pos_encoding=True,
                                          return_dict=True).last_hidden_state
                s_soft = core.multi_modal_projector(s_vis)
            return s_vis, s_soft

        @staticmethod
        def _ve_pool_teacher(t_vis, grid):
            B, N, C = t_vis.shape
            g0 = int(round(N ** 0.5))
            if g0 == grid:
                return t_vis
            x = t_vis.transpose(1, 2).reshape(B, C, g0, g0)
            x = torch.nn.functional.adaptive_avg_pool2d(x.float(), (grid, grid)).to(t_vis.dtype)
            return x.flatten(2).transpose(1, 2)

        def _ve_losses(self, s_vis, s_soft, t_vis, t_soft, grid):
            t_p = self._ve_pool_teacher(t_vis, grid)
            dense = 1.0 - torch.nn.functional.cosine_similarity(s_vis.float(), t_p.float(), dim=-1).mean()
            soft = 1.0 - torch.nn.functional.cosine_similarity(s_soft.float(), t_soft.float(), dim=-1).mean()
            return dense, soft

        @staticmethod
        def _masked_shift(labels):
            """Causal-shift supervised positions. Returns (mask (B,S-1) bool,
            seq_id (N,) long) where seq_id[k] = which sequence the k-th supervised
            token belongs to — the segment index used for PER-EXAMPLE reduction."""
            mask = labels[:, 1:] != -100                       # (B, S-1)
            B = labels.shape[0]
            seq_idx = torch.arange(B, device=labels.device)[:, None].expand(-1, mask.shape[1])
            return mask, seq_idx[mask]

        @staticmethod
        def _mean_over_examples(vals, seq_id, B):
            """PER-EXAMPLE normalization (codex P0 #2): mean over SEQUENCES of each
            sequence's mean-over-its-supervised-tokens. Every example contributes
            ONE unit regardless of answer length, so a ~30-token U5 box answer no
            longer carries ~30× the gradient of a 1-token U3 Yes/No — which, stacked
            on the A2 tau=0.5 P(task)∝√N row-frequency skew, was drowning the closed
            tasks. `vals` (N,) are per-supervised-token losses; `seq_id` (N,) their
            sequence index in [0,B). Returns a scalar (0 keep-graph if none)."""
            counts = torch.zeros(B, device=vals.device, dtype=vals.dtype).scatter_add_(
                0, seq_id, torch.ones_like(vals))
            sums = torch.zeros(B, device=vals.device, dtype=vals.dtype).scatter_add_(
                0, seq_id, vals)
            present = counts > 0
            if not bool(present.any()):
                return vals.sum() * 0.0
            return (sums[present] / counts[present]).mean()

        def _ce_per_example(self, logits, labels):
            """Hard-label CE, PER-EXAMPLE normalized (replaces HF's token-mean
            out.loss). Gathers supervised positions first (N ≪ B·S; assistant
            answers are short) so the [N,V] softmax stays a small fraction of the
            full-logits spike, then averages per sequence then over sequences."""
            mask, seq_id = self._masked_shift(labels)
            if mask.sum() == 0:
                return logits.sum() * 0.0
            sl = logits[:, :-1, :][mask].float()               # (N, V)
            tgt = labels[:, 1:][mask]                           # (N,)
            ce = torch.nn.functional.cross_entropy(sl, tgt, reduction="none")  # (N,)
            return self._mean_over_examples(ce, seq_id, labels.shape[0])

        def _ce_per_example_vec(self, logits, labels):
            """(B,) per-example CE over each example's supervised tokens (NaN if none). Pass DETACHED
            logits — this is a router/surrogate TARGET, not a backbone-training loss."""
            mask, seq_id = self._masked_shift(labels)
            B = labels.shape[0]
            out = torch.full((B,), float("nan"), device=logits.device, dtype=torch.float32)
            if mask.sum() == 0:
                return out
            sl = logits[:, :-1, :][mask].float()
            tgt = labels[:, 1:][mask]
            ce = torch.nn.functional.cross_entropy(sl, tgt, reduction="none")
            cnt = torch.zeros(B, device=ce.device, dtype=ce.dtype).scatter_add_(
                0, seq_id, torch.ones_like(ce))
            summ = torch.zeros(B, device=ce.device, dtype=ce.dtype).scatter_add_(0, seq_id, ce)
            ok = cnt > 0
            out[ok] = (summ[ok] / cnt[ok]).float()
            return out

        def _kd_kl(self, student_logits, teacher_logits, labels, temp):
            """Temperature-scaled KD divergence KL(p_teacher || p_student) on the
            ASSISTANT positions only (labels != -100), PER-EXAMPLE normalized to
            match _ce_per_example (codex P0 #2 — was reduction='batchmean' =
            token-mean, same length bias as the old CE). Gathering the valid
            positions first keeps it cheap (N ≪ B·S). Standard Hinton ×T²."""
            mask, seq_id = self._masked_shift(labels)
            if mask.sum() == 0:
                return student_logits.sum() * 0.0              # keep-graph zero
            sl = student_logits[:, :-1, :][mask]               # (N, V)
            tl = teacher_logits[:, :-1, :][mask]               # (N, V)
            logp_s = torch.log_softmax(sl.float() / temp, dim=-1)
            logp_t = torch.log_softmax(tl.float() / temp, dim=-1)
            p_t = logp_t.exp()
            kl_tok = (p_t * (logp_t - logp_s)).sum(-1)         # (N,) KL(p_t||p_s)/token
            # ── CORRECTNESS GATE ("reject sampling", `kd_correctness_gate: true`) ──────────
            # Distill ONLY where the self-teacher is actually right: keep the KL at answer tokens
            # whose teacher top-1 == gold, drop it elsewhere so we never distill the teacher's
            # mistakes (there the CE on gold is the only signal). Same rule as the DFT line's
            # `kl_correctness_gate` (train_v2/distill_dft.py) — the project's meaning of "reject
            # sampling" is this inline per-token gate, NOT generate-and-filter. Free: the teacher
            # logits are already in hand. Filtering the token list (rather than zeroing) keeps
            # _mean_over_examples averaging each example over its KEPT tokens, matching
            # distill_dft's sum/keep_count; examples with zero kept tokens drop out there.
            # NOT ported: distill_dft's `gate_full_ce` (full-strength CE on teacher-wrong tokens),
            # because the sandwich path computes CE as a scalar, not per token.
            if kd_correctness_gate:
                tgt = labels[:, 1:][mask]                      # (N,) gold token ids
                keep = tl.argmax(-1) == tgt
                if not bool(keep.any()):
                    return student_logits.sum() * 0.0          # keep-graph zero
                kl_tok = kl_tok[keep]
                seq_id = seq_id[keep]
                self._mf_last_kd_gate_keep = float(keep.float().mean())
            kl = self._mean_over_examples(kl_tok, seq_id, labels.shape[0])
            return kl * (temp * temp)

        def _any_rank_empty(self, is_empty):
            """DDP-safe (codex round-2 caveat): if ANY rank's batch is empty
            (all images unreadable), ALL ranks must skip in lockstep — else the
            empty rank does 0 backwards while others do 4 → all-reduce count
            mismatch → hang. One tiny MAX all-reduce/step coordinates this."""
            if not (dist.is_available() and dist.is_initialized()):
                return is_empty
            t = torch.tensor([1.0 if is_empty else 0.0], device=self.args.device)
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            return bool(t.item() > 0)

        def training_step(self, model, inputs, num_items_in_batch=None):
            model.train()
            inputs = self._prepare_inputs(inputs)
            # ROUTER guides: pop BEFORE any model() forward (they are not model kwargs). None when
            # the collator did not emit them (router off) or on an empty batch.
            self._router_pillar = inputs.pop("router_pillar", None)
            self._router_task = inputs.pop("router_task", None)
            local_empty = inputs.get("input_ids") is None or inputs["input_ids"].numel() == 0
            if self._any_rank_empty(local_empty):
                # All ranks skip in lockstep, so global_step stays aligned (no
                # all-reduce-COUNT mismatch / hang). BUT under grad_accum>1, HF defers
                # the accumulation group's all-reduce to the FINAL microbatch; if the
                # skipped one IS that final microbatch, a plain `return zeros` would let
                # optimizer.step() run on the earlier microbatch's UN-synced local grads
                # → DDP replicas diverge permanently (all-reduce syncs grads, never
                # re-syncs weights). So fire a zero-valued backward touching every
                # trainable param: it adds nothing to the accumulated grad but triggers
                # the deferred all-reduce (and correctly no-ops under HF's ambient
                # no_sync when this is a NON-final microbatch). Rare path (unreadable
                # batch); affects joint and stochastic identically.
                zero = sum(p.sum() for p in model.parameters() if p.requires_grad) * 0.0
                self.accelerator.backward(zero)
                return zero.detach()
            ga = self.args.gradient_accumulation_steps

            if sampling == "mixed":
                # ── Arm A: mixing-aware hard-label SFT ─────────────────────
                # ONE per-layer heterogeneous config per optimizer step (drawn
                # deterministically from (seed, global_step)); normal masked-CE
                # loss. Extends stochastic (which only ever set a UNIFORM width)
                # to the per-layer axis + guarantees the narrow widths are drawn.
                kind, wl = _sample_perlayer_config(
                    self._mf_seed, self.state.global_step, widths, weights,
                    n_layers, mix_kind_weights, mix_high_late)
                self._mf_last_kind = kind
                self._mf_last_meanw = sum(wl) / n_layers
                _set_perlayer(wl)
                loss = self._fwd_loss(model, inputs) / ga
                self.accelerator.backward(loss)
                return loss.detach()

            if sampling in ("sandwich", "mixed_sandwich") and router_enable and router_mode == "gumbel" \
                    and getattr(self.accelerator.unwrap_model(model), "width_router", None) is not None:
                # ── NEMOTRON-ELASTIC: co-train backbone + router (Gumbel-ST distillation) ──
                # Route one or more registered budgets for the batch. Router → Gumbel-ST per-neuron
                # GATE. TEACHER = moving full-width SELF or an opt-in frozen initial copy (no_grad).
                # STUDENT = the GATED (routed) forward: KD gradient flows THROUGH the gate to the
                # router AND through the FFN to the backbone (true co-train). No surrogate.
                # loss = kd_alpha·CE + (1-kd_alpha)·KD + budget·|meanwidth-budget|; a full-width CE
                # anchor keeps the self-teacher good (and gives the DDP-synced final backward).
                bundle = self.accelerator.unwrap_model(model).width_router
                labels = inputs["labels"]
                dev = labels.device
                vision_hidden = self._cached_vision_hidden(model, inputs)
                # NEMOTRON discrete budget. Which LEVELS to forward this microbatch:
                # "one" is random, "stratified" balances rank x accumulation draws deterministically,
                # and "minmax"/"all" forwards several levels on the same data. Teacher is shared; each
                # level gets its own gated forward + backward so the graph is freed between levels.
                n_levels = int(bundle.budget_levels.numel())
                # ── PER-RANK routing RNG ──────────────────────────────────────────────────────
                # BUG FIXED 2026-07-27: the default device RNG is IDENTICAL across DDP ranks
                # (verified from the saved rng_state of checkpoints 2500/5000/10000/30000 of
                # router_cotrain_scale), so every rank drew the SAME level and the SAME Gumbels.
                # That run therefore had at most `grad_accum` distinct routing draws per optimizer
                # step — not the `grad_accum x world_size` its config header claimed.
                _rgen = getattr(self, "_router_gen", None)
                _rk = (torch.distributed.get_rank()
                       if (torch.distributed.is_available() and torch.distributed.is_initialized())
                       else 0)
                _router_step = int(self.state.global_step)
                # Re-seed at each optimizer-step boundary. Gradient-accumulation microbatches then
                # consume distinct draws from the same step-local stream, while a checkpoint resume
                # reproduces exactly the stream that an uninterrupted run would start for that step.
                # Rank remains part of the seed, preserving the independent DDP exploration fix.
                if _rgen is None or getattr(self, "_router_rng_step", None) != _router_step:
                    _rgen = torch.Generator(device=dev)
                    _rgen.manual_seed(
                        router_seed_base * 1_000_003
                        + 9973 * int(_rk)
                        + 104729 * _router_step
                        + 1
                    )
                    self._router_gen = _rgen
                    self._router_rng_step = _router_step
                    self._router_microbatch_index = 0
                # EXPLORATION temperature schedule (NOT tau — tau divides out of the argmax and so
                # cannot schedule exploration; see route_gumbel's docstring).
                _prog = min(1.0, float(self.state.global_step) / max(1.0, router_total_steps))
                _t_samp = router_t_sample0 + (router_t_sample1 - router_t_sample0) * _prog
                _tau = exponential_router_temperature(router_tau0, router_tau1, _prog)
                self._mf_router_tau = _tau
                self._mf_router_kappa = 1.0 / max(_t_samp, 1e-12)
                _microbatch_index = int(self._router_microbatch_index)
                self._router_microbatch_index = _microbatch_index + 1
                _ws = (
                    torch.distributed.get_world_size()
                    if (torch.distributed.is_available() and torch.distributed.is_initialized())
                    else 1
                )
                if router_multi_budget == "minmax":
                    active_levels = [0, n_levels - 1]
                elif router_multi_budget == "all":
                    active_levels = list(range(n_levels))
                elif router_multi_budget == "stratified":
                    active_levels = [
                        stratified_budget_level(
                            global_step=_router_step,
                            microbatch_index=_microbatch_index,
                            rank=int(_rk),
                            world_size=int(_ws),
                            microbatches_per_step=int(ga),
                            n_levels=n_levels,
                        )
                    ]
                else:
                    active_levels = [int(torch.randint(0, n_levels, (1,), device=dev, generator=_rgen))]

                def _guide1(ids, drop):
                    if ids is None:
                        return torch.zeros(1, dtype=torch.long, device=dev)
                    keep = torch.rand(1, device=dev) >= drop
                    return torch.where(keep, ids.to(dev).long()[:1] + 1,
                                       torch.zeros(1, dtype=torch.long, device=dev))
                # sample the guide ids ONCE (same guide across levels); rebuild ctx INSIDE the loop so
                # each level owns a live embedding graph (per-level backward frees the previous one).
                pill_ids = _guide1(getattr(self, "_router_pillar", None), router_pillar_drop)
                task_ids = _guide1(getattr(self, "_router_task", None), router_task_drop)
                # Teacher is computed once and shared across levels. Historical runs use the
                # moving full-width self teacher. ``frozen_initial`` uses a separate immutable
                # copy of the starting checkpoint, matching the successful Nemotron-Elastic
                # experiment rather than letting early routing choices move their own target.
                _clear_gates(); _set_uniform(widths[-1])
                teacher_logits = None
                if not ce_only_control:
                    teacher_source = (
                        router_teacher_model
                        if router_teacher_mode == "frozen_initial"
                        else model
                    )
                    if teacher_source is None:
                        raise RuntimeError("frozen router teacher was not initialized")
                    with torch.no_grad():
                        t_out = self._fwd_out_cached_vision(
                            teacher_source, inputs, vision_hidden
                        )
                    teacher_logits = t_out.logits.detach(); del t_out
                total = torch.zeros((), device=dev)
                meanws, last_ce, last_kd = [], 0.0, 0.0
                regret_values, regret_maxes = [], []
                regret_active, regret_present, uniform_ces = [], [], []
                n_act = len(active_levels)
                for li in active_levels:
                    lvl = torch.tensor([li], device=dev)
                    budget_vec = bundle.level_onehot(lvl)                       # (1, n_levels) one-hot
                    target_frac = bundle.budget_levels[lvl]                     # (1,)
                    uniform_ce_vec = None
                    if router_uniform_regret_w > 0.0:
                        _clear_gates()
                        _set_uniform(router_uniform_widths[li])
                        with torch.no_grad():
                            uniform_out = self._fwd_out_cached_vision(
                                model, inputs, vision_hidden
                            )
                        uniform_ce_vec = self._ce_per_example_vec(
                            uniform_out.logits.detach(), labels
                        )
                        finite_uniform = torch.isfinite(uniform_ce_vec)
                        uniform_ces.append(
                            float(uniform_ce_vec[finite_uniform].mean())
                            if bool(finite_uniform.any())
                            else float("nan")
                        )
                        del uniform_out
                    # DDP: every backward before the FINAL full-width one must run under no_sync, or
                    # DDP starts a second all-reduce before the first has finished ("Expected to have
                    # finished reduction in the prior iteration"). This path had only ever run on ONE
                    # GPU before 2026-07-26, where DDP is absent and the omission was invisible.
                    # NOTE the router-gradient consequence, handled after the full-width backward:
                    # the ROUTER's parameters receive gradient ONLY here, and the final synced
                    # backward (full width) does not touch them — so DDP never reduces them and the
                    # ranks would silently diverge. They are all-reduced explicitly below.
                    with self.accelerator.no_sync(model):
                        ctx = bundle.context(pill_ids, task_ids)                # live graph per level
                        route_fn = (
                            bundle.route_gumbel_exact_compute
                            if router_exact_compute_gumbel
                            else bundle.route_gumbel
                        )
                        gate, p_layer = route_fn(
                            budget_vec,
                            ctx,
                            tau=_tau,
                            t_sample=_t_samp,
                            generator=_rgen,
                        )  # (1,L,full),(1,L,K)
                        _set_gates(gate[0])
                        s_out = self._fwd_out_cached_vision(model, inputs, vision_hidden)
                        _clear_gates()
                        if uniform_ce_vec is None:
                            ce = self._ce_per_example(s_out.logits, labels)
                            ce_vec = None
                        else:
                            ce_vec = self._ce_per_example_vec(s_out.logits, labels)
                            finite_ce = torch.isfinite(ce_vec)
                            ce = (
                                ce_vec[finite_ce].mean()
                                if bool(finite_ce.any())
                                else s_out.logits.sum() * 0.0
                            )
                        frac = (p_layer[0] * bundle.widths).sum(-1).mean() / bundle.widths[-1]   # scalar, diff
                        bud_pen = (frac - target_frac.squeeze()) ** 2           # L2 cost-matching to level
                        if teacher_logits is not None:
                            kd = self._kd_kl(s_out.logits, teacher_logits, labels, kd_temp)
                            student_loss = kd_alpha * ce + (1.0 - kd_alpha) * kd
                            last_kd = float(kd.detach())
                        else:
                            student_loss = ce
                            last_kd = 0.0
                        regret_loss = torch.zeros((), device=dev)
                        if uniform_ce_vec is not None:
                            if self._router_task is None or self._router_pillar is None:
                                raise RuntimeError(
                                    "matched-uniform regret requires collated task and pillar ids"
                                )
                            # DISTINCT names from the `task_ids`/`pill_ids` GUIDE tensors bound
                            # above: those are (1,) and +1-shifted for the NULL-slot embedding,
                            # these are the raw (B,) ids used only to form regret cells. Reusing
                            # the name `task_ids` here rebound the guide, so a second pass of this
                            # loop would have rebuilt `ctx` from the wrong tensor (wrong shape,
                            # unshifted ids). Latent only because `stratified` forwards one level
                            # per microbatch; it would fire under router_multi_budget: all|minmax.
                            cell_task_ids = self._router_task.to(device=dev, dtype=torch.long)
                            pillar_ids = self._router_pillar.to(device=dev, dtype=torch.long)
                            n_tasks = len(TASK2ID)
                            n_pillars = len(PILLAR2ID)
                            # cell_task_ids, NOT task_ids: the latter is the (1,)-shaped router
                            # guide bound above, which is identically 0 under
                            # router_task_dropout=1.0. Reading it here silently collapses the
                            # regret cells from budget x task x pillar down to budget x pillar.
                            # That is exactly what the F7 rename broke on 2026-08-05 — the
                            # binding was renamed and this use site was not — so keep the two
                            # names distinct and never reintroduce `task_ids` on this line.
                            cell_ids = cell_task_ids * n_pillars + pillar_ids
                            group_ids = (
                                int(li) * n_tasks * n_pillars + cell_ids
                            )
                            regret_loss, regret_stats = groupwise_uniform_regret_hinge(
                                ce_vec,
                                uniform_ce_vec,
                                group_ids,
                                n_groups=n_levels * n_tasks * n_pillars,
                                margin=router_uniform_regret_margin,
                            )
                            regret_values.append(float(regret_loss.detach()))
                            regret_maxes.append(float(regret_stats["max_regret"]))
                            regret_active.append(float(regret_stats["n_active"]))
                            regret_present.append(float(regret_stats["n_present"]))
                        # /n_act keeps the per-step backbone update magnitude ~constant across #levels;
                        # per-level backward frees the student graph before the next level.
                        loss = (
                            student_loss
                            + router_budget_w * bud_pen
                            + router_uniform_regret_w * regret_loss
                        ) / (ga * n_act)
                        self.accelerator.backward(loss)
                    total = total + loss.detach()
                    last_ce = float(ce.detach())
                    meanws.append(float(frac.detach()) * float(widths[-1]))
                if regret_values:
                    self._mf_last_uniform_regret = sum(regret_values) / len(regret_values)
                    self._mf_last_uniform_regret_max = max(regret_maxes)
                    self._mf_last_uniform_regret_active = sum(regret_active) / len(regret_active)
                    self._mf_last_uniform_regret_present = sum(regret_present) / len(regret_present)
                    self._mf_last_uniform_ce = sum(uniform_ces) / len(uniform_ces)
                # ── ALL-LEVEL budget loss: router-only, no extra 4B-param forward ─────────────
                # The per-level loop above penalises ONLY the level it sampled, so each level's
                # parameters saw the budget gradient ~1/n_levels of the time while any shared
                # trunk saw it always — the asymmetry that let the trunk outgrow the conditioning
                # in router_cotrain_scale. This term supervises EVERY level EVERY step at the cost
                # of one 5-row router call. `router_budget_t` is deliberately high so this term's
                # softmax stays unsaturated and keeps gradient after the sampling path has
                # committed (that run's penalty derivative was a healthy 0.941; it was the
                # fraction-to-logit jacobian at ~1e-14 that killed it).
                if router_budget_all and router_budget_all_w > 0.0:
                    with self.accelerator.no_sync(model):     # DDP: not the final backward
                        if router_budget_mode == "ceiling":
                            _bl, _bfr = bundle.budget_hinge_exact(t_soft=router_budget_t)
                        else:
                            _bl, _bfr = bundle.budget_loss_all_levels(t_budget=router_budget_t,
                                                                      hard=router_budget_hard)
                        _blw = (router_budget_all_w * _bl) / ga
                        self.accelerator.backward(_blw)
                    total = total + _blw.detach()
                    self._mf_router_budget_all = float(_bl.detach())
                # ── honest collapse diagnostic, from the deterministic ARGMAX ─────────────────
                if router_readout_every > 0 and (int(self.state.global_step) % router_readout_every == 0):
                    if router_readout_mode == "constrained":
                        _ro = bundle.constrained_readout(
                            min_utilization=router_readout_min_util)
                    else:
                        _ro = bundle.deterministic_readout()
                    self._mf_router_ndistinct = float(_ro["n_distinct"])
                    self._mf_router_target_err = float(_ro["target_err"])
                    # A constrained decision is global, so a local top1-vs-top2 group margin is
                    # not its decision margin. Keep the legacy fields defined without inventing
                    # that statistic; the exported integer config remains the authoritative output.
                    self._mf_router_min_margin = float(_ro.get("min_margin", 0.0))
                    self._mf_router_max_logit = float(_ro.get("max_abs_logit", 0.0))
                    self._mf_router_argmax_fracs = _ro["fracs"]
                    # ── the only two live fields with information content ────────────────────
                    # Under `constrained` + min_utilization 1.0, n_distinct / target_err /
                    # n_over_budget / frac_L* are CONSTANT for ANY table (verified on an all-zero
                    # table, audit §3.4), so a run monitored on them cannot show anything. The
                    # table norm and the best-vs-second FEASIBLE-config gap do move; log them.
                    self._mf_router_min_decision_gap = _ro.get("min_decision_gap")
                    self._mf_router_decision_gaps = _ro.get("decision_gap")
                    # CEILING diagnostics: how many levels EXCEED their budget, and where the
                    # decision is tightest. `target_err` alone hides this — a level 0.015 over its
                    # ceiling reads as "accurate" and is in fact infeasible (§3.11).
                    self._mf_router_n_over = float(_ro.get("n_over_budget", 0))
                    self._mf_router_min_margin_raw = float(_ro.get("min_margin", 0.0))
                    self._mf_router_max_logit_raw = float(_ro.get("max_abs_logit_raw", 0.0))
                # SANDWICH: explicit SMALLEST-width student (uniform widths[0]), inplace-distilled from
                # the same full-width teacher. Without this the narrowest width receives no gradient at
                # all in the router path — the budget levels are interior, so the router never requests
                # it — and the backbone can drift there over a long run. That drift would also corrupt
                # the gain eval, which uses uni_1280 as a measured baseline.
                if router_train_smallest:
                    _clear_gates(); _set_uniform(widths[0])
                    with self.accelerator.no_sync(model):     # DDP: not the final backward
                        sm_out = self._fwd_out_cached_vision(model, inputs, vision_hidden)
                        sm_ce = self._ce_per_example(sm_out.logits, labels)
                        if teacher_logits is not None:
                            sm_kd = self._kd_kl(sm_out.logits, teacher_logits, labels, kd_temp)
                            sm_loss = kd_alpha * sm_ce + (1.0 - kd_alpha) * sm_kd
                            self._mf_last_smallest_kd = float(sm_kd.detach())
                        else:
                            sm_loss = sm_ce
                            self._mf_last_smallest_kd = 0.0
                        self._mf_last_smallest_ce = float(sm_ce.detach())
                        smloss = (router_smallest_weight * sm_loss) / ga
                        self.accelerator.backward(smloss)
                    total = total + smloss.detach()
                else:
                    # No dedicated smallest-width forward ran. Report the LAST budget level's CE/KD under
                    # a name that says so — calling it "smallest" would be false (it is whatever width the
                    # router chose for the last active level).
                    self._mf_last_level_kd = last_kd
                    self._mf_last_level_ce = last_ce
                self._mf_last_router_meanw = sum(meanws) / len(meanws)
                self._mf_last_meanw = self._mf_last_router_meanw
                # Spread over the levels FORWARDED THIS MICROBATCH. Its old comment called this
                # "the LIVE budget-responsiveness signal", which was FALSE and cost us a 21-hour
                # run's worth of blind monitoring: in `router_multi_budget: one` mode `meanws` has
                # length 1, so min == max identically and collapse is undetectable in principle.
                # Judge budget responsiveness from `router/n_distinct` (deterministic argmax over
                # ALL levels, set by router_readout_every above) — never from these two.
                self._mf_router_meanw_min = min(meanws)
                self._mf_router_meanw_max = max(meanws)
                # Full-width anchor. In joint mode its CE updates/preserves the moving
                # teacher and provides DDP's final synced backward. In frozen-backbone
                # direct mode there is no full-width trainable path: reuse the no-grad
                # teacher logits for monitoring and let only routed students train the
                # Router. Backward on full CE would be both meaningless and invalid.
                _set_uniform(widths[-1])
                if router_freeze_backbone:
                    if teacher_logits is None:
                        with torch.no_grad():
                            f_out = self._fwd_out_cached_vision(
                                model, inputs, vision_hidden
                            )
                        full_logits = f_out.logits.detach()
                        del f_out
                    else:
                        full_logits = teacher_logits
                    full_ce = self._ce_per_example(full_logits, labels)
                    floss = None
                else:
                    f_out = self._fwd_out_cached_vision(model, inputs, vision_hidden)
                    full_ce = self._ce_per_example(f_out.logits, labels)
                    floss = (full_loss_weight * full_ce) / ga
                    self.accelerator.backward(floss)
                    del f_out
                self._mf_last_full_ce = float(full_ce.detach())
                # ── DDP: all-reduce the ROUTER's gradients by hand ────────────────────────────
                # The full-width backward above is DDP's single synced backward, and it covers every
                # FFN parameter — but it does NOT touch `width_router.*`, whose only gradient came
                # from the no_sync'd level backwards. Without this the four ranks would each keep a
                # private router and drift apart silently (no error, just a meaningless router).
                # Mean, matching DDP's own reduction. Safe under grad-accum: re-reducing an already
                # identical tensor is a no-op, so accumulate-then-reduce and reduce-per-microbatch
                # give the same result.
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    ws = torch.distributed.get_world_size()
                    if ws > 1:
                        rb = getattr(self.accelerator.unwrap_model(model), "width_router", None)
                        if rb is not None:
                            for prm in rb.parameters():
                                if prm.grad is not None:
                                    torch.distributed.all_reduce(prm.grad, op=torch.distributed.ReduceOp.SUM)
                                    prm.grad /= ws
                if teacher_logits is not None:
                    del teacher_logits
                return total if floss is None else total + floss.detach()

            if sampling in ("sandwich", "mixed_sandwich") and router_freeze_backbone \
                    and getattr(self.accelerator.unwrap_model(model), "width_router", None) is not None:
                # ── FROZEN SURROGATE PATH: train only RouterBundle ──
                # Do NOT train the backbone: FORWARD a diverse config set under no_grad to collect
                # per-example CE targets, then two-phase-train the heads. Surrogate target is
                # stationary (frozen backbone) and only heads carry grad (no clip pollution).
                bundle = self.accelerator.unwrap_model(model).width_router
                labels = inputs["labels"]
                vision_hidden = self._cached_vision_hidden(model, inputs)
                student_cfgs = _sandwich_student_configs(
                    self._mf_seed, self.state.global_step, widths, weights, n_layers,
                    per_layer_mixed=(sampling == "mixed_sandwich"), n_random=sw_n_random,
                    always_smallest=sw_smallest, kind_weights=mix_kind_weights,
                    high_late_bias=mix_high_late, budget_levels=sw_budget_levels,
                    budget_min_utilization=sw_budget_min_util,
                    budget_n_groups=sw_budget_groups)
                r_cfgs, r_cev = [], []
                with torch.no_grad():
                    for wl in student_cfgs:
                        _set_perlayer(wl)
                        s_out = self._fwd_out_cached_vision(model, inputs, vision_hidden)
                        r_cfgs.append(list(wl))
                        r_cev.append(self._ce_per_example_vec(s_out.logits, labels))
                    _set_uniform(widths[-1])
                    f_out = self._fwd_out_cached_vision(model, inputs, vision_hidden)
                    r_cfgs.append([widths[-1]] * n_layers)
                    r_cev.append(self._ce_per_example_vec(f_out.logits, labels))
                full_cev = r_cev[-1]
                fin = torch.isfinite(full_cev)
                self._mf_last_full_ce = float(full_cev[fin].mean()) if bool(fin.any()) else 0.0
                self._mf_last_meanw = (sum(sum(c) for c in student_cfgs)
                                       / (len(student_cfgs) * n_layers)) if student_cfgs else float(widths[-1])
                dev = labels.device
                B = labels.shape[0]

                def _guide(ids, drop):
                    if ids is None:
                        return torch.zeros(B, dtype=torch.long, device=dev)
                    keep = torch.rand(B, device=dev) >= drop
                    return torch.where(keep, ids.to(dev).long() + 1,
                                       torch.zeros(B, dtype=torch.long, device=dev))
                ctx = bundle.context(_guide(getattr(self, "_router_pillar", None), router_pillar_drop),
                                     _guide(getattr(self, "_router_task", None), router_task_drop))
                ret = torch.zeros((), device=dev)
                # Phase A — surrogate Huber on (CE - CE_full), per example, NaN-masked.
                s_loss, nseen = None, 0
                for c, cev in zip(r_cfgs, r_cev):
                    tgt = cev - full_cev
                    m = torch.isfinite(tgt)
                    if not bool(m.any()):
                        continue
                    oh = bundle.layer_onehot(c, dev).expand(B, -1, -1)
                    pred = bundle.surrogate(oh, ctx)
                    term = torch.nn.functional.huber_loss(pred[m], tgt[m].detach())
                    s_loss = term if s_loss is None else s_loss + term
                    nseen += 1
                # ── CAN a config-only predictor work AT ALL? (audit F11 / §3.10) ──────────────
                # Every student scores the SAME examples, so this is a BLOCKED (repeated-measures)
                # layout, not independent groups. Treating it as independent puts the shared
                # "how sensitive is THIS example to being narrowed" variance into the error term,
                # which biases the config effect toward zero -- observed on job 38635725, where the
                # naive ratio sat consistently BELOW its own independence null (0.048 vs 0.106),
                # the signature of exactly that block effect. So do the two-way decomposition:
                # remove the example main effect first, then test the config main effect against
                # the config x example interaction.
                #   F ~ 1 and partial omega^2 ~ 0  ==>  at fixed compute, configuration identity
                #   does not predict loss; no surrogate can help and the Gumbel task gradient every
                #   prior arm relied on is noise.
                # The full-width row is excluded: its target is identically 0 by construction.
                with torch.no_grad():
                    keep = torch.isfinite(full_cev)
                    for _cev in r_cev[:-1]:
                        keep = keep & torch.isfinite(_cev)
                    n_ex = int(keep.sum())
                    rows = [(_cev - full_cev)[keep].float() for _cev in r_cev[:-1]]
                    k_cfg = len(rows)
                    if k_cfg >= 2 and n_ex >= 2:
                        M = torch.stack(rows)                       # (k_cfg, n_ex)
                        gm = M.mean()
                        m_c = M.mean(dim=1, keepdim=True)           # config main effect
                        m_i = M.mean(dim=0, keepdim=True)           # EXAMPLE main effect (block)
                        ss_tot = ((M - gm) ** 2).sum()
                        ss_cfg = float(n_ex) * ((m_c - gm) ** 2).sum()
                        ss_ex = float(k_cfg) * ((m_i - gm) ** 2).sum()
                        ss_res = ((M - m_c - m_i + gm) ** 2).sum()
                        df_cfg = max(k_cfg - 1, 1)
                        df_res = max((k_cfg - 1) * (n_ex - 1), 1)
                        ms_cfg = ss_cfg / df_cfg
                        ms_res = ss_res / df_res
                        if float(ms_res) > 0.0:
                            self._mf_surr_F = float(ms_cfg / ms_res)
                            denom_w = df_cfg * (ms_cfg - ms_res) + k_cfg * n_ex * ms_res
                            if float(denom_w) != 0.0:
                                self._mf_surr_omega_sq = float(
                                    df_cfg * (ms_cfg - ms_res) / denom_w)
                        if float(ss_tot) > 0.0:
                            # kept for continuity with the first probe run, NOT the criterion
                            self._mf_surr_var_between = float(ss_cfg / ss_tot)
                            self._mf_surr_var_between_null = (
                                float(k_cfg - 1) / float(max(k_cfg * n_ex - 1, 1)))
                            # how much of the variance the EXAMPLE block takes -- if this is large,
                            # the naive independent-groups statistic was badly biased.
                            self._mf_surr_var_example_block = float(ss_ex / ss_tot)
                        self._mf_surr_target_std = float(M.std())
                        self._mf_surr_n_configs = float(k_cfg)
                        self._mf_surr_n_examples = float(n_ex)
                        # RAW DUMP: so no future statistical question ever needs the GPU again.
                        if int(os.environ.get("RANK", "0")) == 0:
                            try:
                                with open(output_dir / "surrogate_probe_raw.jsonl", "a") as _fh:
                                    _fh.write(json.dumps({
                                        "step": int(self.state.global_step),
                                        "configs": [list(map(int, c)) for c in r_cfgs[:-1]],
                                        "targets": M.detach().cpu().tolist(),
                                        "full_ce": full_cev[keep].detach().cpu().tolist(),
                                    }) + "\n")
                            except Exception:
                                pass
                if s_loss is not None:
                    s_loss = s_loss / nseen
                    self._mf_last_surr_loss = float(s_loss.detach())
                    # The frozen path has two disjoint backwards: surrogate/context first, router
                    # second. Neither backward touches every width_router parameter, so DDP cannot
                    # safely reduce either one as the iteration's single synchronized backward.
                    # Accumulate both locally and explicitly average the complete head below.
                    with self.accelerator.no_sync(model):
                        self.accelerator.backward((surrogate_w * s_loss) / ga)
                    ret = ret + s_loss.detach()
                # Phase B — after warmup: freeze surrogate net, train the router (ST hard config).
                if self.state.global_step >= router_warmup_steps:
                    for p in bundle.surrogate.parameters():
                        p.requires_grad_(False)
                    # Exact, resume-deterministic discrete levels. Each rank receives a rotated
                    # sequence, so an 8-row batch covers every level and the four ranks add useful
                    # diversity without relying on an unsaved RNG stream.
                    _rk = (torch.distributed.get_rank()
                           if (torch.distributed.is_available()
                               and torch.distributed.is_initialized()) else 0)
                    level = (
                        torch.arange(B, device=dev)
                        + int(self.state.global_step) * B
                        + int(_rk) * B
                    ) % bundle.n_levels
                    budget_vec = bundle.level_onehot(level)
                    target = bundle.budget_levels[level]
                    z, _wl = bundle.route(budget_vec, ctx.detach(), tau=router_tau)
                    frac = bundle.mean_width_frac(z)
                    r_loss = (bundle.surrogate(z, ctx.detach()).mean()
                              + router_budget_w * (frac - target).abs().mean())
                    self._mf_last_router_loss = float(r_loss.detach())
                    self._mf_last_router_meanw = float(frac.mean().detach()) * float(widths[-1])
                    with self.accelerator.no_sync(model):
                        self.accelerator.backward((router_w * r_loss) / ga)
                    ret = ret + r_loss.detach()
                    for p in bundle.surrogate.parameters():
                        p.requires_grad_(True)
                # Both phases ran under no_sync. Average every accumulated head gradient by hand;
                # this is also correct under grad accumulation because already-averaged history is
                # unchanged by a subsequent all-reduce and only the new local contribution differs.
                bundle.average_distributed_gradients()
                if (router_readout_every > 0
                        and int(self.state.global_step) % router_readout_every == 0):
                    if router_readout_mode == "constrained":
                        _ro = bundle.constrained_readout(
                            min_utilization=router_readout_min_util)
                    else:
                        _ro = bundle.deterministic_readout()
                    self._mf_router_ndistinct = float(_ro["n_distinct"])
                    self._mf_router_target_err = float(_ro["target_err"])
                    self._mf_router_n_over = float(_ro.get("n_over_budget", 0))
                    self._mf_router_argmax_fracs = _ro["fracs"]
                    self._mf_router_min_margin_raw = float(_ro.get("min_margin", 0.0))
                    self._mf_router_max_logit_raw = float(
                        _ro.get("max_abs_logit_raw", 0.0))
                    # See the gumbel path: under an exact cost band these are the only two
                    # readout fields that are not constant for ANY table (audit §3.4).
                    self._mf_router_min_decision_gap = _ro.get("min_decision_gap")
                    self._mf_router_decision_gaps = _ro.get("decision_gap")
                return ret

            if sampling in ("sandwich", "mixed_sandwich"):
                # ── Arm B / C: inplace self-distillation (sandwich rule) ───
                # Full width (10240) = TEACHER: one no_grad forward → detached
                # soft logits (the best config, so a fixed high-quality target).
                # Students = {smallest-uniform 1280, N random sub-configs}, each
                # trained to MATCH the teacher (KD KL) + hard labels (CE). Arm B's
                # random students are UNIFORM widths; Arm C's are PER-LAYER MIXED.
                # The full-width STUDENT runs LAST with CE (KD-vs-self ≈ 0) so (i)
                # full stays anchored on GT (no regression) and (ii) its synced
                # backward covers every FFN param → DDP-correct like joint.
                per_layer_mixed = (sampling == "mixed_sandwich")
                labels = inputs["labels"]
                ve_stack = None          # vision-elastic: holds the student (res, depth) context for the step
                ve_s_vis = ve_s_soft = ve_t_vis = None
                ve_dense = ve_soft = None
                if vision_elastic and inputs.get("pixel_values") is not None:
                    if sandwich_teacher_model is None:
                        raise RuntimeError("vision_elastic: frozen sandwich teacher was not initialized")
                    t_core = self._gemma_core(sandwich_teacher_model)
                    with torch.no_grad():
                        ve_t_vis = t_core.vision_tower(pixel_values=inputs["pixel_values"],
                                                       return_dict=True).last_hidden_state.detach()
                        ve_t_soft = t_core.multi_modal_projector(ve_t_vis).detach()
                    ve_res_i, ve_depth_i = self._ve_sample()
                    core = self._gemma_core(model)
                    ve_stack = contextlib.ExitStack()
                    ve_grid = ve_stack.enter_context(self._ve_student_config(core, ve_res_i, ve_depth_i))
                    ve_s_vis, ve_s_soft = self._ve_student_vision(core, inputs["pixel_values"], ve_res_i)
                    ve_dense, ve_soft = self._ve_losses(ve_s_vis, ve_s_soft, ve_t_vis, ve_t_soft, ve_grid)
                    self._mf_last_ve_res = float(ve_res_i)
                    self._mf_last_ve_depth = float(ve_depth_i)
                    self._mf_last_ve_dense = float(ve_dense.detach())
                    self._mf_last_ve_soft = float(ve_soft.detach())
                    vision_hidden = ve_s_vis.detach()      # narrow LM students: student tower output, no tower grad
                else:
                    vision_hidden = self._cached_vision_hidden(model, inputs)
                # teacher: full width, no grad → detached logits. It is either the
                # historical moving self-teacher or an immutable dense parent.
                # SKIPPED for the CE-only control arm (kd_alpha>=1.0).
                teacher_logits = None
                if not ce_only_control:
                    teacher_source = (
                        sandwich_teacher_model
                        if sandwich_teacher_mode == "frozen_initial"
                        else model
                    )
                    if teacher_source is None:
                        raise RuntimeError("frozen sandwich teacher was not initialized")
                    if teacher_source is model:
                        _set_uniform(widths[-1])
                    with torch.no_grad():
                        t_out = self._fwd_out_cached_vision(
                            teacher_source, inputs,
                            ve_t_vis if ve_t_vis is not None else vision_hidden
                        )
                    teacher_logits = t_out.logits.detach()
                    del t_out
                student_cfgs = _sandwich_student_configs(
                    self._mf_seed, self.state.global_step, widths, weights, n_layers,
                    per_layer_mixed=per_layer_mixed, n_random=sw_n_random,
                    always_smallest=sw_smallest, kind_weights=mix_kind_weights,
                    high_late_bias=mix_high_late, budget_levels=sw_budget_levels,
                    budget_min_utilization=sw_budget_min_util,
                    budget_n_groups=sw_budget_groups)
                self._mf_last_kind = ("mixed_sandwich" if per_layer_mixed else "sandwich")
                self._mf_last_meanw = (sum(sum(c) for c in student_cfgs)
                                       / (len(student_cfgs) * n_layers)) if student_cfgs else float(widths[-1])
                total = None
                # ROUTER: every sandwich student is a (config, per-example CE) pair — the SURROGATE's
                # free training data, collected from forwards we already do.
                bundle = getattr(self.accelerator.unwrap_model(model), "width_router", None)
                r_cfgs = [] if bundle is not None else None    # per-config per-LAYER width lists
                r_cev = [] if bundle is not None else None     # per-config per-EXAMPLE CE vectors (B,)
                # sub-full students under no_sync (defer all-reduce); each: KD + CE
                # (all PER-EXAMPLE normalized). CE-only control drops the KD term.
                for wl in student_cfgs:
                    is_smallest = all(w == widths[0] for w in wl)
                    _set_perlayer(wl)
                    _live = (ve_s_vis is not None and ve_narrow_live)
                    with self.accelerator.no_sync(model):
                        s_out = self._fwd_out_cached_vision(
                            model, inputs, ve_s_vis if _live else vision_hidden)
                        ce = self._ce_per_example(s_out.logits, labels)
                        if ce_only_control:
                            kd = None
                            loss = ce / ga
                        else:
                            kd = self._kd_kl(s_out.logits, teacher_logits, labels, kd_temp)
                            loss = (kd_alpha * ce + (1.0 - kd_alpha) * kd) / ga
                        # retain_graph keeps the shared student-tower graph alive for the next student /
                        # the full-width backward (vision-elastic narrow_students_train_tower only).
                        self.accelerator.backward(loss, retain_graph=_live)
                    if is_smallest:                            # diagnostics (codex P1 #6)
                        self._mf_last_smallest_ce = float(ce.detach())
                        self._mf_last_smallest_kd = float(kd.detach()) if kd is not None else 0.0
                    if r_cfgs is not None:
                        r_cfgs.append(list(wl))
                        r_cev.append(self._ce_per_example_vec(s_out.logits.detach(), labels))
                    total = loss.detach() if total is None else total + loss.detach()
                # full-width student LAST → single SYNCED backward covers all params.
                # CE ×full_loss_weight (codex P0 #3): full gets ~50% of the trained mass
                # (teacher fwd above is zero-grad), anchoring the moving self-teacher.
                _set_uniform(widths[-1])
                f_out = self._fwd_out_cached_vision(
                    model, inputs, ve_s_vis if ve_s_vis is not None else vision_hidden)
                full_ce = self._ce_per_example(f_out.logits, labels)
                self._mf_last_full_ce = float(full_ce.detach())
                loss = (full_loss_weight * full_ce) / ga
                if ve_dense is not None:
                    # vision-elastic: the tower's gradient = dense + soft cosine to the frozen 896/full-depth
                    # teacher + the full-width CE above, all in this one synced backward.
                    loss = loss + (ve_w_dense * ve_dense + ve_w_soft * ve_soft) / ga
                self.accelerator.backward(loss)
                if ve_stack is not None:
                    ve_stack.close()     # restore full tower + native projector before the optimizer step / save
                if teacher_logits is not None:
                    del teacher_logits
                total = loss.detach() if total is None else total + loss.detach()
                if r_cfgs is not None:
                    r_cfgs.append([widths[-1]] * n_layers)
                    r_cev.append(self._ce_per_example_vec(f_out.logits.detach(), labels))
                # ── ROUTER + SURROGATE (budget-first · two-phase · straight-through) ────────────
                # Phase A: fit the SURROGATE to predict each forwarded config's (CE - CE_full) per
                #   example (Huber). Targets + config one-hots are DETACHED, so only the surrogate
                #   (and its context embeddings) learn; the router is untouched.
                # Phase B: FREEZE the surrogate net, then train the ROUTER to pick — per
                #   (budget[,pillar][,task]) context — the config the frozen surrogate calls cheapest,
                #   plus a budget-miss penalty. Queried via a STRAIGHT-THROUGH HARD config (never soft
                #   probs). The hard FFN slice is never differentiated and NOTHING here is forwarded
                #   through the backbone (off-policy) → backbone configs stay deterministic. On 1 GPU
                #   the extra backward is fine; multi-GPU needs the router grads handled outside the
                #   sandwich no_sync (codex DDP note) — TODO before the 4-GPU run.
                if bundle is not None and r_cfgs is not None and len(r_cfgs) >= 2:
                    dev = f_out.logits.device
                    B = labels.shape[0]
                    def _guide(ids, drop):                       # allow-missing: id+1, or 0 (=missing)
                        if ids is None:
                            return torch.zeros(B, dtype=torch.long, device=dev)
                        keep = torch.rand(B, device=dev) >= drop
                        return torch.where(keep, ids.to(dev).long() + 1,
                                           torch.zeros(B, dtype=torch.long, device=dev))
                    pill_p1 = _guide(getattr(self, "_router_pillar", None), router_pillar_drop)
                    task_p1 = _guide(getattr(self, "_router_task", None), router_task_drop)
                    ctx = bundle.context(pill_p1, task_p1)                    # (B, ctx_dim)
                    full_cev = r_cev[-1]                                      # (B,)
                    # Phase A — surrogate Huber on (CE - CE_full), per example, NaN-masked.
                    s_loss, nseen = None, 0
                    for c, cev in zip(r_cfgs, r_cev):
                        tgt = cev - full_cev
                        m = torch.isfinite(tgt)
                        if not bool(m.any()):
                            continue
                        oh = bundle.layer_onehot(c, dev).expand(B, -1, -1)    # (B,L,K) detached
                        pred = bundle.surrogate(oh, ctx)                      # (B,)
                        term = torch.nn.functional.huber_loss(pred[m], tgt[m].detach())
                        s_loss = term if s_loss is None else s_loss + term
                        nseen += 1
                    if s_loss is not None:
                        s_loss = s_loss / nseen
                        self._mf_last_surr_loss = float(s_loss.detach())
                        self.accelerator.backward((surrogate_w * s_loss) / ga)
                    # Phase B — freeze surrogate net; router grad still flows THROUGH it to z.
                    for p in bundle.surrogate.parameters():
                        p.requires_grad_(False)
                    wmin = float(widths[0]) / float(widths[-1])
                    budget = torch.rand(B, device=dev) * (1.0 - wmin) + wmin  # target mean-width frac
                    z, _wl = bundle.route(budget, ctx.detach(), tau=router_tau)   # (B,L,K) straight-through
                    frac = bundle.mean_width_frac(z)                         # (B,)
                    r_loss = (bundle.surrogate(z, ctx.detach()).mean()
                              + router_budget_w * (frac - budget).abs().mean())
                    self._mf_last_router_loss = float(r_loss.detach())
                    self._mf_last_router_meanw = float(frac.mean().detach()) * float(widths[-1])
                    self.accelerator.backward((router_w * r_loss) / ga)
                    for p in bundle.surrogate.parameters():
                        p.requires_grad_(True)
                return total

            if sampling == "stochastic":
                # paper-aligned (Eq. 2): ONE granularity per OPTIMIZER step.
                # Derived from (seed, global_step) — NOT a stateful RNG — so:
                #   * constant across the grad-accum microbatches of one update
                #     (a stateful per-call RNG would mix widths inside a single
                #     update = a mini-joint average, breaking Eq. 2);
                #   * identical on every DDP rank (global_step is);
                #   * reproducible across resume (global_step is checkpointed,
                #     a Python RNG object is not).
                idx = self._mf_width_idx(self.state.global_step)
                self._mf_last_width = widths[idx]
                controller.set(widths[idx])
                loss = self._fwd_loss(model, inputs) / ga
                self.accelerator.backward(loss)
                return loss.detach()

            # joint: forward every granularity (ascending → xl LAST so the
            # synced backward's graph covers all FFN params → no unused-param
            # search). For the first K-1 widths the WHOLE forward+loss+backward
            # runs under no_sync(), so DDP defers all-reduce to the single final
            # full-width backward (codex P0: no_sync must wrap the forward too).
            vision_hidden = self._cached_vision_hidden(model, inputs)
            total = None
            for i, (wd, w) in enumerate(zip(widths, weights)):
                controller.set(wd)
                is_last = (i == n_g - 1)
                if is_last:
                    loss = self._fwd_loss_cached_vision(model, inputs, vision_hidden) * w / ga
                    self.accelerator.backward(loss)
                else:
                    with self.accelerator.no_sync(model):
                        loss = self._fwd_loss_cached_vision(model, inputs, vision_hidden) * w / ga
                        self.accelerator.backward(loss)
                total = loss.detach() if total is None else total + loss.detach()
            return total

    trainer = MatformerTrainer(
        model=model, args=sft_cfg,
        train_dataset=train_ds, eval_dataset=None,
        data_collator=collator,
        processing_class=processor.tokenizer,
    )
    if sandwich_teacher_mode == "frozen_initial":
        teacher_path = cfg.get("sandwich_teacher_model_override") or cfg.get(
            "base_model_override"
        )
        sandwich_teacher_model, _teacher_processor, teacher_controller = (
            _lm.load_matformer_pure(
                vision_freeze=True,
                attn_impl=cfg.get("attn_impl", "eager"),
                factor=int(cfg.get("matformer_factor", 8)),
                base_path=teacher_path,
                widths=cfg.get("matformer_widths"),
            )
        )
        teacher_controller.set(widths[-1])
        sandwich_teacher_model.config.use_cache = False
        sandwich_teacher_model.eval()
        sandwich_teacher_model.requires_grad_(False)
        sandwich_teacher_model.to(trainer.accelerator.device)
        if rank0:
            print(
                "[matformer_pure] sandwich teacher: frozen checkpoint "
                f"{teacher_path} on {trainer.accelerator.device}",
                flush=True,
            )
    if router_teacher_mode == "frozen_initial":
        teacher_path = cfg.get("router_teacher_model_override") or cfg.get(
            "base_model_override"
        )
        router_teacher_model, _teacher_processor, teacher_controller = (
            _lm.load_matformer_pure(
                vision_freeze=True,
                attn_impl=cfg.get("attn_impl", "eager"),
                factor=int(cfg.get("matformer_factor", 8)),
                base_path=teacher_path,
                widths=cfg.get("matformer_widths"),
            )
        )
        teacher_controller.set(widths[-1])
        router_teacher_model.config.use_cache = False
        router_teacher_model.eval()
        router_teacher_model.requires_grad_(False)
        router_teacher_model.to(trainer.accelerator.device)
        if rank0:
            print(
                "[matformer_pure] router teacher: frozen initial checkpoint "
                f"{teacher_path} on {trainer.accelerator.device}",
                flush=True,
            )
    # stochastic-mode width draws are keyed off this + global_step (stateless)
    trainer._mf_seed = int(cfg.get("seed", 0))

    resume = _resolve_resume(output_dir, want_resume=args.resume)
    if rank0:
        print(f"[matformer_pure] resume_from_checkpoint = {resume}", flush=True)
    result = trainer.train(resume_from_checkpoint=resume)

    # full-width final save (xl weights == standard medgemma state-dict layout).
    # NB: save_model writes the FULL-size params regardless of controller width —
    # the controllers only gate the FORWARD slice — so the saved ckpt is always the
    # complete xl state-dict. Set widths to full anyway for post-save hygiene.
    controller.set_largest()
    if layer_ctrls is not None:
        for c in layer_ctrls:
            c.set_largest()
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    if rank0:
        (output_dir / "train_metrics.json").write_text(
            json.dumps(result.metrics, indent=2, default=str))
        router_bundle = getattr(trainer.accelerator.unwrap_model(model), "width_router", None)
        if router_bundle is not None:
            if router_readout_mode == "constrained":
                router_readout = router_bundle.constrained_readout(
                    min_utilization=router_readout_min_util)
            else:
                router_readout = router_bundle.deterministic_readout()
            unwrapped = trainer.accelerator.unwrap_model(model)
            trainable = {
                name: parameter.numel()
                for name, parameter in unwrapped.named_parameters()
                if parameter.requires_grad
            }
            table = getattr(router_bundle.router, "table", None)
            router_artifact = {
                "schema": "elastic-router-training-artifact/v1",
                "global_step": int(trainer.state.global_step),
                "max_steps": int(sft_cfg.max_steps),
                "base_model_override": str(cfg.get("base_model_override")),
                # Continuation provenance. `base_model_override` seeds only the BACKBONE; the
                # router head comes from here or it starts at zero (see width_router.load_router_state).
                # `null` therefore means "this run's table began at zero", which is the fact a
                # reader needs to interpret any trajectory claim.
                "router_init_from": router_init_from_report,
                # Which lr envelope the 160-parameter head actually saw. `null` = it inherited the
                # backbone's `lr_scheduler` and, for a cosine, was annealed to exactly 0.
                "router_lr_scheduler": router_lr_scheduler,
                "router_head_lr": router_head_lr,
                "backbone_lr_scheduler": str(sft_cfg.lr_scheduler_type),
                "backbone_lr": float(sft_cfg.learning_rate),
                "router_mode": router_mode,
                "router_teacher_mode": router_teacher_mode,
                "router_exact_compute_gumbel": router_exact_compute_gumbel,
                "router_freeze_backbone": router_freeze_backbone,
                "router_uniform_regret_weight": router_uniform_regret_w,
                "router_uniform_regret_margin": router_uniform_regret_margin,
                "router_uniform_widths": router_uniform_widths,
                "router_multi_budget": router_multi_budget,
                "router_tau_start": router_tau0,
                "router_tau_end": router_tau1,
                "router_t_sample_start": router_t_sample0,
                "router_t_sample_end": router_t_sample1,
                "router_kappa_start": 1.0 / max(router_t_sample0, 1e-12),
                "router_kappa_end": 1.0 / max(router_t_sample1, 1e-12),
                "trainable_parameters": trainable,
                "trainable_parameter_count": sum(trainable.values()),
                "router_table_l2": (
                    float(table.detach().float().norm()) if table is not None else None
                ),
                "router_table": (
                    table.detach().float().cpu().tolist() if table is not None else None
                ),
                "surrogate_l2": float(
                    sum(
                        parameter.detach().float().pow(2).sum()
                        for parameter in router_bundle.surrogate.parameters()
                    ).sqrt()
                ),
                "readout": router_readout,
                "log_history": trainer.state.log_history,
            }
            (output_dir / "router_artifact.json").write_text(
                json.dumps(router_artifact, indent=2, default=str) + "\n"
            )
        print(f"[matformer_pure] saved final to {final_dir}", flush=True)
        print(f"[matformer_pure] metrics: {result.metrics}", flush=True)


if __name__ == "__main__":
    main()
