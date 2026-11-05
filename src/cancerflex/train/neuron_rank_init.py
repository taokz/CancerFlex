"""Neuron-rank init for matformer route ①b — standalone single-GPU pre-step.

PROVENANCE: this is Flextron's mechanism (Cai et al., arXiv:2406.10260). Its finding is that
a nested prefix slice is near-worthless on an UNPERMUTED model and near-free on a permuted one
— PPL 1902 -> 193 — because "the first d neurons" only means "the d most important neurons"
after you sort them that way. It is the one Flextron component we adopt; its surrogate-loss
router, per-token adaptive routing and attention-head elasticity are not used
(docs/methodology/elastic_router_lessons.md §2). Nothing here is locally invented, and the
step is MANDATORY, not optional: every elastic run initialises from a permuted parent.

SCORING CRITERION, corrected 2026-07: score the quantity that actually enters `down_proj`,
    S_h = sum |h|,  h = act(gate) * up          <- what this script does now
not the cheaper proxy it inherited,
    S_gateup = sum |gate_out| + sum |up_out|    <- superseded
The two disagree because SwiGLU gates multiplicatively: a large `up` on a gated-off neuron
contributes nothing downstream but scores high under the proxy. See
`scripts/train/importance_method_probe.py`. Runs record which was used as
`neuron_rank_meta.json:score_variant` ("h" for every current checkpoint).

WHAT: permute each MedGemma FFN's intermediate neurons by ACTIVATION IMPORTANCE so
the matformer nested prefix-slice `[:d]` keeps the most-important neurons at every
width. Scores the chosen criterion over a subsample of the de-leaked corpus, argsorts
per layer (most-important first), then `index_select`s rows of gate/up (dim 0) + cols
of down (dim 1). Saves the permuted MedGemma in HF format; ①b training wraps THIS base
(matformer `base_model_override`) instead of the plain base.

DEFAULT (changed 2026-09-24): `--score-variant` now DEFAULTS to `h`, the criterion every current
checkpoint used (the launchers pass it explicitly). Before this date the default was the
superseded `gateup`, so a direct call without the flag silently gave the OLD criterion; retired
launchers that relied on the old default pass `--score-variant gateup` explicitly and still reproduce
their historical permutation. Each run records what it used in `neuron_rank_meta.json:score_variant`;
the earliest base, `medgemma_neuron_ranked_init`, has no such field, but its launcher (1,024 prefix
samples, no `--balanced`, no `--score-variant`) shows it used `gateup`.
NOTE: `--balanced` is still OFF by default (a prefix draw, which lets the largest pillar dominate);
the current launchers pass `--balanced --num-per-cell 8 --num-samples 296` explicitly.

WHY standalone (not in-process): the permutation must be identical across all DDP
ranks or training silently corrupts (cf. docs/operations/performance.md §7). Doing
it once in a single process → one permuted checkpoint all ranks load → zero cross-rank
risk, and it's independently verifiable.

CORRECTNESS: a consistent neuron reorder is VALUE-PRESERVING at full width (the MLP
sum over neurons is permutation-invariant). This script ASSERTS full-width logits are
unchanged pre/post (§11: prove the transform is identity end-to-end, not by unit test).
So ①b's xl weights == ①a's base; only the sub-width neuron ordering differs.

The end-to-end check MUST run in fp32. The permutation is exact in real arithmetic, but a
bf16 forward reorders each 10240-wide FFN reduction; that non-associativity is genuine and
compounds over 34 layers to |Δlogit| ~ O(1-30) — NOT an error (index_select reorders the
bf16 weights losslessly). Verified 2026-07-07: bf16 end-to-end Δ=27.6 (false alarm) vs fp32
Δ~1e-4 (true identity). We upcast for the check, then restore bf16 for saving —
bf16(reorder(fp32(w))) == reorder(bf16 w), so the saved base is bit-identical to permuting
the bf16 weights directly. (An earlier implementation applied the permutation with no such check.)

Port of preliminary/archive/Med-Matformer/med-matformer/medgemma_4b/
train_medgemma_deepspeed.py:441 (maybe_apply_neuron_ranking_permutation), adapted for
the multimodal Gemma3ForConditionalGeneration layer path + the de-leaked VLM corpus.

USAGE (single GPU; run AFTER the ①a run frees the GPUs, or on any 1 free GPU):
  CUDA_VISIBLE_DEVICES=0 python -m cancerflex.train.neuron_rank_init \
    --out-dir runs/medgemma_neuron_ranked_init --num-samples 1024
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from cancerflex.paths import ROOT as REPO  # noqa: E402  (set CANCERFLEX_ROOT)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from cancerflex.train import dataset as _ds
from cancerflex.train import load_model as _lm
from cancerflex.train.collator import VLMSFTCollator


def _decoder_layers(model):
    """Multimodal Gemma3ForConditionalGeneration → model.model.language_model.layers;
    text-only fallback → model.model.layers."""
    m = model.model
    if hasattr(m, "language_model") and hasattr(m.language_model, "layers"):
        return m.language_model.layers
    return m.layers


def _to_dev(batch, dev):
    b = {k: v.to(dev) for k, v in batch.items() if isinstance(v, torch.Tensor)}
    b.pop("labels", None)          # scoring only needs the forward activations, not loss
    return b


# P0-a: positions that write the answer. Hidden state at position t predicts token t+1, so the
# scoring mask for "answer-writing" positions is labels[:, 1:] != -100, shifted onto position t.
_TOKEN_MASK = {"m": None}        # set per batch in the scoring loop; None = score every position


def _answer_positions(labels: torch.Tensor) -> torch.Tensor:
    m = torch.zeros_like(labels, dtype=torch.bool)
    m[:, :-1] = labels[:, 1:] != -100
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--corpus-dir", default="Experiment/datasets/v1_final_deleaked")
    ap.add_argument("--num-samples", type=int, default=1024)
    ap.add_argument("--score-source", default="both", choices=["both", "gate", "up"])
    # ── 2026-10-08 P0-a additions (docs/SMALL_WIDTH_LOCALIZATION_PLAN_20261008.md §4). All default
    # to the historical behaviour; every shipped permutation still reproduces with no new flag. ──
    ap.add_argument("--token-mask", default="all", choices=["all", "answer"],
                    help="all = every sequence position incl. prompt/image/padding (historical); "
                         "answer = only positions whose NEXT token is an answer token "
                         "(labels[:,1:] != -100), i.e. the positions that WRITE the answer.")
    ap.add_argument("--task-filter", default=None,
                    help="comma list of tasks (e.g. U5,U6); with --balanced, draw only from those shards.")
    ap.add_argument("--score-only", action="store_true",
                    help="dump per-layer score vectors to <out-dir>/scores.pt (+ meta) and STOP: no "
                         "permutation, no identity check, no model save. For ranking diagnostics.")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--attn-impl", default="eager")
    # Importance must be scored on the model the permutation will be APPLIED to. All three
    # reference works sort BEFORE elastic training; the 2026-07 permutation was scored on the
    # stock base while every current arm runs on the width-sampled a2 checkpoint, so that
    # ordering reflects a different model. Point this at the checkpoint actually in use.
    ap.add_argument("--score-variant", default="h", choices=["gateup", "h", "hw"],
                    help="gateup = |gate_out|+|up_out| (the original); h = |act(gate)*up|, the\n                          quantity that actually enters down_proj; hw = h weighted by the\n                          down_proj column norm, i.e. the neuron's effect on the residual.\n                          Chosen by measurement -- see scripts/train/importance_method_probe.py\n                          and elastic_router_lessons.md 3.3b, not inherited.")
    ap.add_argument("--balanced", action="store_true",
                    help="draw --num-samples-per-cell rows from EVERY (task,pillar) shard instead\n                          of a prefix of the split. The corpus is heavily imbalanced, so a prefix\n                          lets the largest pillar dominate the importance estimate.")
    ap.add_argument("--num-per-cell", type=int, default=32)
    ap.add_argument("--base-model", default=None,
                    help="local checkpoint to score+permute (default: stock MEDGEMMA_BASE)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) raw MedGemma (multimodal). Vision tower is untouched by neuron-rank.
    model, processor = _lm.load_medgemma(vision_freeze=True, attn_impl=args.attn_impl,
                                         base_path=args.base_model)
    print(f"[neuron-rank] scoring base = {args.base_model or 'MEDGEMMA_BASE (stock)'}", flush=True)
    model.eval().to(dev)
    layers = _decoder_layers(model)
    num_layers = len(layers)
    inter = layers[0].mlp.gate_proj.weight.shape[0]        # intermediate_size (10240)
    print(f"[neuron-rank] layers={num_layers} intermediate_size={inter} dev={dev}", flush=True)

    # 2) scoring data — same corpus + collator as training, deterministic subsample
    if args.balanced:
        import itertools, json as _json, random as _random
        from concurrent.futures import ThreadPoolExecutor
        shard_root = REPO / args.corpus_dir / "shards" / "train"
        files = sorted(shard_root.glob("U*/*.jsonl"))
        if args.task_filter:
            keep_tasks = {t.strip() for t in args.task_filter.split(",") if t.strip()}
            files = [f for f in files if f.parent.name in keep_tasks]
            print(f"[neuron-rank] --task-filter {sorted(keep_tasks)} -> {len(files)} shards", flush=True)
        if not files:
            raise SystemExit(f"--balanced: no shards under {shard_root}")
        def _draw(fp):
            rng = _random.Random(f"{args.seed}:{fp.parent.name}:{fp.stem}")
            keep = []
            with open(fp, errors="ignore") as fh:
                for n, line in enumerate(fh):
                    if n < args.num_per_cell:
                        keep.append(line)
                    else:
                        j = rng.randint(0, n)
                        if j < args.num_per_cell:
                            keep[j] = line
            return [_json.loads(x) for x in keep]
        with ThreadPoolExecutor(max_workers=min(16, len(files))) as ex:
            ds = list(itertools.chain.from_iterable(ex.map(_draw, files)))
        print(f"[neuron-rank] BALANCED draw: {len(files)} (task,pillar) shards x "
              f"{args.num_per_cell} = {len(ds)} rows", flush=True)
    else:
        ds = _ds.load_v1_split("train", corpus_dir=args.corpus_dir, limit=args.num_samples * 3)
    collate = VLMSFTCollator(processor, max_length=args.max_length)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    # 3) hooks accumulate |gate/up output| per neuron (float64 for a stable sum)
    score = [torch.zeros(inter, dtype=torch.float64, device=dev) for _ in range(num_layers)]
    hooks = []

    def _masked_abs_sum(t):
        a = t.detach().abs()
        m = _TOKEN_MASK["m"]
        if m is None:
            return a.sum(dim=(0, 1), dtype=torch.float64)
        return (a * m.to(a.device).unsqueeze(-1)).sum(dim=(0, 1), dtype=torch.float64)

    def mk(i):
        def _h(_m, _in, out):
            score[i] += _masked_abs_sum(out)
        return _h

    def mk_pre(i):
        # input to down_proj IS h = act(gate(x)) * up(x) — the quantity a neuron contributes.
        def _h(_m, inp):
            score[i] += _masked_abs_sum(inp[0])
        return _h

    if args.score_variant == "gateup":
        for i, lyr in enumerate(layers):
            if args.score_source in ("both", "gate"):
                hooks.append(lyr.mlp.gate_proj.register_forward_hook(mk(i)))
            if args.score_source in ("both", "up"):
                hooks.append(lyr.mlp.up_proj.register_forward_hook(mk(i)))
    else:                                    # "h" and "hw" both accumulate |h|
        for i, lyr in enumerate(layers):
            hooks.append(lyr.mlp.down_proj.register_forward_pre_hook(mk_pre(i)))

    seen = 0
    positions_scored = 0
    with torch.no_grad():
        for batch in loader:
            if batch["input_ids"].numel() == 0:
                continue
            if args.token_mask == "answer":
                if "labels" not in batch:
                    raise RuntimeError("--token-mask answer needs `labels` from the collator")
                _TOKEN_MASK["m"] = _answer_positions(batch["labels"]).to(dev)
                positions_scored += int(_TOKEN_MASK["m"].sum())
            else:
                positions_scored += int(batch["input_ids"].numel())
            b = _to_dev(batch, dev)
            model(**b, use_cache=False)
            seen += int(b["input_ids"].size(0))
            if seen >= args.num_samples:
                break
    _TOKEN_MASK["m"] = None
    print(f"[neuron-rank] token_mask={args.token_mask} positions_scored={positions_scored}", flush=True)
    for h in hooks:
        h.remove()
    print(f"[neuron-rank] scored {seen} samples", flush=True)

    if args.score_variant == "hw":
        # weight each neuron by ||down_proj[:, j]||_2 — converts "how large is this neuron's
        # activation" into "how large is its effect on the residual stream".
        for i, lyr in enumerate(layers):
            score[i] = score[i] * lyr.mlp.down_proj.weight.detach().float().norm(dim=0).double()
    print(f"[neuron-rank] score_variant={args.score_variant}", flush=True)
    perms = [torch.argsort(s, descending=True).to(torch.long) for s in score]
    if args.score_only:
        out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
        torch.save([s.cpu() for s in score], out / "scores.pt")
        (out / "score_meta.json").write_text(json.dumps({
            "num_layers": num_layers, "intermediate_size": inter, "num_samples_scored": seen,
            "positions_scored": positions_scored, "token_mask": args.token_mask,
            "task_filter": args.task_filter, "score_source": args.score_source,
            "base_model": args.base_model or "MEDGEMMA_BASE", "score_variant": args.score_variant,
            "balanced": bool(args.balanced), "num_per_cell": args.num_per_cell,
            "corpus_dir": args.corpus_dir, "seed": args.seed}, indent=2))
        print(f"[neuron-rank] --score-only: wrote {out}/scores.pt (no permutation applied)", flush=True)
        return 0

    # 4) identity check probe — text-only (images are irrelevant to the FFN-neuron permutation,
    #    and text-only avoids multimodal dtype juggling while still exercising all decoder MLPs).
    probe = processor.tokenizer(
        "A 54-year-old patient presents for oncologic evaluation of a suspicious skin lesion.",
        return_tensors="pt",
    ).to(dev)

    # 5) upcast to fp32 for the end-to-end identity check (bf16 forward reorder-noise compounds
    #    to O(1-30) over 34 layers and false-alarms; see CORRECTNESS in the module docstring).
    model.float()
    with torch.no_grad():
        ref = model(**probe, use_cache=False).logits.float().clone()

    # 6) apply the permutation: rows of gate/up (dim 0), cols of down (dim 1)
    with torch.no_grad():
        for i, lyr in enumerate(layers):
            p = perms[i].to(lyr.mlp.gate_proj.weight.device)
            lyr.mlp.gate_proj.weight.copy_(lyr.mlp.gate_proj.weight.index_select(0, p))
            lyr.mlp.up_proj.weight.copy_(lyr.mlp.up_proj.weight.index_select(0, p))
            lyr.mlp.down_proj.weight.copy_(lyr.mlp.down_proj.weight.index_select(1, p))

    # 7) VERIFY the transform is identity at full width, in fp32 (§11 end-to-end gate)
    with torch.no_grad():
        new = model(**probe, use_cache=False).logits.float()
    max_abs = (new - ref).abs().max().item()
    print(f"[neuron-rank] full-width logit max|Δ| pre/post permute (fp32) = {max_abs:.3e} (want ~0)", flush=True)
    if not (max_abs < 1e-1):
        raise AssertionError(f"permutation NOT value-preserving (fp32 Δ={max_abs}) — bug, refusing to save")
    model.to(torch.bfloat16)   # restore bf16 for saving (reorder commutes with the cast → lossless)

    # 8) save permuted MedGemma (HF format, bf16) + processor + provenance
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    processor.save_pretrained(str(out))
    torch.save(perms, out / "neuron_permutation.pt")
    (out / "neuron_rank_meta.json").write_text(json.dumps({
        "num_layers": num_layers, "intermediate_size": inter, "num_samples_scored": seen,
        "score_source": args.score_source,
        "base_model": args.base_model or "MEDGEMMA_BASE", "score_variant": args.score_variant,
        "balanced": bool(args.balanced), "num_per_cell": args.num_per_cell, "corpus_dir": args.corpus_dir,
        "full_width_logit_max_delta": max_abs, "seed": args.seed,
    }, indent=2))
    print(f"[neuron-rank] saved permuted MedGemma → {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
