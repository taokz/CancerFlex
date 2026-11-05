"""Cell (task x pillar) statistics, two-level resampling, and per-cell loss weights.

All pure / NumPy — unit-testable without torch. The (task,pillar) pair is the
"cell" = the imbalance unit for the multi-task corpus (proxy_protocol §1).

Two-level resample (A2): realized as a PHYSICAL resample to a target cell mix,
exactly like train_v2.dataset.temperature_balance (which Phase-1 used). The
result is a plain HF Dataset consumed by the default RandomSampler, so DDP
sharding is the proven path — no custom Sampler x accelerate interaction.

Per-cell loss weights (A4): rarity from the NATURAL counts (inverse-sqrt or
effective-number), normalized so the mean weight over the *resampled* training
mix is 1 (keeps the loss scale ~constant across the DRW defer transition), with
an optional cap to bound gradient-norm spikes (proxy_protocol §1).
"""
from __future__ import annotations

import numpy as np


def cell_key(task: str, pillar: str) -> str:
    return f"{task}__{pillar}"


def build_cell_index(ds) -> dict[str, int]:
    """Stable, sorted {cell_key: idx} over all (task,pillar) present in ds.
    MUST be shared between the reweight table and the collator's per-row id."""
    keys = sorted({cell_key(t, p) for t, p in zip(ds["task"], ds["pillar"])})
    return {k: i for i, k in enumerate(keys)}


def cell_ids_for(ds, cell_index: dict[str, int]) -> np.ndarray:
    """Per-row cell index array (-1 for an unseen cell, which should not occur)."""
    return np.array([cell_index.get(cell_key(t, p), -1)
                     for t, p in zip(ds["task"], ds["pillar"])], dtype=np.int64)


def cell_counts(ds, cell_index: dict[str, int]) -> np.ndarray:
    """Row count per cell index (length = len(cell_index))."""
    counts = np.zeros(len(cell_index), dtype=np.int64)
    for cid in cell_ids_for(ds, cell_index):
        if cid >= 0:
            counts[cid] += 1
    return counts


def two_level_balance(ds, tau: float = 0.5, seed: int = 0):
    """Physical resample so P(cell) = P(task) * P(pillar|task), each level
    temperature-flattened: P(task) ∝ N_task^tau, P(pillar|task) ∝ N_cell^tau.
    tau=1 → natural, tau→0 → uniform-per-level. Target total ≈ len(ds).
    Deterministic given seed. Returns ds.select(picked) (shuffled).
    """
    rng = np.random.default_rng(seed)
    tasks = list(ds["task"])
    pillars = list(ds["pillar"])
    cells: dict[tuple[str, str], list[int]] = {}
    for i, (t, p) in enumerate(zip(tasks, pillars)):
        cells.setdefault((t, p), []).append(i)

    task_tot: dict[str, int] = {}
    for (t, _p), ix in cells.items():
        task_tot[t] = task_tot.get(t, 0) + len(ix)

    t_keys = sorted(task_tot)
    t_w = np.array([task_tot[t] for t in t_keys], dtype=float) ** tau
    t_w /= t_w.sum()
    p_task = dict(zip(t_keys, t_w))

    total = len(ds)
    picked: list[int] = []
    for t in t_keys:
        plist = sorted({p for (tt, p) in cells if tt == t})
        p_w = np.array([len(cells[(t, p)]) for p in plist], dtype=float) ** tau
        p_w /= p_w.sum()
        for p, wp in zip(plist, p_w):
            tgt = int(round(total * p_task[t] * wp))
            if tgt <= 0:
                continue
            ix = np.asarray(cells[(t, p)])
            sel = rng.choice(ix, size=tgt, replace=(tgt > len(ix)))
            picked.extend(sel.tolist())
    rng.shuffle(picked)
    return ds.select(picked)


def label_balance(ds, field: str = "u4_label", tau: float = 0.5, seed: int = 0,
                  options_field: str | None = "u4_eval_options"):
    """Physical resample so P(class) ∝ N_class^tau, within a SINGLE cell.

    `two_level_balance` above balances task×pillar. That is the right unit for the multi-task
    corpus and it is what the deployed model used, but it says nothing about the class mix INSIDE
    a cell, and for a single-cell corpus it is a no-op. A continue-training run aimed at a tail
    class needs the other axis: oral U4 is 75.6% `healthy`, so an epoch of uniform-over-rows spends
    75.6% of its gradient re-teaching the majority class while the classes the run exists to fix
    get 5.1% and 2.2% -- and the failure being fixed is that the model ALREADY over-predicts
    `healthy` externally, where that class falls to 16.9%.

    Same mechanism and same contract as `two_level_balance`, deliberately: a temperature-flattened
    PHYSICAL resample to ~len(ds) rows, returning a plain HF Dataset for the default RandomSampler,
    so DDP sharding stays the proven path. tau=1 is natural, tau=0 is uniform over classes,
    tau=0.5 is the sqrt-temperature setting A2 uses at the cell level.

    NOT a floor. Like the two-level sampler this compresses the tail upward, it does not equalise
    it: a class with a 260x deficit comes back with a 16x one. Compute the resulting share before
    quoting an exposure number; do not report "1/K of the batch".

    OFF-LIST LABELS ARE HELD AT THEIR NATURAL COUNT, not rebalanced. When `options_field` names a
    per-row column holding that row's own candidate list, any row whose gold label is absent from
    its own list is copied through once and excluded from the temperature draw. Such a row is not a
    rare class that needs protecting; it is a row whose prompt offers K choices and whose target is
    a K+1th string, so it cannot be answered correctly and it trains the model to reply outside the
    option space. Balancing would treat its rarity as a deficit and amplify it: oral U4 carries 158
    such rows (gold `oral potentially malignant disorder (OPMD)`, the PARENT class of four options,
    demoted from the candidate list on 2026-09-16 eval-side only), and a plain sqrt resample lifts
    them from 0.96% to 4.77% of the mix -- a 5x increase in exactly the behaviour a neighbouring
    pillar's repair was spent removing. Pass `options_field=None` to disable this and balance every
    label, which is correct for a corpus with no fixed candidate list.
    """
    rng = np.random.default_rng(seed)
    if field not in ds.column_names:
        raise KeyError(
            f"label_balance_field={field!r} is not a column of the training set "
            f"(have: {sorted(ds.column_names)}). Balancing silently on a missing "
            f"field would return the natural mix while the log claimed otherwise.")
    labels = list(ds[field])
    offlist: list[int] = []
    if options_field and options_field in ds.column_names:
        opts = list(ds[options_field])
        offlist = [i for i, (lab, o) in enumerate(zip(labels, opts))
                   if o is not None and str(lab) not in set(o)]
    keep = set(offlist)
    groups: dict[str, list[int]] = {}
    for i, lab in enumerate(labels):
        if i in keep:
            continue
        groups.setdefault(str(lab), []).append(i)
    if len(groups) < 2:
        raise ValueError(
            f"label_balance on {field!r} found {len(groups)} distinct value(s); "
            f"there is nothing to balance and the config asks for something it cannot do.")

    keys = sorted(groups)
    w = np.array([len(groups[k]) for k in keys], dtype=float) ** tau
    w /= w.sum()
    # Off-list rows keep their natural count; the temperature draw fills what is left, so the
    # resampled set is still ~len(ds) and the two arms stay the same size.
    picked: list[int] = list(offlist)
    total = len(ds) - len(offlist)
    for k, wk in zip(keys, w):
        tgt = int(round(total * wk))
        if tgt <= 0:
            continue
        ix = np.asarray(groups[k])
        picked.extend(rng.choice(ix, size=tgt, replace=(tgt > len(ix))).tolist())
    rng.shuffle(picked)
    return ds.select(picked)


def label_balance_report(ds, field: str, tau: float,
                         options_field: str | None = "u4_eval_options") -> str:
    """The before/after class shares, for the run log.

    Exists because an exposure number quoted from the formula instead of from the realised mix has
    been wrong here before. This prints what the resample actually produced.
    """
    from collections import Counter
    labels = [str(x) for x in ds[field]]
    off = set()
    if options_field and options_field in ds.column_names:
        off = {i for i, (lab, o) in enumerate(zip(labels, ds[options_field]))
               if o is not None and lab not in set(o)}
    nat = Counter(labels)
    n = len(labels)
    inl = Counter(lab for i, lab in enumerate(labels) if i not in off)
    budget = n - len(off)
    w = {k: v ** tau for k, v in inl.items()}
    z = sum(w.values()) or 1.0
    lines = [f"{'class':46} {'rows':>7} {'natural':>9} {'balanced':>9}  note"]
    for k, v in sorted(nat.items(), key=lambda kv: -kv[1]):
        if k in inl:
            bal, note = w[k] / z * budget / n, ""
        else:
            bal, note = v / n, "OFF-LIST: held at natural count, not rebalanced"
        lines.append(f"{k[:46]:46} {v:7d} {v / n:8.2%} {bal:8.2%}  {note}")
    return "\n".join(lines)


def inverse_sqrt_weights(counts: np.ndarray) -> np.ndarray:
    """Raw per-cell rarity weight ∝ 1/sqrt(n). Gentle (16x for a 260x imbalance);
    less gradient variance than inverse-freq (proxy_protocol §1)."""
    counts = np.asarray(counts, dtype=float)
    return 1.0 / np.sqrt(np.maximum(counts, 1.0))


def effective_number_weights(counts: np.ndarray, beta: float = 0.99999) -> np.ndarray:
    """Cui CVPR'19 class-balanced weight wₙ=(1−β)/(1−βⁿ). NOTE: with tail cells
    ~2k here, β must be ≈1−1/N_total (≈0.99999) to actually spread the effective
    numbers — β=0.999 degenerates to ~uniform. Provided as an alternative to
    inverse_sqrt (default arm uses inverse_sqrt)."""
    counts = np.asarray(counts, dtype=float)
    return (1.0 - beta) / (1.0 - np.power(beta, np.maximum(counts, 1.0)))


def normalize_reweight(raw_by_cell: np.ndarray, resampled_cell_ids: np.ndarray,
                       cap: float | None = 8.0) -> np.ndarray:
    """Scale raw per-cell weights so the MEAN weight over the resampled training
    rows is 1 (loss-scale-preserving), then cap at `cap`x and renormalize."""
    w = np.asarray(raw_by_cell, dtype=float).copy()
    valid = resampled_cell_ids[resampled_cell_ids >= 0]
    w /= w[valid].mean()
    if cap is not None:
        w = np.minimum(w, cap)
        w /= w[valid].mean()
    return w


def oversample_datasets(ds, dataset_ids, target_frac: float, seed: int = 0):
    """Append extra copies of rows whose `dataset_id` is in `dataset_ids` until
    those rows make up `target_frac` of the returned dataset, then shuffle.

    Purpose: during a WARM-START continue-train, force faster absorption of a small,
    freshly-added data slice (e.g. Cairo/ThyroidXL, ~4% of the corpus). The rest of
    `ds` is the already sampler-balanced replay backbone — every task stays present,
    so the saturated old tasks are pinned (near-zero gradient = anti-forgetting no-op)
    while the non-saturated new slice gets real gradient. Extra copies are drawn WITH
    REPLACEMENT from the matching rows already in `ds` (target_frac is normally >>
    their natural share). Returns a shuffled dataset so the appended block interleaves.

    k solves (n_new0 + k) / (n_total + k) = target_frac. If the slice already meets
    the target, returns a shuffled `ds` unchanged. Raises if nothing matches (guards a
    silently-misspelled dataset_id from becoming a no-op)."""
    from datasets import concatenate_datasets
    want = set(dataset_ids)
    ids = list(ds["dataset_id"])
    new_idx = np.array([i for i, d in enumerate(ids) if d in want], dtype=np.int64)
    n_total = len(ds)
    n_new0 = int(new_idx.size)
    if n_new0 == 0:
        raise ValueError(
            f"oversample_datasets: no rows match dataset_ids={sorted(want)} "
            f"(examples present: {sorted(set(ids))[:8]})")
    if not (0.0 < target_frac < 1.0):
        raise ValueError(f"oversample_datasets: target_frac must be in (0,1), got {target_frac}")
    k = int(round((target_frac * n_total - n_new0) / (1.0 - target_frac)))
    if k <= 0:
        return ds.shuffle(seed=seed)
    rng = np.random.default_rng(seed)
    extra = ds.select(rng.choice(new_idx, size=k, replace=True).tolist())
    return concatenate_datasets([ds, extra]).shuffle(seed=seed)
