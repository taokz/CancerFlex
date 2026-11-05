"""v1 corpus → HuggingFace Datasets loader for TRL SFTTrainer.

Each JSONL record carries `prompt`, `answer`, `image_path` (+ metadata). The
prompt already contains a literal `<image>` placeholder produced by
`Experiment/core_code/prompt_renderer.render_prompt(...)`; the collator
replaces it with the processor's image token (`<image_soft_token>` for Gemma3
/ `<image>` for Gemma-3n native — handled per-processor in `collator.py`).

Why HF Datasets (not raw torch Dataset): TRL SFTTrainer reads HF Datasets
out of the box, and `streaming=True` reads JSONL line-by-line so the 2M-row
train file never sits in RAM (≈ 2.6 GB raw, OK to load fully too — we let
the user choose via `stream=True`).

Balancing (skin = 81% of corpus) is NOT applied here. To balance, set
`balance_temperature=τ ∈ (0,1]` on the config: P(pillar) ∝ count^τ, τ=0.5
flattens skin to ≈ 47% — matches the eval-time recommendation in
`docs/methodology/eval_split_convention.md`.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterator, Optional

from datasets import Dataset, IterableDataset

from cancerflex.paths import ROOT  # noqa: E402  (set CANCERFLEX_ROOT)


def _split_jsonl_files(root: Path, split: str) -> list[Path]:
    """Resolve a split from either legacy flat JSONL or sharded JSONL layout."""
    flat = root / f"{split}.jsonl"
    if flat.exists():
        return [flat]
    shard_root = root / "shards" / split
    if shard_root.exists():
        files = sorted(shard_root.rglob("*.jsonl"))
        if files:
            return files
    raise FileNotFoundError(f"no JSONL files for split={split}: {flat} or {shard_root}")


def _read_jsonl(path: Path) -> Iterator[dict]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _read_jsonl_many(paths: list[Path]) -> Iterator[dict]:
    for path in paths:
        yield from _read_jsonl(path)


def _read_u6_image_path_allowlist(path_value: str | Path | None) -> set[str] | None:
    """Return an optional U6-only image-path allowlist.

    This is deliberately a loader-side selection rather than a corpus edit:
    all non-U6 rows remain available and a training config can opt in without
    mutating the de-leaked source shards. Relative paths resolve from the repo
    root so YAML configs stay portable within this workspace.
    """
    if path_value in (None, ""):
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f"U6 image-path allowlist does not exist: {path}")
    allowed = {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if not allowed:
        raise ValueError(f"U6 image-path allowlist is empty: {path}")
    return allowed


def _keep_u6_allowlisted(record: dict, allowed_u6_paths: set[str] | None) -> bool:
    """Keep every task unchanged; restrict only U6 when an allowlist is set."""
    return (
        allowed_u6_paths is None
        or record.get("task") != "U6"
        or record.get("image_path") in allowed_u6_paths
    )


def load_v1_split(split: str, corpus_dir: str | Path = "Experiment/datasets/v1_final_deleaked",
                  stream: bool = False, limit: int | None = None,
                  u6_image_path_allowlist: str | Path | None = None) -> Dataset | IterableDataset:
    """Load one split (train / val / test_id / test_ood) as an HF Dataset.

    stream=False (default): all rows in RAM (~2.6 GB for the 2M-row train).
    stream=True            : IterableDataset (line-by-line, lower RAM but no
                             shuffle/select; use for very large corpora).
    limit                  : optional row cap (smoke tests).
    u6_image_path_allowlist: optional newline-delimited list. When set, retain
                             every non-U6 row and only listed U6 image paths.
                             Applied before ``limit`` and any caller-side
                             balancing; do not use it for eval splits.
    """
    path = Path(corpus_dir) if Path(corpus_dir).is_absolute() else ROOT / corpus_dir
    files = _split_jsonl_files(path, split)
    allowed_u6_paths = _read_u6_image_path_allowlist(u6_image_path_allowlist)

    if stream:
        def _gen():
            emitted = 0
            for rec in _read_jsonl_many(files):
                if not _keep_u6_allowlisted(rec, allowed_u6_paths):
                    continue
                if limit is not None and emitted >= limit:
                    return
                emitted += 1
                yield rec
        return IterableDataset.from_generator(_gen)

    # gold_raw is heterogeneous (int for U1/U2/U3, str for U4) — pyarrow's
    # type inference fails on mixed types. Stringify it (we only use answer/
    # prompt/image_path for SFT; gold_raw is metadata for downstream eval).
    records = []
    for rec in _read_jsonl_many(files):
        if not _keep_u6_allowlisted(rec, allowed_u6_paths):
            continue
        if limit is not None and len(records) >= limit:
            break
        if "gold_raw" in rec and rec["gold_raw"] is not None:
            rec["gold_raw"] = str(rec["gold_raw"])
        records.append(rec)
    return Dataset.from_list(records)


def temperature_balance(ds: Dataset, by: str = "pillar", tau: float = 0.5,
                        seed: int = 0) -> Dataset:
    """Resample so P(group) ∝ count^τ. τ=1 keeps natural frequencies,
    τ→0 uniformizes. Skin is 81% of raw corpus → τ=0.5 flattens to ~47%.

    Implementation: per-group resample with replacement to a target size
    chosen so total ≈ len(ds). Deterministic given seed.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    groups: dict[str, list[int]] = {}
    for i, g in enumerate(ds[by]):
        groups.setdefault(g, []).append(i)
    counts = {g: len(ix) for g, ix in groups.items()}
    raw = np.array(list(counts.values()), dtype=float)
    keys = list(counts.keys())
    weights = raw ** tau
    weights = weights / weights.sum()
    total = len(ds)
    targets = (weights * total).astype(int)
    # Resample indices with replacement per group up to its target.
    picked: list[int] = []
    for g, t in zip(keys, targets):
        ix = np.array(groups[g])
        if t <= len(ix):
            sel = rng.choice(ix, size=t, replace=False)
        else:
            sel = rng.choice(ix, size=t, replace=True)
        picked.extend(sel.tolist())
    rng.shuffle(picked)
    return ds.select(picked)
