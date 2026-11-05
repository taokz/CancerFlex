"""Convenience iterator for Med-Gemma (or any model) baseline evaluation.

Yields per-row tuples ready to be fed to a VLM:
    (row_dict, PIL_image, prompt_text, gold_label, metadata_dict)

Usage:
    from eval_iterator import iter_eval_rows
    for row, img, prompt, label, meta in iter_eval_rows("skin", task="U3",
                                                       split="test_ood"):
        pred = medgemma_predict(prompt, img)
        ...

Handles:
- Reading split and source metadata from the manifest parquet, then applying
  the active task-eligibility contract shared with ``build_task_views.py``
- Filtering by split (`test_id` / `test_ood` / both)
- Filtering rows lacking the requested U-task label
- Image loading (PIL, RGB-converted, gracefully skips on read failure)
- Prompt rendering via `prompt_renderer.render_prompt`

Metadata returned per row includes everything needed to recover stratified
metrics: dataset_id, patient_id, split, subsplit, ood_axis, image_path.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd
from PIL import Image

# Make sibling imports work whether run as a script or imported as a module.
_THIS_DIR = Path(__file__).resolve().parent
_CORE_DIR = _THIS_DIR.parent
for _p in (_CORE_DIR, _THIS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from prompt_renderer import render_prompt
import canonical_image
import u4_axis

from cancerflex.paths import ROOT  # noqa: E402  (set CANCERFLEX_ROOT)


# ---------------------------------------------------------------------------
# Label resolution per U-task
# ---------------------------------------------------------------------------

_BINARY_TASKS = {"U1", "U2", "U3"}
_MULTICLASS_TASKS = {"U4"}
_BBOX_TASKS = {"U5"}
_MASK_TASKS = {"U6"}


def _label_for_task(row: dict, task: str):
    """Return the gold label for the given task, or None if unavailable."""
    if task == "U1":
        return row.get("u1_label")
    if task == "U2":
        return row.get("u2_label")
    if task == "U3":
        return row.get("u3_label")
    if task == "U4":
        v = row.get("u4_label")
        if isinstance(v, str) and v.strip().lower() in {"", "nan", "none", "null"}:
            return None
        return v
    if task == "U5":
        bb = row.get("bbox")
        # bb can be: None, NaN scalar, str JSON, list, or numpy ndarray
        # (ndarray of ndarrays for multi-box). Truth-test ndarrays manually.
        if bb is None:
            return None
        if isinstance(bb, str):
            if bb.strip() in {"", "nan", "None", "null"}:
                return None
            try:
                return json.loads(bb)
            except Exception:
                return None
        # ndarray / list / tuple path
        if hasattr(bb, "__len__"):
            return bb if len(bb) > 0 else None
        # plain scalar (only NaN expected); pd.isna safe here
        try:
            if pd.isna(bb):
                return None
        except (TypeError, ValueError):
            pass
        return bb
    if task == "U6":
        mp = row.get("mask_path")
        if mp is None:
            return None
        if isinstance(mp, str):
            return mp if mp.strip() and mp.strip().lower() not in {"nan", "none", "null"} else None
        try:
            if pd.isna(mp):
                return None
        except (TypeError, ValueError):
            pass
        return mp
    if task == "U7":
        # Report task — gold label is the context_json + ground-truth labels.
        return row.get("context_json")
    return None


def _eval_label_for_task(row: dict, task: str):
    if task == "U4" and row.get("u4_eval_tier") == "primary":
        v = row.get("u4_eval_label")
        if isinstance(v, str) and v.strip().lower() in {"", "nan", "none", "null"}:
            return None
        return v
    return _label_for_task(row, task)


def _label_present(label) -> bool:
    if label is None:
        return False
    if isinstance(label, float):
        import math
        return not math.isnan(label)
    if isinstance(label, str):
        return label.strip() != ""
    if isinstance(label, list):
        return len(label) > 0
    return True


def _task_eligibility_mask(
    df: pd.DataFrame,
    pillar: str,
    task: str,
    gate_col: str | None,
) -> pd.Series:
    """Return the active task-contract eligibility mask.

    Keep this aligned with ``data_build/build_task_views.py``. Binary endpoints
    use every row carrying the requested endpoint label. SLICE-3D ``lesion_id``
    records only whether a lesion was manually tagged; it is not a lesion-
    presence gate, because all released tiles are lesion-centred crops.
    """
    if gate_col and gate_col in df.columns:
        eligible = df[gate_col].notna()
    else:
        eligible = pd.Series(True, index=df.index, dtype=bool)

    if task == "U4" and "u4_eval_unit" in df.columns:
        eligible &= df["u4_eval_unit"].notna() & df["u4_eval_tier"].eq("primary")

    if pillar == "skin" and task == "U1" and "dataset_id" in df.columns:
        eligible &= ~df["dataset_id"].eq("SLICE-3D")

    return eligible


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def _load_image(rel_path: str) -> Optional[Image.Image]:
    """Load an image as RGB via the shared canonical loader; None on failure.

    Delegates to `canonical_image.load_cached`, the single source of truth shared
    with `train_v2/collator.py`. This replaces the previous local decoder, which
    rendered DICOM differently from train (raw min-max, no VOI LUT / MONOCHROME1
    inversion) — the source of the breast CMMD train/eval drift. DICOM is now
    rendered clinically (VOI LUT + MONOCHROME1), identically to training.

    `LOCAL_IMAGE_ROOT` (byte-identical node-local mirror for Lustre small-file
    latency) is still honored for the original-decode fallback path.
    """
    _local_root = os.environ.get("LOCAL_IMAGE_ROOT")
    try:
        return canonical_image.load_cached(rel_path, root=ROOT, local_root=_local_root)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public iterator
# ---------------------------------------------------------------------------

def load_manifest(pillar: str) -> pd.DataFrame:
    """Load the manifest, honoring the eval override environment variables."""
    manifest_override = (
        os.environ.get(f"EVAL_{pillar.upper()}_MANIFEST")
        or os.environ.get("EVAL_MANIFEST_PATH")
    )
    manifest_path = Path(manifest_override) if manifest_override else ROOT / f"Experiment/datasets/{pillar}/manifest.parquet"
    mf = pd.read_parquet(manifest_path)
    for col in ("subsplit", "ood_axis"):
        if col not in mf.columns:
            mf[col] = ""
    return mf.reset_index(drop=True)


def load_eval_manifest(pillar: str) -> pd.DataFrame:
    """Load the manifest filtered to test rows.

    The manifest parquet is the source of truth for row metadata: it carries the
    'split' / 'subsplit' / 'ood_axis' columns written by split_utils.assign_splits
    (deterministic hash split). Task-specific eligibility is applied later by
    ``_task_eligibility_mask`` so it remains aligned with ``build_task_views``.
    Returns a position-indexed DataFrame of the test_id + test_ood rows.
    """
    mf = load_manifest(pillar)
    return mf[mf["split"].isin(["test_id", "test_ood"])].reset_index(drop=True)


def _row_lock_file(task: str, pillar: str | None = None) -> Path | None:
    """Return the optional row-lock file for a task.

    ``EVAL_ROW_LOCK_DIR`` is intentionally opt-in. It lets finished models be
    rescored on the exact rows used by an older baseline after the active
    manifest has changed. Default evaluation remains manifest-driven.

    If ``EVAL_ROW_LOCK_DIR`` contains the literal ``{pillar}``, it is a TEMPLATE
    resolved per-pillar — this lets ONE eval process (model loaded once) score
    all pillars, each with its own lock dir, instead of one process per pillar
    (which re-loads the 15.7GB model for every pillar).
    """
    lock_dir = os.environ.get("EVAL_ROW_LOCK_DIR")
    if not lock_dir:
        return None
    if pillar is not None and "{pillar}" in lock_dir:
        lock_dir = lock_dir.replace("{pillar}", pillar)
    root = Path(lock_dir)
    names = [f"{task}_rows.parquet", f"{task}_rows.csv"]
    if task in {"U5", "U6"}:
        names = [f"{task}_opengen_rows.parquet", f"{task}_opengen_rows.csv", *names]
    for name in names:
        p = root / name
        if p.exists():
            return p
    raise FileNotFoundError(f"EVAL_ROW_LOCK_DIR is set but no row lock exists for {task}: {root}")


def _apply_row_lock(df: pd.DataFrame, pillar: str, task: str) -> pd.DataFrame:
    try:
        p = _row_lock_file(task, pillar)
    except FileNotFoundError:
        if os.environ.get("EVAL_SKIP_MISSING_ROW_LOCK") == "1":
            import sys as _sys
            print(
                f"[eval_iterator] pillar={pillar} task={task}: no row lock; "
                "skipping pair because EVAL_SKIP_MISSING_ROW_LOCK=1.",
                file=_sys.stderr,
            )
            return df.iloc[0:0].copy()
        raise
    if p is None:
        return df
    lock = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    if "image_path" not in lock.columns:
        raise ValueError(f"row lock {p} has no image_path column")
    join_cols = ["image_path"]
    if "split" in lock.columns and "split" in df.columns:
        join_cols.append("split")
    order = lock[join_cols].drop_duplicates().reset_index().rename(columns={"index": "_row_lock_order"})
    out = df.merge(order, on=join_cols, how="inner").sort_values("_row_lock_order")
    out = out.drop(columns=["_row_lock_order"]).reset_index(drop=True)
    import sys as _sys
    print(
        f"[eval_iterator] pillar={pillar} task={task}: applied row lock {p} "
        f"({len(out)}/{len(df)} eligible rows kept).",
        file=_sys.stderr,
    )
    return out


def iter_eval_rows(
    pillar: str,
    task: str,
    split: str = "both",
    template_idx: int = 0,
    seed: Optional[int] = None,
    limit: Optional[int] = None,
    skip_missing_images: bool = True,
    verbose: bool = False,
    strip_leak_fields: bool = True,
    shard: Optional[tuple] = None,
) -> Iterator[tuple]:
    """Yield evaluation tuples for (pillar, task).

    Args:
        pillar: one of skin / breast / cervix / oral / thyroid / blood / liver
        task: U0 / U1 / U2 / U3 / U4 / U5 / U6 / U7
        split: 'test_id', 'test_ood', or 'both' (default)
        template_idx: which prompt template to use (0 = canonical)
        seed: if not None, randomly pick template per row (reproducible)
        limit: optional cap on number of yielded rows
        skip_missing_images: if True, silently skip rows whose image file is
            missing or unreadable. Default True for ergonomics; the iterator
            still tracks skipped_no_image and emits a one-line warning if
            >5% of rows are dropped (so silent CMMD-style misses surface).
        strip_leak_fields: if True (default), pass
            ``prompt_renderer.LEAK_FIELDS_BY_PILLAR[pillar]`` to the renderer
            so gold-correlated context fields (e.g. thyroid `tirads`) are
            omitted. Set False for training where the leak is fine.

    Yields:
        (row_dict, image, prompt_text, gold_label, metadata_dict)

        row_dict: dict of the full manifest row (use only if you need extra fields).
        image: PIL.Image RGB.
        prompt_text: rendered prompt string.
        gold_label: per-task label (binary/string/list-of-bboxes/path).
        metadata_dict: subset useful for results-CSV (image_path, patient_id,
            dataset_id, split, subsplit, ood_axis, cancer_type).

    Bbox dtype heterogeneity (U5): the manifest stores bbox in pillar-native
    encoding — skin/breast/cervix use ndarray of `[cx, cy, w, h]` normalized
    [0,1]; thyroid uses ndarray of `[x1, y1, x2, y2]` in pixels; oral uses
    a JSON-serialized string `'[x1, y1, x2, y2]'` in pixels. Per-pillar
    decoding lives in `scripts/run_medgemma_open_gen.py::_decode_gold_bbox`.
    """
    if task == "U4":
        full_df = u4_axis.attach_primary_eval_columns(load_manifest(pillar), pillar)
        train_primary = full_df[
            full_df["split"].eq("train")
            & full_df["u4_eval_tier"].eq("primary")
            & full_df["u4_eval_unit"].notna()
            & full_df["u4_eval_label"].notna()
        ]
        u4_eval_options_by_unit: dict[str, list[str]] = {}
        for unit, labels in train_primary.groupby("u4_eval_unit")["u4_eval_label"]:
            seen = set(labels.dropna().astype(str))
            ordered = [opt for opt in u4_axis.primary_options(str(unit)) if opt in seen]
            if ordered:
                u4_eval_options_by_unit[str(unit)] = ordered
        df = full_df[full_df["split"].isin(["test_id", "test_ood"])].reset_index(drop=True)
    else:
        u4_eval_options_by_unit = {}
        df = load_eval_manifest(pillar)
    if split != "both":
        df = df[df["split"] == split]
    n_split = len(df)

    # Pre-filter to U-task-eligible rows (vectorized) so the loop never has
    # to silently drop "no label" rows. Different datasets within a pillar
    # are intentionally task-incomplete by data lineage (e.g. thyroid TN3K /
    # TN5000 / Hou ship binary labels only — no TI-RADS, so they are *not*
    # U4-eligible). We cannot physically drop these rows from the manifest:
    # they still participate in U1/U2/U3. So we gate per-task here, using
    # the same eligibility column as v1_corpus/build_v1_corpus.py (and for
    # U4 also `u4_unit IS NOT NULL`, the canonical U4 admission gate). The
    # ineligible count is surfaced as an INFO line, not a "dropped" WARN —
    # it is by-design, not a data-quality issue.
    _ELIG_COL = {"U0": "is_degraded",
                 "U1": "u1_label", "U2": "u2_label", "U3": "u3_label",
                 "U4": "u4_eval_label", "U5": "bbox",     "U6": "mask_path",
                 "U7": "context_json"}
    gate_col = _ELIG_COL.get(task)
    elig = _task_eligibility_mask(df, pillar, task, gate_col)
    df = df[elig].reset_index(drop=True)
    df = _apply_row_lock(df, pillar, task)
    n_total = len(df)               # eligible count (post-filter)
    n_ineligible = n_split - n_total
    if n_ineligible > 0:
        import sys as _sys
        print(f"[eval_iterator] pillar={pillar} task={task} split={split}: "
              f"{n_ineligible}/{n_split} rows ineligible for {task} "
              f"(no {gate_col}" + (" / no u4_eval_unit" if task == "U4" else "")
              + f"); eligible n={n_total}.", file=_sys.stderr)

    yielded = 0
    skipped_no_label = 0  # should stay 0 after vectorized pre-filter
    skipped_no_image = 0
    # --- Bounded, order-preserving multithreaded image prefetch -------------
    # Eval is I/O-bound: per-image disk read + DICOM/PIL decode is 80-209ms
    # (Lustre latency dominated) while the GPU forward/logit-scoring is ~0ms.
    # Image IO + PIL/pydicom decode RELEASE the GIL, so a ThreadPool genuinely
    # overlaps loading of upcoming rows with the GPU scoring of the current one
    # (same trick the training collator uses for a 5-7x load speedup). We keep
    # the *exact* serial semantics: only rows that pass the cheap label check
    # have their image loaded, rows are yielded in manifest order, and a None
    # image (load failure) is handled identically to before. Worker count is
    # configurable via EVAL_LOAD_WORKERS (default 8).
    try:
        _n_workers = int(os.environ.get("EVAL_LOAD_WORKERS", "8"))
    except (TypeError, ValueError):
        _n_workers = 8
    _n_workers = max(1, _n_workers)

    def _label_passing_rows():
        """Yield (row_dict, label) for rows passing the label check, in manifest
        order, incrementing skipped_no_label exactly as the serial loop did.

        If ``shard=(i,N)`` is set, keep only every N-th label-passing row BEFORE
        its image is decoded (notes §8) — so a sharded run decodes 1/N of the
        images instead of all N shards decoding all of them. The shard counter
        counts label-passing rows; with no image-load failures (the norm under
        the image cache) this equals the consumer's post-yield ``seen_rows``
        index, so it is OUTPUT-IDENTICAL to post-yield sharding. (Under image
        failures the two indices drift, but each remains a valid 1/N partition.)"""
        nonlocal skipped_no_label
        _sidx = 0
        for _, _row in df.iterrows():
            _rd = _row.to_dict()
            _lb = _eval_label_for_task(_rd, task)
            if not _label_present(_lb):
                # Defensive: pre-filter above should have caught all ineligible
                # rows. Hitting this branch means an unusual NaN representation
                # slipped past `.notna()` (e.g. literal "nan" string) — log it.
                skipped_no_label += 1
                continue
            if shard is not None:
                _keep = (_sidx % shard[1] == shard[0])
                _sidx += 1
                if not _keep:
                    continue
            yield _rd, _lb

    def _prefetch_images(pairs, n_workers):
        """Map `_load_image` over (row_dict, label) pairs in input order with a
        bounded look-ahead window, yielding (row_dict, label, image). Ordering
        and per-row image identity match the serial path exactly; only *when*
        the load happens (overlapped) changes."""
        if n_workers <= 1:
            for _rd, _lb in pairs:
                yield _rd, _lb, _load_image(_rd["image_path"])
            return
        from collections import deque
        from concurrent.futures import ThreadPoolExecutor
        window = max(2 * n_workers, 16)  # max rows kept in flight (look-ahead)
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            inflight = deque()
            src = iter(pairs)
            exhausted = False
            while True:
                while not exhausted and len(inflight) < window:
                    try:
                        _rd, _lb = next(src)
                    except StopIteration:
                        exhausted = True
                        break
                    fut = ex.submit(_load_image, _rd["image_path"])
                    inflight.append((_rd, _lb, fut))
                if not inflight:
                    break
                _rd, _lb, fut = inflight.popleft()
                yield _rd, _lb, fut.result()

    for row_dict, label, image in _prefetch_images(
        _label_passing_rows(), _n_workers
    ):
        if image is None:
            skipped_no_image += 1
            if skip_missing_images:
                continue
        per_row_seed = (seed + yielded) if seed is not None else None
        # U4: prompt must list the SAME per-report-unit option set the scorer uses
        # (u4_axis.options(u4_unit)), not the static PILLAR_CONFIG list — else the
        # prompt's classes mismatch the gold/scoring for thyroid/breast.
        u4_opts = None
        if task == "U4":
            unit = row_dict.get("u4_eval_unit")
            if isinstance(unit, str) and unit:
                opts_json = row_dict.get("u4_eval_options_json")
                if isinstance(opts_json, str) and opts_json.strip():
                    try:
                        parsed_opts = json.loads(opts_json)
                        if isinstance(parsed_opts, list) and parsed_opts:
                            u4_opts = [str(v) for v in parsed_opts]
                    except Exception:
                        u4_opts = None
                u4_opts = u4_opts or u4_eval_options_by_unit.get(unit) or u4_axis.primary_options(unit)
        # Prompt-stem version (prompt-design audit 2026-09-16): env
        # EDGE_PROMPT_STEM_VERSION, default 1 = the frozen v1 rendering. Only a
        # model TRAINED on a version-2 corpus should be evaluated with version 2;
        # the version is recorded in `meta` so the eval CSV/log can prove which
        # prompt wording was scored.
        _stem_version = u4_axis.prompt_stem_version()
        _u4_unit = row_dict.get("u4_eval_unit") if task == "U4" else None
        prompt = render_prompt(
            row_dict,
            task=task,
            template_idx=template_idx if seed is None else None,
            seed=per_row_seed,
            strip_leak_fields=strip_leak_fields,
            u4_options=u4_opts,
            u4_unit=_u4_unit if isinstance(_u4_unit, str) else None,
            prompt_stem_version=_stem_version,
        )
        meta = {
            "image_path":  row_dict["image_path"],
            "patient_id":  row_dict.get("patient_id"),
            "dataset_id":  row_dict.get("dataset_id"),
            "cancer_type": row_dict.get("cancer_type"),
            "split":       row_dict["split"],
            "subsplit":    row_dict.get("subsplit"),
            "ood_axis":    row_dict.get("ood_axis"),
            "u4_unit":     row_dict.get("u4_unit"),
            "u4_label":    row_dict.get("u4_label"),
            "u4_eval_unit": row_dict.get("u4_eval_unit"),
            "u4_eval_label": row_dict.get("u4_eval_label"),
            "u4_eval_tier": row_dict.get("u4_eval_tier"),
            "u4_eval_options": u4_opts,
            # canonical option list stays in u4_eval_options (scorer key); the
            # rendered surface forms differ from it only under stem version >= 2.
            "prompt_stem_version": _stem_version,
            "u4_eval_options_rendered": (
                u4_axis.surface_options(_u4_unit, u4_opts, _stem_version)
                if (task == "U4" and u4_opts and isinstance(_u4_unit, str)) else u4_opts),
        }
        yield row_dict, image, prompt, label, meta
        yielded += 1
        if limit is not None and yielded >= limit:
            break
    import sys as _sys
    # Pre-filter handles no-label upfront, so skipped_no_label should be 0
    # in normal operation. Anything > 0 is a defensive-catch oddity worth
    # surfacing (e.g. a literal "nan" string that bypassed `.notna()`).
    if n_total > 0:
        drop_image = skipped_no_image / n_total
        if skipped_no_label > 0:
            print(
                f"[eval_iterator WARN] pillar={pillar} task={task} split={split}: "
                f"{skipped_no_label} eligible rows had unparseable labels "
                f"(post pre-filter). Investigate — pre-filter should have caught these.",
                file=_sys.stderr,
            )
        if drop_image > 0.05:
            print(
                f"[eval_iterator WARN] pillar={pillar} task={task} split={split}: "
                f"{skipped_no_image}/{n_total} rows ({drop_image:.1%}) failed to "
                f"load (likely DICOM without pydicom, or missing files). Install "
                f"`pydicom>=3.0` for CMMD/INbreast, and verify data/<Pillar>/ paths.",
                file=_sys.stderr,
            )
    if verbose:
        print(f"[eval_iterator] pillar={pillar} task={task} split={split}: "
              f"yielded={yielded} skipped_no_label={skipped_no_label} "
              f"skipped_no_image={skipped_no_image}")


# ---------------------------------------------------------------------------
# Tiny answer-parsing helpers (model-agnostic)
# ---------------------------------------------------------------------------

import re

_YES_REGEX = re.compile(r"\b(yes|positive|malignant|cancer|refer|biopsy|abnormal|present|adequate|use|accept)\b", re.I)
_NO_REGEX  = re.compile(r"\b(no|negative|benign|normal|surveillance|not|absent|inadequate|reject|re-?acquire)\b", re.I)


def parse_binary(text: str) -> Optional[int]:
    """Parse a model's free-form binary answer to {0, 1, None}.

    None signals parse failure (track these per task — high parse-failure rate
    is a real signal about Med-Gemma's zero-shot fitness).
    """
    if not isinstance(text, str):
        return None
    t = text.strip().lower()
    if not t:
        return None
    # Look for the FIRST decisive token. Yes-words check before no-words
    # because phrases like "Yes, not benign" should resolve to yes.
    yes_match = _YES_REGEX.search(t)
    no_match = _NO_REGEX.search(t)
    if yes_match and (not no_match or yes_match.start() < no_match.start()):
        return 1
    if no_match:
        return 0
    return None


def parse_multiclass(text: str, class_list: list[str]) -> Optional[str]:
    """Match the model output against a class list (longest-substring, case-insensitive).

    Returns the matched class name (canonical form), or None if no match.
    """
    if not isinstance(text, str):
        return None
    t = text.lower()
    matches = []
    for cls in class_list:
        cls_clean = cls.lower().strip()
        if not cls_clean:
            continue
        if cls_clean in t:
            matches.append((len(cls_clean), cls))
    if not matches:
        return None
    return max(matches, key=lambda x: x[0])[1]


_BBOX_REGEX = re.compile(r"\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]")


def parse_bboxes(text: str) -> list[list[float]]:
    """Extract bbox tuples from free-form model output."""
    if not isinstance(text, str):
        return []
    boxes = []
    for m in _BBOX_REGEX.finditer(text):
        boxes.append([float(g) for g in m.groups()])
    return boxes


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pillar = sys.argv[1] if len(sys.argv) > 1 else "skin"
    task   = sys.argv[2] if len(sys.argv) > 2 else "U3"
    limit  = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    print(f"Iterating {pillar} / {task} / both splits / limit={limit}\n")
    for i, (row, img, prompt, label, meta) in enumerate(
        iter_eval_rows(pillar, task, split="both", limit=limit, verbose=True)
    ):
        print(f"=== row {i} ===")
        print(f"  meta: {meta}")
        print(f"  gold label: {label!r}")
        print(f"  image size: {img.size if img else 'IMAGE MISSING'}")
        print(f"  prompt (first 400 chars):")
        print("  " + prompt[:400].replace("\n", "\n  "))
        print()
