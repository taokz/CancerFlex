"""MedGemma free-form generation driver for U5 / U7.

Dataset boundary:
- Active-current eval reads current v1_final JSONL shards directly from
  `$CANCERFLEX_ROOT/Experiment/datasets/v1_final_deleaked/shards/{test_id,test_ood}/`.
- Difference eval can restrict rows by setting `EVAL_ROW_LOCK_DIR` to the
  generated legacy-to-active row-lock directory. This scores only rows that
  cannot reuse legacy predictions.
- U5 uses the MedGemma localization prompt format documented by
  https://github.com/Google-Health/medgemma: bounding boxes are emitted as
  `[y0, x0, y1, x1]` normalized to [0, 1000].
- U6 is standard pixel-mask segmentation in the current active data. MedGemma's
  text-generation interface used here cannot emit pixel masks, so this runner
  deliberately does not evaluate U6.

Tasks supported:

- U5 (bbox): per-pillar localization query, parse Final Answer JSON, compute
  IoU against gold bbox.
- U7 (structured open generation): current v1_final JSONL rows provide the
  prompt and structured gold answer. The runner records raw model output and
  parse fields so later exact/structured/LLM-as-judge scoring can be applied.

Per-row output: CSV with `predicted_raw`, parsed fields, task score columns,
`parse_ok`, plus metadata.

Per-pair output: metrics_<pillar>_<task>_opengen.json with mIoU, mAP@0.5,
parse_failure_rate, per-OOD-axis breakdown where applicable.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import re
import sys
import time
import traceback
from pathlib import Path


def _preflight_unsupported_tasks() -> None:
    tasks_arg: str | None = None
    for idx, arg in enumerate(sys.argv[1:], start=1):
        if arg == "--tasks" and idx + 1 < len(sys.argv):
            tasks_arg = sys.argv[idx + 1]
            break
        if arg.startswith("--tasks="):
            tasks_arg = arg.split("=", 1)[1]
            break
    if tasks_arg is None:
        return
    tasks = {part.strip() for part in tasks_arg.split(",") if part.strip()}
    # U6 used to be rejected because active v1_final stored U6 as a pixel mask.
    # Our multi-task corpus trains U6 as **box-from-mask** (seg prompt -> box_2d),
    # and v1_final_deleaked materializes U6 test rows in that box form. The body
    # already scores U6 exactly like U5 (box-IoU; gold via _decode_gold_objects_as_boxes
    # with a mask->bbox fallback). So U6 is supported when the shard source carries
    # box rows — point EVAL_OPENGEN_SHARDS at v1_final_deleaked. No rejection here.


_preflight_unsupported_tasks()

import numpy as np
import pandas as pd
import torch
from PIL import Image

from cancerflex.paths import ROOT  # noqa: E402  (set CANCERFLEX_ROOT)
# the evaluation modules import each other by flat name; make this directory importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_iterator import iter_eval_rows  # noqa: E402
import canonical_image  # noqa: E402

OUT_DIR = Path(os.environ.get("EVAL_OUT_DIR", str(ROOT / "eval_results" / "medgemma_baseline")))
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUT_DIR / "run_log.jsonl"
MODEL_PATH = os.environ.get("CANCERFLEX_BASE_MODEL", "google/medgemma-1.5-4b-it")


# ─────────────────────────────────────────────────────────────────────────────
# Load-once / multi-width helpers (mirror run_closed.py:173-238, PERF §18).
# A MatFormer checkpoint is loaded+wrapped ONCE and each width is an in-place
# controller.set() slice of the SAME weights, so a K-width open-gen sweep pays
# one cold load instead of K. Each width routes its raw CSVs/metrics/log to its
# own EVAL_OUT_DIR so resume/dedup stays width-local (a scalar width is unchanged).
# ─────────────────────────────────────────────────────────────────────────────
def _parse_matformer_widths() -> list[int]:
    """Parse the backward-compatible scalar-or-list EVAL_MATFORMER_WIDTH knob.

    A scalar behaves exactly as before. A comma-list is the load-once mode and
    requires EVAL_MATFORMER_OUT_DIRS in main() so one width's raw rows can never
    trigger a resume-skip for another width.
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

    The JSON environment form is intentionally explicit, e.g.
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
# Open-gen reads pre-materialized JSONL shards. Default = active v1_final (U5/U7,
# U6 as mask). Override to eval a model on the corpus it was trained on, e.g. the
# de-leaked multi-task corpus (which has U6 as box-from-mask). Unified knob
# EVAL_SHARDS_ROOT is shared with run_u0 so one env redirects U0/U5/U6 together;
# EVAL_OPENGEN_SHARDS kept as an open-gen-only alias.
#   EVAL_SHARDS_ROOT=Experiment/datasets/v1_final_deleaked/shards
ACTIVE_SHARDS = Path(
    os.environ.get("EVAL_SHARDS_ROOT")
    or os.environ.get("EVAL_OPENGEN_SHARDS")
    or str(ROOT / "Experiment/datasets/v1_final_deleaked/shards"))  # de-leaked default (was v1_final; repointed 2026-07-19, DEPRECATIONS §row v1_final); env still overrides
if not ACTIVE_SHARDS.is_absolute():
    ACTIVE_SHARDS = ROOT / ACTIVE_SHARDS


def log_event(category: str, **fields) -> None:
    rec = {"ts": dt.datetime.utcnow().isoformat() + "Z", "category": category, **fields}
    try:
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
    except Exception:
        pass


def _row_lock_file(task: str, pillar: str | None = None) -> Path | None:
    lock_dir = os.environ.get("EVAL_ROW_LOCK_DIR")
    if not lock_dir:
        return None
    if pillar is not None and "{pillar}" in lock_dir:
        lock_dir = lock_dir.replace("{pillar}", pillar)
    root = Path(lock_dir)
    names = [f"{task}_opengen_rows.parquet", f"{task}_opengen_rows.csv", f"{task}_rows.parquet", f"{task}_rows.csv"]
    for name in names:
        p = root / name
        if p.exists():
            return p
    raise FileNotFoundError(f"EVAL_ROW_LOCK_DIR is set but no row lock exists for {task}: {root}")


def _load_row_lock(task: str, pillar: str) -> pd.DataFrame | None:
    try:
        p = _row_lock_file(task, pillar)
    except FileNotFoundError:
        if os.environ.get("EVAL_SKIP_MISSING_ROW_LOCK") == "1":
            log_event(
                "opengen_row_lock_missing_skip",
                pillar=pillar,
                task=task,
            )
            return pd.DataFrame(columns=["record_id", "image_path", "_row_lock_order"])
        raise
    if p is None:
        return None
    lock = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    if "image_path" not in lock.columns and "record_id" not in lock.columns:
        raise ValueError(f"row lock {p} has neither image_path nor record_id")
    keep = [c for c in ["record_id", "image_path"] if c in lock.columns]
    out = lock[keep].drop_duplicates().reset_index().rename(columns={"index": "_row_lock_order"})
    return out


def _active_jsonl_records(pillar: str, task: str) -> list[dict]:
    """Load current active U5/U7 records exactly as materialized for v1_final."""
    rows: list[dict] = []
    _es = os.environ.get("EVAL_SPLIT", "both")
    _splits = (_es,) if _es in ("test_id", "test_ood") else ("test_id", "test_ood")
    for split in _splits:
        path = ACTIVE_SHARDS / split / task / f"{pillar}.jsonl"
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                rec["_source_jsonl"] = str(path.relative_to(ROOT))
                rows.append(rec)
    if not rows:
        return rows
    lock = _load_row_lock(task, pillar)
    if lock is not None:
        df = pd.DataFrame(rows)
        if "record_id" in lock.columns and "record_id" in df.columns:
            df = df.merge(lock[["record_id", "_row_lock_order"]], on="record_id", how="inner")
        else:
            df = df.merge(lock[["image_path", "_row_lock_order"]], on="image_path", how="inner")
        df = df.sort_values("_row_lock_order").drop(columns=["_row_lock_order"]).reset_index(drop=True)
        rows = df.to_dict("records")
        log_event("opengen_row_lock_applied", pillar=pillar, task=task, n=len(rows))
    return rows


def iter_active_opengen_rows(
    pillar: str,
    task: str,
    shard: tuple[int, int] | None = None,
    completed: set[str] | None = None,
):
    """Yield (row, image, prompt, gold, meta) from active v1_final JSONL.

    This is the current-data SSOT for U5/U7. The old manifest iterator loses
    task-view-only details such as mask-derived U5 boxes and U7 eligible fields.

    Apply deterministic row selection before image loading.  The previous order
    decoded every skipped image, so a 100-row stride sample from blood/U5 still
    performed thousands of GPFS cache reads while the GPU sat idle.
    """
    for seen_rows, rec in enumerate(_active_jsonl_records(pillar, task)):
        if shard is not None and seen_rows % shard[1] != shard[0]:
            continue
        rel = rec.get("image_path")
        meta = {
            "image_path": rel or "",
            "patient_id": rec.get("patient_id"),
            "dataset_id": rec.get("dataset_id"),
            "split": rec.get("split"),
            "subsplit": rec.get("subsplit"),
            "ood_axis": rec.get("ood_axis"),
            "record_id": rec.get("record_id"),
            "source_jsonl": rec.get("_source_jsonl"),
        }
        row_key = (
            f"{meta.get('source_jsonl') or ''}::{meta.get('record_id')}"
            if meta.get("record_id")
            else str(meta["image_path"])
        )
        if completed and row_key in completed:
            continue
        image = None
        if rel:
            try:
                image = canonical_image.load_cached(
                    rel, root=ROOT, local_root=os.environ.get("LOCAL_IMAGE_ROOT"))
            except Exception as exc:
                log_event("image_load_failure", pillar=pillar, task=task,
                          image_path=rel, error=str(exc))
        yield rec, image, rec.get("prompt") or build_text_prompt(pillar, task), rec.get("gold_raw"), meta


# ─────────────────────────────────────────────────────────────────────────────
# Prompt construction
# ─────────────────────────────────────────────────────────────────────────────

LOC_INSTRUCTIONS = (
    "Instructions:\n"
    "The following user query will require outputting bounding boxes. The format of "
    "bounding boxes coordinates is [y0, x0, y1, x1] where (y0, x0) must be top-left "
    "corner and (y1, x1) the bottom-right corner. This implies that x0 < x1 and "
    "y0 < y1. Always normalize the x and y coordinates the range [0, 1000], meaning "
    "that a bounding box starting at 15% of the image width would be associated with "
    "an x coordinate of 150. You MUST output a single parseable json list of objects "
    "enclosed into ```json...``` brackets, for instance ```json[{\"box_2d\": [800, 3, "
    "840, 471], \"label\": \"car\"}, {\"box_2d\": [400, 22, 600, 73], \"label\": "
    "\"dog\"}]``` is a valid output. Now answer to the user query.\n\nQuery:\n"
)

U5_QUERY = {
    "skin":    "Where is the visible skin lesion? Output the final answer in the format \"Final Answer: X\" where X is a JSON list of objects. Each object needs a \"box_2d\" and \"label\" key.",
    "breast":  "Where is the breast lesion or mass on this ultrasound? Output the final answer in the format \"Final Answer: X\" where X is a JSON list of objects. Each object needs a \"box_2d\" and \"label\" key.",
    "cervix":  "Where is the cervical abnormality or abnormal cell region in this image? Output the final answer in the format \"Final Answer: X\" where X is a JSON list of objects. Each object needs a \"box_2d\" and \"label\" key.",
    "oral":    "Where is the oral lesion in this image? Output the final answer in the format \"Final Answer: X\" where X is a JSON list of objects. Each object needs a \"box_2d\" and \"label\" key.",
    "thyroid": "Where is the thyroid nodule on this ultrasound? Output the final answer in the format \"Final Answer: X\" where X is a JSON list of objects. Each object needs a \"box_2d\" and \"label\" key.",
    "blood":   "Where are the white blood cells (WBCs) in this peripheral blood smear? Output the final answer in the format \"Final Answer: X\" where X is a JSON list of objects. Each object needs a \"box_2d\" and \"label\" key.",
}

U7_THYROID_PROMPT = (
    "You are reviewing a thyroid ultrasound. Examine the image and report the "
    "following sonographic features in JSON. Use exactly these keys: \"margin\", "
    "\"echo_pattern\", \"calcifications\". \n\n"
    "- margin: one of {\"well defined\", \"ill defined\", \"spiculated\", \"microlobulated\"} or null.\n"
    "- echo_pattern: one of {\"anechoic\", \"hyperechogenicity\", \"isoechogenicity\", "
    "\"hypoechogenicity\", \"marked hypoechogenicity\"} or null.\n"
    "- calcifications: one of {\"none\", \"macrocalcifications\", \"microcalcifications\", "
    "\"peripheral\", \"rim\"} or null.\n\n"
    "Output the final answer in the format \"Final Answer: X\" where X is a single "
    "JSON object with those three keys."
)


# U7 structured-report fields per pillar (field-level accuracy task). thyroid = TI-RADS
# sonographic features; breast = BI-RADS US descriptors (BrEaST); skin = 7-point dermoscopic
# checklist (Derm7pt/PH2, parsed from the serialized lesion_description_text).
U7_FIELDS = {
    "thyroid": ["margin", "echo_pattern", "calcifications"],
    "breast":  ["shape", "margin", "echo_pattern", "calcifications", "posterior_features"],
    "skin":    ["pigment network", "streaks", "pigmentation", "regression structures",
                "dots and globules", "blue whitish veil", "vascular structures"],
}
U7_OPTIONS = {
    "breast": {
        "shape": ["oval", "round", "irregular"],
        "margin": ["circumscribed", "indistinct", "angular", "microlobulated", "spiculated"],
        "echo_pattern": ["anechoic", "hypoechoic", "isoechoic", "hyperechoic",
                         "complex cystic/solid", "heterogeneous"],
        "calcifications": ["no", "in a mass", "intraductal", "indefinable"],
        "posterior_features": ["no", "enhancement", "shadowing", "combined"],
    },
    "skin": {
        "pigment network": ["typical", "atypical", "absent"],
        "streaks": ["absent", "regular", "irregular"],
        "pigmentation": ["absent", "diffuse regular", "diffuse irregular",
                         "localized regular", "localized irregular"],
        "regression structures": ["absent", "blue areas", "white areas", "combinations"],
        "dots and globules": ["absent", "regular", "irregular"],
        "blue whitish veil": ["absent", "present"],
        "vascular structures": ["absent", "arborizing", "comma", "dotted",
                                "linear irregular", "within regression"],
    },
}
U7_MODALITY = {"breast": "breast ultrasound", "skin": "dermoscopy image"}


def build_u7_prompt(pillar: str) -> str:
    if pillar == "thyroid":
        return U7_THYROID_PROMPT  # preserve exact wording (thyroid U7 already scored)
    fields = U7_FIELDS[pillar]
    opts = U7_OPTIONS[pillar]
    keys = ", ".join(f'"{k}"' for k in fields)
    lines = "\n".join(f'- {k}: one of {{{", ".join(opts[k])}}} or null.' for k in fields)
    return (f"You are reviewing a {U7_MODALITY[pillar]}. Examine the image and report the "
            f"following features in JSON. Use exactly these keys: {keys}.\n\n{lines}\n\n"
            'Output the final answer in the format "Final Answer: X" where X is a single '
            "JSON object with those keys.")


def u7_gold_fields(pillar: str, gold_label) -> dict | None:
    """Gold finding fields for one U7 row. thyroid/breast read discrete imaging_findings
    keys; skin parses the serialized 7-point checklist in lesion_description_text."""
    try:
        cj = gold_label if isinstance(gold_label, dict) else json.loads(gold_label)
    except Exception:
        return None
    f = (cj.get("imaging_findings") or {}) if isinstance(cj, dict) else {}
    if not isinstance(f, dict):
        return None
    fields = U7_FIELDS[pillar]
    if pillar == "skin":
        kv = {}
        for part in str(f.get("lesion_description_text") or "").split(";"):
            if ":" in part:
                k, v = part.split(":", 1)
                kv[k.strip().lower()] = v.strip().lower()
        gold = {k: kv[k] for k in fields if kv.get(k)}
    else:
        gold = {k: str(f[k]).strip().lower() for k in fields
                if f.get(k) not in (None, "", "null")}
    return gold or None


def build_text_prompt(pillar: str, task: str) -> str:
    if task in ("U5", "U6"):
        return LOC_INSTRUCTIONS + U5_QUERY[pillar]
    if task == "U7":
        return build_u7_prompt(pillar)
    raise ValueError(task)


# ─────────────────────────────────────────────────────────────────────────────
# Gold bbox / mask decoding
# ─────────────────────────────────────────────────────────────────────────────


def _to_pred_xyxy_norm(box_2d_ykxk_thousand) -> list[float] | None:
    """MedGemma output [y0, x0, y1, x1] in [0,1000] → [x1, y1, x2, y2] in [0,1]."""
    try:
        y0, x0, y1, x1 = [float(v) for v in box_2d_ykxk_thousand]
    except Exception:
        return None
    if not (math.isfinite(y0) and math.isfinite(x0) and math.isfinite(y1) and math.isfinite(x1)):
        return None
    return [x0 / 1000.0, y0 / 1000.0, x1 / 1000.0, y1 / 1000.0]


def _decode_gold_bbox(pillar: str, raw_bbox, image_size: tuple[int, int]) -> list[list[float]] | None:
    """Return list of [x1, y1, x2, y2] boxes in [0,1] normalised coords.

    Canonical manifest schema (enforced 2026-05-19 across all builders):
      `bbox` = list of `[cx, cy, w, h]` in [0,1] normalised image coords.

    The `pillar` and `image_size` args are kept for backward compatibility
    but no longer dispatch per-pillar branches.
    """
    if raw_bbox is None:
        return None
    if isinstance(raw_bbox, str):
        try:
            raw_bbox = json.loads(raw_bbox)
        except Exception:
            return None
    out: list[list[float]] = []
    for b in raw_bbox:
        try:
            cx, cy, w, h = [float(v) for v in b]
        except Exception:
            continue
        out.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    return out if out else None


def _decode_gold_mask_as_bbox(mask_path: str | None) -> list[float] | None:
    """Read a mask PNG, return its tight bbox in [0,1] normalised coords."""
    if not isinstance(mask_path, str) or not mask_path:
        return None
    abs_path = (ROOT / mask_path).resolve()
    if not abs_path.exists():
        return None
    try:
        m = np.array(Image.open(str(abs_path)).convert("L"))
    except Exception:
        return None
    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        return None
    H, W = m.shape
    return [float(xs.min()) / W, float(ys.min()) / H,
            float(xs.max()) / W, float(ys.max()) / H]


def _decode_gold_objects_as_boxes(gold_raw) -> list[list[float]] | None:
    """v1_final U5 gold_raw objects -> [x1,y1,x2,y2] normalised boxes."""
    if isinstance(gold_raw, str):
        try:
            gold_raw = json.loads(gold_raw)
        except Exception:
            return None
    if isinstance(gold_raw, dict):
        gold_raw = [gold_raw]
    if not isinstance(gold_raw, list):
        return None
    boxes = []
    for obj in gold_raw:
        if not isinstance(obj, dict):
            continue
        bb = _to_pred_xyxy_norm(obj.get("box_2d", []))
        if bb is not None:
            boxes.append(bb)
    return boxes or None


def _normalise_json_value(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return f"{float(value):.6g}"
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).lower()
    return str(value).strip().lower()


def _nested_lookup(obj: dict, dotted: str):
    if not isinstance(obj, dict):
        return None
    if dotted in obj:
        return obj[dotted]
    parts = dotted.split(".")
    candidates = [parts]
    if parts and parts[0] == "context_json":
        candidates.append(parts[1:])
    for path in candidates:
        cur = obj
        ok = True
        for part in path:
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                ok = False
                break
        if ok:
            return cur
    return None


def compare_u7_fields(pred: dict | None, gold: dict) -> dict[str, bool]:
    if not isinstance(pred, dict) or not isinstance(gold, dict):
        return {}
    out = {}
    for key, g in gold.items():
        p = _nested_lookup(pred, str(key))
        out[str(key)] = _normalise_json_value(p) == _normalise_json_value(g)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Output parsing
# ─────────────────────────────────────────────────────────────────────────────

_REASONING_RE = re.compile(r"<unused94>.*?<unused95>", re.DOTALL)
_FINAL_ANSWER_RE = re.compile(r"Final Answer:\s*", re.IGNORECASE)
_JSON_FENCE_RE = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)
_JSON_OBJ_HEAD_RE = re.compile(r"[\[{]")


def _strip_reasoning(text: str) -> str:
    text = _REASONING_RE.sub("", text)
    if "<unused95>" in text:
        text = text.split("<unused95>", 1)[1].lstrip()
    return text


# A single flat box object {"box_2d":[...], "label":"..."} (box objects have no
# nested braces; the box_2d list has no braces) — used to salvage complete leading
# objects from a TRUNCATED / unclosed list (model hit max_new_tokens mid-JSON).
_BOX_OBJ_RE = re.compile(r'\{[^{}]*?"box_2d"\s*:\s*\[[^\]]*\][^{}]*?\}')


def _dedup_boxes(objs: list[dict]) -> list[dict]:
    """Keep dict objects with a box_2d, dropping EXACT-duplicate boxes (degenerate
    repetition inflates predictions). Distinct multi-target boxes are preserved."""
    seen, out = set(), []
    for o in objs:
        if not (isinstance(o, dict) and "box_2d" in o):
            continue
        key = json.dumps(o.get("box_2d"))
        if key in seen:
            continue
        seen.add(key)
        out.append(o)
    return out


def parse_bboxes(raw: str) -> list[dict]:
    """Extract list of dicts {box_2d:[y0,x0,y1,x1], label:str} from model output.

    Robust to (a) repetition degeneration (dedup exact-duplicate boxes) and (b)
    truncated/unclosed JSON lists from hitting max_new_tokens (salvage the complete
    leading objects via regex). These make the box-IoU eval fair for multi-target
    pillars (skin/cervix) without rewarding the degeneration."""
    if not isinstance(raw, str):
        return []
    txt = _strip_reasoning(raw)
    # Prefer Final Answer section if present.
    m = _FINAL_ANSWER_RE.search(txt)
    if m:
        txt = txt[m.end():]
    # JSON fence first.
    fence = _JSON_FENCE_RE.search(txt)
    if fence:
        body = fence.group(1)
    else:
        head = _JSON_OBJ_HEAD_RE.search(txt)
        body = txt[head.start():] if head else txt
    # Try strict parse then progressively trim (handles trailing junk after the JSON).
    for end in range(len(body), 0, -1):
        try:
            obj = json.loads(body[:end])
            if isinstance(obj, list):
                return _dedup_boxes([o for o in obj if isinstance(o, dict) and "box_2d" in o])
            if isinstance(obj, dict) and "box_2d" in obj:
                return [obj]
        except Exception:
            continue
    # SALVAGE: strict parse failed (e.g. truncated unclosed list) -> recover every
    # complete {box_2d,...} object from the fragment, then dedup.
    salvaged = []
    for mo in _BOX_OBJ_RE.finditer(body):
        try:
            salvaged.append(json.loads(mo.group(0)))
        except Exception:
            continue
    return _dedup_boxes(salvaged)


def parse_u7_thyroid(raw: str) -> dict | None:
    if not isinstance(raw, str):
        return None
    txt = _strip_reasoning(raw)
    m = _FINAL_ANSWER_RE.search(txt)
    if m:
        txt = txt[m.end():]
    fence = _JSON_FENCE_RE.search(txt)
    if fence:
        body = fence.group(1)
    else:
        head = _JSON_OBJ_HEAD_RE.search(txt)
        body = txt[head.start():] if head else txt
    for end in range(len(body), 0, -1):
        try:
            obj = json.loads(body[:end])
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


# ─────────────────────────────────────────────────────────────────────────────
# IoU
# ─────────────────────────────────────────────────────────────────────────────


def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    if ax2 <= ax1 or ay2 <= ay1 or bx2 <= bx1 or by2 <= by1:
        return 0.0
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    a_area = (ax2 - ax1) * (ay2 - ay1)
    b_area = (bx2 - bx1) * (by2 - by1)
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


def best_iou(pred_boxes: list[list[float]], gold_boxes: list[list[float]]) -> float:
    """Best IoU over the bipartite assignment (simple greedy)."""
    if not pred_boxes or not gold_boxes:
        return 0.0
    best = 0.0
    for g in gold_boxes:
        for p in pred_boxes:
            i = iou_xyxy(p, g)
            if i > best:
                best = i
    return best


# Rows/pairs DROPPED by inference-level failures (OOM, inference error, pair crash).
# Nonzero => results are PARTIAL; the run exits nonzero so run_multitask's rc-gate
# marks the eval invalid instead of silently publishing partial metrics.
_RUN_DROPS = 0


def match_counts(pred_boxes: list[list[float]], gold_boxes: list[list[float]],
                 thr: float = 0.5) -> tuple[int, int, int]:
    """Greedy ONE-TO-ONE matching at an IoU threshold -> (tp, fp, fn).

    Unlike best_iou (which credits only the single best-matching box and never
    penalizes spurious/duplicate predictions), this is the standard detection
    accounting used when the model emits no per-box confidence: each pred and each
    gold is matched at most once (highest-IoU pairs first, IoU>=thr), so extra
    boxes become FP and missed gold become FN. It is the no-confidence degeneration
    of COCO-AP (single operating point) and mirrors DETR's bipartite matching at
    eval time. Micro-aggregating tp/fp/fn over rows -> precision/recall/F1@thr."""
    n_pred, n_gold = len(pred_boxes), len(gold_boxes)
    if n_pred == 0 or n_gold == 0:
        return 0, n_pred, n_gold
    pairs = []
    for pi, p in enumerate(pred_boxes):
        for gi, g in enumerate(gold_boxes):
            i = iou_xyxy(p, g)
            if i >= thr:
                pairs.append((i, pi, gi))
    pairs.sort(reverse=True)
    used_p: set[int] = set()
    used_g: set[int] = set()
    tp = 0
    for _i, pi, gi in pairs:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        tp += 1
    return tp, n_pred - tp, n_gold - tp


def _as_box_list(v) -> list[list[float]]:
    """Coerce a parsed JSON value into well-formed [x1,y1,x2,y2] float boxes,
    dropping anything malformed. Guards _prf_from_df against a column value that
    is valid JSON but the wrong shape/type (a dict, a scalar, ragged lists) —
    which would otherwise crash or miscount inside match_counts."""
    if not isinstance(v, list):
        return []
    out = []
    for b in v:
        if isinstance(b, (list, tuple)) and len(b) == 4:
            try:
                out.append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])
            except (TypeError, ValueError):
                continue
    return out


def _prf_from_df(df, thr: float = 0.5) -> dict:
    """Micro precision/recall/F1 over greedy one-to-one box matches at IoU>=thr.

    Reads the stored pred/gold box JSON columns, so it is purely ADDITIVE to
    mIoU/AP (those are untouched and stay comparable to prior runs) and can be
    recomputed on existing CSVs. This is the metric that penalizes the
    'spam many boxes, keep the one that hits' degeneration. `bad_json` counts rows
    whose stored box column failed to parse (should be 0 for harness-written CSVs;
    nonzero = a corrupted/foreign CSV worth investigating, not silently trusting)."""
    TP = FP = FN = 0
    bad_json = 0
    for _, r in df.iterrows():
        try:
            pred = _as_box_list(json.loads(r.get("pred_box_xyxy_norm") or "[]"))
            gold = _as_box_list(json.loads(r.get("gold_box_xyxy_norm") or "[]"))
        except Exception:
            bad_json += 1
            continue
        tp, fp, fn = match_counts(pred, gold, thr)
        TP += tp
        FP += fp
        FN += fn
    prec = TP / (TP + FP) if (TP + FP) else 0.0
    rec = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    out = {"precision@0.5": round(prec, 4), "recall@0.5": round(rec, 4),
           "f1@0.5": round(f1, 4), "tp": int(TP), "fp": int(FP), "fn": int(FN)}
    if bad_json:
        out["bad_json_rows"] = int(bad_json)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Inference helper (free-form generation)
# ─────────────────────────────────────────────────────────────────────────────


def _has_chat_template(processor) -> bool:
    """medgemma's processor ships a chat template; gemma-3n's does NOT. Used to pick
    the input-building path below."""
    return bool(getattr(processor, "chat_template", None)
                or getattr(getattr(processor, "tokenizer", None), "chat_template", None))


def _g3n_native_inputs(processor, model, image, prompt_text):
    """Build model inputs for a processor WITHOUT a chat template (gemma-3n). gemma-3n's
    multimodal image expansion is driven by a DIRECT processor(text, images) call with the
    image placeholder token prepended — NOT apply_chat_template, which raises ("processor
    does not have a chat template") and, if a template were borrowed, mismatches image
    tokens. Mirrors the proven train + closed-eval native path
    (strategy_a._iter_batches / _LogProbAdapter.generate) so the eval prompt is
    byte-identical to training: _inject_image_token + images=[[img]] (nested) + bf16 cast."""
    from cancerflex.legacy_unreleased.strategy_a import (  # noqa: E501
        _inject_image_token, _cast_inputs_for_model)
    text = _inject_image_token(processor, prompt_text)
    images = [[image]] if image is not None else None
    raw = dict(processor(text=[text], images=images, return_tensors="pt"))
    return _cast_inputs_for_model(raw, model)


@torch.no_grad()
def generate_text(model, processor, image, prompt_text, device, max_new_tokens=1000):
    """Single-row generation. max_new_tokens=1000 matches Google's CXR
    localization notebook — enough headroom for <unused94>thought</unused95>
    optional prose plus a JSON 'Final Answer:' block."""
    if _has_chat_template(processor):
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text":  prompt_text},
        ]}]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_tensors="pt", return_dict=True,
        ).to(device)
    else:
        inputs = _g3n_native_inputs(processor, model, image, prompt_text)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                  do_sample=False, repetition_penalty=1.08)
    return processor.decode(out_ids[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=False)


@torch.no_grad()
def generate_text_batch(model, processor, batch, device, max_new_tokens=1000):
    """Batched generation. batch: list of (image, prompt_text). Returns list
    of decoded strings in the same order. Falls back to per-row gen if the
    processor doesn't accept a list of message lists."""
    if not batch:
        return []
    if not _has_chat_template(processor):
        # gemma-3n: no chat template + batched cross-image generation is unverified and
        # known to diverge (Gemma3 variable pan-and-scan crops). Per-row is the only
        # verified-correct path — mirrors the closed scorer's per-record rule.
        return [generate_text(model, processor, img, pt, device, max_new_tokens)
                for img, pt in batch]
    if len(batch) == 1:
        image, pt = batch[0]
        return [generate_text(model, processor, image, pt, device, max_new_tokens)]
    all_messages = []
    for image, pt in batch:
        all_messages.append([{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text":  pt},
        ]}])
    try:
        inputs = processor.apply_chat_template(
            all_messages, add_generation_prompt=True, tokenize=True,
            return_tensors="pt", return_dict=True, padding=True,
        ).to(device)
    except Exception:
        # Processor doesn't accept list-of-messages — fall back to per-row.
        return [generate_text(model, processor, img, pt, device, max_new_tokens)
                for img, pt in batch]
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                  do_sample=False, repetition_penalty=1.08)
    # Generation appends to the (left-)padded prompt, so the generated tokens start at the
    # PADDED prompt width for every row — NOT attention_mask.sum() (real-token count), which
    # with left padding would re-include the prompt tail (the prompt literally contains the
    # "Final Answer:" instruction → parser could grab it). Slice at in_ids.shape[1], same as
    # the single-row path. (Latent today: fixed image-token count ⇒ uniform length ⇒ no
    # padding; defensive fix per codex review.)
    prompt_width = inputs["input_ids"].shape[1]
    return [processor.decode(out_ids[i][prompt_width:], skip_special_tokens=False)
            for i in range(len(batch))]


# ─────────────────────────────────────────────────────────────────────────────
# Per-pair runner
# ─────────────────────────────────────────────────────────────────────────────


def _csv_path(pillar: str, task: str, shard: tuple[int, int] | None = None) -> Path:
    """Per-shard CSV path. Race-free: each shard writes its own file; aggregator
    concats all shards' files at end. If shard is None, single canonical path
    (legacy / single-process invocation)."""
    if shard is None:
        return OUT_DIR / f"{pillar}_{task}_opengen.csv"
    return OUT_DIR / f"{pillar}_{task}_opengen_s{shard[0]}of{shard[1]}.csv"


def _all_shard_csvs(pillar: str, task: str) -> list[Path]:
    """All per-shard CSVs for a (pillar, task) — used at aggregation time."""
    base = OUT_DIR
    out = []
    canonical = base / f"{pillar}_{task}_opengen.csv"
    if canonical.exists():
        out.append(canonical)
    out.extend(sorted(base.glob(f"{pillar}_{task}_opengen_s*of*.csv")))
    return out


def _completed_keys(csv_path: Path) -> set[str]:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()
    try:
        cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
        failure_col = ["failure_reason"] if "failure_reason" in cols else []
        if "source_jsonl" in cols and "record_id" in cols:
            d = pd.read_csv(
                csv_path, usecols=["source_jsonl", "record_id", *failure_col],
                low_memory=False)
            if failure_col:
                # Explicit failure rows preserve source coverage, but must remain
                # retryable: a resumed shard should replace them with real scores.
                d = d[d["failure_reason"].fillna("").astype(str).str.strip() == ""]
            return {
                f"{src}::{rid}"
                for src, rid in zip(d["source_jsonl"].fillna("").astype(str), d["record_id"].fillna("").astype(str))
                if rid
            }
        key_col = "record_id" if "record_id" in cols else "image_path"
        d = pd.read_csv(csv_path, usecols=[key_col, *failure_col], low_memory=False)
        if failure_col:
            d = d[d["failure_reason"].fillna("").astype(str).str.strip() == ""]
        return set(d[key_col].dropna().astype(str).tolist())
    except Exception:
        return set()


# ─────────────────────────────────────────────────────────────────────────────
# OPT-IN multi-view U5 inference (EVAL_U5_VIEWS). Diagnostic only; unset = identity.
# ─────────────────────────────────────────────────────────────────────────────
# WHY (2026-09-23, weak-cell audit RESULTS section 22): HMCHH cervical cells are ~66 px and
# iToBoS skin lesions ~27 px after the whole-image 896 resize, i.e. about one pooled visual
# token. EVAL_U5_VIEWS asks the SAME U5 prompt on sub-windows cut from the ORIGINAL-resolution
# image, maps each predicted box back to full-image normalised coordinates, and scores the union
# against the UNCHANGED gold with the unchanged best_iou. Values:
#   tile2x2      four non-overlapping half-width x half-height tiles.
#   oracle_crop  ONE half-size window centred on the largest GOLD box (reads the gold: an upper
#                bound on what a 2x zoom can buy, never a deployable method).
# The views, per-view raw outputs and per-view boxes are written to an extra `views_json` column.
U5_VIEWS_ENV = "EVAL_U5_VIEWS"


def _u5_views(mode: str, gold_boxes_norm) -> list[list[float]]:
    """Return view windows as normalised [x0, y0, x1, y1] on the full image."""
    if mode == "tile2x2":
        return [[0.0, 0.0, 0.5, 0.5], [0.5, 0.0, 1.0, 0.5],
                [0.0, 0.5, 0.5, 1.0], [0.5, 0.5, 1.0, 1.0]]
    if mode == "oracle_crop":
        g = max(gold_boxes_norm, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        cx, cy = (g[0] + g[2]) / 2, (g[1] + g[3]) / 2
        out = []
        for c, lo, hi in ((cx, g[0], g[2]), (cy, g[1], g[3])):
            half = max(0.25, 0.6 * (hi - lo))          # half-size window, grown if the box is larger
            half = min(half, 0.5)
            a = min(max(c - half, 0.0), 1.0 - 2 * half)  # clamp inside the image, keep the size
            out.append((a, a + 2 * half))
        return [[out[0][0], out[1][0], out[0][1], out[1][1]]]
    raise ValueError(f"{U5_VIEWS_ENV}={mode!r}: expected tile2x2 or oracle_crop")


def _view_pixel_box(view: list[float], size: tuple[int, int]) -> tuple[int, int, int, int]:
    W, H = size
    return (int(round(view[0] * W)), int(round(view[1] * H)),
            int(round(view[2] * W)), int(round(view[3] * H)))


def _view_to_full(box: list[float], view_px: tuple[int, int, int, int],
                  size: tuple[int, int]) -> list[float]:
    """Map a box normalised to the CROP back to full-image normalised coordinates, using the
    exact integer pixel window that was cropped (so rounding cannot shift the mapping)."""
    W, H = size
    x0, y0, x1, y1 = view_px
    cw, ch = x1 - x0, y1 - y0
    return [(x0 + box[0] * cw) / W, (y0 + box[1] * ch) / H,
            (x0 + box[2] * cw) / W, (y0 + box[3] * ch) / H]


def _strip_pad(raw: str) -> str:
    return raw.replace("<pad>", "") if isinstance(raw, str) else raw


def run_pair(model, processor, device, pillar: str, task: str,
             limit: int | None = None, shard: tuple[int, int] | None = None,
             batch_size: int = 4, max_new_tokens: int = 1536) -> None:
    if task in ("U5", "U6", "U7") and not _active_jsonl_records(pillar, task):
        log_event("opengen_pair_skipped", pillar=pillar, task=task,
                  reason="no active v1_final shard for this pillar/task")
        return

    csv_path = _csv_path(pillar, task, shard=shard)
    completed = _completed_keys(csv_path)
    log_event("opengen_pair_start", pillar=pillar, task=task, csv=str(csv_path),
              shard=str(shard) if shard else None,
              already_done=len(completed))

    if task == "U7":
        fieldnames = ["record_id", "source_jsonl", "image_path", "patient_id", "dataset_id", "split", "subsplit",
                      "ood_axis", "raw_output", "predicted_json",
                      "gold_imaging_findings_json", "fields_correct", "parse_ok",
                      "failure_reason", "elapsed_ms"]
    else:
        fieldnames = ["record_id", "source_jsonl", "image_path", "patient_id", "dataset_id", "split", "subsplit",
                      "ood_axis", "raw_output", "n_pred_boxes",
                      "gold_box_xyxy_norm", "pred_box_xyxy_norm", "best_iou",
                      "parse_ok", "failure_reason", "elapsed_ms"]
    views_mode = os.environ.get(U5_VIEWS_ENV) if task == "U5" else None
    if views_mode:
        fieldnames = fieldnames + ["views_json"]
        log_event("opengen_u5_views_on", pillar=pillar, task=task, mode=views_mode)

    # Resume may target a CSV created before explicit failure rows existed.
    # Upgrade that per-shard file before appending; otherwise the extra field
    # would make new rows wider than the historical header.
    if csv_path.exists() and csv_path.stat().st_size > 0:
        old_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
        if "failure_reason" not in old_cols:
            old = pd.read_csv(csv_path, low_memory=False)
            old["failure_reason"] = ""
            old.reindex(columns=fieldnames).to_csv(csv_path, index=False)

    is_new = not csv_path.exists() or csv_path.stat().st_size == 0
    f = open(csv_path, "a", newline="", buffering=1)
    w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    if is_new:
        w.writeheader()

    yielded = 0
    parsed_ok_n = 0
    iou_sum = 0.0
    iou_thresh_hits = 0
    t_run = time.time()
    last_log = time.time()

    # Buffer for batched inference. Each entry: {image, gold_boxes_norm,
    # gold_findings, meta, queued_t}.
    buffer: list[dict] = []

    def _failure_row(b: dict, reason: str, elapsed_ms: float,
                     raw_output: str = "") -> None:
        """Persist one unscored source row so failures are counted and auditable."""
        nonlocal yielded, parsed_ok_n, iou_sum, iou_thresh_hits
        global _RUN_DROPS
        meta_b = b["meta"]
        common = {
            "record_id": meta_b.get("record_id"),
            "source_jsonl": meta_b.get("source_jsonl"),
            "image_path": meta_b.get("image_path", ""),
            "patient_id": meta_b.get("patient_id"),
            "dataset_id": meta_b.get("dataset_id"),
            "split": meta_b.get("split"),
            "subsplit": meta_b.get("subsplit"),
            "ood_axis": meta_b.get("ood_axis"),
            # Preserve a model response when parsing/scoring fails; inference
            # failures have no response and deliberately write an empty string.
            "raw_output": raw_output[:6000],
            "parse_ok": False,
            "failure_reason": reason,
            "elapsed_ms": round(elapsed_ms, 1),
        }
        if task in ("U5", "U6"):
            w.writerow({
                **common,
                "n_pred_boxes": 0,
                "gold_box_xyxy_norm": json.dumps(b.get("gold_boxes_norm") or []),
                "pred_box_xyxy_norm": "[]",
                "best_iou": 0.0,
            })
        else:
            w.writerow({
                **common,
                "predicted_json": "",
                "gold_imaging_findings_json": json.dumps(b.get("gold_findings")),
                "fields_correct": "{}",
            })
        yielded += 1
        _RUN_DROPS += 1
        log_event("opengen_failure_row", pillar=pillar, task=task,
                  record_id=meta_b.get("record_id"),
                  image_path=meta_b.get("image_path"), reason=reason)

    def _score_row(b: dict, raw: str, per_row_ms: float) -> None:
        """Parse, score, and persist one model response."""
        nonlocal yielded, parsed_ok_n, iou_sum, iou_thresh_hits
        meta_b = b["meta"]
        try:
            if task in ("U5", "U6"):
                boxes = parse_bboxes(raw)
                pred_norm = []
                for o in boxes:
                    bb = _to_pred_xyxy_norm(o.get("box_2d", []))
                    if bb is not None:
                        pred_norm.append(bb)
                gold_boxes_norm = b["gold_boxes_norm"]
                best = best_iou(pred_norm, gold_boxes_norm) if (pred_norm and gold_boxes_norm) else 0.0
                parse_ok = bool(boxes)
                w.writerow({
                    "record_id": meta_b.get("record_id"),
                    "source_jsonl": meta_b.get("source_jsonl"),
                    "image_path": meta_b["image_path"],
                    "patient_id": meta_b.get("patient_id"),
                    "dataset_id": meta_b.get("dataset_id"),
                    "split": meta_b.get("split"),
                    "subsplit": meta_b.get("subsplit"),
                    "ood_axis": meta_b.get("ood_axis"),
                    "raw_output": raw[:6000],
                    "n_pred_boxes": len(pred_norm),
                    "gold_box_xyxy_norm": json.dumps(gold_boxes_norm),
                    "pred_box_xyxy_norm": json.dumps(pred_norm),
                    "best_iou": round(best, 4),
                    "parse_ok": parse_ok,
                    "failure_reason": "",
                    "elapsed_ms": round(per_row_ms, 1),
                })
                if parse_ok:
                    parsed_ok_n += 1
                iou_sum += best
                if best >= 0.5:
                    iou_thresh_hits += 1
            else:  # U7
                gold_findings = b["gold_findings"]
                pred = parse_u7_thyroid(raw)
                fields_correct = compare_u7_fields(pred, gold_findings)
                w.writerow({
                    "record_id": meta_b.get("record_id"),
                    "source_jsonl": meta_b.get("source_jsonl"),
                    "image_path": meta_b["image_path"],
                    "patient_id": meta_b.get("patient_id"),
                    "dataset_id": meta_b.get("dataset_id"),
                    "split": meta_b.get("split"),
                    "subsplit": meta_b.get("subsplit"),
                    "ood_axis": meta_b.get("ood_axis"),
                    "raw_output": raw[:6000],
                    "predicted_json": json.dumps(pred) if pred else "",
                    "gold_imaging_findings_json": json.dumps(gold_findings),
                    "fields_correct": json.dumps(fields_correct),
                    "parse_ok": pred is not None,
                    "failure_reason": "",
                    "elapsed_ms": round(per_row_ms, 1),
                })
                if pred is not None:
                    parsed_ok_n += 1
            yielded += 1
        except Exception as exc:
            log_event("opengen_row_processing_error", pillar=pillar, task=task,
                      record_id=meta_b.get("record_id"),
                      image_path=meta_b.get("image_path"), error=str(exc),
                      traceback=traceback.format_exc(limit=3))
            _failure_row(b, "row_processing_error", per_row_ms, raw_output=raw)

    def _infer_entries(entries: list[dict]) -> None:
        """Infer all entries, recursively bisecting failures down to one row."""
        if not entries:
            return
        batch_inputs = [(b["image"], b["prompt_text"]) for b in entries]
        t0 = time.time()
        try:
            raws = generate_text_batch(model, processor, batch_inputs, device,
                                        max_new_tokens=max_new_tokens)
            if len(raws) != len(entries):
                raise RuntimeError(
                    f"generate_text_batch returned {len(raws)} rows for {len(entries)} inputs")
        except torch.cuda.OutOfMemoryError as exc:
            log_event("oom", pillar=pillar, task=task,
                      batch_size=len(entries), error=str(exc))
            torch.cuda.empty_cache()
            if len(entries) == 1:
                _failure_row(entries[0], "oom_unrecoverable",
                             (time.time() - t0) * 1000.0)
                return
            # Greedy decoding is deterministic, but batch-vs-single numerical
            # equivalence must be verified per model/runtime. Regardless, retrying
            # both ordered halves is lossless and strictly safer than dropping rows.
            mid = len(entries) // 2
            _infer_entries(entries[:mid])
            _infer_entries(entries[mid:])
            return
        except Exception as exc:
            log_event("inference_error", pillar=pillar, task=task,
                      batch_size=len(entries), error=str(exc),
                      traceback=traceback.format_exc(limit=3))
            if len(entries) == 1:
                _failure_row(entries[0], "inference_error",
                             (time.time() - t0) * 1000.0)
                return
            # Isolate a bad record instead of turning one exception into a
            # silent whole-batch loss.
            mid = len(entries) // 2
            _infer_entries(entries[:mid])
            _infer_entries(entries[mid:])
            return
        per_row_ms = (time.time() - t0) * 1000.0 / max(1, len(entries))
        for b, raw in zip(entries, raws):
            _score_row(b, raw, per_row_ms)

    def _process_batch():
        """Drain `buffer`, preserving order through recursive inference retries."""
        if not buffer:
            return
        entries = list(buffer)
        buffer.clear()
        _infer_entries(entries)

    vbuffer: list[dict] = []   # EVAL_U5_VIEWS only: one entry per source row (its views inside)

    def _gen_views_safe(items: list[tuple]) -> list:
        """generate_text_batch with OOM/error bisection; a failed single view -> None."""
        if not items:
            return []
        try:
            raws = generate_text_batch(model, processor, items, device,
                                       max_new_tokens=max_new_tokens)
            if len(raws) != len(items):
                raise RuntimeError("generate_text_batch length mismatch")
            return raws
        except Exception as exc:
            log_event("views_inference_error", pillar=pillar, task=task,
                      batch_size=len(items), error=str(exc)[:300])
            torch.cuda.empty_cache()
            if len(items) == 1:
                return [None]
            mid = len(items) // 2
            return _gen_views_safe(items[:mid]) + _gen_views_safe(items[mid:])

    def _process_views():
        nonlocal yielded, parsed_ok_n, iou_sum, iou_thresh_hits
        global _RUN_DROPS
        if not vbuffer:
            return
        groups = list(vbuffer)
        vbuffer.clear()
        flat = [(v["image"], g["prompt_text"]) for g in groups for v in g["views"]]
        t0 = time.time()
        raws = _gen_views_safe(flat)
        per_row_ms = (time.time() - t0) * 1000.0 / max(1, len(groups))
        k = 0
        for g in groups:
            meta_b = g["meta"]
            pred_full, vrecs, failed = [], [], False
            for v in g["views"]:
                raw = raws[k]
                k += 1
                if raw is None:
                    failed = True
                    vrecs.append({"view": v["view"], "view_px": v["view_px"], "raw": None})
                    continue
                boxes = parse_bboxes(raw)
                vb = []
                for o in boxes:
                    bb = _to_pred_xyxy_norm(o.get("box_2d", []))
                    if bb is not None:
                        vb.append(bb)
                        pred_full.append(_view_to_full(bb, v["view_px"], g["size"]))
                vrecs.append({"view": v["view"], "view_px": v["view_px"],
                              "raw": _strip_pad(raw), "n_boxes": len(vb), "boxes_view_norm": vb})
            if failed:
                _RUN_DROPS += 1
            gold_boxes_norm = g["gold_boxes_norm"]
            best = best_iou(pred_full, gold_boxes_norm) if (pred_full and gold_boxes_norm) else 0.0
            parse_ok = bool(pred_full)
            w.writerow({
                "record_id": meta_b.get("record_id"),
                "source_jsonl": meta_b.get("source_jsonl"),
                "image_path": meta_b["image_path"],
                "patient_id": meta_b.get("patient_id"),
                "dataset_id": meta_b.get("dataset_id"),
                "split": meta_b.get("split"),
                "subsplit": meta_b.get("subsplit"),
                "ood_axis": meta_b.get("ood_axis"),
                "raw_output": "\n|||\n".join(str(r["raw"]) for r in vrecs)[:6000],
                "n_pred_boxes": len(pred_full),
                "gold_box_xyxy_norm": json.dumps(gold_boxes_norm),
                "pred_box_xyxy_norm": json.dumps(pred_full),
                "best_iou": round(best, 4),
                "parse_ok": parse_ok,
                "failure_reason": "view_inference_error" if failed else "",
                "elapsed_ms": round(per_row_ms, 1),
                "views_json": json.dumps({"mode": views_mode, "size": list(g["size"]),
                                          "views": vrecs}),
            })
            yielded += 1
            if parse_ok:
                parsed_ok_n += 1
            iou_sum += best
            if best >= 0.5:
                iou_thresh_hits += 1

    for row, image, row_prompt, gold_label, meta in iter_active_opengen_rows(
        pillar, task, shard=shard, completed=completed
    ):
        if limit is not None and yielded + len(buffer) >= limit:
            break
        if image is None:
            _failure_row({"meta": meta, "gold_boxes_norm": None,
                          "gold_findings": None}, "image_load_failure", 0.0)
            continue
        gold_boxes_norm = None
        gold_findings = None
        if task in ("U5", "U6"):
            gold_boxes_norm = _decode_gold_objects_as_boxes(gold_label)
            if gold_boxes_norm is None and task == "U5":
                gold_boxes_norm = _decode_gold_bbox(pillar, gold_label, image.size)
            if gold_boxes_norm is None and task == "U6":
                gb = _decode_gold_mask_as_bbox(gold_label)
                gold_boxes_norm = [gb] if gb is not None else None
            if gold_boxes_norm is None:
                # Tripwire: a row whose gold can't be decoded is dropped from the
                # denominator. Silent on the de-leaked corpus (all U5/U6 rows are
                # box-able), but a schema mismatch would otherwise inflate metrics
                # invisibly — log it so an official run can be checked.
                log_event("gold_decode_skipped", pillar=pillar, task=task,
                          record_id=meta.get("record_id"),
                          image_path=meta.get("image_path"))
                _failure_row({"meta": meta, "gold_boxes_norm": None,
                              "gold_findings": None}, "gold_decode_failure", 0.0)
                continue
        elif task == "U7":
            if isinstance(gold_label, str):
                try:
                    gold_findings = json.loads(gold_label)
                except Exception:
                    gold_findings = None
            elif isinstance(gold_label, dict):
                gold_findings = gold_label
            if not isinstance(gold_findings, dict) or not gold_findings:
                _failure_row({"meta": meta, "gold_boxes_norm": None,
                              "gold_findings": gold_findings},
                             "gold_decode_failure", 0.0)
                continue

        if views_mode:
            # Views are cut from the ORIGINAL-resolution decode (not the 896 cache).
            orig = canonical_image.load_image_rgb(meta["image_path"], root=ROOT,
                                                  local_root=os.environ.get("LOCAL_IMAGE_ROOT"))
            vlist = []
            for view in _u5_views(views_mode, gold_boxes_norm):
                vpx = _view_pixel_box(view, orig.size)
                vlist.append({"view": view, "view_px": list(vpx), "image": orig.crop(vpx)})
            vbuffer.append({"meta": meta, "prompt_text": row_prompt, "size": orig.size,
                            "gold_boxes_norm": gold_boxes_norm, "views": vlist})
            if sum(len(g["views"]) for g in vbuffer) >= batch_size:
                _process_views()
            continue
        buffer.append({
            "image": image, "prompt_text": row_prompt, "meta": meta,
            "gold_boxes_norm": gold_boxes_norm, "gold_findings": gold_findings,
        })
        if len(buffer) >= batch_size:
            _process_batch()
            if time.time() - last_log > 60:
                f.flush()
                rate = yielded / max(1.0, time.time() - t_run)
                extras = ({"miou": round(iou_sum / max(1, yielded), 3),
                           "hit_at_0.5": iou_thresh_hits}
                          if task != "U7" else {})
                log_event("opengen_progress", pillar=pillar, task=task, done=yielded,
                          parse_ok=parsed_ok_n, rate_per_sec=round(rate, 2), **extras)
                last_log = time.time()

    # Drain final partial batch
    _process_batch()
    if views_mode:
        _process_views()
    f.close()
    log_event("opengen_pair_done", pillar=pillar, task=task, processed=yielded,
              parse_ok=parsed_ok_n,
              parse_failure_rate=round(1 - parsed_ok_n / max(1, yielded), 3),
              elapsed_sec=round(time.time() - t_run, 1))

    # Aggregate. Read CSV via tempfile snapshot to avoid pandas-races against
    # any concurrent appenders (own driver, or sibling shards if any).
    # Concat all per-shard files for this (pillar, task).
    # Shards own raw rows only. Canonical metrics are written by the driver's
    # completeness-gated --aggregate_only pass; never publish a partial shard.
    if yielded > 0 and shard is None and _RUN_DROPS == 0:
        _write_aggregated_metric(pillar, task)


def _safe_read_csv(csv_path: Path) -> pd.DataFrame:
    """Snapshot the CSV to a tempfile, then read — sidesteps the
    'TypeError: raise: exception class must be a subclass of BaseException'
    pandas race we saw when reading a file mid-append."""
    import shutil, tempfile
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        shutil.copy(csv_path, tmp_path)
        return pd.read_csv(tmp_path, low_memory=False)
    finally:
        try: os.unlink(tmp_path)
        except Exception: pass


def _aggregate_dataframe(pillar: str, task: str) -> tuple[pd.DataFrame | None, list[Path]]:
    """Concat all available CSVs for (pillar, task) and deduplicate rows."""
    paths = _all_shard_csvs(pillar, task)
    if not paths:
        return None, []
    frames = []
    for p in paths:
        try:
            frames.append(_safe_read_csv(p))
        except Exception as exc:
            log_event("opengen_aggregate_csv_read_error", pillar=pillar, task=task,
                      path=str(p), error=str(exc))
    if not frames:
        return None, paths
    df = pd.concat(frames, ignore_index=True)
    if {"source_jsonl", "record_id"}.issubset(df.columns):
        df = df.drop_duplicates(subset=["source_jsonl", "record_id"], keep="last")
    else:
        dedup_key = "record_id" if "record_id" in df.columns else "image_path"
        df = df.drop_duplicates(subset=[dedup_key], keep="last")
    return df, paths


def _write_canonical_csv(pillar: str, task: str) -> None:
    """Materialize the canonical CSV after sharded open-generation runs."""
    df, paths = _aggregate_dataframe(pillar, task)
    if df is None:
        log_event("opengen_aggregate_only_no_csv", pillar=pillar, task=task)
        return
    canonical = _csv_path(pillar, task, shard=None)
    df.to_csv(canonical, index=False)
    log_event("opengen_aggregate_only_merged", pillar=pillar, task=task,
              canonical=str(canonical), n_shards=len(paths), n_rows=len(df))


def _write_aggregated_metric(pillar: str, task: str) -> None:
    """Concat all per-shard CSVs for (pillar, task), dedup rows,
    compute metrics, write metrics_<pillar>_<task>_opengen.json."""
    df, paths = _aggregate_dataframe(pillar, task)
    if df is None:
        return
    metric_path = OUT_DIR / f"metrics_{pillar}_{task}_opengen.json"
    if "failure_reason" in df.columns:
        failed = df["failure_reason"].fillna("").astype(str).str.strip().ne("")
        if failed.any():
            # Raw rows remain canonical/auditable, but an unrecovered failure
            # can never produce an official-looking metric JSON.
            if metric_path.exists():
                metric_path.unlink()
            log_event("opengen_metrics_withheld_incomplete", pillar=pillar, task=task,
                      n_failure_rows=int(failed.sum()))
            return
    if task in ("U5", "U6"):
        # best_iou-based mIoU/AP are LENIENT (best single match, no FP penalty);
        # kept as a localization-quality signal + comparable to prior runs.
        # f1@0.5 (one-to-one match, micro P/R/F1) is the STRICT detection metric
        # that penalizes spurious/duplicate boxes. See match_counts / _prf_from_df.
        out = {"pillar": pillar, "task": task, "n_unique": int(len(df)),
                "n_shards": len(paths),
                "parse_failure_rate": float((~df["parse_ok"].astype(bool)).mean()),
                "mIoU": float(df["best_iou"].mean()),
                "AP@0.5": float((df["best_iou"] >= 0.5).mean()),
                "AP@0.3": float((df["best_iou"] >= 0.3).mean()),
                **_prf_from_df(df)}
        for split in ("test_id", "test_ood"):
            sub = df[df["split"] == split]
            if len(sub):
                out[split] = {"n": int(len(sub)),
                              "mIoU": float(sub["best_iou"].mean()),
                              "AP@0.5": float((sub["best_iou"] >= 0.5).mean()),
                              "parse_failure_rate": float((~sub["parse_ok"].astype(bool)).mean()),
                              **_prf_from_df(sub)}
    else:  # U7
        by_field: dict[str, list[bool]] = {}
        for _, r in df.iterrows():
            try:
                fc = json.loads(r.get("fields_correct") or "{}")
            except Exception:
                fc = {}
            if not isinstance(fc, dict):
                continue
            for k, v in fc.items():
                by_field.setdefault(str(k), []).append(bool(v))
        per_field = {
            k: {"n": len(v), "accuracy": float(np.mean(v))}
            for k, v in sorted(by_field.items())
            if v
        }
        macro = ([v["accuracy"] for v in per_field.values()])
        out = {"pillar": pillar, "task": task, "n_unique": int(len(df)),
                "n_shards": len(paths),
                "parse_failure_rate": float((~df["parse_ok"].astype(bool)).mean()),
                "macro_field_accuracy": float(np.mean(macro)) if macro else None,
                "per_field": per_field}
    metric_path.write_text(json.dumps(out, indent=2, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="U5,U7")
    ap.add_argument("--pillars", default="skin,breast,cervix,oral,thyroid")
    ap.add_argument("--limit_per_pair", type=int, default=None)
    ap.add_argument("--shard", default=None, help="i/N — process only rows where row_idx%N==i")
    ap.add_argument("--aggregate_only", action="store_true",
                    help="skip model loading; merge sharded open-gen CSVs into canonical CSVs and metrics")
    ap.add_argument("--batch_size", type=int, default=12)
    ap.add_argument("--max_new_tokens", type=int, default=1536,
                    help="generation budget per row (default 1536; MedGemma's "
                         "<unused94>thought... prefix eats 300-500 before JSON, and "
                         "multi-target box lists are long — 1024 truncated skin/cervix)")
    args = ap.parse_args()
    shard = None
    if args.shard:
        i, N = args.shard.split("/")
        shard = (int(i), int(N))

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    pillars = [p.strip() for p in args.pillars.split(",") if p.strip()]
    # U6 is supported as box-from-mask (box-IoU, same path as U5) when the shard
    # source carries box rows — set EVAL_OPENGEN_SHARDS=v1_final_deleaked/shards.
    # (Old active v1_final stores U6 as a pixel mask; rows without a box gold are
    # skipped during scoring.)

    # Load-once / multi-width (PERF §18, mirrors run_closed): a comma-list
    # EVAL_MATFORMER_WIDTH loads+wraps the MatFormer once and slices each width in
    # place, routing each width's raw CSVs to its own EVAL_OUT_DIR. A scalar width
    # (or none) is unchanged. Activate the first width's dir now so the pre-load
    # run_start already lands there; the rest are activated inside the run loop.
    requested_widths = _parse_matformer_widths()
    if len(requested_widths) > 1:
        if args.aggregate_only:
            raise ValueError(
                "multi-width --aggregate_only is unsupported; aggregate each width's "
                "persisted EVAL_OUT_DIR separately"
            )
        width_out_dirs = _multi_width_out_dirs(requested_widths)
        _activate_out_dir(width_out_dirs[requested_widths[0]])
    else:
        width_out_dirs = {requested_widths[0]: OUT_DIR} if requested_widths else {}

    if args.aggregate_only:
        log_event("opengen_aggregate_only_start", tasks=tasks, pillars=pillars)
        for task in tasks:
            for pillar in pillars:
                _write_canonical_csv(pillar, task)
                _write_aggregated_metric(pillar, task)
        log_event("opengen_aggregate_only_done")
        return

    from transformers import AutoProcessor, AutoModelForImageTextToText
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_event("opengen_run_start", tasks=args.tasks, pillars=args.pillars,
              limit=args.limit_per_pair, pid=os.getpid(),
              cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"))
    # EVAL_PROCESSOR overrides the processor source — REQUIRED for non-medgemma archs
    # (e.g. gemma-3n-E4B-it) whose tokenizer/chat-template/image-token handling differs
    # from medgemma. Defaults to MODEL_PATH (medgemma SFT/matformer share its processor).
    processor = AutoProcessor.from_pretrained(os.environ.get("EVAL_PROCESSOR", MODEL_PATH))
    try:
        from flash_attn import flash_attn_func  # noqa: F401
        attn_impl = "flash_attention_2"
    except (ImportError, OSError):
        attn_impl = "eager"
    # EVAL_ATTN forces an attn implementation. gemma-3n's vision tower (TimmWrapperModel)
    # supports ONLY eager, but its text decoder supports sdpa — pass a per-submodule JSON dict
    # like {"text_config":"sdpa","vision_config":"eager","audio_config":"eager"} to speed up
    # the (decode-dominated) language model while keeping vision eager.
    _attn_env = os.environ.get("EVAL_ATTN")
    if _attn_env:
        attn_impl = json.loads(_attn_env) if _attn_env.strip().startswith("{") else _attn_env
    # EVAL_CKPT (a trained medgemma SFT/matformer final/) overrides the base model;
    # the processor (with chat template) still comes from the base MODEL_PATH.
    model_src = os.environ.get("EVAL_CKPT", MODEL_PATH)
    # _matformer_controller stays None for non-MatFormer paths; the width run-loop
    # below only slices when it is set (i.e. EVAL_MATFORMER_WIDTH was given).
    _matformer_controller = None
    if requested_widths:
        # ── Matformer-pure UNIFORM nested-width slice eval (mirrors run_closed.py:1160) ──
        # One global FFN width applied to ALL layers (controller.set) — NOT Mix'n'Match / no
        # per-layer mixing. See docs/methodology/matformer_width_eval.md.
        # LOAD-ONCE: wrap the model ONCE here; each requested width is an in-place
        # controller.set() slice applied in the run loop below (PERF §18). width==full
        # (10240) is bit-identical to the plain HF load (the identity check we verify
        # before trusting sub-width numbers).
        # NOTE: `sys` is module-level (top of file); do NOT re-import it here — a local
        # `import sys` would shadow it across all of main() and make the later
        # `sys.exit(1)` raise UnboundLocalError.
        factor = int(os.environ.get("EVAL_MATFORMER_FACTOR", "8"))
        model = AutoModelForImageTextToText.from_pretrained(
            model_src, dtype=torch.bfloat16, attn_implementation=attn_impl)
        PURE = str(ROOT / "preliminary" / "Med-Matformer-pure")
        if PURE not in sys.path:
            sys.path.insert(0, PURE)
        from cancerflex.matformer.model import wrap_medgemma_mlps
        from cancerflex.matformer.granularity import GranularityController
        text_cfg = getattr(model.config, "text_config", model.config)
        ckpt_dir = Path(model_src)
        gran_path = ckpt_dir / "matformer_granularities.json"
        if not gran_path.exists():
            gran_path = ckpt_dir.parent / "matformer_granularities.json"
        if gran_path.exists():
            controller = GranularityController(
                widths=[int(w) for w in json.loads(gran_path.read_text())["widths"]])
            gran_src = str(gran_path)
        else:
            controller = GranularityController.from_exponential_factor(
                full_width=text_cfg.intermediate_size, factor=factor)
            gran_src = f"recomputed(factor={factor})"
        if controller.widths[-1] != text_cfg.intermediate_size:
            raise ValueError(f"granularity full width {controller.widths[-1]} != "
                             f"intermediate_size {text_cfg.intermediate_size}")
        for _w in requested_widths:
            if _w not in controller.widths:
                raise ValueError(f"width {_w} not in {controller.widths} (from {gran_src})")
        n_wrapped = wrap_medgemma_mlps(model, controller)
        if n_wrapped == 0:
            raise RuntimeError("wrap_medgemma_mlps replaced 0 MLPs — arch mismatch")
        model = model.to(device).eval()
        _matformer_controller = controller
        # Keep the historical scalar `width` field for single-width callers (a scalar
        # EVAL_MATFORMER_WIDTH must log byte-for-byte as before); multi-width sets it None
        # and reports the list in `requested_widths` instead.
        log_event("opengen_model_loaded", model_src=model_src, arch="matformer_pure",
                  width=(requested_widths[0] if len(requested_widths) == 1 else None),
                  requested_widths=requested_widths, widths=list(controller.widths),
                  gran_src=gran_src, n_mlps_wrapped=n_wrapped)
    else:
        # gemma-3n distill students (train_v2/distill_dft) save a raw state_dict as
        # `final/model.pt`, NOT an HF dir — from_pretrained(model_src) below would fail
        # (no config.json) and would also build medgemma, not gemma-3n. Detect that case
        # and rebuild base gemma-3n + overlay the trained weights, mirroring the proven
        # run_closed.py iter_01 dense path (lines ~1285-1292). Processor already came from
        # EVAL_PROCESSOR above (gemma-3n's own tokenizer/chat-template).
        _arch = os.environ.get("EVAL_ARCH", "medgemma")
        _ckpt_dir = Path(model_src)
        _is_hf = (_ckpt_dir / "config.json").exists() or \
                 (_ckpt_dir / "model.safetensors").exists() or \
                 (_ckpt_dir / "model.safetensors.index.json").exists()
        if _arch == "gemma3n" and os.environ.get("EVAL_CKPT") and not _is_hf:
            from cancerflex.legacy_unreleased.strategy_a import load_gemma3n as _load_g3n
            from cancerflex.legacy_unreleased.common.checkpoint import load_checkpoint as _load_ckpt
            base = os.environ.get("EVAL_BASE", "google/gemma-3n-E4B")
            model, _ = _load_g3n(base, vision_freeze=True, apply_lora=False)  # keep EVAL_PROCESSOR processor
            _load_ckpt(model, model_src, strict=False)                       # overlay trained model.pt
            model = model.to(device).eval()
            log_event("opengen_model_loaded", model_src=model_src, arch="gemma3n", fmt="pt")
        else:
            # Real bnb 4-bit (NF4) quant-on-load — NO-OP unless EVAL_QUANT=nf4 (scripts/eval/bnb_quant.py).
            # Real int4 is full-width-only (plain load, this branch); it must NOT be combined with the
            # matformer width wrapper (the requested_widths branch above) — bnb can't slice packed weights.
            from bnb_quant import maybe_bnb_config_from_env, assert_real_4bit
            _qcfg = maybe_bnb_config_from_env()
            # device_map="auto" can silently skip legacy-key conversion for some weights on old
            # ckpts (matformer keys are legacy `language_model.model.*`, pre the `model.` wrapper) ->
            # unloaded MLPs -> mIoU collapse. EVAL_NO_DEVICE_MAP=1 loads then .to(device) instead
            # (matches the wrapped path). bnb 4-bit needs a device_map, so keep it when quantizing.
            _no_dm = os.environ.get("EVAL_NO_DEVICE_MAP") == "1" and _qcfg is None
            model = AutoModelForImageTextToText.from_pretrained(
                model_src, dtype=torch.bfloat16, attn_implementation=attn_impl,
                device_map=(None if _no_dm else "auto"), quantization_config=_qcfg,
            )
            if _no_dm:
                model = model.to(device)
            model.eval()
            # Per-layer Mix'n'Match hook — NO-OP unless EVAL_GEMMA3N_PERLAYER is set.
            # Installs heterogeneous FFN-width prefix-slicing on the native MatFormer
            # (E4B). See scripts/eval/gemma3n_mnm.py.
            _evdir = str(ROOT / "scripts" / "eval")
            if _evdir not in sys.path:
                sys.path.insert(0, _evdir)
            from gemma3n_mnm import apply_perlayer_from_env as _apply_mnm
            _mnm_ctrl = _apply_mnm(model)
            if _mnm_ctrl is not None:
                log_event("opengen_gemma3n_perlayer_mnm",
                          spec=os.environ.get("EVAL_GEMMA3N_PERLAYER"),
                          summary=_mnm_ctrl.summary())
            if _qcfg is not None:
                log_event("opengen_model_loaded", model_src=model_src, quant="nf4",
                          real_4bit=assert_real_4bit(model))
            else:
                log_event("opengen_model_loaded", model_src=model_src)

    # INT4 PTQ quantize-on-load hook (NO-OP unless EVAL_PTQ_BITS is set); see scripts/eval/ptq_hook.py
    from ptq_hook import maybe_quantize_from_env
    maybe_quantize_from_env(model)
    # Vision-tower input-resolution hook (NO-OP unless EVAL_VISION_RES is set); see
    # scripts/eval/vision_res_hook.py (P0 probe, docs/VISION_ENCODER_COMPRESSION_20261008.md).
    from vision_res_hook import maybe_set_vision_resolution_from_env
    maybe_set_vision_resolution_from_env(model, processor, log_event)

    global _RUN_DROPS
    # One ordinary run when no width is requested; otherwise one in-place controller
    # slice per requested MatFormer width, each routed to its own EVAL_OUT_DIR so
    # resume/dedup stays width-local (PERF §18). The pre-load opengen_run_start above
    # already covers the first width's dir; later widths re-emit it in their own dir.
    width_runs = requested_widths or [None]
    for run_index, width in enumerate(width_runs):
        if width is not None:
            if _matformer_controller is None:
                raise RuntimeError(
                    "EVAL_MATFORMER_WIDTH set but the MatFormer-pure controller was not built")
            if len(width_runs) > 1:
                _activate_out_dir(width_out_dirs[width])
            if run_index > 0 and torch.cuda.is_available():
                torch.cuda.synchronize()
            _matformer_controller.set(width)
            if run_index > 0:
                log_event("opengen_run_start", tasks=args.tasks, pillars=args.pillars,
                          limit=args.limit_per_pair, pid=os.getpid(),
                          cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"),
                          matformer_width=width)
        drops_before_width = _RUN_DROPS
        for task in tasks:
            for pillar in pillars:
                try:
                    run_pair(model, processor, device, pillar, task,
                             limit=args.limit_per_pair, shard=shard,
                             batch_size=args.batch_size,
                             max_new_tokens=args.max_new_tokens)
                except Exception as exc:
                    _RUN_DROPS += 1
                    log_event("opengen_pair_crash", pillar=pillar, task=task,
                              error=str(exc), traceback=traceback.format_exc(limit=5))

        # Per-width completion in the CURRENT (width-local) dir, mirroring run_closed:
        # multi-width writes matformer_width + width_run_drops so a per-dir completion
        # checker sees each width's own run_done and its own drop count; the scalar /
        # no-width path keeps the exact single aggregate event as before.
        if len(width_runs) > 1:
            log_event("opengen_run_done", matformer_width=width,
                      width_run_drops=_RUN_DROPS - drops_before_width,
                      run_drops=_RUN_DROPS)
        else:
            log_event("opengen_run_done", run_drops=_RUN_DROPS)

    if _RUN_DROPS > 0:
        print(f"[run_open_gen] *** {_RUN_DROPS} row(s)/pair(s) DROPPED by inference "
              f"failures -> results PARTIAL; exiting nonzero ***", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
