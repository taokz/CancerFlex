"""Per-field structured-output metrics for v0-derm evaluation.

These metrics are designed to defeat the two failure modes that pure
LM-loss eval is blind to on this corpus:

1. **Template memorization** — most assistant tokens are JSON template
   (keys, punctuation, fixed placeholders). Once the schema is learned the
   loss collapses across flags. The fix here is to stop scoring the
   template at all: extract the JSON, score field-by-field.

2. **Label-imbalance memorization** — if 80% of rows have
   ``malignancy_risk="low"``, plain accuracy is dominated by the majority
   class. We use **macro-F1** (per-class F1 averaged) so rare classes get
   equal weight.

API:
- ``parse_json_head(text)`` -> dict | None
- ``per_field_metrics(gt_jsons, pred_jsons, fields=None)`` -> dict[str, float]
- ``macro_f1(gt, pred, classes)`` -> float
- ``confusion_matrix(gt, pred, classes)`` -> np.ndarray[K, K]

All pure Python / NumPy — no torch dependency, fully unit-testable on CPU.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Default categorical fields to score for the v0-derm response schema.
# These match the keys produced by ``Experiment/core_code/v0_derm_corpus/build_v0_derm_corpus.py``.
DEFAULT_FIELDS: Tuple[str, ...] = (
    "quality",
    "target_present",
    "screen_positive",
    "malignancy_risk",
    "fine_label",
    "recommendation",
)

_JSON_HEAD_RE = re.compile(r"\{")


def parse_json_head(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first balanced JSON object at the start of ``text``.

    Tolerates leading whitespace, trailing T4S++ block (``<ref>...<seg>...``),
    or trailing free-form text after the JSON.
    """
    if not text:
        return None
    s = text.lstrip()
    if not s.startswith("{"):
        m = _JSON_HEAD_RE.search(s)
        if m is None:
            return None
        s = s[m.start():]
    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(s):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(s[: i + 1])
                except (TypeError, ValueError):
                    return None
    return None


def _norm(value: Any) -> Any:
    """Normalize values for comparison: strip strings, leave bool/int/None as-is."""
    if isinstance(value, str):
        v = value.strip().lower()
        return v if v else None
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return tuple(_norm(x) for x in value)
    return value


def _get_field(obj: Optional[Dict[str, Any]], field: str) -> Any:
    """Read a possibly-nested field from a parsed JSON dict.

    For the common case the field is a top-level key. For ``localization``
    we surface ``localization.target`` if asked.
    """
    if obj is None:
        return None
    if "." in field:
        parts = field.split(".")
        cur: Any = obj
        for p in parts:
            if isinstance(cur, dict):
                cur = cur.get(p)
            else:
                return None
        return _norm(cur)
    return _norm(obj.get(field))


def macro_f1(gt: Sequence[Any], pred: Sequence[Any], classes: Sequence[Any]) -> float:
    """Macro-F1 across ``classes``. Items where pred is None count as wrong."""
    if not classes:
        return 0.0
    f1s: List[float] = []
    for c in classes:
        tp = sum(1 for g, p in zip(gt, pred) if g == c and p == c)
        fp = sum(1 for g, p in zip(gt, pred) if g != c and p == c)
        fn = sum(1 for g, p in zip(gt, pred) if g == c and p != c)
        if tp + fp == 0 and tp + fn == 0:
            # Class never appears in GT or pred — skip rather than emit a 0.
            continue
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        if precision + recall == 0:
            f1s.append(0.0)
        else:
            f1s.append(2 * precision * recall / (precision + recall))
    return sum(f1s) / max(1, len(f1s))


def confusion_matrix(
    gt: Sequence[Any], pred: Sequence[Any], classes: Sequence[Any],
) -> np.ndarray:
    """K x K confusion matrix; row=gt class, col=pred class. Missing pred → ignored."""
    idx = {c: i for i, c in enumerate(classes)}
    K = len(classes)
    m = np.zeros((K, K), dtype=np.int64)
    for g, p in zip(gt, pred):
        if g not in idx or p not in idx:
            continue
        m[idx[g], idx[p]] += 1
    return m


def per_field_metrics(
    gt_jsons: Sequence[Optional[Dict[str, Any]]],
    pred_jsons: Sequence[Optional[Dict[str, Any]]],
    fields: Iterable[str] = DEFAULT_FIELDS,
    min_class_support: int = 1,
    field_statuses: Optional[Sequence[Optional[Dict[str, str]]]] = None,
) -> Dict[str, Any]:
    """Per-field accuracy + macro-F1 + class support distribution.

    Eligibility (under the field-aware contract):
      A row contributes to a field's metrics ONLY when its field_status is
      ``observed_value`` or ``observed_null``. Rows with ``unobserved`` or
      ``not_applicable`` for that field are skipped — we should not penalize
      the model for outputting (or omitting) a field the dataset never
      supplied.

    If ``field_statuses`` is None, falls back to the legacy GT-non-null
    eligibility, which is correct for v0-derm rows that consistently use
    None for unobserved fields.

    ``min_class_support``: classes appearing in fewer than this many GT rows
    are excluded from macro-F1 (so macro-F1 isn't dominated by ultra-rare
    labels with 1 GT example).
    """
    out: Dict[str, Any] = {}
    n = max(len(gt_jsons), len(pred_jsons))
    if n == 0:
        return out
    out["n_samples"] = n
    for field in fields:
        # Eligibility per row.
        if field_statuses is not None:
            eligible = [
                fs.get(field) in {"observed_value", "observed_null"}
                if isinstance(fs, dict) else False
                for fs in field_statuses
            ]
        else:
            eligible = [True] * n

        gt_vals: List[Any] = []
        pred_vals: List[Any] = []
        for i, ok in enumerate(eligible):
            if not ok:
                continue
            gt_vals.append(_get_field(gt_jsons[i], field))
            pred_vals.append(_get_field(pred_jsons[i], field))

        n_eligible = len(gt_vals)
        out[f"{field}_n_eligible"] = n_eligible
        if n_eligible == 0:
            continue

        gt_has = sum(1 for g in gt_vals if g is not None)
        pred_has = sum(1 for p in pred_vals if p is not None)
        out[f"{field}_gt_coverage"] = gt_has / n_eligible
        out[f"{field}_pred_coverage"] = pred_has / n_eligible

        if gt_has == 0:
            continue

        em_num = sum(1 for g, p in zip(gt_vals, pred_vals) if g is not None and g == p)
        out[f"{field}_em"] = em_num / gt_has

        gt_counter = Counter(g for g in gt_vals if g is not None)
        kept_classes = sorted(
            (c for c, k in gt_counter.items() if k >= min_class_support),
            key=str,
        )
        out[f"{field}_n_classes"] = len(kept_classes)
        if kept_classes:
            out[f"{field}_macro_f1"] = macro_f1(gt_vals, pred_vals, kept_classes)
        out[f"{field}_class_support"] = dict(gt_counter)
    return out


def per_dataset_metrics(
    gt_jsons: Sequence[Optional[Dict[str, Any]]],
    pred_jsons: Sequence[Optional[Dict[str, Any]]],
    dataset_ids: Sequence[str],
    fields: Iterable[str] = DEFAULT_FIELDS,
    min_class_support: int = 1,
    field_statuses: Optional[Sequence[Optional[Dict[str, str]]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Same as ``per_field_metrics`` but stratified by ``dataset_id``."""
    by_ds: Dict[str, List[int]] = {}
    for i, ds in enumerate(dataset_ids):
        by_ds.setdefault(str(ds), []).append(i)
    out: Dict[str, Dict[str, Any]] = {}
    for ds, idxs in by_ds.items():
        sub_gt = [gt_jsons[i] for i in idxs]
        sub_pred = [pred_jsons[i] for i in idxs]
        sub_fs = ([field_statuses[i] for i in idxs]
                  if field_statuses is not None else None)
        out[ds] = per_field_metrics(
            sub_gt, sub_pred, fields=fields, min_class_support=min_class_support,
            field_statuses=sub_fs,
        )
    return out


def binary_sens_at_spec(
    gt: Sequence[bool], pred: Sequence[bool], target_spec: float = 0.95,
) -> Optional[float]:
    """Sensitivity at fixed specificity for a binary metric.

    Greedy decoding gives only hard predictions, so we report the realized
    operating point: sens and spec computed from the confusion matrix. We
    return sens iff spec >= target_spec, else None (with a flag so the
    caller can downgrade).
    """
    tp = sum(1 for g, p in zip(gt, pred) if g and p)
    fp = sum(1 for g, p in zip(gt, pred) if (not g) and p)
    fn = sum(1 for g, p in zip(gt, pred) if g and (not p))
    tn = sum(1 for g, p in zip(gt, pred) if (not g) and (not p))
    sens = tp / max(1, tp + fn)
    spec = tn / max(1, tn + fp)
    if spec < target_spec:
        return None
    return sens


# ─────────────────────────────────────────────────────────────────────────────
# Imbalance-robust metrics (added 2026-05-19 — NOT YET WIRED INTO AGGREGATORS)
#
# Defined here so callers can pull them in once we decide to refresh metric
# JSONs. To enable in `run_medgemma_baseline._aggregate_pair`, import these
# and add to the binary / multiclass blocks; the call sites are marked
# `TODO(imbalance-metrics)`. Until then they are dormant. Rationale + when
# each fires:
#
#   mcc_binary / mcc_multiclass     single scalar [-1, 1]; random baseline 0
#                                    irrespective of class prevalence — best
#                                    headline for skewed test sets.
#   average_precision_binary        area under PR curve; uninflated by
#                                    99%-positive priors (where ROC-AUC saturates).
#                                    Report both AP-positive AND AP-negative
#                                    so the minority-as-negative pillars
#                                    (e.g. Raabin all u3=0) also get scored.
#   balanced_accuracy_*             (sens + spec)/2 for binary; mean of per-class
#                                    recall for multiclass. random baseline 0.5
#                                    binary / 1/K multiclass.
#
# All pure numpy, O(n log n).
# ─────────────────────────────────────────────────────────────────────────────


def mcc_binary(y: Sequence[int], pred: Sequence[int]) -> Optional[float]:
    """Matthews Correlation Coefficient for binary classification.

    Robust to class imbalance because it uses all four cells of the
    confusion matrix symmetrically. Random baseline = 0 regardless of
    prevalence. Returns None when the denominator is 0 (every prediction
    constant OR every gold constant).
    """
    y_arr = np.asarray(y, dtype=np.int64)
    p_arr = np.asarray(pred, dtype=np.int64)
    if y_arr.size == 0 or y_arr.size != p_arr.size:
        return None
    tp = int(((y_arr == 1) & (p_arr == 1)).sum())
    tn = int(((y_arr == 0) & (p_arr == 0)).sum())
    fp = int(((y_arr == 0) & (p_arr == 1)).sum())
    fn = int(((y_arr == 1) & (p_arr == 0)).sum())
    num = tp * tn - fp * fn
    denom_sq = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom_sq <= 0:
        return None
    return float(num / math.sqrt(denom_sq))


def mcc_multiclass(
    gt: Sequence[Any], pred: Sequence[Any], classes: Sequence[Any],
) -> Optional[float]:
    """Multiclass MCC (Gorodkin 2004 generalisation).

    Same range [-1, 1], random baseline 0. Symmetric over classes — useful
    when reporting U4 alongside per-class AUROC.
    """
    gt = list(gt)
    pred = list(pred)
    if not gt or len(gt) != len(pred):
        return None
    idx = {c: i for i, c in enumerate(classes)}
    K = len(classes)
    C = np.zeros((K, K), dtype=np.int64)
    for g, p in zip(gt, pred):
        if g in idx and p in idx:
            C[idx[g], idx[p]] += 1
    t = C.sum(axis=1).astype(np.float64)  # gold counts per class
    pkt = C.sum(axis=0).astype(np.float64)  # pred counts per class
    n = C.sum()
    if n <= 0:
        return None
    s = float(np.trace(C))
    num = s * n - float(t @ pkt)
    denom_sq = (n * n - float(pkt @ pkt)) * (n * n - float(t @ t))
    if denom_sq <= 0:
        return None
    return float(num / math.sqrt(denom_sq))


def average_precision_binary(
    y: Sequence[int], score: Sequence[float], positive_class: int = 1,
) -> Optional[float]:
    """Area under the precision-recall curve (a.k.a. average precision).

    Uses the step-wise definition: AP = Σ (R_n − R_{n−1}) · P_n, summed
    over thresholds — the same convention as sklearn's average_precision_score.
    Pass positive_class=0 to score the minority-as-negative pillars.

    Returns None for single-class slices.
    """
    y_arr = np.asarray(y).astype(np.int64)
    s_arr = np.asarray(score, dtype=np.float64)
    if y_arr.size == 0 or y_arr.size != s_arr.size:
        return None
    # Flip score sign if scoring the negative class.
    if positive_class == 0:
        y_arr = (y_arr == 0).astype(np.int64)
        s_arr = -s_arr
    else:
        y_arr = (y_arr == positive_class).astype(np.int64)
    pos = int(y_arr.sum())
    if pos == 0 or pos == y_arr.size:
        return None  # single-class — AP undefined / trivial
    # Sort by score descending.
    order = np.argsort(-s_arr, kind="mergesort")
    y_sorted = y_arr[order]
    # Group ties: cumulative TP / FP at each unique score threshold.
    tp_cum = np.cumsum(y_sorted)
    fp_cum = np.cumsum(1 - y_sorted)
    precision = tp_cum / np.maximum(1, tp_cum + fp_cum)
    recall = tp_cum / pos
    # AP = Σ (R_n − R_{n−1}) · P_n, with R_0 = 0.
    recall_prev = np.concatenate(([0.0], recall[:-1]))
    ap = float(np.sum((recall - recall_prev) * precision))
    return ap


def balanced_accuracy_binary(
    y: Sequence[int], pred: Sequence[int],
) -> Optional[float]:
    """(sensitivity + specificity) / 2 — rescales accuracy by prevalence.

    Random baseline 0.5 regardless of class skew. Returns None for
    single-class gold.
    """
    y_arr = np.asarray(y, dtype=np.int64)
    p_arr = np.asarray(pred, dtype=np.int64)
    if y_arr.size == 0 or y_arr.size != p_arr.size:
        return None
    pos = int((y_arr == 1).sum())
    neg = int((y_arr == 0).sum())
    if pos == 0 or neg == 0:
        return None
    sens = float(((y_arr == 1) & (p_arr == 1)).sum()) / pos
    spec = float(((y_arr == 0) & (p_arr == 0)).sum()) / neg
    return 0.5 * (sens + spec)


def balanced_accuracy_multiclass(
    gt: Sequence[Any], pred: Sequence[Any], classes: Sequence[Any],
) -> Optional[float]:
    """Mean of per-class recall (i.e. sensitivity averaged over classes).

    Random baseline 1/K. Returns None if no class has any gold rows.
    """
    gt = list(gt); pred = list(pred)
    if not gt or len(gt) != len(pred):
        return None
    recalls = []
    for c in classes:
        n_c = sum(1 for g in gt if g == c)
        if n_c == 0:
            continue
        tp_c = sum(1 for g, p in zip(gt, pred) if g == c and p == c)
        recalls.append(tp_c / n_c)
    if not recalls:
        return None
    return float(sum(recalls) / len(recalls))


# ─────────────────────────────────────────────────────────────────────────────
# Label-token logit aggregators (see docs/eval_metrics.md §1–§5)
#
# Pure numpy. sklearn / scipy are imported lazily inside the bootstrap and
# DeLong paths so the module stays importable in minimal envs.
# ─────────────────────────────────────────────────────────────────────────────


def softmax_from_scores(scores: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along the last axis.

    Input shape [..., K]; output same shape, rows sum to 1.
    """
    s = np.asarray(scores, dtype=np.float64)
    m = np.max(s, axis=-1, keepdims=True)
    e = np.exp(s - m)
    return e / np.sum(e, axis=-1, keepdims=True)


def single_token_verbalizer_guard(tokenizer: Any, verbalizers: Sequence[str]) -> Dict[str, Any]:
    """Decide if a single-token-logit fast path is safe for ``verbalizers``.

    Safe iff every verbalizer tokenises to exactly one piece AND no two
    verbalizers share the same first token id. Returns a dict with
    ``ok``, ``reason``, and ``token_ids`` so callers can log the failure.
    """
    if tokenizer is None:
        return {"ok": False, "reason": "no tokenizer provided", "token_ids": None}
    ids: List[List[int]] = []
    multi: List[Tuple[str, int]] = []
    for v in verbalizers:
        try:
            t = tokenizer(v, add_special_tokens=False)["input_ids"]
        except Exception as exc:  # pragma: no cover
            return {"ok": False, "reason": f"tokenizer error on {v!r}: {exc}", "token_ids": None}
        ids.append(list(t))
        if len(t) != 1:
            multi.append((v, len(t)))
    if multi:
        return {
            "ok": False,
            "reason": f"multi-token verbalizers: {multi}",
            "token_ids": ids,
        }
    firsts = [i[0] for i in ids]
    if len(set(firsts)) != len(firsts):
        return {
            "ok": False,
            "reason": f"first-token collision: {dict(zip(verbalizers, firsts))}",
            "token_ids": ids,
        }
    return {"ok": True, "reason": "", "token_ids": ids}


def _midranks(x: np.ndarray) -> np.ndarray:
    """Midranks of x (ties get the average of their ranks). Length-preserving."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(x)
    i = 0
    n = len(x)
    while i < n:
        j = i
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1
        # Average rank for ties; ranks here are 1-based by convention.
        avg = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def auroc_binary(y: Sequence[int], score: Sequence[float]) -> Optional[float]:
    """Binary AUROC via the rank formula. Returns None if y is single-class.

    ``y`` is interpreted as truthy=positive class. ``score`` is any monotone
    score where larger = more positive (e.g. logit_pos - logit_neg, or
    softmax probability of the positive class).
    """
    y_arr = np.asarray([1 if v else 0 for v in y], dtype=np.int64)
    s_arr = np.asarray(score, dtype=np.float64)
    if y_arr.size != s_arr.size or y_arr.size == 0:
        return None
    n_pos = int(y_arr.sum())
    n_neg = int(y_arr.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    r = _midranks(s_arr)
    sum_pos_ranks = float(r[y_arr == 1].sum())
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def auroc_ovo_macro(
    y: Sequence[Any], p: np.ndarray, classes: Sequence[Any],
) -> Optional[float]:
    """One-vs-one macro AUROC via Hand & Till (2001).

    For each unordered pair (i, j) compute
    ``A_hat(i, j) = 0.5 * (A(i | i, j) + A(j | i, j))`` where
    ``A(i | i, j)`` is the binary AUROC on the subset gold ∈ {i, j}
    ranking by ``p[:, i]`` alone with positives = (gold == i). This
    matches ``sklearn.metrics.roc_auc_score(..., multi_class="ovo")``.

    Ranking by the margin ``p[:, i] - p[:, j]`` is *not* equivalent: when
    a third class absorbs different mass across rows, the margin can
    reorder samples that ``p[:, i]`` alone would separate correctly.
    """
    y_arr = np.asarray(y)
    p = np.asarray(p, dtype=np.float64)
    K = len(classes)
    if K < 2 or p.shape != (len(y_arr), K):
        return None
    aucs: List[float] = []
    for i in range(K):
        for j in range(i + 1, K):
            ci, cj = classes[i], classes[j]
            mask = (y_arr == ci) | (y_arr == cj)
            if mask.sum() == 0:
                continue
            y_pair = (y_arr[mask] == ci).astype(np.int64)
            if y_pair.sum() == 0 or y_pair.sum() == y_pair.size:
                continue
            a_ij = auroc_binary(y_pair, p[mask, i])
            a_ji = auroc_binary(1 - y_pair, p[mask, j])
            if a_ij is None or a_ji is None:
                continue
            aucs.append(0.5 * (a_ij + a_ji))
    if not aucs:
        return None
    return float(np.mean(aucs))


def auroc_ovr_per_class(
    y: Sequence[Any], p: np.ndarray, classes: Sequence[Any],
) -> Dict[Any, Optional[float]]:
    """One-vs-rest AUROC per class. Class with no positives → None."""
    y_arr = np.asarray(y)
    p = np.asarray(p, dtype=np.float64)
    out: Dict[Any, Optional[float]] = {}
    for k, c in enumerate(classes):
        out[c] = auroc_binary((y_arr == c).astype(np.int64), p[:, k])
    return out


def qwk(gt: Sequence[Any], pred: Sequence[Any], labels: Sequence[Any]) -> Optional[float]:
    """Quadratic-weighted Cohen's kappa over ordered ``labels``.

    Items with gt or pred outside ``labels`` are dropped (the caller should
    surface a coverage statistic separately).
    """
    idx = {c: i for i, c in enumerate(labels)}
    pairs = [(idx[g], idx[p]) for g, p in zip(gt, pred) if g in idx and p in idx]
    if not pairs:
        return None
    K = len(labels)
    obs = np.zeros((K, K), dtype=np.float64)
    for g, p in pairs:
        obs[g, p] += 1.0
    n = obs.sum()
    if n == 0:
        return None
    g_marg = obs.sum(axis=1)
    p_marg = obs.sum(axis=0)
    exp = np.outer(g_marg, p_marg) / n
    # Quadratic weights.
    w = (np.arange(K)[:, None] - np.arange(K)[None, :]) ** 2 / max(1, (K - 1) ** 2)
    num = (w * obs).sum()
    den = (w * exp).sum()
    if den == 0:
        return None
    return float(1.0 - num / den)


def ece(p_max: Sequence[float], correct: Sequence[bool], n_bins: int = 15) -> Optional[float]:
    """Equal-frequency ECE on top-class confidences.

    ``p_max[i]`` is the model's confidence for record i (max softmax),
    ``correct[i]`` is whether the argmax matched gold. Equal-frequency
    binning ensures no bin is empty.
    """
    p = np.asarray(p_max, dtype=np.float64)
    c = np.asarray([1.0 if v else 0.0 for v in correct], dtype=np.float64)
    n = p.size
    if n == 0 or n_bins < 1:
        return None
    order = np.argsort(p)
    p = p[order]
    c = c[order]
    bin_size = n / n_bins
    total = 0.0
    for b in range(n_bins):
        lo = int(np.floor(b * bin_size))
        hi = int(np.floor((b + 1) * bin_size)) if b < n_bins - 1 else n
        if hi <= lo:
            continue
        conf = p[lo:hi].mean()
        acc = c[lo:hi].mean()
        total += (hi - lo) * abs(conf - acc)
    return float(total / n)


def brier_binary(y: Sequence[int], p_pos: Sequence[float]) -> Optional[float]:
    """Standard binary Brier score: mean over rows of ``(p_pos - y)**2``.

    Use this on K=2 tasks. ``brier_multiclass`` on a 2-class one-hot would
    return double this value (it sums ``(p_k - y_k)**2`` over both classes,
    each contributing the same squared error).
    """
    y_arr = np.asarray([1 if v else 0 for v in y], dtype=np.float64)
    p_arr = np.asarray(p_pos, dtype=np.float64)
    if y_arr.size == 0 or p_arr.shape != y_arr.shape:
        return None
    diff = p_arr - y_arr
    return float((diff * diff).mean())


def brier_multiclass(y_idx: Sequence[int], p: np.ndarray, K: int) -> Optional[float]:
    """Multi-class Brier score. ``y_idx`` are gold class indices in [0, K)."""
    y_arr = np.asarray(y_idx, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    if y_arr.size == 0 or p.shape != (y_arr.size, K):
        return None
    onehot = np.zeros_like(p)
    valid = (y_arr >= 0) & (y_arr < K)
    if valid.sum() == 0:
        return None
    onehot[np.arange(y_arr.size)[valid], y_arr[valid]] = 1.0
    diff = (p - onehot)[valid]
    return float((diff * diff).sum(axis=1).mean())


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Wilson interval for a proportion. Returns (estimate, low, high) or Nones."""
    if n <= 0:
        return None, None, None
    # 1.96 for 95% — keep dep-free, hard-code the normal inverse for common alphas.
    z = {0.05: 1.9599639845400545, 0.10: 1.6448536269514722, 0.01: 2.5758293035489004}.get(alpha)
    if z is None:
        # Approximate via scipy if available.
        try:
            from scipy.stats import norm  # noqa
            z = float(norm.ppf(1 - alpha / 2))
        except Exception:
            z = 1.9599639845400545
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, float(centre - half), float(centre + half)


def delong_ci(
    y: Sequence[int], score: Sequence[float], alpha: float = 0.05,
) -> Dict[str, Optional[float]]:
    """DeLong CI for binary AUROC via Sun & Xu (2014) placement values.

    Returns ``{auc, low, high, se}``; any field is None when y is single-class.
    """
    y_arr = np.asarray([1 if v else 0 for v in y], dtype=np.int64)
    s_arr = np.asarray(score, dtype=np.float64)
    n_pos = int(y_arr.sum())
    n_neg = int(y_arr.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return {"auc": None, "low": None, "high": None, "se": None}
    s_pos = s_arr[y_arr == 1]
    s_neg = s_arr[y_arr == 0]
    # Placement values via midranks.
    r_all_pos = _midranks(np.concatenate([s_pos, s_neg]))[:n_pos]
    r_pos_alone = _midranks(s_pos)
    r_all_neg = _midranks(np.concatenate([s_neg, s_pos]))[:n_neg]
    r_neg_alone = _midranks(s_neg)
    V10 = (r_all_pos - r_pos_alone) / n_neg
    V01 = 1.0 - (r_all_neg - r_neg_alone) / n_pos
    auc = float(V10.mean())
    var10 = float(V10.var(ddof=1)) if n_pos > 1 else 0.0
    var01 = float(V01.var(ddof=1)) if n_neg > 1 else 0.0
    se = float(np.sqrt(var10 / n_pos + var01 / n_neg))
    z = {0.05: 1.9599639845400545, 0.10: 1.6448536269514722, 0.01: 2.5758293035489004}.get(alpha, 1.9599639845400545)
    return {
        "auc": auc,
        "low": float(max(0.0, auc - z * se)),
        "high": float(min(1.0, auc + z * se)),
        "se": se,
    }


def bootstrap_auroc_ci(
    y: Sequence[int], score: Sequence[float], groups: Sequence[Any],
    B: int = 1000, alpha: float = 0.05, seed: int = 20260527,
) -> Dict[str, Optional[float]]:
    """Cluster (group) bootstrap 95% CI for binary AUROC.

    Resamples whole GROUPS with replacement (``groups`` = the natural sampling
    unit, e.g. ``patient_id`` / ``source_image``), so rows correlated within a
    group stay together — the honest CI when the data is clustered. When every
    row is its own group this reduces to an ordinary per-row bootstrap. Returns
    ``{low, high, point, n_boot}``; low/high are None when single-class or empty.

    Unifies binary-AUROC CIs across U0-U3 on ONE bootstrap method (U0 groups by
    ``source_image`` in run_u0; U1-U3 group by ``patient_id`` in run_closed) and
    matches U4's bootstrap F1 CI. It replaces the earlier DeLong CI. The point
    AUROC estimate is unaffected — only the CI method changes.
    """
    y_arr = np.asarray([1 if v else 0 for v in y], dtype=np.int64)
    s_arr = np.asarray(score, dtype=np.float64)
    g_arr = np.asarray([str(g) for g in groups])
    n = y_arr.size
    if n == 0 or g_arr.size != n or int(y_arr.sum()) in (0, n):
        return {"low": None, "high": None, "point": None, "n_boot": 0}
    uniq = np.unique(g_arr)
    by_g = {g: np.where(g_arr == g)[0] for g in uniq}
    rng = np.random.RandomState(seed)
    boots: List[float] = []
    for _ in range(B):
        samp = rng.choice(uniq, size=uniq.size, replace=True)
        idx = np.concatenate([by_g[g] for g in samp])
        a = auroc_binary(y_arr[idx].tolist(), s_arr[idx].tolist())
        if a is not None and np.isfinite(a):
            boots.append(float(a))
    if not boots:
        return {"low": None, "high": None, "point": None, "n_boot": 0}
    arr = np.asarray(boots)
    return {
        "low": float(np.quantile(arr, alpha / 2)),
        "high": float(np.quantile(arr, 1 - alpha / 2)),
        "point": float(arr.mean()),
        "n_boot": int(arr.size),
    }


def stratified_bootstrap_ci(
    metric_fn,
    payload: Dict[str, np.ndarray],
    strata: Sequence[Any],
    B: int = 2000,
    alpha: float = 0.05,
    seed: int = 20260513,
) -> Dict[str, Optional[float]]:
    """Generic stratified bootstrap.

    ``metric_fn(payload_resampled)`` -> float | None.
    ``payload`` is a dict of arrays all sharing axis 0 = sample dimension.
    Resampling is done with replacement within each stratum so prevalence
    per stratum is preserved.
    """
    rng = np.random.default_rng(seed)
    n = len(strata)
    if n == 0:
        return {"low": None, "high": None, "point": None, "n_boot": 0}
    strata_arr = np.asarray(strata)
    groups: Dict[Any, np.ndarray] = {}
    for s in np.unique(strata_arr):
        groups[s] = np.where(strata_arr == s)[0]
    boots: List[float] = []
    for _ in range(B):
        parts = []
        for s, idxs in groups.items():
            parts.append(rng.choice(idxs, size=len(idxs), replace=True))
        sample = np.concatenate(parts)
        resampled = {k: v[sample] for k, v in payload.items()}
        try:
            val = metric_fn(resampled)
        except Exception:
            val = None
        if val is None or not np.isfinite(val):
            continue
        boots.append(float(val))
    if not boots:
        return {"low": None, "high": None, "point": None, "n_boot": 0}
    arr = np.array(boots)
    return {
        "low": float(np.quantile(arr, alpha / 2)),
        "high": float(np.quantile(arr, 1 - alpha / 2)),
        "point": float(np.mean(arr)),
        "n_boot": int(arr.size),
    }


# Random-baseline reference values; doc §6 requires explicit reporting.
RANDOM_BASELINES: Dict[str, float] = {
    "auroc_binary": 0.5,
    "auroc_ovo_macro": 0.5,
    "auroc_ovr": 0.5,
    "qwk": 0.0,
    # accuracy_random depends on K — compute at call site as 1/K.
}
