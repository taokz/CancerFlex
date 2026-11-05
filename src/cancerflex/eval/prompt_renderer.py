"""Renders a VQA-style training/eval prompt from a manifest row + U-task.

Usage:
    from prompt_renderer import render_prompt
    row = pd.read_parquet("Experiment/datasets/skin/manifest.parquet").iloc[0]
    prompt = render_prompt(row, task="U3", template_idx=0)
    # or random template:
    prompt = render_prompt(row, task="U3", seed=42)

The output is a string with placeholders for the image token ("<image>") that
downstream code (tokenizer / data collator) replaces with the actual image
embedding.
"""
from __future__ import annotations

import inspect
import json
import random
from typing import Any

try:
    from prompt_templates import TEMPLATES, PILLAR_CONFIG, get_template
except ImportError:
    from .prompt_templates import TEMPLATES, PILLAR_CONFIG, get_template
try:
    import u4_axis
except ImportError:
    from . import u4_axis


# Version-2 U4 template (prompt-design audit 2026-09-16; see
# `u4_axis.U4_UNIT_STEMS_V2` for the rationale). Same 4-block layout and the same
# "ANSWER: One class name." contract as template 0; ONLY the TASK sentence changes:
# `{u4_stem}` is the report-unit-specific stem that names the label space, so two
# units of one pillar no longer share a stem. Used only when the caller asks for
# prompt-stem version >= 2; version 1 keeps `TEMPLATES["U4"][template_idx]`.
U4_TEMPLATE_V2 = (
    "IMAGE: {image}\n\n"
    "TASK: {u4_stem} Choose exactly one of: {u4_classes}.\n\n"
    "CONTEXT:\n{context_block}\n\n"
    "ANSWER: One class name."
)


# ---------------------------------------------------------------------------
# Pillar resolution: map (cancer_type, modality, dataset_id) → pillar config key.
# ---------------------------------------------------------------------------

def _resolve_pillar_key(row: dict) -> str:
    def _safe(v):
        if v is None:
            return ""
        if isinstance(v, float):
            import math
            if math.isnan(v):
                return ""
        return str(v)
    cancer = _safe(row.get("cancer_type")).lower()
    modality = _safe(row.get("modality")).lower()
    dataset_id = _safe(row.get("dataset_id"))

    if cancer in {"skin"}:
        return "skin"
    if cancer in {"breast"}:
        if "mammography" in modality or "mammo" in modality or "ffdm" in modality:
            return "breast_mammo"
        return "breast_us"
    if cancer in {"cervix", "cervical"}:
        if "colpo" in modality or "via" in modality:
            return "cervix_via"
        return "cervix_cytology"
    if cancer in {"oral"}:
        return "oral"
    if cancer in {"thyroid"}:
        return "thyroid"
    if cancer in {"blood", "leukemia"}:
        return "blood"
    if cancer in {"liver"}:
        return "liver"
    # Fallback by dataset_id
    if dataset_id in {"VinDr-Mammo", "INbreast", "CMMD", "Mammo-MX"}:
        return "breast_mammo"
    if dataset_id == "BCMID":
        if "mammography" in modality or "mammo" in modality or "ffdm" in modality:
            return "breast_mammo"
        return "breast_us"
    if dataset_id.startswith("BUS"):
        return "breast_us"
    # Blood pillar — datasets known
    if dataset_id in {
        "C-NMC-2019", "Raabin-WBC", "APL-Madrid", "BCCD", "ALL-IDB-Subtypes",
        "AML-Cytomorphology_LMU", "MLL23", "PBC-Barcelona", "WBCAttPlus",
        "RV-PBS", "LLD",
        # TXL-PBC (ingested 2026-09-16) carries no cancer_type, so without this
        # entry every one of its 597 rows raised here. Latent, not active: TXL
        # only has u5/u6 labels and run_open_gen builds its own U5/U6 prompt
        # without calling render_prompt — but any U0-U4 admission, or any caller
        # that renders a blood row generically, would have crashed.
        "TXL-PBC",
    }:
        return "blood"
    raise ValueError(f"Cannot resolve pillar for row: cancer={cancer!r}, modality={modality!r}, dataset_id={dataset_id!r}")


# ---------------------------------------------------------------------------
# Context block rendering: skip null fields, fixed order.
# ---------------------------------------------------------------------------

# Order: broad → specific
_CONTEXT_BULLET_SPECS = [
    # (label, dotted_path, render_fn or None)
    ("Demographics", "demographics", "demographics"),
    ("Site", "site_anatomy", "site_anatomy"),
    ("Device", "device", "device"),
    ("Clinical history", "clinical_priors", "clinical_priors"),
    ("Visible features", "imaging_findings", "imaging_findings"),
]


def _is_null(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, float):
        import math
        return math.isnan(v)
    if isinstance(v, str):
        return v.strip().lower() in {"", "nan", "none", "null", "na", "-", "unknown"}
    return False


def _render_demographics(d: dict) -> str:
    pieces = []
    age = d.get("age")
    sex = d.get("sex")
    if not _is_null(age) and not _is_null(sex):
        pieces.append(f"{int(float(age))}-year-old {sex}")
    elif not _is_null(age):
        pieces.append(f"{int(float(age))}-year-old patient")
    elif not _is_null(sex):
        pieces.append(f"{sex} patient")
    for k, label in [
        ("race", "race"), ("ethnicity", "ethnicity"),
        ("fitzpatrick", "Fitzpatrick"), ("skin_tone", "skin tone"),
    ]:
        v = d.get(k)
        if not _is_null(v):
            pieces.append(f"{label}={v}")
    return ", ".join(pieces) if pieces else ""


def _render_site_anatomy(d: dict, strip: set[str] | None = None) -> str:
    skip = strip or set()
    pieces = []
    laterality = d.get("laterality")
    # anatomical_site is droppable via the strip set. Notably oral, whose source
    # free-text site field is polluted with read-after findings / quality /
    # diagnosis tokens (dental caries, out of focus, pigmentation, ...), is
    # stripped in render_prompt so none of that leaks into the CONTEXT block.
    anatomical_site = None if "anatomical_site" in skip else d.get("anatomical_site")
    view = d.get("view")
    if not _is_null(laterality) and not _is_null(anatomical_site):
        pieces.append(f"{laterality} {anatomical_site}")
    elif not _is_null(anatomical_site):
        pieces.append(str(anatomical_site))
    elif not _is_null(laterality):
        pieces.append(f"{laterality} side")
    if not _is_null(view):
        pieces.append(f"view: {view}")
    for k, label in [
        ("lesion_count", "lesion count"),
        ("lesion_size_mm", "lesion size (mm)"),
    ]:
        if k in skip:
            continue
        v = d.get(k)
        if not _is_null(v):
            pieces.append(f"{label}={v}")
    return ", ".join(pieces) if pieces else ""


# Dataset names that leaked into the `scanner` field, which is otherwise legitimate
# acquisition metadata a deployed system would have. "LLD high_resolution 100X C1" is three
# real descriptors (resolution, magnification, camera id) prefixed by the DATASET's name.
#
# WHY IT MATTERS. blood U4 holds two report units — `blood:wbc:morphology_primary` (75,590 rows)
# and `blood:leukemia:LLD` (14,969) — that rendered a byte-identical question stem, the same
# collision that broke cervix U4. blood did not show the failure only because every LLD row
# carried its own dataset name in the prompt, so the model could recover the label space from
# the leak instead of from the option list. That is masking, not solving: external OOV is 0/27,217
# for a reason we would not want to write down. Version-2 stems name the label space in the
# question itself, so the crutch is no longer needed and is removed with them.
#
# Version-gated ON PURPOSE: every published blood number was produced with the leak present, so
# version 1 must keep rendering it byte-identically or those numbers stop being reproducible.
_SCANNER_DATASET_PREFIXES = ("LLD ",)


def _strip_dataset_name_from_scanner(v: str) -> str:
    for p in _SCANNER_DATASET_PREFIXES:
        if isinstance(v, str) and v.startswith(p):
            return v[len(p):]
    return v


def _render_device(d: dict, prompt_stem_version: int = 1) -> str:
    pieces = []
    for k in ["modality", "scanner", "manufacturer", "device_family", "image_type"]:
        v = d.get(k)
        if _is_null(v):
            continue
        if k == "scanner" and prompt_stem_version >= 2:
            v = _strip_dataset_name_from_scanner(v)
            if _is_null(v) or not str(v).strip():
                continue
        pieces.append(f"{k}={v}")
    return ", ".join(pieces) if pieces else ""


def _render_clinical_priors(d: dict, strip: set[str] | None = None) -> str:
    skip = strip or set()
    pieces = []
    for k, label in [
        ("birads_density", "BIRADS density"),
        ("tirads", "TI-RADS"),
        ("bethesda", "Bethesda"),
        ("biopsy_history", "prior biopsy"),
        ("fhx_cancer", "family history of cancer"),
        ("smoking", "smoking"),
        ("drinking", "drinking"),
        ("fibroscan_stiffness_kpa", "FibroScan VCTE (kPa)"),
        ("cap_score", "CAP score"),
    ]:
        if k in skip:
            continue
        v = d.get(k)
        if not _is_null(v):
            pieces.append(f"{label}={v}")
    if "weighted_condition_probs" not in skip:
        wcp = d.get("weighted_condition_probs")
        if not _is_null(wcp) and isinstance(wcp, dict) and wcp:
            top3 = sorted(wcp.items(), key=lambda kv: -kv[1])[:3]
            pieces.append("dermatologist differential=" + " / ".join(f"{k}({v:.2f})" for k, v in top3))
    return ", ".join(pieces) if pieces else ""


def _render_imaging_findings(d: dict, strip: set[str] | None = None) -> str:
    skip = strip or set()
    pieces = []
    for k, label in [
        ("birads_assessment", "BIRADS assessment"),
        ("shape", "shape"),
        ("margin", "margin"),
        ("echo_pattern", "echo pattern"),
        ("calcifications", "calcifications"),
        ("posterior_features", "posterior features"),
        ("color", "color"),
        ("surface_texture", "surface texture"),
    ]:
        if k in skip:
            continue
        v = d.get(k)
        if not _is_null(v):
            pieces.append(f"{label}={v}")
    if "lesion_description_text" not in skip:
        ldt = d.get("lesion_description_text")
        if not _is_null(ldt):
            pieces.append(f"description: {ldt}")
    return "; ".join(pieces) if pieces else ""


_RENDERERS = {
    "demographics": _render_demographics,
    "site_anatomy": _render_site_anatomy,
    "device": _render_device,
    "clinical_priors": _render_clinical_priors,
    "imaging_findings": _render_imaging_findings,
}


def render_context_block(context_json: Any,
                          strip_context_fields: set[str] | None = None,
                          prompt_stem_version: int = 1) -> str:
    """Render context_json into a bulleted multi-line block. Null fields
    are skipped entirely; empty context returns a placeholder line.

    ``strip_context_fields``: optional set of context-leaf field names to
    omit (e.g. {"tirads"} for thyroid eval to avoid gold-label leak in
    Clinical history block).
    """
    if _is_null(context_json):
        return "(no prior information available)"
    if isinstance(context_json, str):
        try:
            context = json.loads(context_json)
        except Exception:
            return "(no prior information available)"
    elif isinstance(context_json, dict):
        context = context_json
    else:
        return "(no prior information available)"

    strip = set(strip_context_fields or [])
    bullets = []
    for label, key, renderer_key in _CONTEXT_BULLET_SPECS:
        sub = context.get(key, {})
        if not isinstance(sub, dict):
            continue
        renderer = _RENDERERS[renderer_key]
        # Pass each optional kwarg to the renderers that actually declare it, read off the
        # signature rather than from a hand-maintained list.
        #
        # An earlier version of this dispatch hard-coded `strip` to clinical_priors and
        # imaging_findings and sent everything else through a bare `renderer(sub)`. That silently
        # dropped `strip` for `_render_site_anatomy`, which declares it — and `render_prompt`
        # unconditionally strips `anatomical_site` for oral because that field carries
        # read-after-findings text. The result was a leaked `- Site: ...` line on 16,997 of 27,770
        # oral rows at EVERY task, at version 1, i.e. a change to the supposedly frozen path.
        # Introduced and caught the same day (2026-09-16); the oral cells in
        # eval_results/{stemv2_smoke,stemv2_closed,refined_closed}_20260916 were rendered with it.
        #
        # The replaced try/except-TypeError form had the opposite failure mode (it swallowed a
        # mis-spelled kwarg the renderer wanted). Introspection avoids both.
        params = inspect.signature(renderer).parameters
        kwargs = {}
        if "strip" in params:
            kwargs["strip"] = strip
        if "prompt_stem_version" in params:
            kwargs["prompt_stem_version"] = prompt_stem_version
        rendered = renderer(sub, **kwargs)
        if rendered:
            bullets.append(f"- {label}: {rendered}")
    if not bullets:
        return "(no prior information available)"
    return "\n".join(bullets)


# Per-pillar gold-leak fields to strip from CONTEXT when evaluating zero-shot
# baselines. Each entry is the set of context-leaf keys whose value is (or
# is highly correlated with) the gold label for the U-task in question.
LEAK_FIELDS_BY_PILLAR: dict[str, set[str]] = {
    # tirads is DDTI's u4_label and the derivation source for u2/u3_label on
    # all thyroid datasets — strip it for any zero-shot thyroid eval.
    "thyroid": {"tirads"},
    # birads_density / birads_assessment correlate with u2/u3 on VinDr/CMMD.
    # Currently null in context_json for most rows; strip defensively when
    # eval-leaning callers want a paranoid prompt.
    "breast_mammo": {"birads_density", "birads_assessment"},
    "breast_us": set(),
    "cervix_cytology": set(),  # bethesda would leak if populated; not in current splits.
    "cervix_via": set(),
    "skin": set(),
    "oral": set(),
    "blood": set(),
    "liver": set(),
}


# Read-AFTER-the-image context fields: these encode the answer (or its source)
# for the classification tasks and MUST NOT appear in the CONTEXT block when the
# model is supposed to infer them from the image. Stripped for U1-U4 (see
# render_prompt). This mirrors high-quality practice (MAIRA-2 / Med-PaLM M feed
# read-BEFORE priors — indication, technique, demographics — but never the
# read-out Findings/assessment); leaving these in is label leakage / shortcut
# learning. NOT applied to U7 (its descriptor target is handled in the U7
# redesign) nor to U0/U5/U6 (no CONTEXT block).
CLASSIFICATION_TASKS: set[str] = {"U1", "U2", "U3", "U4"}
POST_READ_CONTEXT_FIELDS: set[str] = {
    # imaging_findings — the visual read-out of the image (the model's job)
    "birads_assessment", "shape", "margin", "echo_pattern", "calcifications",
    "posterior_features", "color", "surface_texture", "lesion_description_text",
    # clinical_priors that are post-read assessments / diagnoses (NOT history):
    # tirads, bethesda, birads_density (assessment categories) and
    # weighted_condition_probs (the dermatologist differential = diagnosis).
    "tirads", "bethesda", "birads_density", "weighted_condition_probs",
    # site_anatomy findings that reveal presence/extent
    "lesion_count", "lesion_size_mm",
}


# ---------------------------------------------------------------------------
# Main render function.
# ---------------------------------------------------------------------------

def render_prompt(
    row: dict,
    task: str,
    template_idx: int | None = None,
    seed: int | None = None,
    image_token: str = "<image>",
    strip_leak_fields: bool = False,
    extra_strip_fields: set[str] | None = None,
    u4_options: list[str] | None = None,
    u4_unit: str | None = None,
    prompt_stem_version: int | str | None = None,
) -> str:
    """Render a VQA prompt for a manifest row and a given U-task.

    Args:
        row: dict-like (pd.Series.to_dict() works) with manifest fields
        task: one of U0..U7
        template_idx: deterministic template index; if None, random
        seed: optional seed for random template choice (reproducibility)
        image_token: placeholder for image embedding
        strip_leak_fields: if True, omit pillar-specific gold-leak fields
            from the CONTEXT block (see ``LEAK_FIELDS_BY_PILLAR``).
            Recommended for zero-shot eval; default False to preserve
            backward-compat for training pipelines.
        extra_strip_fields: optional additional context-leaf field names to
            strip on top of the per-pillar default.
        u4_options: for task U4 only — the per-report-unit candidate list
            (u4_axis.options(u4_unit)) to list in the prompt instead of the
            static PILLAR_CONFIG["u4_classes"]. Keeps the prompt's class set
            identical to the scorer's option set. None → legacy static list.
        u4_unit: for task U4 only — the row's report unit (`u4_eval_unit`). Needed
            only under prompt-stem version >= 2, where it selects the unit-specific
            stem (`u4_axis.U4_UNIT_STEMS_V2`) and the option surface forms.
        prompt_stem_version: None → `u4_axis.prompt_stem_version()` (env
            EDGE_PROMPT_STEM_VERSION, default 1). Version 1 is byte-identical to the
            pre-versioning renderer for every task; version 2 changes ONLY the U4
            TASK sentence (and the cervix Bethesda option surface forms). Asking for
            version >= 2 on a U4 row without a known unit raises, on purpose: a
            silent fallback to the colliding version-1 stem is the defect this
            version exists to remove.

    Returns:
        Rendered prompt string with 4 blocks (TASK / CONTEXT / IMAGE / ANSWER).
    """
    if task not in TEMPLATES:
        raise ValueError(f"Unknown task '{task}'. Valid: {list(TEMPLATES)}")
    stem_version = u4_axis.prompt_stem_version(prompt_stem_version)
    u4_stem = None
    if task == "U4" and stem_version >= 2:
        u4_stem = u4_axis.unit_stem(u4_unit, stem_version)
        if not u4_stem:
            raise ValueError(
                f"prompt-stem version {stem_version} requires a known U4 report unit; "
                f"got u4_unit={u4_unit!r} (known: {sorted(u4_axis.U4_UNIT_STEMS_V2)})")
        u4_options = list(u4_options) if u4_options else list(u4_axis.primary_options(u4_unit))
        u4_options = u4_axis.surface_options(u4_unit, u4_options, stem_version)
    pillar = _resolve_pillar_key(row)
    cfg = PILLAR_CONFIG[pillar]
    strip_set: set[str] = set(extra_strip_fields or [])
    if strip_leak_fields:
        strip_set |= LEAK_FIELDS_BY_PILLAR.get(pillar, set())
        # U1-U4: also strip every read-AFTER-the-image field so the answer (or its
        # source) cannot leak into the CONTEXT block. Keeps read-before priors
        # (demographics / site / device / history) intact.
        if task in CLASSIFICATION_TASKS:
            strip_set |= POST_READ_CONTEXT_FIELDS
    # Oral's free-text anatomical_site is polluted with read-after findings /
    # quality / diagnosis tokens (dental caries, out of focus, pigmentation, ...);
    # it is a low-value site prior and unsafe to render at any task. Drop it.
    if pillar == "oral":
        strip_set.add("anatomical_site")

    # Determine modality string (with optional image_type override for skin)
    modality_str = cfg["modality_default"]
    context_value = row.get("context_json")
    if pillar == "skin":
        # SLICE-3D's historical manifest mislabeled these lesion-centred 3D-TBP
        # crops as dermoscopy. Dataset identity takes precedence so frozen
        # manifests can be evaluated with the corrected acquisition wording.
        if str(row.get("dataset_id", "")).strip().upper() == "SLICE-3D":
            modality_str = cfg["modality_by_image_type"]["3d_total_body_photography_crop"]
        # Prefer the row-level acquisition modality. Fall back to image_type for
        # legacy rows, then to the pillar default.
        cj = context_value
        if cj and isinstance(cj, str):
            try:
                cj = json.loads(cj)
            except Exception:
                cj = None
        if isinstance(cj, dict) and str(row.get("dataset_id", "")).strip().upper() == "SLICE-3D":
            cj = dict(cj)
            cj["device"] = dict(cj.get("device", {}))
            cj["device"]["modality"] = "3d_total_body_photography_crop"
            cj["device"]["device_family"] = "3d_total_body_photography"
            context_value = cj
        elif isinstance(cj, dict):
            device = cj.get("device", {})
            it = device.get("modality") or device.get("image_type")
            if it and str(it).lower() in cfg.get("modality_by_image_type", {}):
                modality_str = cfg["modality_by_image_type"][str(it).lower()]

    # Site
    site = row.get("anatom_site")
    if _is_null(site):
        site = cfg["site_default"]

    # Render context block
    context_block = render_context_block(context_value,
                                          strip_context_fields=strip_set or None,
                                          prompt_stem_version=stem_version)

    # Pick template. Blood localization targets cells/blasts, not a generic
    # solid-organ lesion contour.
    if pillar == "blood" and task == "U5":
        template = (
            "IMAGE: {image}\n\n"
            "TASK: Locate the visible white blood cells or blast cells in this {modality}. "
            "Output the final answer in the format \"Final Answer: X\" where X is a JSON list of objects. "
            "Each object needs a \"box_2d\" key in [y0, x0, y1, x1] coordinates normalized to [0,1000] "
            "and a \"label\" key.\n\n"
            "CONTEXT:\n{context_block}\n\n"
            "ANSWER: Final Answer: ```json[{{\"box_2d\":[...],\"label\":\"white blood cell\"}}]```."
        )
    elif pillar == "blood" and task == "U6":
        template = (
            "IMAGE: {image}\n\n"
            "TASK: Localize the visible white blood cell or blast-cell mask in this {modality}. "
            "Return the tight mask box as Gemma-style object JSON with \"box_2d\" [y0, x0, y1, x1] "
            "normalized to [0,1000] and a \"label\" key.\n\n"
            "CONTEXT:\n{context_block}\n\n"
            "ANSWER: Final Answer: ```json[{{\"box_2d\":[...],\"label\":\"white blood cell\"}}]```."
        )
    elif u4_stem is not None:
        # Version-2 U4: one template, unit-specific stem (ignores template_idx/seed —
        # version 2 is defined as a single canonical template, like the frozen
        # template_idx=0 materialisation of version 1).
        template = U4_TEMPLATE_V2
    else:
        template = get_template(task, idx=template_idx, seed=seed)

    # Format
    return template.format(
        modality=modality_str,
        site=site,
        u2_action=cfg["u2_action"],
        u4_classes=(" / ".join(u4_options) if u4_options else cfg["u4_classes"]),
        u4_stem=(u4_stem.format(modality=modality_str) if u4_stem else ""),
        report_fields=cfg.get("report_fields", "diagnosis, recommended action"),
        context_block=context_block,
        image=image_token,
    )


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pandas as pd
    import sys
    pillar_arg = sys.argv[1] if len(sys.argv) > 1 else "skin"
    task_arg   = sys.argv[2] if len(sys.argv) > 2 else "U3"
    df = pd.read_parquet(f"Experiment/datasets/{pillar_arg}/manifest.parquet")
    # Try to find a row with non-null context_json (most rows have it)
    sample = df[df["context_json"].notna()].sample(1, random_state=42).iloc[0].to_dict()
    print("--- ROW ---")
    print(f"dataset_id={sample.get('dataset_id')}, cancer_type={sample.get('cancer_type')}, modality={sample.get('modality')}")
    print()
    print(f"--- PROMPT ({task_arg}, template idx=0) ---")
    print(render_prompt(sample, task=task_arg, template_idx=0))
    print()
    print(f"--- PROMPT ({task_arg}, random template, seed=7) ---")
    print(render_prompt(sample, task=task_arg, seed=7))
