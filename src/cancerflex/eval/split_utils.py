"""Shared deterministic train/test split assignment — the single source of truth.

Writes ``split``, ``subsplit`` and ``ood_axis`` directly into each pillar's
``manifest.parquet``. ``eval_iterator`` reads these columns straight from the
parquet; there is no separate CSV split layer.

Split rule, applied per row:

1. ``_force_split`` wins when set — this carries source-native splits
   (``train`` / ``val`` / ``test_id``) and forced out-of-distribution cohorts
   (``test_ood``) that the builders mark in their readers.
2. otherwise a deterministic hash decides train vs test_id::

       test_id  ⟺  int(md5(str(patient_id)), 16) % 1000 < TEST_ID_THRESH[pillar]

   (default threshold 100 ⇒ ~10%). When ``patient_id`` is null/empty the hash
   falls back to ``image_path`` (per-image, still deterministic).

This is **growth-stable**: a pure fixed threshold on a per-row hash means adding
rows / patients / whole datasets never moves an existing row's assignment — there
are no folds, quantiles or population-wide shuffles to drift. (The old
``StratifiedGroupKFold`` first-fold approach was row-order dependent and so not
reproducible after the manifest was regenerated.)

No ``val`` is ever synthesized. ``val`` appears only where a source dataset
provides it, carried through ``_force_split``.

``OOD_AXIS`` below is **label-only**: it maps a dataset to its OOD-axis tag for
rows that are already ``test_ood``. It does NOT itself force test_ood membership
(that is per-row via ``_force_split``), so partially-OOD datasets — e.g. blood
``Raabin-WBC`` where only the TestB scanner-shift subset is OOD — are handled
correctly.
"""
from __future__ import annotations

import hashlib

import pandas as pd

# Dataset → OOD-axis label. LABEL ONLY (see module docstring): test_ood
# membership is decided per-row by `_force_split`, this just tags the axis.
OOD_AXIS: dict[str, dict[str, str]] = {
    "skin": {
        "DDI": "demographic_darkskin",
        "PH2": "geographic_portugal",
        "SCIN": "device_smartphone_demographic",
        "MIDAS": "demographic_fitz_race",
    },
    "breast": {
        "CMMD": "geographic_taxonomy_molecular",
        "BUS-CoT": "source_device_histology_bus",
        "Mammo-MX": "geographic_mexico_birads_density_dose",
    },
    "cervix": {
        "RIVA": "geographic_institution_brazil_multiexpert",
        "MobileODT": "geographic_taxonomy_cervix_type",
        "AnnoCerv": "geographic_colposcopy_romania_feature_masks",
        "IARCImageBankVIA": "geographic_iarc_via_histology",
    },
    "oral": {
        "Piyarathne": "geographic_srilanka",
        "MODID": "acquisition_multispectral",
    },
    "thyroid": {
        "DDTI": "geographic_colombia_device_era",
        "StanfordCineclip": "geographic_stanford_us_histopath",
        "Hou": "cohort_jiaxing_batch2_histopath",
        "SegThy": "healthy_volunteer_no_nodule_us",
    },
    "liver": {},  # all v1.5 candidate
    "blood": {
        "Raabin-WBC": "source_geographic_iran",
        "BCCD": "task_taxonomy",  # detection-only
        "AML-Cytomorphology_LMU": "source_reference_lab_munich_aml",
        "RV-PBS": "source_smartphone_microscope_instance_masks",
        "LLD": "source_multi_camera_leukemiaattri",
    },
}

# Per-pillar test_id hash threshold out of 1000 (default 100 == 10%).
# Large pillars are budgeted down to keep test_id near ~10K; skin: 18 ≈ 1.8%.
# Still a pure fixed threshold ⇒ still growth-stable.
TEST_ID_THRESH: dict[str, int] = {"skin": 18}

_EMPTY_FORCE = {"", "none", "nan"}


def hash_is_test_id(key, *, mod: int = 1000, thresh: int = 100) -> bool:
    """Deterministic, growth-stable test_id membership for a single hash key."""
    return int(hashlib.md5(str(key).encode("utf-8")).hexdigest(), 16) % mod < thresh


def _subsplit(split: str, dataset_id, forced: bool, pillar: str | None) -> str:
    if pillar == "liver":
        return "v1_5_candidate"
    if split == "test_ood":
        return "test_ood_detection_only" if dataset_id == "BCCD" else "v1_test_ood_external"
    if split == "test_id":
        return f"v1_test_id_native_{dataset_id}" if forced else "v1_test_id_holdout"
    if split == "val":
        return f"v1_val_native_{dataset_id}"
    return "v1_train"


def assign_splits(
    df: pd.DataFrame,
    *,
    pillar: str | None = None,
    group_col: str = "patient_id",
    force_col: str = "_force_split",
    fallback_col: str = "image_path",
) -> pd.DataFrame:
    """Assign ``split`` / ``subsplit`` / ``ood_axis`` columns (see module docstring).

    Parameters
    ----------
    df : manifest with ``dataset_id``, ``group_col`` and ``fallback_col`` columns,
        optionally a per-row ``force_col`` of native/forced split labels.
    pillar : pillar name, used for the ``TEST_ID_THRESH`` and ``OOD_AXIS`` lookups.
    """
    out = df.copy().reset_index(drop=True)
    thresh = TEST_ID_THRESH.get(pillar, 100)

    # Hash key: patient_id when present and non-empty, else fall back to image_path.
    if group_col in out.columns:
        pid = out[group_col].tolist()
    else:
        pid = [None] * len(out)
    fb = out[fallback_col].astype(str).tolist()
    keys = [
        (str(p).strip() if (p is not None and not pd.isna(p) and str(p).strip()) else fb[i])
        for i, p in enumerate(pid)
    ]
    default = pd.Series(
        ["test_id" if hash_is_test_id(k, thresh=thresh) else "train" for k in keys],
        index=out.index,
        dtype=object,
    )

    if force_col in out.columns:
        forced_vals = out[force_col]
        forced_mask = forced_vals.map(lambda v: str(v).strip().lower() not in _EMPTY_FORCE)
        forced_vals = forced_vals.astype(object).where(forced_mask, "")
    else:
        forced_mask = pd.Series([False] * len(out), index=out.index)
        forced_vals = pd.Series([""] * len(out), index=out.index, dtype=object)

    out["split"] = default.where(~forced_mask, forced_vals.map(lambda v: str(v).strip())).values

    # ood_axis: label-only lookup, only for rows that ended up test_ood.
    axis_map = (OOD_AXIS.get(pillar) or {}) if pillar else {}
    out["ood_axis"] = ""
    ood_rows = out["split"] == "test_ood"
    if ood_rows.any():
        out.loc[ood_rows, "ood_axis"] = (
            out.loc[ood_rows, "dataset_id"].map(axis_map).fillna("")
        )

    out["subsplit"] = [
        _subsplit(s, d, bool(f), pillar)
        for s, d, f in zip(out["split"], out["dataset_id"], forced_mask)
    ]

    if force_col in out.columns:
        out = out.drop(columns=[force_col])
    return out
