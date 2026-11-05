"""Shared manifest schema contract for active v1 pillars.

Builders should call ``normalize_manifest`` immediately before writing
``manifest.parquet``. The function keeps the semantic source columns intact,
adds nullable common columns when they are absent, validates the required core
contract, and writes columns in a stable order.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd


ACTIVE_V1_PILLARS = ("skin", "breast", "cervix", "oral", "thyroid", "blood")

REQUIRED_CORE_COLUMNS = [
    "isic_id",
    "dataset_id",
    "cancer_type",
    "modality",
    "device_family",
    "role",
    "patient_id",
    "image_path",
    "u1_label",
    "u2_label",
    "u3_label",
    "u4_label",
    "u5_label",
    "bbox",
    "mask_path",
    "u6_label",
    "age",
    "sex",
    "anatom_site",
    "split",
    "ood_axis",
    "subsplit",
    "context_json",
    "u4_label_raw",
    "u4_unit",
]

COMMON_NULLABLE_COLUMNS = [
    "has_region_annotation",
    "has_lesion_annotation",
    "corpus_layer",
    "split_curation_note",
    "perceptual_duplicate_group_id",
    "perceptual_duplicate_group_size",
    "exact_duplicate_group_id",
    "exact_duplicate_group_size",
]


def _missing(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    present = set(df.columns)
    return [col for col in columns if col not in present]


def validate_manifest_contract(df: pd.DataFrame, pillar: str) -> None:
    """Raise if an active-v1 manifest is missing required core columns."""
    if pillar not in ACTIVE_V1_PILLARS:
        raise ValueError(
            f"manifest contract applies only to active v1 pillars: {ACTIVE_V1_PILLARS}; got {pillar!r}"
        )
    missing = _missing(df, REQUIRED_CORE_COLUMNS)
    if missing:
        raise ValueError(f"{pillar} manifest missing required core columns: {missing}")


def normalize_manifest(df: pd.DataFrame, pillar: str) -> pd.DataFrame:
    """Return a manifest with the active-v1 common schema and stable column order."""
    validate_manifest_contract(df, pillar)
    out = df.copy()
    for col in COMMON_NULLABLE_COLUMNS:
        if col not in out.columns:
            out[col] = None

    ordered = [*REQUIRED_CORE_COLUMNS, *COMMON_NULLABLE_COLUMNS]
    extras = [col for col in out.columns if col not in set(ordered)]
    return out.loc[:, [*ordered, *extras]]
