"""Central, auditable definition of the U4 (fine-grained subtype) benchmark.

The old design pooled, within one pillar's "U4", datasets that classify along
*incompatible axes* (breast BI-RADS vs molecular) plus binary-only datasets
(≡ U3) and pseudo-labels (segmentation_only). This module makes U4 honest.

Design (owner decisions, updated 2026-06-11):
  - A dataset NOT in `REPORT_UNIT` below has its `u4_label` nulled → removed from
    U4 (it is binary-only → use U3, or a pseudo-label → use U2/U5/U6).
  - Each kept dataset maps to a **report unit** = the group within which the U4
    metric is comparable. Multiple datasets share a unit ONLY when they share one
    clinical taxonomy (skin dermoscopy oncology; breast BI-RADS; oral oncology).
    Where datasets carry the same axis at *incompatible granularity* (thyroid
    DDTI 4a/b/c vs Stanford scalar; blood APL maturation stages vs Raabin coarse),
    each dataset is its **own** unit — native granularity kept, comparable only
    within that dataset (owner's explicit choice over collapsing for cross-dataset
    pooling).
  - `UNIT_OPTIONS[unit]` = the native/fine option set for audit and auxiliary use.
  - `PRIMARY_UNIT_OPTIONS[unit]` = the default headline-eval option set. For blood,
    primary U4 eval uses train-supported coarse classes: LLD remains a 5-way
    leukemia subtype task, while WBC morphology datasets share one coarse unit.
  - A row is U4-eligible iff its dataset has a unit AND its gold harmonizes onto
    that unit's option set.

Builders call `apply(df, pillar)`: null dropped u4_label, stamp `u4_unit`.
Builders/eval call `attach_primary_eval_columns(df, pillar)`: add
`u4_eval_label`, `u4_eval_unit`, `u4_eval_tier`, and `u4_eval_options_json`.
Default eval/training uses those primary columns; `u4_label`/`u4_unit` stay as
fine/native audit fields.
"""
from __future__ import annotations
import json
import math
import os
import re

# dataset → report unit (the comparable scoring/reporting group). Datasets absent
# here are dropped from U4 (binary-only or pseudo-label).
REPORT_UNIT: dict[str, dict[str, str]] = {
    "skin": {ds: "skin:derm_subtype" for ds in (  # shared dermoscopy oncology taxonomy
        "BCN-20K", "DDI", "DERM12345", "Derm7pt", "DermNet", "HAM10000", "HIBA",
        "ISIC2020", "MIDAS", "MILK10K", "PAD-UFES-20", "PH2", "SCIN", "SLICE-3D")},
    "breast": {"VinDr-Mammo": "breast:birads", "INbreast": "breast:birads", "BCMID": "breast:birads", "Mammo-MX": "breast:birads",  # shared BI-RADS
               "CMMD": "breast:molecular"},                                   # single dataset
    "cervix": {"SIPaKMeD": "cervix:cytology", "RIVA": "cervix:bethesda",
               "Cervix93": "cervix:bethesda_coarse"},                         # MobileODT dropped (anatomy≠pathology)
    "oral": {ds: "oral:oral_subtype" for ds in ("CODE", "SMART-OM", "MODID", "Piyarathne",
                                                 "RayHealth", "RayGrade", "JavedMOD")},
    # per-dataset (incompatible native granularity — owner: keep native, report per-dataset)
    "thyroid": {"DDTI": "thyroid:tirads:DDTI", "StanfordCineclip": "thyroid:tirads:Stanford"},
    "blood": {"APL-Madrid": "blood:wbc:APL", "Raabin-WBC": "blood:wbc:Raabin",
              "ALL-IDB-Subtypes": "blood:wbc:ALL-IDB",
              "AML-Cytomorphology_LMU": "blood:wbc:AML-Cytomorphology_LMU",
              "MLL23": "blood:wbc:MLL23",
              "PBC-Barcelona": "blood:wbc:PBC-Acevedo",
              "WBCAttPlus": "blood:wbc:PBC-Acevedo",
              "RV-PBS": "blood:wbc:RV-PBS",
              "LLD": "blood:leukemia:LLD"},
}

# unit → candidate option set scored at inference. Native granularity preserved.
UNIT_OPTIONS: dict[str, list[str]] = {
    "skin:derm_subtype": ["melanoma", "basal cell carcinoma", "squamous cell carcinoma",
        "nevus", "seborrheic keratosis", "actinic keratosis", "dermatofibroma", "other"],
    "breast:birads": ["BIRADS 1", "BIRADS 2", "BIRADS 3", "BIRADS 4", "BIRADS 5", "BIRADS 6"],
    # molecular subtype is a property of malignant tumours only → benign + unknown
    # subtype are NOT this task (covered by U3); kept native to CMMD.
    "breast:molecular": ["Luminal A", "Luminal B", "HER2-enriched", "TNBC"],
    "cervix:cytology": ["superficial-intermediate", "parabasal", "metaplastic",
        "koilocytotic", "dyskeratotic"],
    "cervix:bethesda": ["NILM", "ASCUS", "ASC-H", "LSIL", "HSIL", "SCC"],
    "cervix:bethesda_coarse": ["negative", "LSIL", "high_grade_or_worse"],
    "oral:oral_subtype": ["oral cancer", "oral potentially malignant disorder (OPMD)",
        "leukoplakia", "erythroplakia", "lichen planus", "oral submucous fibrosis (OSMF)",
        "healthy", "benign variation"],
    # thyroid: native per-dataset granularity (DDTI sub-levels kept; Stanford scalar)
    "thyroid:tirads:DDTI": ["TI-RADS 2", "TI-RADS 3", "TI-RADS 4a", "TI-RADS 4b",
        "TI-RADS 4c", "TI-RADS 5"],
    "thyroid:tirads:Stanford": ["TI-RADS 1", "TI-RADS 2", "TI-RADS 3", "TI-RADS 4", "TI-RADS 5"],
    # blood: native per-dataset granularity (APL maturation stages kept) — frozen
    # from APL-Madrid's distinct gold vocab (identity-matched).
    "blood:wbc:APL": ["Smudge cells", "Blast, no lineage spec", "Lymphocyte", "Monocyte",
        "Segmented neutrophils", "Lymphocyte, variant", "Erythroblast", "Promyelocyte",
        "Eosinophils", "Band neutrophils", "Myelocyte", "Giant thrombocyte", "Basophil",
        "Plasma cells", "Metamyelocyte", "Thrombocyte aggregation", "Promonocyte"],
    "blood:wbc:Raabin": ["Neutrophil", "Lymphocyte", "Monocyte", "Eosinophil", "Basophil"],
    "blood:wbc:ALL-IDB": ["ALL_L1", "ALL_L2", "ALL_L3"],
    # AML-Cytomorphology_LMU and MLL23 are source-specific morphology axes:
    # keep native classes rather than collapsing near-synonyms across releases.
    "blood:wbc:AML-Cytomorphology_LMU": ["Basophil", "Erythroblast", "Eosinophil",
        "Smudge cell", "Lymphocyte atypical", "Lymphocyte typical", "Metamyelocyte",
        "Monoblast", "Monocyte", "Myelocyte", "Myeloblast", "Neutrophil band",
        "Neutrophil segmented", "Promyelocyte bilobled", "Promyelocyte"],
    "blood:wbc:MLL23": ["Basophil", "Eosinophil", "Hairy cell", "Lymphocyte typical",
        "Large granular lymphocyte", "Neoplastic lymphocyte", "Reactive lymphocyte",
        "Metamyelocyte", "Monocyte", "Myeloblast", "Myelocyte", "Neutrophil band",
        "Neutrophil segmented", "Normoblast", "Plasma cell", "Promyelocyte",
        "Atypical promyelocyte", "Smudge cell"],
    "blood:wbc:PBC-Acevedo": ["Basophil", "Eosinophil", "Erythroblast",
        "Immature granulocyte", "Lymphocyte", "Monocyte", "Neutrophil", "Platelet"],
    "blood:wbc:RV-PBS": ["Band cell", "Basophil", "Blast cell", "Eosinophil",
        "Lymphocyte", "Metamyelocyte", "Monocyte", "Myelocyte", "Neutrophil",
        "Promyelocyte"],
    "blood:leukemia:LLD": ["ALL", "AML", "CML", "CLL", "APML"],
}

# oral conditions that are NOT neoplastic/pre-neoplastic → excluded from U4 (A6);
# they remain valid for U1/U2.
_ORAL_NEOPLASTIC = re.compile(
    r"cancer|oscc|carcinoma|malign|opmd|leukoplak|leukplak|erythroplak|lichen|olp|"
    r"osmf|osf|submucous|verruc|dysplasia|precancer", re.I)
_ORAL_BASELINE = re.compile(r"normal|healthy|variation", re.I)


def report_unit(pillar: str, dataset: str) -> str | None:
    return REPORT_UNIT.get(pillar, {}).get(dataset)


def options(unit: str) -> list[str]:
    return UNIT_OPTIONS.get(unit, [])


# Primary-eval policy.
#
# Fine-grained U4 stays available in `u4_label` / `u4_unit`, but headline eval
# should not score blood as a long-tail native-vocabulary benchmark. Blood uses
# one disease-subtype axis (LLD) plus one shared coarse WBC morphology axis.
# Rows without a primary-eval label remain available as auxiliary/fine data but
# are not included in default U4 eval metrics.
BLOOD_PRIMARY_WBC_UNIT = "blood:wbc:morphology_primary"
PRIMARY_UNIT_OPTIONS: dict[str, list[str]] = {
    **UNIT_OPTIONS,
    BLOOD_PRIMARY_WBC_UNIT: [
        "blast",
        "promyelocyte",
        "immature granulocyte",
        "neutrophil lineage",
        "lymphocyte lineage",
        "monocyte lineage",
        "eosinophil",
        "basophil",
        "erythroid precursor",
        "plasma cell",
        "platelet/thrombocyte",
        "smudge cell",
    ],
}
BLOOD_PRIMARY_WBC_SOURCE_UNITS = {
    "blood:wbc:APL",
    "blood:wbc:Raabin",
    "blood:wbc:AML-Cytomorphology_LMU",
    "blood:wbc:MLL23",
    "blood:wbc:PBC-Acevedo",
    "blood:wbc:RV-PBS",
}
BLOOD_AUXILIARY_U4_UNITS = {
    # Tiny subtype axis in current eval (L1/L2/L3); keep as auxiliary.
    "blood:wbc:ALL-IDB",
}
BLOOD_U4_PRIMARY_MIN_SUPPORT = 20
PRIMARY_EXCLUDED_LABELS: dict[str, set[str]] = {
    "skin": {"other"},
}

# Labels demoted to auxiliary ONLY under prompt-stem version >= 2, because the version-1
# corpus and every number published from it were scored with them in the primary set.
#
# oral `OPMD` is the residual bucket for "pre-neoplastic, subtype not stated" (u4_axis's oral
# normaliser sends any unmatched neoplastic term there). It is therefore the PARENT of four of
# its own siblings — leukoplakia, erythroplakia, lichen planus and OSMF are all OPMDs — and the
# option list was not mutually exclusive. Measured cost on the deployed checkpoint: the external
# oral U4 gold set contains **zero** OPMD rows (OSMF 86 / oral cancer 56 / leukoplakia 40 /
# healthy 37) while the model answered OPMD on 20 of 219, i.e. 9.1% of the cell was wrong by
# construction and the "wrong" answers were clinically defensible. Demoting the parent makes the
# remaining seven options mutually exclusive. The external gold set is untouched by this (it had
# no OPMD rows); it removes 158 train and 11 test_id rows from the primary tier, exactly as
# cervix ASC-US is already handled.
PRIMARY_EXCLUDED_LABELS_V2: dict[str, set[str]] = {
    "oral": {"oral potentially malignant disorder (OPMD)"},
}


def _excluded_labels(pillar: str, version: int | str | None = None) -> set[str]:
    out = set(PRIMARY_EXCLUDED_LABELS.get(pillar, set()))
    if prompt_stem_version(version) >= 2:
        out |= PRIMARY_EXCLUDED_LABELS_V2.get(pillar, set())
    return out


def primary_options(unit: str, version: int | str | None = None) -> list[str]:
    """Headline option set for `unit`. Under version >= 2, labels demoted by
    PRIMARY_EXCLUDED_LABELS_V2 are dropped so the emitted option list matches the
    label space the gold side will actually score."""
    opts = PRIMARY_UNIT_OPTIONS.get(unit, [])
    pillar = unit.split(":", 1)[0] if unit else ""
    drop = _excluded_labels(pillar, version)
    return [o for o in opts if o not in drop] if drop else opts


def primary_unit(pillar: str, unit: str | None) -> str | None:
    if not unit:
        return None
    if pillar == "blood":
        if unit == "blood:leukemia:LLD":
            return unit
        if unit in BLOOD_PRIMARY_WBC_SOURCE_UNITS:
            return BLOOD_PRIMARY_WBC_UNIT
        return None
    if pillar == "cervix":
        if unit in {"cervix:bethesda", "cervix:bethesda_coarse"}:
            return "cervix:bethesda_coarse"
        return unit
    return unit


def _cervix_bethesda_coarse_label(label: str | None) -> str | None:
    n = _norm(label)
    if not n:
        return None
    if n in {"negative", "nilm", "normal"}:
        return "negative"
    if n == "lsil":
        return "LSIL"
    if n in {"hsil", "asc h", "asch", "scc", "high grade or worse"} or "squamous cell" in n:
        return "high_grade_or_worse"
    return None


def _blood_primary_wbc_label(label: str | None) -> str | None:
    n = _norm(label)
    if not n:
        return None
    if n in {"blast cell", "blast, no lineage spec", "blast no lineage spec", "myeloblast", "monoblast"}:
        return "blast"
    if "promyelocyte" in n:
        return "promyelocyte"
    if n in {"myelocyte", "metamyelocyte", "immature granulocyte"}:
        return "immature granulocyte"
    if n in {
        "neutrophil",
        "segmented neutrophils",
        "neutrophil segmented",
        "neutrophil band",
        "band neutrophils",
        "band cell",
    }:
        return "neutrophil lineage"
    if n in {
        "lymphocyte",
        "lymphocyte typical",
        "lymphocyte atypical",
        "lymphocyte variant",
        "lymphocyte, variant",
        "hairy cell",
        "large granular lymphocyte",
        "neoplastic lymphocyte",
        "reactive lymphocyte",
    }:
        return "lymphocyte lineage"
    if n in {"monocyte", "promonocyte"}:
        return "monocyte lineage"
    if n in {"eosinophil", "eosinophils"}:
        return "eosinophil"
    if n == "basophil":
        return "basophil"
    if n in {"erythroblast", "normoblast"}:
        return "erythroid precursor"
    if n in {"plasma cell", "plasma cells"}:
        return "plasma cell"
    if n in {"platelet", "giant thrombocyte", "thrombocyte aggregation"}:
        return "platelet/thrombocyte"
    if n in {"smudge cell", "smudge cells"}:
        return "smudge cell"
    return None


def primary_label(pillar: str, unit: str | None, label: str | None,
                  version: int | str | None = None) -> str | None:
    if label in _excluded_labels(pillar, version):
        return None
    punit = primary_unit(pillar, unit)
    if punit is None:
        return None
    if pillar == "blood":
        if unit == "blood:leukemia:LLD":
            return label if label in UNIT_OPTIONS["blood:leukemia:LLD"] else None
        if unit in BLOOD_PRIMARY_WBC_SOURCE_UNITS:
            return _blood_primary_wbc_label(label)
        return None
    if pillar == "cervix" and unit in {"cervix:bethesda", "cervix:bethesda_coarse"}:
        return _cervix_bethesda_coarse_label(label)
    opts = PRIMARY_UNIT_OPTIONS.get(punit, [])
    return label if label in opts else None


# Native option names that the GOLD normaliser folds into a coarsened eval unit.
# DERIVED, never hand-written: each fine-grained option is pushed through the same
# `primary_label` the gold side uses, and whatever bucket it lands in becomes an
# accepted surface form for that bucket.
#
# WHY THIS EXISTS. The gold side coarsens with a clinical synonym table
# (`_cervix_bethesda_coarse_label`: HSIL / ASC-H / SCC -> high_grade_or_worse), but the
# prediction side used to match the bucket NAME as a literal substring. A model that
# answered "HSIL" — clinically correct, and a synonym the gold table itself accepts —
# scored `<unparsed>`, i.e. wrong. Both sides must read the same vocabulary; this
# returns it.
#
# SCOPE CORRECTION (2026-09-16, raw text persisted in
# eval_results/cervix_u4_rawtext_20260916/cervix_U4.csv): the 253/851 external
# `<unparsed>` rows are NOT Bethesda synonyms. 240 of them are SIPaKMeD cell-type
# names (metaplastic 208, superficial-intermediate 20, parabasal 10, dyskeratotic 2)
# and none contains HSIL / ASC-H / SCC / NILM, so this map recovers 0 of the 253. The
# cause is the shared question stem across the two cervix units — see
# U4_UNIT_STEMS_V2 below and docs/methodology/prompt_design_audit_20260916.md. This
# map stays correct for what it does (accept clinical synonyms of a coarsened bucket).
#
# Scope is per-unit ON PURPOSE. cervix folds "squamous cell" into high grade, while skin
# has "squamous cell carcinoma" as its own option — applying one unit's synonyms globally
# would corrupt the other. Units with no coarsening return {} and behave exactly as before.
NATIVE_SOURCE_UNIT: dict[str, list[str]] = {
    # coarsened eval unit -> the native unit(s) whose option names may be uttered instead
    "cervix:bethesda_coarse": ["cervix:bethesda", "cervix:bethesda_coarse"],
}


def eval_surface_forms(pillar: str, eval_unit: str | None,
                       version: int | str | None = None) -> dict[str, str]:
    """{normalised surface form -> canonical eval option} for a coarsened unit.

    Keys are `_norm`-ed (underscores/hyphens collapsed to spaces) so that a generated
    "ASC-H" matches the gold table's "asc h". An option that the gold normaliser refuses
    to map (cervix ASCUS -> None) is deliberately absent: it stays unparsed on both sides.

    `version` (prompt-stem version, see `prompt_stem_version`): under version >= 2 the
    version-2 surface forms of `OPTION_SURFACE_FORMS_V2` are added as accepted spellings
    of their canonical option, so a model trained on a version-2 corpus that answers
    "HSIL or worse (...)" scores as `high_grade_or_worse`. Under version 1 (the default)
    the returned map is exactly what it was before versioning existed.
    """
    v = prompt_stem_version(version)
    # Version 1 is the FROZEN scoring path and must stay byte-identical to what produced the
    # published numbers, which used a literal match against the option strings alone. The clinical
    # synonym map below (NATIVE_SOURCE_UNIT: "HSIL"/"ASC-H"/"SCC" -> high_grade_or_worse) was added
    # on 2026-09-16 and, until this guard, was active at version 1 as well — an undeclared change
    # to the frozen path. It happened not to move any number: on all 1,376 persisted cervix U4 raw
    # generations the literal and synonym matchers agree 1,376/1,376, because the deployed model
    # never uttered a bare "HSIL". "Did not matter on this data" is not the same guarantee as
    # "cannot matter", so the map is now version-gated like everything else. Found by the
    # adversarial review, docs/methodology/review_20260916.md D7.
    if v < 2:
        return {}
    v2_forms = OPTION_SURFACE_FORMS_V2.get(eval_unit or "", {}) if v >= 2 else {}
    if not eval_unit or (eval_unit not in NATIVE_SOURCE_UNIT and not v2_forms):
        # Not a coarsened unit: return {} so the caller keeps the original literal
        # matcher untouched. This is what makes "every other pillar is unchanged"
        # a guarantee rather than a hope.
        return {}
    out: dict[str, str] = {}
    for native in NATIVE_SOURCE_UNIT.get(eval_unit, []):
        for opt in UNIT_OPTIONS.get(native, []):
            canon = primary_label(pillar, native, opt)
            if canon:
                out[_norm(opt)] = canon
    # the canonical bucket names themselves, normalised (so "high_grade_or_worse"
    # still matches when the model echoes the prompt verbatim)
    for opt in PRIMARY_UNIT_OPTIONS.get(eval_unit, UNIT_OPTIONS.get(eval_unit, [])):
        out[_norm(opt)] = opt
    # version-2 surface forms (full string) -> canonical. The abbreviations inside them
    # (NILM / LSIL / HSIL / ASC-H) are already keys via NATIVE_SOURCE_UNIT for cervix.
    for canon, surface in v2_forms.items():
        out[_norm(surface)] = canon
    return out


# ---------------------------------------------------------------------------
# Prompt-stem versioning (prompt-design audit 2026-09-16,
# docs/methodology/prompt_design_audit_20260916.md).
#
# WHY. Version-1 U4 prompts render ONE question stem per pillar
# ("Classify the most likely diagnosis for this <modality> among: <options>.")
# for every report unit of that pillar, so two units that ask semantically
# different questions share a byte-identical stem and only the option list
# tells them apart. Measured on the de-leaked corpus: cervix `cervix:cytology`
# (SIPaKMeD cell morphology, 4,514 train rows) and `cervix:bethesda_coarse`
# (Bethesda severity, 69 train rows) collide, and the model learned to answer
# the stem and ignore the option list: 253/851 external severity rows were
# answered with a SIPaKMeD cell-type name (`<unparsed>`, scored wrong). Blood
# `blood:wbc:morphology_primary` / `blood:leukemia:LLD` collide in the same
# way (75,590 vs 14,969 train rows) but happen to be rescued by a
# `scanner=LLD ...` token in the CONTEXT block.
#
# WHAT. Version 2 gives every report unit its OWN stem that names the label
# space in the question itself, and (for the one underscore-coded option
# string in the benchmark, `high_grade_or_worse`) a clinical surface form.
# The canonical option strings in UNIT_OPTIONS / PRIMARY_UNIT_OPTIONS are
# UNCHANGED: `u4_eval_label`, `predicted_class`, `classes_json` and every
# metric stay keyed on the canonical names; only the text SHOWN to the model
# (prompt option list + training answer) changes, and `eval_surface_forms`
# maps the surface form back to the canonical name for scoring.
#
# GATING. Nothing changes unless a caller asks for version >= 2 explicitly or
# sets EDGE_PROMPT_STEM_VERSION=2. The default is 1 and reproduces the frozen
# `v1_final_deleaked` prompts byte-for-byte, so every checkpoint, shard and
# eval in flight is unaffected. A version-2 corpus must be built into a NEW
# directory and needs a RETRAIN before the train-side fix can matter.
# ---------------------------------------------------------------------------
PROMPT_STEM_VERSION_DEFAULT = 1
PROMPT_STEM_VERSION_ENV = "EDGE_PROMPT_STEM_VERSION"


def prompt_stem_version(explicit: int | str | None = None) -> int:
    """Resolve the prompt-stem version: explicit argument > env var > default 1."""
    if explicit is not None:
        return int(explicit)
    raw = os.environ.get(PROMPT_STEM_VERSION_ENV)
    return int(raw) if raw not in (None, "") else PROMPT_STEM_VERSION_DEFAULT


# unit -> version-2 question stem. `{modality}` is filled by the renderer with the
# same per-pillar modality string version 1 uses. Each stem names the label
# space (what KIND of answer is wanted) so the question is unambiguous even if
# the option list were dropped. Written in clinical language; the trailing
# parenthetical states what the question is NOT, for the units that collide.
U4_UNIT_STEMS_V2: dict[str, str] = {
    "skin:derm_subtype":
        "Give the specific dermatological diagnosis of the lesion in this {modality} "
        "(the skin tumour subtype, not just benign versus malignant).",
    "breast:birads":
        "Assign the BI-RADS assessment category to the finding in this {modality}.",
    "breast:molecular":
        "Determine the molecular subtype of the malignant breast tumour in this {modality}.",
    "cervix:cytology":
        "Identify the predominant cell type in this {modality} by its cell morphology "
        "(a cytomorphology call, not a Bethesda severity grade).",
    "cervix:bethesda":
        "Report the Bethesda System diagnostic category for this {modality}.",
    "cervix:bethesda_coarse":
        "Grade the severity of the squamous lesion in this {modality} using the Bethesda "
        "System (a severity grade, not a cell-type call).",
    "oral:oral_subtype":
        "Name the oral mucosal condition shown in this {modality} (oral cancer, a "
        "potentially malignant disorder, or a non-neoplastic finding).",
    "thyroid:tirads:DDTI":
        "Assign the TI-RADS risk category, including the 4a/4b/4c sub-levels, to the "
        "thyroid nodule in this {modality}.",
    "thyroid:tirads:Stanford":
        "Assign the TI-RADS risk category (level 1 to 5) to the thyroid nodule in this "
        "{modality}.",
    BLOOD_PRIMARY_WBC_UNIT:
        "Identify the type of the white blood cell shown in this {modality} by its "
        "morphology (a cell-lineage call, not a leukemia diagnosis).",
    "blood:leukemia:LLD":
        "Identify the leukemia type suggested by this {modality} (a haematological "
        "diagnosis, not a single-cell morphology call).",
    "blood:wbc:ALL-IDB":
        "Identify the FAB morphological subtype (L1, L2 or L3) of the acute "
        "lymphoblastic leukaemia cells in this {modality}.",
    # native per-dataset WBC morphology units (never rendered as primary; kept so a
    # fine/native corpus can be built without a KeyError)
    **{u: "Identify the type of the white blood cell shown in this {modality} by its "
          "morphology (a cell-lineage call, not a leukemia diagnosis)."
       for u in ("blood:wbc:APL", "blood:wbc:Raabin", "blood:wbc:AML-Cytomorphology_LMU",
                 "blood:wbc:MLL23", "blood:wbc:PBC-Acevedo", "blood:wbc:RV-PBS")},
}


# unit -> {canonical option -> surface form shown to the model} under version 2.
# Only options whose canonical string is NOT natural clinical language are
# listed; every other option is shown verbatim. Keep this table SMALL: each entry
# changes the training target and the first-token id used for the OvO companion.
OPTION_SURFACE_FORMS_V2: dict[str, dict[str, str]] = {
    "cervix:bethesda_coarse": {
        # `high_grade_or_worse` is the only underscore-coded option in the U4
        # benchmark; the model echoed it verbatim on 33/851 external rows and
        # never produced HSIL. Bethesda abbreviations lead so the first
        # generated token differs across the three options (NIL / LS / HS).
        "negative": "NILM (negative for intraepithelial lesion or malignancy)",
        "LSIL": "LSIL (low-grade squamous intraepithelial lesion)",
        "high_grade_or_worse": "HSIL or worse (high-grade squamous intraepithelial "
                               "lesion, ASC-H or carcinoma)",
    },
}


def unit_stem(unit: str | None, version: int | str | None = None) -> str | None:
    """Version-2 stem for `unit` (with `{modality}` unfilled), or None under
    version 1 / for an unknown unit."""
    if prompt_stem_version(version) < 2 or not unit:
        return None
    return U4_UNIT_STEMS_V2.get(unit)


def surface_form(unit: str | None, canonical: str, version: int | str | None = None) -> str:
    """Text shown to the model for a canonical option (identity under version 1)."""
    if prompt_stem_version(version) < 2 or not unit:
        return canonical
    return OPTION_SURFACE_FORMS_V2.get(unit, {}).get(canonical, canonical)


def surface_options(unit: str | None, options: list[str],
                    version: int | str | None = None) -> list[str]:
    """Map a canonical option list to its rendered surface forms, order preserved."""
    return [surface_form(unit, o, version) for o in options]


def attach_primary_eval_columns(df, pillar: str, min_support: int | None = None):
    """Add explicit primary-eval U4 columns.

    `u4_eval_label`/`u4_eval_unit` are the labels and units used by default
    evaluation. `u4_label`/`u4_unit` remain the fine/native labels. For blood,
    primary eval also drops split-local classes with support below
    `BLOOD_U4_PRIMARY_MIN_SUPPORT`, so rare native tails cannot accidentally
    enter headline metrics.
    """
    out = df.copy()
    if "u4_label" not in out.columns or "u4_unit" not in out.columns:
        out["u4_eval_label"] = None
        out["u4_eval_unit"] = None
        out["u4_eval_tier"] = "auxiliary"
        out["u4_eval_options_json"] = None
        return out

    eval_units: list[str | None] = []
    eval_labels: list[str | None] = []
    for unit, label in zip(out["u4_unit"], out["u4_label"]):
        unit_s = unit if isinstance(unit, str) else None
        label_s = label if isinstance(label, str) else None
        punit = primary_unit(pillar, unit_s)
        plabel = primary_label(pillar, unit_s, label_s)
        eval_units.append(punit if plabel is not None else None)
        eval_labels.append(plabel)

    out["u4_eval_unit"] = eval_units
    out["u4_eval_label"] = eval_labels
    out["u4_eval_tier"] = out["u4_eval_label"].map(lambda v: "primary" if isinstance(v, str) and v else "auxiliary")

    threshold = BLOOD_U4_PRIMARY_MIN_SUPPORT if min_support is None else min_support
    if threshold and "split" in out.columns:
        primary = out["u4_eval_tier"].eq("primary")
        if out["split"].eq("train").any():
            train_primary = primary & out["split"].eq("train")
            train_keys = set(
                zip(
                    out.loc[train_primary, "u4_eval_unit"],
                    out.loc[train_primary, "u4_eval_label"],
                )
            )
            keys = list(zip(out["u4_eval_unit"], out["u4_eval_label"]))
            no_train_support = [
                tier == "primary" and split != "train" and key not in train_keys
                for tier, split, key in zip(out["u4_eval_tier"], out["split"], keys)
            ]
            if any(no_train_support):
                out.loc[no_train_support, "u4_eval_tier"] = "auxiliary"
                out.loc[no_train_support, "u4_eval_unit"] = None
                out.loc[no_train_support, "u4_eval_label"] = None
            train_unit_labels = out.loc[train_primary].groupby("u4_eval_unit")["u4_eval_label"].nunique()
            insufficient_train_units = set(
                str(unit)
                for unit, n_labels in train_unit_labels.items()
                if int(n_labels) < 2
            )
            if insufficient_train_units:
                insufficient = [
                    tier == "primary" and isinstance(unit, str) and unit in insufficient_train_units
                    for tier, unit in zip(out["u4_eval_tier"], out["u4_eval_unit"])
                ]
                if any(insufficient):
                    out.loc[insufficient, "u4_eval_tier"] = "auxiliary"
                    out.loc[insufficient, "u4_eval_unit"] = None
                    out.loc[insufficient, "u4_eval_label"] = None

    if pillar == "blood" and threshold and "split" in out.columns:
        primary = out["u4_eval_tier"].eq("primary")
        counts = (
            out.loc[primary]
            .groupby(["split", "u4_eval_unit", "u4_eval_label"], dropna=False)
            .size()
        )
        keys = list(zip(out["split"], out["u4_eval_unit"], out["u4_eval_label"]))
        low_support = [
            tier == "primary" and counts.get(key, 0) < threshold
            for tier, key in zip(out["u4_eval_tier"], keys)
        ]
        if any(low_support):
            out.loc[low_support, "u4_eval_tier"] = "auxiliary"
            out.loc[low_support, "u4_eval_unit"] = None
            out.loc[low_support, "u4_eval_label"] = None
    primary = out["u4_eval_tier"].eq("primary")
    if "split" in out.columns and out["split"].eq("train").any():
        option_source = out.loc[primary & out["split"].eq("train")]
    else:
        option_source = out.loc[primary]
    options_by_unit: dict[str, list[str]] = {}
    for unit, labels in option_source.groupby("u4_eval_unit")["u4_eval_label"]:
        seen = set(labels.dropna().astype(str))
        ordered = [opt for opt in primary_options(str(unit)) if opt in seen]
        if ordered:
            options_by_unit[str(unit)] = ordered
    out["u4_eval_options_json"] = [
        json.dumps(options_by_unit.get(str(unit), []), ensure_ascii=False)
        if tier == "primary" and isinstance(unit, str) and options_by_unit.get(str(unit))
        else None
        for unit, tier in zip(out["u4_eval_unit"], out["u4_eval_tier"])
    ]
    return out


def _norm(s) -> str:
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"[_\-]+", " ", s)
    return re.sub(r"\s+", " ", s)


# skin oncology taxonomy patterns (shared dermoscopy axis)
_SKIN = [
    (re.compile(r"\bmelanoma\b|^mel$|malignant.*melanoma"), "melanoma"),
    (re.compile(r"basal.cell|^bcc$|malignant.*bcc"), "basal cell carcinoma"),
    (re.compile(r"squamous.cell|^scc$|malignant.*scc"), "squamous cell carcinoma"),
    (re.compile(r"\bnev(us|i)?\b|\bnv\b|melanocytic.nev"), "nevus"),   # incl. bare 'nev'
    # solar lentigo is a benign keratosis-like lesion: ISIC/HAM10000 group it with seborrheic keratosis
    # (BKL). It was mapped to actinic keratosis until 2026-09-23.
    (re.compile(r"seborrheic.kerato|^sek$|pigmented benign kerato|benign.seborrheic|solar.lentigo"), "seborrheic keratosis"),
    (re.compile(r"actinic.kerato|\back\b|solar.or.actinic|^akiec$"), "actinic keratosis"),
    (re.compile(r"dermatofibroma|\bdf\b"), "dermatofibroma"),
]

# Other neoplastic skin entities (beyond the 7 named classes) → "other"; anything NOT
# matching is a non-neoplastic condition (eczema/psoriasis/infection/…) → EXCLUDED from
# U4 (skin A6, mirrors oral). Audited to avoid keratosis pilaris / keratoderma /
# hyperkeratosis (uses specific kerato terms only).
_SKIN_NEOPLASTIC = re.compile(
    r"carcinoma|sarcoma|lymphoma|leukem|leukaem|mycosis.fungoides|kaposi|merkel|paget|bowen|"
    r"\bnev\b|spitz|reed|blue.nev|dysplastic|atypical.*melanocytic|aimp|epidermal.nev|melanotic.macule|"
    r"keratoacanthoma|pigmented.benign.kerato|lichenoid.kerato|lichen.planus.like.kerato|"
    r"inverted.follicular.kerato|benign.kerato|arsenical.kerato|alveolar.kerato|lentig|acanthoma|"
    r"angioma|angiokerato|angiofibroma|hemangioma|lymphangioma|vascular|varix|glomangioma|"
    r"pyogenic.granuloma|leiomyoma|fibroma|dermatofibrosarcoma|fibromatosis|fibrous.papule|"
    r"fibroepithelial|acrochordon|fibrokeratoma|lipoma|neurofibroma|neuroma|neurothekeoma|"
    r"xanthoma|xanthogranuloma|mastocytos|histiocytos|sebaceous|trichoblastoma|trichofolliculoma|"
    r"trichilemmoma|tricholemmoma|syringoma|syringocystadenoma|poroma|hidradenoma|adnexal|\bcyst\b|"
    r"porokeratos|dyskeratoma|metasta|skin.tag|\bmilia\b|lymphoid|lymphocytic.infiltrat|"
    r"blastic.plasmacytoid", re.I)


def harmonize(unit: str, raw) -> str | None:
    """Map a dataset-native gold label onto `unit`'s option set, or None if the
    row is not part of that unit's task (→ dropped from U4)."""
    n = _norm(raw)
    if not n:
        return None
    opts = UNIT_OPTIONS.get(unit, [])
    norm_opts = {_norm(o): o for o in opts}
    if n in norm_opts:                       # identity match (native units)
        return norm_opts[n]

    if unit == "skin:derm_subtype":
        # DermNet folder "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions" mixes AK and
        # BCC with no per-image label, so no single class is correct -> not a U4 row (2026-09-23).
        if "actinic keratosis/bcc" in n or ("actinic" in n and "basal cell" in n):
            return None
        for pat, tgt in _SKIN:
            if pat.search(n):
                return tgt
        if _SKIN_NEOPLASTIC.search(n):       # other neoplastic entity → "other"
            return "other"
        return None                          # non-neoplastic skin condition → excluded from U4 (A6)
    if unit == "breast:birads":
        m = re.match(r"birads ?([1-6])", n)
        return f"BIRADS {m.group(1)}" if m else None
    if unit == "breast:molecular":
        if "luminal a" in n: return "Luminal A"
        if "luminal b" in n: return "Luminal B"
        if "her2" in n: return "HER2-enriched"
        if "tnbc" in n or "triple negative" in n: return "TNBC"
        return None                          # benign / unknown-subtype → not this task
    if unit == "cervix:bethesda":
        if "ascus" in n: return "ASCUS"
        if "asch" in n or "asc h" in n: return "ASC-H"
        if "lsil" in n: return "LSIL"
        if "hsil" in n: return "HSIL"
        if "scc" in n or "squamous cell" in n: return "SCC"
        if n in ("nilm", "negative", "normal"): return "NILM"
        return None
    if unit == "cervix:bethesda_coarse":
        return _cervix_bethesda_coarse_label(raw)
    if unit == "oral:oral_subtype":
        if _ORAL_BASELINE.search(n):
            if "variation" in n or "variant" in n:
                return "benign variation"
            return "healthy" if re.search(r"normal|healthy", n) else "benign variation"
        if not _ORAL_NEOPLASTIC.search(n):
            return None                      # non-neoplastic condition → dropped (A6)
        if "cancer" in n or "oscc" in n or "carcinoma" in n or n == "malignant": return "oral cancer"
        if "leukoplak" in n or "leukplak" in n: return "leukoplakia"     # A5 typo
        if "erythroplak" in n: return "erythroplakia"
        if "lichen" in n or "olp" in n: return "lichen planus"           # A5: OLP/LR → lichen planus
        if "osmf" in n or n == "osf" or "submucous" in n: return "oral submucous fibrosis (OSMF)"
        if "opmd" in n: return "oral potentially malignant disorder (OPMD)"
        return "oral potentially malignant disorder (OPMD)"  # other pre-neoplastic → OPMD
    if unit == "thyroid:tirads:DDTI":
        m = re.match(r"tirads ([2-5][abc]?)$", n)
        return f"TI-RADS {m.group(1)}" if m else None      # tirads_unknown → None
    if unit == "thyroid:tirads:Stanford":
        m = re.match(r"tirads(\d)", n)        # composite code → leading level
        return f"TI-RADS {m.group(1)}" if (m and m.group(1) in "12345") else None
    # blood units are native identity (handled by norm_opts above)
    return None


def apply(df, pillar: str):
    """Builder hook: preserve raw gold in `u4_label_raw`, then stamp `u4_unit` and
    rewrite `u4_label` to the unit's option label (or None if dropped from U4).
    Idempotent: re-uses an existing `u4_label_raw` as the source, so a manifest can
    be re-refined (e.g. a future U4 redefinition) WITHOUT rerunning the builder."""
    raw_src = (df["u4_label_raw"] if "u4_label_raw" in df.columns else df["u4_label"]).tolist()
    units, labels = [], []
    for ds, raw in zip(df["dataset_id"], raw_src):
        unit = report_unit(pillar, str(ds))
        if unit is None:
            units.append(None); labels.append(None); continue
        h = harmonize(unit, raw)
        units.append(unit if h is not None else None)
        labels.append(h)
    df = df.copy()
    df["u4_label_raw"] = raw_src
    df["u4_unit"] = units
    df["u4_label"] = labels
    return df
