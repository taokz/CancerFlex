"""Prompt templates for VQA-style CancerFlex training.

Templates follow the 4-block format defined in
`docs/standards/corpus-prompt-contract.md`. Current `v1_final` materialization uses
`template_idx=0` for U0-U4/U7, dedicated Gemma `box_2d` localization prompts
for U5, and `mask_path` pixel-mask records for U6. The remaining templates are legacy prompt
augmentation candidates and must not be treated as the active train/eval data
format without first aligning their answer contract. Order is **image-first** to match
Med-Gemma's pretrained input layout (LLaVA-family convention):
    IMAGE: <image>
    TASK: ...
    CONTEXT: ...
    ANSWER: ...

Each template is a plain f-string-style format with {placeholders}. Render via
`prompt_renderer.render_prompt(row, task, template_idx)`.

Placeholders:
    {modality}        — per-pillar config (e.g. "dermoscopy image", "mammogram")
    {site}            — row's anatomical_site or pillar default
    {u2_action}       — pillar-specific referral action
    {u4_classes}      — pillar-specific U4 taxonomy class list
    {context_block}   — rendered patient-context bullets (multi-line, may be empty)
    {image}           — image token placeholder (e.g. "<image>")
"""

# ---------------------------------------------------------------------------
# Per-pillar configuration — modality strings, U4 class lists, referral actions.
# ---------------------------------------------------------------------------

PILLAR_CONFIG = {
    "skin": {
        "modality_default": "dermoscopy image",
        "modality_by_image_type": {
            "dermoscopic": "dermoscopy image",
            "clinical": "clinical skin photograph",
            "dermoscopy": "dermoscopy image",
            "clinical_photo": "clinical skin photograph",
            "3d_total_body_photography_crop": "3D total-body photography lesion crop",
        },
        "site_default": "skin lesion",
        "u2_action": "dermatology referral",
        "u4_classes": "melanoma / basal cell carcinoma / squamous cell carcinoma / nevus / seborrheic keratosis / actinic keratosis / dermatofibroma / other",
        "report_fields": "skin tone / Fitzpatrick where available, clinical priors, lesion descriptor text, dermoscopic checklist terms, paired-view context",
    },
    "breast_mammo": {
        "modality_default": "mammogram",
        "site_default": "breast",
        "u2_action": "biopsy or further imaging",
        "u4_classes": "BIRADS 1 / BIRADS 2 / BIRADS 3 / BIRADS 4 / BIRADS 5 / molecular subtype (Luminal A, Luminal B, HER2-enriched, TNBC)",
        "report_fields": "density category, view, laterality, acquisition context, source lexicon descriptors",
    },
    "breast_us": {
        "modality_default": "breast ultrasound",
        "site_default": "breast",
        "u2_action": "biopsy or mammography referral",
        "u4_classes": "benign / malignant / normal",
        "report_fields": "shape, margin, echogenicity, posterior features, calcifications, device/acquisition context",
    },
    "cervix_cytology": {
        "modality_default": "Pap smear cytology field",
        "site_default": "cervical Pap smear",
        "u2_action": "colposcopy referral",
        "u4_classes": "superficial / intermediate / parabasal / metaplastic / koilocyte / dyskeratotic / LSIL / HSIL / SCC",
        "report_fields": "Bethesda field where source-labeled, acquisition metadata, annotator/magnification context",
    },
    "cervix_via": {
        "modality_default": "colposcopy image",
        "site_default": "cervix",
        "u2_action": "biopsy or treatment",
        "u4_classes": "cervix type 1 / cervix type 2 / cervix type 3 / acetowhite-positive / VIA-negative",
        "report_fields": "acetic-acid phase, SCJ/transformation-zone fields, acetowhite color/margin/surface/location/size, ablation eligibility",
    },
    "oral": {
        "modality_default": "intraoral smartphone photograph",
        "site_default": "oral mucosa",
        "u2_action": "biopsy or specialist referral",
        "u4_classes": "healthy / benign variation / OPMD (leukoplakia, erythroplakia, lichen planus, OSMF) / oral cancer",
        "report_fields": "lesion location, color, margin, surface texture, source description text, demographics/risk factors",
    },
    "thyroid": {
        "modality_default": "thyroid neck ultrasound",
        "site_default": "thyroid nodule",
        "u2_action": "FNA or structured follow-up",
        "u4_classes": "TI-RADS composition (cystic / spongiform / mixed / solid), echogenicity (anechoic / hyperechoic / isoechoic / hypoechoic / very hypoechoic), shape, margin, echogenic foci",
        "report_fields": "TI-RADS source category, composition, echogenicity, margins, calcifications, age/sex",
    },
    "blood": {
        "modality_default": "peripheral blood smear microscopy field",
        "site_default": "WBC cell",
        "u2_action": "hematology consult",
        # Note: blast lineage (lymphoid / myeloid) was previously expressed as
        # ``Blast (lymphoid) / Blast (myeloid)`` but the prompt-config parser in
        # run_medgemma_baseline._candidate_classes split paren contents into
        # standalone ``lymphoid`` / ``myeloid`` candidate tokens, which the
        # blood gold harmonizer does not target — model scoring against those
        # tokens leaked into every prediction. Collapse to single ``Blast`` to
        # match the U4 option SSOT (u4_axis.options, keyed by u4_unit).
        "u4_classes": "Neutrophil / Lymphocyte / Monocyte / Eosinophil / Basophil / Blast / Promyelocyte / Smudge / Atypical lymphocyte",
        "report_fields": "none for supervised v1 U7; blood morphology labels remain U4/annotation evidence",
    },
    "liver": {
        "modality_default": "liver ultrasound",
        "site_default": "liver",
        "u2_action": "CEUS / MRI workup or fibrosis grading",
        "u4_classes": "HCC / hemangioma / cyst / FNH / metastasis / NAFLD steatosis grade (S0-S3) / fibrosis stage (F0-F4)",
        "report_fields": "focal lesion characterization, diffuse disease grade, recommended action",
    },
}


# ---------------------------------------------------------------------------
# Templates per U-task.
# Each list item is a complete template string with the 4-block format.
# ---------------------------------------------------------------------------

TEMPLATES = {
    # ---- U0 — quality / usability (12 templates) ----
    "U0": [
        "IMAGE: {image}\n\nTASK: Is this {modality} technically evaluable for clinical assessment?\n\nCONTEXT:\n{context_block}\n\nANSWER: Yes / No.",
        "IMAGE: {image}\n\nTASK: Quality check — can this {modality} be used for diagnosis, or should it be re-acquired?\n\nCONTEXT:\n{context_block}\n\nANSWER: Use / Re-acquire.",
        "IMAGE: {image}\n\nTASK: Determine if the image acquisition meets standards for {modality} (focus, framing, exposure, artifacts).\n\nCONTEXT:\n{context_block}\n\nANSWER: Adequate / Inadequate.",
        "IMAGE: {image}\n\nTASK: Assess whether this {modality} has any technical issues that would prevent confident reading.\n\nCONTEXT:\n{context_block}\n\nANSWER: List any issues, or write 'none'.",
        "IMAGE: {image}\n\nTASK: Rate the technical quality of this {modality}.\n\nCONTEXT:\n{context_block}\n\nANSWER: acceptable / borderline / not acceptable.",
        "IMAGE: {image}\n\nTASK: Should this {modality} be re-acquired before diagnosis?\n\nCONTEXT:\n{context_block}\n\nANSWER: Yes / No.",
        "IMAGE: {image}\n\nTASK: Is the image quality of this {modality} sufficient to support a clinical decision?\n\nCONTEXT:\n{context_block}\n\nANSWER: Yes / No.",
        "IMAGE: {image}\n\nTASK: Identify any field-of-view, blur, or artifact problems in this {modality}.\n\nCONTEXT:\n{context_block}\n\nANSWER: Describe issues, or write 'image is clean'.",
        "IMAGE: {image}\n\nTASK: Image-quality triage for {modality}: accept or reject.\n\nCONTEXT:\n{context_block}\n\nANSWER: accept / reject.",
        "IMAGE: {image}\n\nTASK: Given typical {modality} acquisition standards, is this image OK to read?\n\nCONTEXT:\n{context_block}\n\nANSWER: Yes / No.",
        "IMAGE: {image}\n\nTASK: Quality?\n\nCONTEXT:\n{context_block}\n\nANSWER: A (acceptable) / B (not acceptable).",
        "IMAGE: {image}\n\nTASK: As an AI assistant, judge whether the patient's {modality} acquisition is diagnostic-quality.\n\nCONTEXT:\n{context_block}\n\nANSWER: Yes / No.",
    ],

    # ---- U1 — target present (12 templates) ----
    "U1": [
        "IMAGE: {image}\n\nTASK: Is the relevant anatomy ({site}) visible in this {modality} and does it contain a lesion of interest?\n\nCONTEXT:\n{context_block}\n\nANSWER: Yes / No.",
        "IMAGE: {image}\n\nTASK: Determine whether {site} is in frame and a lesion is present.\n\nCONTEXT:\n{context_block}\n\nANSWER: Yes / No.",
        "IMAGE: {image}\n\nTASK: Is there a lesion / abnormality in this {modality} of {site}?\n\nCONTEXT:\n{context_block}\n\nANSWER: Yes / No.",
        "IMAGE: {image}\n\nTASK: Decide if this {modality} shows the target anatomy with a visible lesion, or if it shows normal tissue / wrong anatomy.\n\nCONTEXT:\n{context_block}\n\nANSWER: Lesion-present / Normal / Wrong-anatomy.",
        "IMAGE: {image}\n\nTASK: Lesion presence check on {site}.\n\nCONTEXT:\n{context_block}\n\nANSWER: Yes / No.",
        "IMAGE: {image}\n\nTASK: Identify whether {site} contains a visible lesion in this {modality}.\n\nCONTEXT:\n{context_block}\n\nANSWER: present / not present.",
        "IMAGE: {image}\n\nTASK: As a clinician scanning the image, do you see a lesion on {site}?\n\nCONTEXT:\n{context_block}\n\nANSWER: Yes / No.",
        "IMAGE: {image}\n\nTASK: Is the field-of-view appropriate AND is a lesion visible?\n\nCONTEXT:\n{context_block}\n\nANSWER: A (yes both) / B (anatomy wrong) / C (anatomy correct but no lesion).",
        "IMAGE: {image}\n\nTASK: Predict whether this {modality} contains a lesion worth analyzing.\n\nCONTEXT:\n{context_block}\n\nANSWER: Yes / No.",
        "IMAGE: {image}\n\nTASK: Anatomical-target presence in {modality} of {site}.\n\nCONTEXT:\n{context_block}\n\nANSWER: target present / target absent.",
        "IMAGE: {image}\n\nTASK: Lesion?\n\nCONTEXT:\n{context_block}\n\nANSWER: Yes / No.",
        "IMAGE: {image}\n\nTASK: Given the patient's context, does the {modality} confirm a {site} lesion is present?\n\nCONTEXT:\n{context_block}\n\nANSWER: Yes / No.",
    ],

    # ---- U2 — referral / management-positive (12 templates) ----
    "U2": [
        "IMAGE: {image}\n\nTASK: Should this {modality} of {site} be referred for {u2_action}?\n\nCONTEXT:\n{context_block}\n\nANSWER: Yes / No.",
        "IMAGE: {image}\n\nTASK: Determine if findings warrant {u2_action} based on the {modality}.\n\nCONTEXT:\n{context_block}\n\nANSWER: Yes / No.",
        "IMAGE: {image}\n\nTASK: Decide whether this patient should receive {u2_action} given the {modality}.\n\nCONTEXT:\n{context_block}\n\nANSWER: Refer / Do not refer.",
        "IMAGE: {image}\n\nTASK: Triage decision — refer for {u2_action} or follow standard surveillance?\n\nCONTEXT:\n{context_block}\n\nANSWER: Refer / Surveillance.",
        "IMAGE: {image}\n\nTASK: Is referral for {u2_action} indicated?\n\nCONTEXT:\n{context_block}\n\nANSWER: Yes / No.",
        "IMAGE: {image}\n\nTASK: Predict whether the responsible clinician would refer this patient for {u2_action} after viewing the {modality}.\n\nCONTEXT:\n{context_block}\n\nANSWER: Yes / No.",
        "IMAGE: {image}\n\nTASK: Management decision: should this case proceed to {u2_action}?\n\nCONTEXT:\n{context_block}\n\nANSWER: Yes / No.",
        "IMAGE: {image}\n\nTASK: Refer?\n\nCONTEXT:\n{context_block}\n\nANSWER: Yes / No.",
        "IMAGE: {image}\n\nTASK: Imagine the patient is asking 'do I need to see a specialist or have {u2_action}?' — answer based on this {modality}.\n\nCONTEXT:\n{context_block}\n\nANSWER: Yes / No.",
        "IMAGE: {image}\n\nTASK: Suspicious enough to warrant {u2_action}?\n\nCONTEXT:\n{context_block}\n\nANSWER: A (yes) / B (no).",
        "IMAGE: {image}\n\nTASK: Determine if this {modality} of {site} crosses the clinical-action threshold for {u2_action}.\n\nCONTEXT:\n{context_block}\n\nANSWER: Yes / No.",
        "IMAGE: {image}\n\nTASK: Output your confidence (0–1) that this case requires {u2_action}.\n\nCONTEXT:\n{context_block}\n\nANSWER: A decimal between 0 and 1.",
    ],

    # ---- U3 — malignancy (12 templates) ----
    "U3": [
        "IMAGE: {image}\n\nTASK: Considering the patient context and the image, determine whether the visible lesion on the {modality} is malignant.\n\nCONTEXT:\n{context_block}\n\nANSWER: Yes / No.",
        "IMAGE: {image}\n\nTASK: Is the lesion on this {modality} of {site} malignant?\n\nCONTEXT:\n{context_block}\n\nANSWER: Yes / No.",
        "IMAGE: {image}\n\nTASK: Classify the lesion as benign or malignant.\n\nCONTEXT:\n{context_block}\n\nANSWER: Benign / Malignant.",
        "IMAGE: {image}\n\nTASK: Decide if biopsy-confirmed cancer is the most likely diagnosis here.\n\nCONTEXT:\n{context_block}\n\nANSWER: Yes / No.",
        "IMAGE: {image}\n\nTASK: Predict malignancy likelihood for this {modality} lesion.\n\nCONTEXT:\n{context_block}\n\nANSWER: Malignant / Benign.",
        "IMAGE: {image}\n\nTASK: Malignant?\n\nCONTEXT:\n{context_block}\n\nANSWER: Yes / No.",
        "IMAGE: {image}\n\nTASK: Assess whether this {site} lesion shows features consistent with cancer.\n\nCONTEXT:\n{context_block}\n\nANSWER: Yes / No.",
        "IMAGE: {image}\n\nTASK: Is the lesion concerning for cancer?\n\nCONTEXT:\n{context_block}\n\nANSWER: A (concerning) / B (not concerning).",
        "IMAGE: {image}\n\nTASK: Output the cancer-risk decision for this image.\n\nCONTEXT:\n{context_block}\n\nANSWER: Cancer / No cancer.",
        "IMAGE: {image}\n\nTASK: Imagine you receive a pathology report later — would it show malignancy for this lesion?\n\nCONTEXT:\n{context_block}\n\nANSWER: Yes / No.",
        "IMAGE: {image}\n\nTASK: Cancer-risk assessment for this {modality} of {site}.\n\nCONTEXT:\n{context_block}\n\nANSWER: High-risk (malignant likely) / Low-risk (benign likely).",
        "IMAGE: {image}\n\nTASK: Output your confidence (0–1) that this lesion is malignant.\n\nCONTEXT:\n{context_block}\n\nANSWER: A decimal between 0 and 1.",
    ],

    # ---- U4 — fine subtype / grade (15 templates) ----
    "U4": [
        "IMAGE: {image}\n\nTASK: Classify the most likely diagnosis for this {modality} among: {u4_classes}.\n\nCONTEXT:\n{context_block}\n\nANSWER: One class name.",
        "IMAGE: {image}\n\nTASK: What is the specific lesion type? Choose from: {u4_classes}.\n\nCONTEXT:\n{context_block}\n\nANSWER: One class name.",
        "IMAGE: {image}\n\nTASK: Predict the fine-grained class for this {site} from: {u4_classes}.\n\nCONTEXT:\n{context_block}\n\nANSWER: One class name.",
        "IMAGE: {image}\n\nTASK: Determine the diagnosis class. Available classes: {u4_classes}.\n\nCONTEXT:\n{context_block}\n\nANSWER: One class name.",
        "IMAGE: {image}\n\nTASK: Subtype this lesion. Options: {u4_classes}.\n\nCONTEXT:\n{context_block}\n\nANSWER: One class name.",
        "IMAGE: {image}\n\nTASK: Identify the lesion type from: {u4_classes}.\n\nCONTEXT:\n{context_block}\n\nANSWER: One class name.",
        "IMAGE: {image}\n\nTASK: Provide the most likely diagnosis along with the second-most-likely as differential.\n\nCONTEXT:\n{context_block}\n\nANSWER: 1) <primary>; 2) <differential>. Choose from: {u4_classes}.",
        "IMAGE: {image}\n\nTASK: Classify and explain. Class options: {u4_classes}.\n\nCONTEXT:\n{context_block}\n\nANSWER: Class: <class>. Reasoning: <1-2 sentences>.",
        "IMAGE: {image}\n\nTASK: For this {modality} of {site}, what is the diagnosis?\n\nCONTEXT:\n{context_block}\n\nANSWER: One of: {u4_classes}.",
        "IMAGE: {image}\n\nTASK: Pick the closest matching class from {u4_classes}.\n\nCONTEXT:\n{context_block}\n\nANSWER: <class>.",
        "IMAGE: {image}\n\nTASK: Classify (terse).\n\nCONTEXT:\n{context_block}\n\nANSWER: One of: {u4_classes}.",
        "IMAGE: {image}\n\nTASK: Most-likely diagnosis label?\n\nCONTEXT:\n{context_block}\n\nANSWER: One label from {u4_classes}.",
        "IMAGE: {image}\n\nTASK: Decide the fine subtype.\n\nCONTEXT:\n{context_block}\n\nANSWER: <class from {u4_classes}>.",
        "IMAGE: {image}\n\nTASK: Rank-order the top-3 likely diagnoses among {u4_classes}.\n\nCONTEXT:\n{context_block}\n\nANSWER: 1) <c1>; 2) <c2>; 3) <c3>.",
        "IMAGE: {image}\n\nTASK: From the perspective of a referring clinician, which class best fits?\n\nCONTEXT:\n{context_block}\n\nANSWER: <class> (from {u4_classes}).",
    ],

    # ---- U5 — bbox detection (6 templates) ----
    "U5": [
        "IMAGE: {image}\n\nTASK: Locate any suspicious lesions in this {modality}. Output the final answer in the format \"Final Answer: X\" where X is a JSON list of objects. Each object needs a \"box_2d\" key in [y0, x0, y1, x1] coordinates normalized to [0,1000] and a \"label\" key.\n\nCONTEXT:\n{context_block}\n\nANSWER: Final Answer: ```json[{{\"box_2d\":[...],\"label\":\"target\"}}]```.",
        "IMAGE: {image}\n\nTASK: Predict object boxes for visible lesions on {site}. Use Gemma-style \"box_2d\" coordinates [y0, x0, y1, x1] normalized to [0,1000].\n\nCONTEXT:\n{context_block}\n\nANSWER: Final Answer: ```json[{{\"box_2d\":[...],\"label\":\"target\"}}]```.",
        "IMAGE: {image}\n\nTASK: Identify lesion locations with parseable object JSON. Do not output center-width-height boxes.\n\nCONTEXT:\n{context_block}\n\nANSWER: Final Answer: ```json[{{\"box_2d\":[y0,x0,y1,x1],\"label\":\"target\"}}]```.",
        "IMAGE: {image}\n\nTASK: Where are the suspicious regions in this {modality}? Return only a JSON list of objects using \"box_2d\" and \"label\".\n\nCONTEXT:\n{context_block}\n\nANSWER: Final Answer: ```json[{{\"box_2d\":[...],\"label\":\"target\"}}]```.",
        "IMAGE: {image}\n\nTASK: Lesion detection. Use [y0, x0, y1, x1] integer coordinates normalized to [0,1000].\n\nCONTEXT:\n{context_block}\n\nANSWER: Final Answer: ```json[{{\"box_2d\":[...],\"label\":\"target\"}}]```.",
        "IMAGE: {image}\n\nTASK: Mark the locations of any abnormal regions with Gemma-style object boxes.\n\nCONTEXT:\n{context_block}\n\nANSWER: Final Answer: ```json[{{\"box_2d\":[...],\"label\":\"target\"}}]```.",
    ],

    # ---- U6 — segmentation mask (legacy placeholders; active v1 uses mask_path) ----
    "U6": [
        "IMAGE: {image}\n\nTASK: Segment the target region in this {modality}. Active v1 stores the gold output as a pixel mask referenced by mask_path; this is not a MedGemma text-generation target.\n\nCONTEXT:\n{context_block}\n\nANSWER: <pixel_mask>",
        "IMAGE: {image}\n\nTASK: Produce a pixel-accurate segmentation mask for the lesion in this {modality} of {site}. The supervised target is the dataset mask_path, not a text box or polygon.\n\nCONTEXT:\n{context_block}\n\nANSWER: <pixel_mask>",
        "IMAGE: {image}\n\nTASK: Pixel-mask supervision is available for this row. Score predictions with Dice and mask IoU against mask_path.\n\nCONTEXT:\n{context_block}\n\nANSWER: <pixel_mask>",
        "IMAGE: {image}\n\nTASK: Delineate the target mask. Do not reduce this U6 target to a bounding box for official metrics.\n\nCONTEXT:\n{context_block}\n\nANSWER: <pixel_mask>",
        "IMAGE: {image}\n\nTASK: Segment the abnormal region using a segmentation model or mask decoder. The gold label is a raster mask file.\n\nCONTEXT:\n{context_block}\n\nANSWER: <pixel_mask>",
        "IMAGE: {image}\n\nTASK: Standard segmentation task. Active training/eval should consume mask_path directly and report Dice plus mask IoU.\n\nCONTEXT:\n{context_block}\n\nANSWER: <pixel_mask>",
    ],

    # ---- U7 — structured context / descriptor reasoning (25 templates) ----
    "U7": [
        "IMAGE: {image}\n\nTASK: Return only the source-labeled structured context fields available for this {modality}: {report_fields}. Do not infer diagnosis or management unless that field is explicitly source-labeled.\n\nCONTEXT:\n{context_block}\n\nANSWER: JSON object.",
        "IMAGE: {image}\n\nTASK: Extract the available descriptor fields for this {site} from the image and context. Use null for fields not provided by the source.\n\nCONTEXT:\n{context_block}\n\nANSWER: JSON object with {report_fields}.",
        "IMAGE: {image}\n\nTASK: Produce a machine-readable descriptor record for this {modality}. Include only source-supported fields, not a free-text clinical report.\n\nCONTEXT:\n{context_block}\n\nANSWER: JSON object.",
        "IMAGE: {image}\n\nTASK: Fill the structured context schema for this case using source-provided fields and visible descriptor evidence.\n\nCONTEXT:\n{context_block}\n\nANSWER: Field-by-field structured output.",
        "IMAGE: {image}\n\nTASK: List the source-labeled visual descriptors present for this {site}. Do not add a diagnosis label.\n\nCONTEXT:\n{context_block}\n\nANSWER: Descriptor list.",
        "IMAGE: {image}\n\nTASK: Identify the available acquisition or view context for this {modality}, such as laterality, view, image type, or paired-view information when present.\n\nCONTEXT:\n{context_block}\n\nANSWER: JSON object.",
        "IMAGE: {image}\n\nTASK: Return the available patient or cohort context fields relevant to this image. Do not use them to write a recommendation.\n\nCONTEXT:\n{context_block}\n\nANSWER: JSON object.",
        "IMAGE: {image}\n\nTASK: Generate the structured descriptor fields requested by the source schema: {report_fields}.\n\nCONTEXT:\n{context_block}\n\nANSWER: JSON object.",
        "IMAGE: {image}\n\nTASK: For this {modality}, separate source context from visual descriptors in a structured output.\n\nCONTEXT:\n{context_block}\n\nANSWER: {{\"context_fields\": {{...}}, \"visual_descriptors\": {{...}}}}.",
        "IMAGE: {image}\n\nTASK: Produce a compact source-field record for this case. Exclude diagnosis, malignancy, referral, and treatment-plan fields unless they are part of the provided schema.\n\nCONTEXT:\n{context_block}\n\nANSWER: JSON object.",
        "IMAGE: {image}\n\nTASK: Record the lesion or finding descriptors supported by the source annotation, such as shape, margin, texture, color, echo pattern, calcification, or checklist terms when available.\n\nCONTEXT:\n{context_block}\n\nANSWER: JSON object.",
        "IMAGE: {image}\n\nTASK: Convert the available context into structured key-value pairs for downstream evaluation.\n\nCONTEXT:\n{context_block}\n\nANSWER: Key-value JSON.",
        "IMAGE: {image}\n\nTASK: Return the non-label clinical context fields for this {site}; do not restate U2 referral, U3 malignancy, or U4 class labels.\n\nCONTEXT:\n{context_block}\n\nANSWER: JSON object.",
        "IMAGE: {image}\n\nTASK: Build a structured descriptor summary from the source fields. Keep the output factual and field-based.\n\nCONTEXT:\n{context_block}\n\nANSWER: Field-by-field output.",
        "IMAGE: {image}\n\nTASK: If paired or multi-view context is present, summarize the paired-view relationship as structured fields. Otherwise return only the available single-image fields.\n\nCONTEXT:\n{context_block}\n\nANSWER: JSON object.",
        "IMAGE: {image}\n\nTASK: Extract source-provided demographic and acquisition fields for this case, plus descriptor fields when available.\n\nCONTEXT:\n{context_block}\n\nANSWER: JSON object.",
        "IMAGE: {image}\n\nTASK: Return a structured evidence record. Each value must be grounded in either the image-visible descriptor or the supplied context block.\n\nCONTEXT:\n{context_block}\n\nANSWER: JSON object.",
        "IMAGE: {image}\n\nTASK: Fill only fields that are observable or source-labeled for this {modality}; mark unavailable fields as null.\n\nCONTEXT:\n{context_block}\n\nANSWER: JSON object with {report_fields}.",
        "IMAGE: {image}\n\nTASK: Produce a source-aligned descriptor annotation for this image.\n\nCONTEXT:\n{context_block}\n\nANSWER: Structured annotation JSON.",
        "IMAGE: {image}\n\nTASK: Summarize the available structured context without turning it into an impression or plan.\n\nCONTEXT:\n{context_block}\n\nANSWER: JSON object.",
        "IMAGE: {image}\n\nTASK: Return the structured finding lexicon fields supported by this source dataset.\n\nCONTEXT:\n{context_block}\n\nANSWER: JSON object.",
        "IMAGE: {image}\n\nTASK: For research annotation, output the source-labeled context and descriptor fields in machine-readable form.\n\nCONTEXT:\n{context_block}\n\nANSWER: JSON object.",
        "IMAGE: {image}\n\nTASK: Identify which context fields are present and provide their values. Do not synthesize missing clinical history.\n\nCONTEXT:\n{context_block}\n\nANSWER: JSON object.",
        "IMAGE: {image}\n\nTASK: Generate a structured descriptor record suitable for per-field scoring.\n\nCONTEXT:\n{context_block}\n\nANSWER: JSON object.",
        "IMAGE: {image}\n\nTASK: Return the available U7 structured-context target for this case.\n\nCONTEXT:\n{context_block}\n\nANSWER: JSON object.",
    ],
}


def get_template(task: str, idx: int | None = None, seed: int | None = None) -> str:
    """Get a template for a given task. If idx is None, choose randomly
    (deterministic when seed is given)."""
    import random
    options = TEMPLATES[task]
    if idx is not None:
        return options[idx % len(options)]
    rng = random.Random(seed) if seed is not None else random
    return rng.choice(options)


# Sanity check: counts match the documented spec.
EXPECTED_COUNTS = {"U0": 12, "U1": 12, "U2": 12, "U3": 12, "U4": 15, "U5": 6, "U6": 6, "U7": 25}
assert all(len(TEMPLATES[k]) == EXPECTED_COUNTS[k] for k in EXPECTED_COUNTS), (
    {k: (len(TEMPLATES[k]), EXPECTED_COUNTS[k]) for k in EXPECTED_COUNTS}
)
TOTAL = sum(EXPECTED_COUNTS.values())  # 100
