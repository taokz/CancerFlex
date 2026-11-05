# Release notes — CancerFlex code release 1.0.0

## What this release contains

The code path that produced the released CancerFlex weights and the paper's primary results:

- `src/cancerflex/matformer/` — nested-width SwiGLU block, width controller and model wrapper (shared by training, evaluation and inference).
- `src/cancerflex/train/` — dense-parent SFT (`sft_multitask_balanced.py` + `balanced/`), importance ordering (`neuron_rank_init.py`), elastic / repair training (`matformer_pure_main.py`), data loading and collation.
- `src/cancerflex/eval/` — closed-set and open-generation evaluation drivers and the shared evaluation contract (prompt templates/renderer, task axes, image canonicalisation, metrics).
- `src/cancerflex/export/` — materialisation and verification of stand-alone narrow-width checkpoints.
- `configs/` — the five training configurations of the released recipe, with rewritten headers.
- `src/cancerflex/infer.py`, `src/cancerflex/synthetic.py`, `tests/`, CI — new packaging code only.

## Packaging changes (no change to model behaviour or training logic)

- Fixed filesystem roots and base-model locations became environment variables (`CANCERFLEX_ROOT`, `CANCERFLEX_BASE_MODEL`); import paths were renamed to the `cancerflex` package; `sys.path` manipulation now points at the package's own directories.
- Config `output_dir` / `run_name` values were renamed to neutral names under `runs/`; experiment-tracking project names were neutralised; headers were rewritten.
- Comments naming machines, environments or launch scripts outside this repository were generalised.

## Not included / needs owner confirmation

- **Needs owner confirmation:** that the included configs and entry points are exactly the arms behind the submitted manuscript (they were identified from the research repository's model-lineage records and release card, with high confidence, but not re-run).
- **Needs owner confirmation:** the MIT licence for the code (the weights carry their own terms).
- Mask-pathway (segmentation) scoring and its feature adapter: depend on third-party decoder weights and an adapter not in the code repository.
- Learned width-router, per-layer mix-and-match, vision-tower compression and reference-model (Gemma-3n) code: ablations or reference arms, not the primary results. Lazy imports of these arms remain in the evaluation drivers and fail only if their flags are set.
- Data manifests, training shards, evaluation outputs, paper figure builders and all weights.
- Linting: CI lints the new packaging modules and tests; the research modules are kept as written and are excluded from lint.

## Verification done

- `pip install -e .` and `HF_HUB_OFFLINE=1 pytest -q`: 31 tests pass on CPU (imports, nested-width forward equivalence, multimodal forward and CLI on a synthetic image, using a tiny random Gemma-3 style model).
- The tree was searched for credentials, e-mail addresses other than the author's, IP addresses, machine and account names and internal path fragments; none remain.
