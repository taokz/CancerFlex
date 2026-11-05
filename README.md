# CancerFlex

CancerFlex is a compact multimodal model for six two-dimensional cancer-imaging domains (skin, breast,
cervix, oral, thyroid and blood), fine-tuned from MedGemma 1.5 4B. One set of weights carries four nested
language-decoder feed-forward widths (1,280 / 2,560 / 5,120 / 10,240), so the same checkpoint can run at a
smaller compute budget by selecting a prefix of every feed-forward block. This repository contains the
training, evaluation, width-export and inference code. It contains no weights and no data.
**Research use only; not a medical device and not for the diagnosis or treatment of patients.**

## Install

```bash
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
pip install -e .
```

Python 3.10+; dependencies are pinned in `pyproject.toml`. Training and evaluation need a GPU; the tests
and the quickstart run on CPU.

## Quickstart (no real data needed)

```bash
cancerflex-infer --checkpoint /path/to/cancerflex-weights --synthetic --width 1280 --json
```

`--synthetic` feeds a procedurally generated image through the full pipeline. With a real image:

```bash
cancerflex-infer --checkpoint /path/to/cancerflex-weights --image scan.png \
  --prompt "Is a lesion visible in this image? Answer Yes or No." --width 10240
```

`--width` selects one of the nested widths listed in the checkpoint's `matformer_granularities.json`
(default: full width). The width is a pure prefix slice of every SwiGLU block; at full width the model
is bit-identical to the unwrapped checkpoint.

## Weights

Weights are distributed separately (placeholder: `<model-hub-url>`). They are a derivative of MedGemma 1.5 4B
and are governed by the Health AI Developer Foundations Terms of Use, which is **not** the licence of this
code. Set `CANCERFLEX_BASE_MODEL` to a local copy or Hub id of MedGemma 1.5 4B when a checkpoint
directory does not ship processor files.

## Configuration

All locations are environment variables (see `src/cancerflex/paths.py`):

| variable | meaning | default |
|---|---|---|
| `CANCERFLEX_ROOT` | workspace holding `Experiment/datasets/...` and `eval_results/` | current directory |
| `CANCERFLEX_BASE_MODEL` | MedGemma 1.5 4B base (processor + initial weights) | `google/medgemma-1.5-4b-it` |
| `EVAL_OUT_DIR`, `EVAL_MATFORMER_WIDTH`, `EVAL_ARCH`, `EVAL_SHARDS_ROOT` | evaluation driver knobs (documented in the driver headers) | |

## Expected data layout (under `CANCERFLEX_ROOT`)

```
Experiment/datasets/
  <pillar>/manifest.parquet            # per-pillar labels, splits, provenance (pillar = skin|breast|cervix|oral|thyroid|blood)
  <pillar>/task_views/                 # task eligibility views
  v1_final_deleaked/shards/{train,test_id,test_ood}/{U0..U6}/<pillar>.jsonl   # training / eval records
  v1_combined_repair/shards/...        # corrected-label corpus used by the repair configs
eval_results/                          # written by the evaluation drivers
runs/                                  # training outputs (configs/*.yaml output_dir)
```

Each JSONL record carries `prompt` (with a literal `<image>` placeholder), `answer`, `image_path` and
task metadata; see `src/cancerflex/train/dataset.py` and `src/cancerflex/eval/eval_iterator.py`.

## Training (recipe of the released model)

```bash
# 1. dense parent (two stages)
python -m cancerflex.train.sft_multitask_balanced --config configs/dense_parent_stage1.yaml
python -m cancerflex.train.sft_multitask_balanced --config configs/dense_parent_stage2_augdata.yaml
# 2. importance-order the feed-forward coordinates of the dense parent
python -m cancerflex.train.neuron_rank_init --base-path runs/dense_parent_stage2/final --out-dir runs/importance_permuted
# 3. elastic (nested-width) training with distillation from the frozen dense parent
torchrun --nproc_per_node 4 -m cancerflex.train.matformer_pure_main --config configs/elastic_training.yaml
# 4. one-shot repair on the corrected corpus (dense teacher first, then the elastic model)
python -m cancerflex.train.sft_multitask_balanced --config configs/repair_dense.yaml
torchrun --nproc_per_node 4 -m cancerflex.train.matformer_pure_main --config configs/repair_elastic_gate.yaml
```

Every config header states what the arm is and how it is run. Check the entry scripts' `--help` for the
exact flag names of `neuron_rank_init`.

## Evaluation

```bash
EVAL_ARCH=matformer EVAL_MATFORMER_WIDTH=1280 EVAL_OUT_DIR=eval_results/w1280 \
  python -m cancerflex.eval.run_closed --help      # closed-set tasks (U0-U3 AUROC, U4 generation)
python -m cancerflex.eval.run_open_gen --help      # open generation (U5 boxes, mIoU)
python -m cancerflex.export.export_matformer_width --help   # materialise a stand-alone narrow checkpoint
```

The drivers read `manifest.parquet` / the JSONL shards described above and write per-row predictions plus
metric JSON into `EVAL_OUT_DIR`.

## Tests

```bash
HF_HUB_OFFLINE=1 pytest -q
```

The suite builds a tiny random Gemma-3 style model on CPU and checks imports, the nested-width forward
(full-width equivalence, narrow-width execution) and the inference CLI end to end on a synthetic image.

## Citation

```bibtex
@article{cancerflex2026,
  title   = {CancerFlex: <title placeholder>},
  author  = {<authors placeholder>},
  journal = {<journal placeholder>},
  year    = {2026}
}
```

## Licence

Code: MIT (see `LICENSE`). Model weights: Health AI Developer Foundations Terms of Use (distributed separately).
