# Supplementary Code — UKAD: Deformable Kolmogorov-Arnold Networks for Medical Image Segmentation

This package reproduces every experiment reported in the paper: UKAD (S/B/L), all twelve retrained
baselines, and the ablation arms in Appendix D. It excludes raw datasets, trained checkpoints, and
logs — only the code and configuration needed to rerun the pipeline from scratch.

## Requirements

Python 3.12, PyTorch 2.5, PyTorch Lightning 2.5+. Install with:

```bash
pip install -r requirements/requirements.txt
```

## Environment

Two environment variables must be set before running anything:

- `WORKBENCH` — a writable directory for checkpoints, logs, and split manifests.
- `DATASETS` — the directory holding the raw BUSI, CVC-ClinicDB, GlaS, and ISIC 2018 data (see the
  paper's Appendix A.3 for dataset sources and the expected split protocol).

```bash
export WORKBENCH=/path/to/workbench
export DATASETS=/path/to/datasets
export ALBUMENTATIONS_NO_TELEMETRY=1
```

## Reproducing the paper's headline table (Table 1)

```bash
# UKAD variants
bash scripts/variants/run_ukad_b.sh busi
bash scripts/variants/run_ukad_b.sh cvc
bash scripts/variants/run_ukad_b.sh glas
bash scripts/variants/run_ukad_b.sh isic
bash scripts/variants/run_ukad_l.sh <dataset>   # and run_ukad_s.sh for the small variant

# Baselines (repeat per dataset and per model under configs/baselines/)
bash scripts/train/train_glas.sh attunet
```

Each `train.py` invocation accepts one of the three data-split seeds used throughout the paper
(2981, 6142, 1187); the model/training RNG is fixed internally at 1029. For example:

```bash
python train.py --config configs/variants/ukad_b/busi.yaml --dataseed 2981
```

## Evaluation

```bash
bash scripts/eval/eval_busi.sh ukad_b
```

This reloads the best-validation-IoU checkpoint, runs the held-out validation split, and reports
MedPy's IoU, Dice, HD, HD95, recall, precision, and specificity — the same pipeline that produced
every number in the paper's tables (see Appendix A.2 for exact metric definitions).

## Significance tests (Appendix C)

```bash
python scripts/eval/paired_runs.py
python scripts/eval/per_image_iou.py --models variants/ukad_l variants/ukad_b adakan rollingunet_l ukan
python scripts/eval/paired_bootstrap.py
```

`paired_runs.py` runs a paired Wilcoxon test over the 12 dataset-split pairs with Holm correction.
`per_image_iou.py` reloads the stored checkpoints and writes per-image IoU on each validation split.
`paired_bootstrap.py` pairs those values by image and reports a bootstrap interval and p-value,
resampling within each dataset-split stratum.

## Reproduction check

```bash
python scripts/eval/check_reproduction.py outputs/performance/<model>_<dataset>.json
python scripts/eval/summarize_reproduction.py
```

The first script reloads every checkpoint listed in a result JSON, re-runs the validation split,
and writes the stored value, the recomputed value, and their difference for each metric. It also
counts batches on which MedPy's HD or HD95 would raise. The second script collects the run logs
into one CSV.

## Layout

- `src/models/nets/baselines/ukad.py` — the proposed UKAD architecture.
- `src/models/nets/` — every baseline architecture, registered in `src/models/nets/__init__.py`.
- `src/models/base_model.py`, `src/models/_seg_metrics.py` — the Lightning task wrapper and metrics.
- `configs/variants/` — UKAD-S/B/L configs, one per dataset.
- `configs/baselines/` — one config directory per baseline model, including nnU-Net ResEnc-M, MedNeXt, LKM-UNet, and CMUNeXt configs for additional comparisons.
- `configs/ablation/` — the fourteen ablation arms of Appendix D.
- `scripts/train/`, `scripts/eval/`, `scripts/variants/` — driver scripts.
- `train.py` — the single entry point every experiment in the paper runs through.

## Reproducibility notes

All configs are `!include`-composed YAML resolved by `src/utils/serialization_utils.py`; each node
is `{module, class_name, config}`, instantiated by name rather than hardcoded. This is the same
mechanism the paper's Reproducibility Statement refers to: every reported number comes from one
pipeline, with split manifests written to disk before training and read back identically at
evaluation.
