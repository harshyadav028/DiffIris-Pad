# Reproducing Table 3 — Step-by-Step

All commands run from the project root `/home/teaching/Documents/Geetanjali_PhD_IRIS_PAD/`.
Use the `bbdm_clean` conda environment throughout.

```bash
PYTHON=/home/teaching/miniconda3/envs/bbdm_clean/bin/python
```

---

## Prerequisites

1. Download checkpoints from HuggingFace (see `checkpoint_staging/UPLOAD_INSTRUCTIONS.md`):
   ```bash
   mkdir -p iris_bbdm_pad/checkpoints
   # Replace [HF_URL] with actual HuggingFace URL after upload
   wget [HF_URL]/top_model_epoch_70.pth -P results/iris_bonafide_pad/LBBDM-f4/checkpoint/
   wget [HF_URL]/vq-f4.zip -P BBDM/resources/vq-f4/
   cd BBDM/resources/vq-f4 && unzip vq-f4.zip && cd -
   ```

2. Ensure LivDet-Iris 2025 dataset is in `Images/` with the structure:
   ```
   Images/
   ├── combined_dataset.csv
   ├── train/
   ├── val/
   └── test/
   ```

---

## Step 1 — Preprocessing (skip if `iris_bbdm_pad/data/preprocessed/` exists)

```bash
PYTHONNOUSERSITE=1 $PYTHON \
    iris_bbdm_pad/data/iris_preprocessing.py \
    --input_dir Images/ \
    --output_dir iris_bbdm_pad/data/preprocessed/ \
    --image_size 256 --workers 4 --resume
```

## Step 2 — Bonafide Pair Generation (skip if `iris_bbdm_pad/data/bonafide_pairs/` exists)

```bash
PYTHONNOUSERSITE=1 $PYTHON \
    iris_bbdm_pad/data/prepare_bonafide_pairs.py \
    --input_dir iris_bbdm_pad/data/preprocessed/train/Live/ \
    --output_dir iris_bbdm_pad/data/bonafide_pairs/
```

## Step 3 — BBDM Training (skip if using downloaded checkpoint)

```bash
PYTHONNOUSERSITE=1 $PYTHON \
    iris_bbdm_pad/training/train_bbdm_bonafide.py \
    --config iris_bbdm_pad/configs/bbdm_iris_bonafide.yaml \
    --gpu_ids 0
# Checkpoint saved to: results/iris_bonafide_pad/LBBDM-f4/checkpoint/top_model_epoch_70.pth
```

## Step 4 — ViT Scoring (val + test)

```bash
bash run_vit_scoring_both.sh
# Outputs: IJCB_paper_requirements/scoring/vit_scores_val.csv
#          IJCB_paper_requirements/scoring/vit_scores_test.csv
```

## Step 5 — Downstream Evaluation (Table 3, Diff-IrisPAD row)

```bash
bash run_task1_downstream.sh
# Outputs:
#   IJCB_paper_requirements/scoring/vit_metrics_test.csv      ← Table 3 numbers
#   IJCB_paper_requirements/scoring/dynamic_fusion_metrics_test.csv
#   IJCB_paper_requirements/tables/table_scoring_comparison.csv
```

**Table 3 numbers are in** `IJCB_paper_requirements/scoring/vit_metrics_test.csv` — the "All" row gives overall ACER 0.276.

---

## Reproducing the AnoDDPM Baseline Row

```bash
# Download AnoDDPM checkpoint
wget [HF_URL]/top_model_epoch_best.pth \
    -P iris_td/results/ddpm_run1/DDPM/checkpoint/

# Run AnoDDPM ViT scoring
PYTHONNOUSERSITE=1 $PYTHON \
    iris_td/training/run_ddpm_scoring.py \
    --config iris_td/configs/ddpm_iris.yaml \
    --split test --noise_type simplex --use_ddim --steps 50

# Compute dynamic fusion metrics
PYTHONNOUSERSITE=1 $PYTHON \
    iris_td/evaluation/generate_anoddpm_scoring_ablation.py
```

Results in `iris_td/final_results/anoddpm_ddim_50steps_dynamic_combined_format.csv`.

---

## Expected Results (Table 3)

| Method | ACER | APCER | BPCER |
|---|---|---|---|
| Diff-IrisPAD (ours) | **0.276** | 0.264 | 0.288 |
| AnoDDPM + DDIM Simplex | 0.397 | — | — |

Full per-attack breakdown in `IJCB_paper_requirements/scoring/vit_metrics_test.csv`.
