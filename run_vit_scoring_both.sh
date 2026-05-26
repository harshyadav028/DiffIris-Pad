#!/bin/bash
set -e
cd /home/teaching/Documents/Geetanjali_PhD_IRIS_PAD

PYTHON=PYTHONNOUSERSITE=1 /home/teaching/miniconda3/envs/bbdm_clean/bin/python
LOG=IJCB_paper_requirements/logs/vit_scoring_run.log
mkdir -p IJCB_paper_requirements/logs IJCB_paper_requirements/scoring

echo "[$(date)] Starting ViT scoring — test split" | tee -a "$LOG"
PYTHONNOUSERSITE=1 /home/teaching/miniconda3/envs/bbdm_clean/bin/python \
    iris_bbdm_pad/evaluation/run_vit_scoring.py \
    --split test \
    --output_csv IJCB_paper_requirements/scoring/vit_scores_test.csv \
    --batch_size 32 \
    --num_steps 50 \
    2>&1 | tee -a "$LOG"

echo "[$(date)] Starting ViT scoring — val split" | tee -a "$LOG"
PYTHONNOUSERSITE=1 /home/teaching/miniconda3/envs/bbdm_clean/bin/python \
    iris_bbdm_pad/evaluation/run_vit_scoring.py \
    --split val \
    --output_csv IJCB_paper_requirements/scoring/vit_scores_val.csv \
    --batch_size 32 \
    --num_steps 50 \
    2>&1 | tee -a "$LOG"

echo "[$(date)] All ViT scoring complete." | tee -a "$LOG"
