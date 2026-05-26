#!/usr/bin/env bash
# iris_td/run_ddpm_vit_configs_321.sh
#
# For each DDPM config (3→2→1), at 50 stochastic denoising steps:
#   STEP A: val  LPIPS scoring  (ddpm_anomaly_detector.py)
#   STEP B: test LPIPS scoring
#   STEP C: val  ViT scoring    (ddpm_vit_scorer_unified.py)
#   STEP D: test ViT scoring
#   STEP E: fusion ablation CPU (ViT / Fixed / Dynamic)
#
# Config 3: AnoDDPM   — Simplex,  t*=500,  50 stochastic steps
# Config 2: DDPM-Partial — Gaussian, t*=500,  50 stochastic steps
# Config 1: DDPM-Vanilla — Gaussian, t*=1000, 50 stochastic steps
#
# Run in tmux:
#   cd /home/teaching/Documents/Geetanjali_PhD_IRIS_PAD
#   tmux new -s ddpm321
#   bash iris_td/run_ddpm_vit_configs_321.sh 2>&1 | tee iris_td/final_results/ddpm_vit_321_run.log

set -eo pipefail
cd /home/teaching/Documents/Geetanjali_PhD_IRIS_PAD
export PYTHONNOUSERSITE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
PY=/home/teaching/miniconda3/envs/bbdm_clean/bin/python
export PYTHONPATH=BBDM:$PYTHONPATH

CFG=iris_td/configs/ddpm_iris.yaml
CKPT=iris_td/results/ddpm_run1/DDPM/checkpoint/top_model_epoch_best.pth

ts() { date '+%Y-%m-%d %H:%M:%S'; }

mkdir -p iris_td/final_results iris_td/pad_scores

echo "========================================================"
echo "DDPM ViT Scoring — Configs 3, 2, 1  (50 stochastic steps)"
echo "Started: $(ts)"
echo "========================================================"

# ════════════════════════════════════════════════════════════════
# CONFIG 3: AnoDDPM — Simplex, t*=500, 50 stochastic steps
# ════════════════════════════════════════════════════════════════
echo ""
echo "████████████████████████████████████████████████████████"
echo "CONFIG 3: AnoDDPM — Simplex t*=500, 50 stochastic steps"
echo "████████████████████████████████████████████████████████"

echo "▶ [$(ts)] C3-A: Val LPIPS scoring"
$PY iris_td/models/ddpm_anomaly_detector.py \
    --config $CFG --checkpoint $CKPT \
    --test_dir iris_td/data/evaluation_sets/val/ \
    --output_csv iris_td/pad_scores/ddpm_val_simplex_tstar500_steps50.csv \
    --labels iris_td/labels/val_labels.csv \
    --noise_type simplex --t_star 500 --num_steps 50 \
    --batch_size 32 --split val
echo "✔ [$(ts)] C3-A done"

echo "▶ [$(ts)] C3-B: Test LPIPS scoring"
$PY iris_td/models/ddpm_anomaly_detector.py \
    --config $CFG --checkpoint $CKPT \
    --test_dir iris_td/data/evaluation_sets/test/ \
    --output_csv iris_td/pad_scores/ddpm_test_simplex_tstar500_steps50.csv \
    --labels iris_td/labels/test_labels.csv \
    --noise_type simplex --t_star 500 --num_steps 50 \
    --batch_size 32 --split test
echo "✔ [$(ts)] C3-B done"

echo "▶ [$(ts)] C3-C: Val ViT scoring"
$PY iris_td/models/ddpm_vit_scorer_unified.py \
    --noise_type simplex --t_star 500 --num_steps 50 \
    --split val \
    --output_csv iris_td/pad_scores/ddpm_val_simplex_tstar500_steps50_vit.csv
echo "✔ [$(ts)] C3-C done"

echo "▶ [$(ts)] C3-D: Test ViT scoring"
$PY iris_td/models/ddpm_vit_scorer_unified.py \
    --noise_type simplex --t_star 500 --num_steps 50 \
    --split test \
    --output_csv iris_td/pad_scores/ddpm_test_simplex_tstar500_steps50_vit.csv
echo "✔ [$(ts)] C3-D done"

echo "▶ [$(ts)] C3-E: Fusion ablation"
$PY iris_td/evaluation/generate_ddpm_scoring_ablation.py \
    --config_name AnoDDPM_simplex_steps50 \
    --val_vit    iris_td/pad_scores/ddpm_val_simplex_tstar500_steps50_vit.csv \
    --test_vit   iris_td/pad_scores/ddpm_test_simplex_tstar500_steps50_vit.csv \
    --val_lpips  iris_td/pad_scores/ddpm_val_simplex_tstar500_steps50.csv \
    --test_lpips iris_td/pad_scores/ddpm_test_simplex_tstar500_steps50.csv \
    --output     iris_td/final_results/anoddpm_simplex_steps50_scoring_ablation.csv
echo "✔ [$(ts)] C3-E done → iris_td/final_results/anoddpm_simplex_steps50_scoring_ablation.csv"

# ════════════════════════════════════════════════════════════════
# CONFIG 2: DDPM-Partial — Gaussian, t*=500, 50 stochastic steps
# ════════════════════════════════════════════════════════════════
echo ""
echo "████████████████████████████████████████████████████████"
echo "CONFIG 2: DDPM-Partial — Gaussian t*=500, 50 stochastic steps"
echo "████████████████████████████████████████████████████████"

echo "▶ [$(ts)] C2-A: Val LPIPS scoring"
$PY iris_td/models/ddpm_anomaly_detector.py \
    --config $CFG --checkpoint $CKPT \
    --test_dir iris_td/data/evaluation_sets/val/ \
    --output_csv iris_td/pad_scores/ddpm_val_gaussian_tstar500_steps50.csv \
    --labels iris_td/labels/val_labels.csv \
    --noise_type gaussian --t_star 500 --num_steps 50 \
    --batch_size 32 --split val
echo "✔ [$(ts)] C2-A done"

echo "▶ [$(ts)] C2-B: Test LPIPS scoring"
$PY iris_td/models/ddpm_anomaly_detector.py \
    --config $CFG --checkpoint $CKPT \
    --test_dir iris_td/data/evaluation_sets/test/ \
    --output_csv iris_td/pad_scores/ddpm_test_gaussian_tstar500_steps50.csv \
    --labels iris_td/labels/test_labels.csv \
    --noise_type gaussian --t_star 500 --num_steps 50 \
    --batch_size 32 --split test
echo "✔ [$(ts)] C2-B done"

echo "▶ [$(ts)] C2-C: Val ViT scoring"
$PY iris_td/models/ddpm_vit_scorer_unified.py \
    --noise_type gaussian --t_star 500 --num_steps 50 \
    --split val \
    --output_csv iris_td/pad_scores/ddpm_val_gaussian_tstar500_steps50_vit.csv
echo "✔ [$(ts)] C2-C done"

echo "▶ [$(ts)] C2-D: Test ViT scoring"
$PY iris_td/models/ddpm_vit_scorer_unified.py \
    --noise_type gaussian --t_star 500 --num_steps 50 \
    --split test \
    --output_csv iris_td/pad_scores/ddpm_test_gaussian_tstar500_steps50_vit.csv
echo "✔ [$(ts)] C2-D done"

echo "▶ [$(ts)] C2-E: Fusion ablation"
$PY iris_td/evaluation/generate_ddpm_scoring_ablation.py \
    --config_name DDPM_Partial_gaussian_steps50 \
    --val_vit    iris_td/pad_scores/ddpm_val_gaussian_tstar500_steps50_vit.csv \
    --test_vit   iris_td/pad_scores/ddpm_test_gaussian_tstar500_steps50_vit.csv \
    --val_lpips  iris_td/pad_scores/ddpm_val_gaussian_tstar500_steps50.csv \
    --test_lpips iris_td/pad_scores/ddpm_test_gaussian_tstar500_steps50.csv \
    --output     iris_td/final_results/ddpm_partial_gaussian_steps50_scoring_ablation.csv
echo "✔ [$(ts)] C2-E done → iris_td/final_results/ddpm_partial_gaussian_steps50_scoring_ablation.csv"

# ════════════════════════════════════════════════════════════════
# CONFIG 1: DDPM-Vanilla — Gaussian, t*=1000, 50 stochastic steps
# ════════════════════════════════════════════════════════════════
echo ""
echo "████████████████████████████████████████████████████████"
echo "CONFIG 1: DDPM-Vanilla — Gaussian t*=1000, 50 stochastic steps"
echo "████████████████████████████████████████████████████████"

echo "▶ [$(ts)] C1-A: Val LPIPS scoring"
$PY iris_td/models/ddpm_anomaly_detector.py \
    --config $CFG --checkpoint $CKPT \
    --test_dir iris_td/data/evaluation_sets/val/ \
    --output_csv iris_td/pad_scores/ddpm_val_gaussian_tstar1000_steps50.csv \
    --labels iris_td/labels/val_labels.csv \
    --noise_type gaussian --t_star 1000 --num_steps 50 \
    --batch_size 32 --split val
echo "✔ [$(ts)] C1-A done"

echo "▶ [$(ts)] C1-B: Test LPIPS scoring"
$PY iris_td/models/ddpm_anomaly_detector.py \
    --config $CFG --checkpoint $CKPT \
    --test_dir iris_td/data/evaluation_sets/test/ \
    --output_csv iris_td/pad_scores/ddpm_test_gaussian_tstar1000_steps50.csv \
    --labels iris_td/labels/test_labels.csv \
    --noise_type gaussian --t_star 1000 --num_steps 50 \
    --batch_size 32 --split test
echo "✔ [$(ts)] C1-B done"

echo "▶ [$(ts)] C1-C: Val ViT scoring"
$PY iris_td/models/ddpm_vit_scorer_unified.py \
    --noise_type gaussian --t_star 1000 --num_steps 50 \
    --split val \
    --output_csv iris_td/pad_scores/ddpm_val_gaussian_tstar1000_steps50_vit.csv
echo "✔ [$(ts)] C1-C done"

echo "▶ [$(ts)] C1-D: Test ViT scoring"
$PY iris_td/models/ddpm_vit_scorer_unified.py \
    --noise_type gaussian --t_star 1000 --num_steps 50 \
    --split test \
    --output_csv iris_td/pad_scores/ddpm_test_gaussian_tstar1000_steps50_vit.csv
echo "✔ [$(ts)] C1-D done"

echo "▶ [$(ts)] C1-E: Fusion ablation"
$PY iris_td/evaluation/generate_ddpm_scoring_ablation.py \
    --config_name DDPM_Vanilla_gaussian_steps50 \
    --val_vit    iris_td/pad_scores/ddpm_val_gaussian_tstar1000_steps50_vit.csv \
    --test_vit   iris_td/pad_scores/ddpm_test_gaussian_tstar1000_steps50_vit.csv \
    --val_lpips  iris_td/pad_scores/ddpm_val_gaussian_tstar1000_steps50.csv \
    --test_lpips iris_td/pad_scores/ddpm_test_gaussian_tstar1000_steps50.csv \
    --output     iris_td/final_results/ddpm_vanilla_gaussian_steps50_scoring_ablation.csv
echo "✔ [$(ts)] C1-E done → iris_td/final_results/ddpm_vanilla_gaussian_steps50_scoring_ablation.csv"

# ════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════
echo ""
echo "========================================================"
echo "ALL DONE: $(ts)"
echo "Results:"
echo "  Config 3: iris_td/final_results/anoddpm_simplex_steps50_scoring_ablation.csv"
echo "  Config 2: iris_td/final_results/ddpm_partial_gaussian_steps50_scoring_ablation.csv"
echo "  Config 1: iris_td/final_results/ddpm_vanilla_gaussian_steps50_scoring_ablation.csv"
echo "========================================================"
