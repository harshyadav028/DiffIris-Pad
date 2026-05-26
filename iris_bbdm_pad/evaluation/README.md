# Phase 3: Evaluation, Ablation & Publication Figures

## What This Phase Does
Computes all PAD metrics on the test set using the optimal threshold from Phase 2,
runs an ablation study on BBDM denoising steps (10/25/50/100/200), generates
comparison tables with LivDet-Iris 2025 and supervised baseline results, and
produces all publication-ready figures and LaTeX/Markdown tables for the research paper.

## Prerequisites
- Phase 2 complete: `iris_bbdm_pad/results/test_pad_scores.csv` exists (47,434 rows)
- Phase 2 complete: `iris_bbdm_pad/results/val_pad_scores.csv` exists
- Phase 2 complete: `iris_bbdm_pad/results/threshold.json` exists
- Checkpoint: `results/iris_bonafide_pad/LBBDM-f4/checkpoint/top_model_epoch_70.pth`
- `open_set_summary.csv` in project root (supervised baseline results)
- Install: `pip install scikit-learn seaborn matplotlib scipy`

## Steps to Run

### Step 1: Full PAD Evaluation (required first)
```bash
python iris_bbdm_pad/evaluation/evaluate_pad.py \
    --test_scores iris_bbdm_pad/results/test_pad_scores.csv \
    --threshold_json iris_bbdm_pad/results/threshold.json \
    --output_dir iris_bbdm_pad/results/phase3_evaluation/
```
**Runtime**: ~30 seconds. Generates all core metrics.

### Step 2: Cross-Attack Analysis (requires Step 1)
```bash
python iris_bbdm_pad/evaluation/cross_attack_evaluation.py \
    --test_scores iris_bbdm_pad/results/test_pad_scores.csv \
    --threshold_json iris_bbdm_pad/results/threshold.json \
    --output_dir iris_bbdm_pad/results/phase3_evaluation/
```

### Step 3: Comparison Tables
```bash
python iris_bbdm_pad/evaluation/comparison_tables.py \
    --our_results iris_bbdm_pad/results/phase3_evaluation/metrics_summary.json \
    --our_per_attack iris_bbdm_pad/results/phase3_evaluation/cross_attack_detailed.csv \
    --open_set_csv open_set_summary.csv \
    --output_dir iris_bbdm_pad/results/phase3_evaluation/
```

### Step 4: LaTeX/Markdown Tables
```bash
python iris_bbdm_pad/evaluation/generate_results_tables.py \
    --metrics_json iris_bbdm_pad/results/phase3_evaluation/metrics_summary.json \
    --per_attack_csv iris_bbdm_pad/results/phase3_evaluation/per_attack_metrics.csv \
    --all_methods_csv iris_bbdm_pad/results/phase3_evaluation/all_methods_comparison.csv \
    --output_dir iris_bbdm_pad/results/phase3_evaluation/tables/
```

### Step 5: Generate All Paper Figures (~5 minutes)
```bash
python iris_bbdm_pad/evaluation/generate_paper_figures.py \
    --test_scores iris_bbdm_pad/results/test_pad_scores.csv \
    --val_scores iris_bbdm_pad/results/val_pad_scores.csv \
    --threshold_json iris_bbdm_pad/results/threshold.json \
    --metrics_json iris_bbdm_pad/results/phase3_evaluation/metrics_summary.json \
    --ablation_csv iris_bbdm_pad/results/phase3_evaluation/ablation/ablation_denoising_steps.csv \
    --open_set_csv open_set_summary.csv \
    --cross_attack_csv iris_bbdm_pad/results/phase3_evaluation/cross_attack_detailed.csv \
    --output_dir iris_bbdm_pad/results/phase3_figures/
```

### Step 6: Ablation Study - Denoising Steps (requires GPU, ~2-4 hours)
```bash
PYTHONPATH=BBDM:$PYTHONPATH python iris_bbdm_pad/evaluation/ablation_denoising_steps.py \
    --config iris_bbdm_pad/configs/bbdm_iris_bonafide.yaml \
    --checkpoint results/iris_bonafide_pad/LBBDM-f4/checkpoint/top_model_epoch_70.pth \
    --test_dir iris_bbdm_pad/data/evaluation_sets/val/ \
    --output_dir iris_bbdm_pad/results/phase3_evaluation/ablation/ \
    --steps 10 25 50 100 200 \
    --num_samples 500
```
For synthetic placeholder data (fast testing):
```bash
python iris_bbdm_pad/evaluation/ablation_denoising_steps.py \
    --output_dir iris_bbdm_pad/results/phase3_evaluation/ablation/ \
    --skip_inference
```

### Step 7: Interactive Exploration (optional)
```bash
cd iris_bbdm_pad/notebooks && jupyter notebook 03_paper_figures.ipynb
```

## Expected Outputs

### phase3_evaluation/
| File | Description |
|------|-------------|
| metrics_summary.json | All metrics (ACER, APCER, BPCER, EER, AUC, F1, TDR@FDR) |
| metrics_summary.csv | Same as flat CSV |
| per_attack_metrics.csv | Per-attack APCER, detection rate, score stats |
| all_methods_comparison.csv | Metrics for all 5 scoring methods |
| cross_attack_detailed.csv | Extended per-attack analysis with score separation |
| confusion_matrix.csv | TP/TN/FP/FN counts |
| results_table.md | Quick markdown summary |
| comparison_livdet2025.{md,csv} | vs LivDet-Iris 2025 competitors |
| comparison_livdet2023.{md,csv} | vs LivDet-Iris 2023 results |
| paradigm_comparison.{md,csv} | Supervised vs unsupervised paradigm table |
| comparison_open_set.{md,csv} | vs 10 supervised open-set models (same dataset) |
| comparison_per_attack_vs_supervised.{md,csv} | Per-attack: BBDM vs supervised |
| comparison_all_models.{md,csv} | All 10 supervised + BBDM sorted by ACER |
| ablation/ablation_denoising_steps.{csv,json} | Ablation results |
| tables/table_main_results.{md,tex} | Main results table |
| tables/table_per_attack.{md,tex} | Per-attack results table |
| tables/table_ablation.{md,tex} | Ablation table |
| tables/table_livdet_comparison.{md,tex} | LivDet comparison table |
| tables/table_open_set_comparison.{md,tex} | Open-set baseline comparison |
| tables/table_scoring_methods.{md,tex} | Scoring method comparison |

### phase3_figures/ (15 figures, each as .png + .pdf)
| Figure | Filename | Description |
|--------|----------|-------------|
| Fig 1 | roc_curve | ROC curves for all 5 scoring methods |
| Fig 2 | det_curve | DET curve (log-log, biometrics standard) |
| Fig 3 | confusion_matrix | 2x2 confusion matrix heatmap |
| Fig 4 | score_distributions | KDE of bonafide vs attack score distributions |
| Fig 5 | per_attack_apcer | Per-attack APCER horizontal bars |
| Fig 6 | per_attack_violins | Per-attack violin plots |
| Fig 7 | ablation_steps_acer | ACER & inference time vs denoising steps |
| Fig 8 | ablation_steps_table | Ablation table rendered as figure |
| Fig 9 | comparison_livdet2025 | APCER comparison with LivDet-Iris 2025 |
| Fig 10 | comparison_open_set_overall | ACER vs 10 supervised baselines |
| Fig 11 | comparison_per_attack_heatmap | Per-attack APCER heatmap vs supervised |
| Fig 12 | comparison_per_attack_bars | Grouped bar: BBDM vs top-2 supervised |
| Fig 13 | roc_all_methods | ROC curves overlay for all methods |
| Fig 14 | method_comparison_bars | ACER bar chart across scoring methods |
| Fig 15 | pipeline_diagram | Full inference pipeline flowchart |

## Figures for Paper
- **Main results figure**: Fig 3 (confusion matrix) + Fig 4 (score distributions)
- **Primary comparison**: Fig 10 (vs supervised baselines) + Fig 11 (per-attack heatmap)
- **Method analysis**: Fig 1 (ROC) + Fig 14 (method comparison)
- **Ablation**: Fig 7 (steps vs ACER)
- **System overview**: Fig 15 (pipeline diagram)

## Known Issues
- seaborn style API changed in v0.12: use `sns.set_style('whitegrid')` not `plt.style.use('seaborn-whitegrid')`
- PDF figures require a working LaTeX installation or use `matplotlib.backends.backend_pdf`
- ablation_denoising_steps.py requires PYTHONPATH=BBDM for real inference; use `--skip_inference` for synthetic data
- Artifact attack type is absent from open_set_summary.csv (supervised models never tested on it)
