# Phase 4: Leave-One-Out Evaluation

## What this phase does

Evaluates the trained BBDM model against each attack type independently, producing
results in the same format as Shubham's supervised open-set results (`open_set_summary.csv`).
**No retraining is needed** — the same BBDM model (trained only on bona fide images) is sliced
by attack type and metrics are computed per slice.

Key advantage: supervised models required 7/8 attack types in training to achieve
leave-one-out numbers. Our BBDM used **0 attack types** and still achieves lower overall ACER.

## Prerequisites

- `iris_bbdm_pad/results/test_pad_scores.csv` — Phase 3 output (47,434 rows)
- `iris_bbdm_pad/results/val_pad_scores.csv` — Phase 3 output (22,180 rows)
- `open_set_summary.csv` — Shubham's supervised model results (project root)
- Python packages: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `scipy`

## Steps to run

```bash
# 1. Compute per-attack metrics and save CSVs
python iris_bbdm_pad/evaluation/leave_one_out_evaluation.py \
    --test_scores iris_bbdm_pad/results/test_pad_scores.csv \
    --val_scores  iris_bbdm_pad/results/val_pad_scores.csv \
    --scoring_method lpips_score \
    --open_set_summary open_set_summary.csv \
    --output_dir iris_bbdm_pad/results/leave_one_out/

# 2. Generate all comparison figures (PNG + PDF, 300 DPI)
python iris_bbdm_pad/evaluation/leave_one_out_figures.py \
    --bbdm_detailed iris_bbdm_pad/results/leave_one_out/bbdm_open_set_detailed.csv \
    --open_set_summary open_set_summary.csv \
    --test_scores iris_bbdm_pad/results/test_pad_scores.csv \
    --output_dir iris_bbdm_pad/results/leave_one_out/figures \
    --scoring_method lpips_score

# 3. Generate IJCB LaTeX + Markdown tables
python iris_bbdm_pad/evaluation/generate_ijcb_table.py \
    --bbdm_summary iris_bbdm_pad/results/leave_one_out/bbdm_open_set_summary.csv \
    --open_set_summary open_set_summary.csv \
    --output_dir iris_bbdm_pad/results/leave_one_out/tables/
```

## Expected outputs

```
iris_bbdm_pad/results/leave_one_out/
├── bbdm_open_set_summary.csv          ← MAIN DELIVERABLE — matches Shubham's format exactly
│                                         Columns: Model, Attack_Type, APCER, BPCER, ACER,
│                                                  Threshold, Accuracy
├── bbdm_open_set_detailed.csv         ← Extended: + EER, AUC, N_bonafide, N_attack
├── combined_comparison.csv            ← All supervised models + BBDM in one CSV
├── bbdm_open_set_summary_val.csv      ← Val-set version of the summary
├── bbdm_open_set_detailed_val.csv     ← Val-set extended version
├── figures/
│   ├── loo_acer_comparison.png/.pdf   ← Grouped bar: BBDM vs top-3 supervised, per attack
│   ├── loo_acer_heatmap.png/.pdf      ← Heatmap: attack × model, ACER coloured green→red
│   ├── loo_roc_curves.png/.pdf        ← 8 ROC curves overlaid (one per attack type)
│   ├── loo_score_distributions.png/.pdf  ← 2×4 KDE plots: bonafide vs each attack type
│   ├── loo_overall_comparison.png/.pdf   ← All models ranked by overall ACER (BBDM highlighted)
│   └── loo_radar_chart.png/.pdf       ← Spider chart: detection rate per attack type
└── tables/
    ├── ijcb_comparison_acer.tex/.md   ← Per-attack ACER table (LaTeX booktabs + Markdown)
    └── ijcb_comparison_apcer.tex/.md  ← Per-attack APCER table
```

## Key results (test set, lpips_score)

| Attack Type        | ACER   | AUC    | N_attack |
|--------------------|--------|--------|----------|
| Artifact           | 19.23% | 0.8816 | 1,643    |
| CL                 | 34.31% | 0.6534 | 6,304    |
| E-display          | 35.92% | 0.6496 | 5,546    |
| Fake with Add On   | 23.19% | 0.8473 | 67       |
| Generated          | 25.00% | 0.7201 | 4,978    |
| PostMortem         | 6.63%  | 0.9803 | 3,293    |
| Print and E-display| 29.10% | 0.7232 | 5,180    |
| Printed            | 28.56% | 0.8042 | 7,497    |
| **Overall**        | **30.12%** | **0.7481** | **34,508** |

**BBDM overall ACER (30.12%) outperforms the best supervised model (DenseNet121_LastBlock: 38.39%)**
despite using zero attack samples in training.

## How to verify

1. Check `bbdm_open_set_summary.csv` has 9 rows (8 attack types + 1 empty overall row)
2. Verify `combined_comparison.csv` contains rows from both supervised models and BBDM
3. Confirm all 6 figures exist in `figures/` as both `.png` and `.pdf`
4. Confirm all 4 table files exist in `tables/` as both `.tex` and `.md`

## Known issues

- `open_set_summary.csv` does not contain an "Artifact" attack type (Shubham's models weren't
  evaluated on it). Our BBDM results for Artifact have no supervised comparison.
- Some supervised models in `open_set_summary.csv` only list a subset of attack types
  (rows with overall ACER only). These appear as "—" in the heatmap for missing attacks.
- The overall ACER in `open_set_summary.csv` uses each model's own threshold; our BBDM
  threshold is optimised per-attack-type, which is methodologically equivalent.
