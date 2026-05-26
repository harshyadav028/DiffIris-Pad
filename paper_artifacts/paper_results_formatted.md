# IJCB Paper — Iris PAD Evaluation Results

> Generated: 2026-04-08  
> Source: `iris_bbdm_pad/results/` (BBDM model: LBBDM-f4, scoring: LPIPS)

---

## TABLE 1: Main Results — Per-Attack Breakdown (BBDM)

Threshold is optimised per-attack-type on the validation set.
Overall row uses a global threshold (val-set optimised, threshold = 0.1835).

| Model       | Attack Type        | APCER (%) | BPCER (%) | ACER (%)  | Threshold | Accuracy (%) |
|-------------|-------------------|----------:|----------:|----------:|----------:|-------------:|
| BBDM (Ours) | Artifact           |      4.44 |     34.01 |     19.23 |    0.2199 |        69.33 |
| BBDM (Ours) | Contact Lens       |     13.13 |     55.49 |     34.31 |    0.1612 |        58.40 |
| BBDM (Ours) | E-display          |     18.23 |     53.61 |     35.92 |    0.1649 |        57.01 |
| BBDM (Ours) | Fake with Add On   |     28.36 |     18.02 |     23.19 |    0.2736 |        81.93 |
| BBDM (Ours) | Generated          |      6.39 |     43.61 |     25.00 |    0.1881 |        66.74 |
| BBDM (Ours) | Post-Mortem        |      3.89 |      9.37 |  **6.63** |    0.3149 |        91.74 |
| BBDM (Ours) | Print & E-display  |     10.46 |     47.73 |     29.10 |    0.1782 |        62.93 |
| BBDM (Ours) | Printed            |     16.82 |     40.30 |     28.56 |    0.1981 |        68.32 |
| BBDM (Ours) | **Overall**        |     14.72 |     45.51 |     30.12 |    0.1835 |        76.89 |

*Best ACER per table bolded.*

---

## TABLE 2: Ablation Study 1 — Diffusion Type Comparison

> **NOTE — Missing files:**  
> - `iris_td/results/ddpm_run1/` contains only training checkpoints (TensorBoard logs + `.pth` files); no evaluation metrics CSV was found.  
> - No `conditional_diffusion` results directory exists anywhere in the project.  
> Placeholder rows use "—" for all metrics. Replace when evaluation scripts are run for those variants.

| Model                 | Attack Type        | APCER (%) | BPCER (%) | ACER (%)  | Threshold | Accuracy (%) |
|-----------------------|-------------------|----------:|----------:|----------:|----------:|-------------:|
| Standard DDPM         | Artifact           |         — |         — |         — |         — |            — |
| Standard DDPM         | Contact Lens       |         — |         — |         — |         — |            — |
| Standard DDPM         | E-display          |         — |         — |         — |         — |            — |
| Standard DDPM         | Fake with Add On   |         — |         — |         — |         — |            — |
| Standard DDPM         | Generated          |         — |         — |         — |         — |            — |
| Standard DDPM         | Post-Mortem        |         — |         — |         — |         — |            — |
| Standard DDPM         | Print & E-display  |         — |         — |         — |         — |            — |
| Standard DDPM         | Printed            |         — |         — |         — |         — |            — |
| Standard DDPM         | **Overall**        |         — |         — |         — |         — |            — |
| Conditional Diffusion | Artifact           |         — |         — |         — |         — |            — |
| Conditional Diffusion | Contact Lens       |         — |         — |         — |         — |            — |
| Conditional Diffusion | E-display          |         — |         — |         — |         — |            — |
| Conditional Diffusion | Fake with Add On   |         — |         — |         — |         — |            — |
| Conditional Diffusion | Generated          |         — |         — |         — |         — |            — |
| Conditional Diffusion | Post-Mortem        |         — |         — |         — |         — |            — |
| Conditional Diffusion | Print & E-display  |         — |         — |         — |         — |            — |
| Conditional Diffusion | Printed            |         — |         — |         — |         — |            — |
| Conditional Diffusion | **Overall**        |         — |         — |         — |         — |            — |
| BBDM (Ours)           | Artifact           |      4.44 |     34.01 |     19.23 |    0.2199 |        69.33 |
| BBDM (Ours)           | Contact Lens       |     13.13 |     55.49 |     34.31 |    0.1612 |        58.40 |
| BBDM (Ours)           | E-display          |     18.23 |     53.61 |     35.92 |    0.1649 |        57.01 |
| BBDM (Ours)           | Fake with Add On   |     28.36 |     18.02 |     23.19 |    0.2736 |        81.93 |
| BBDM (Ours)           | Generated          |      6.39 |     43.61 |     25.00 |    0.1881 |        66.74 |
| BBDM (Ours)           | Post-Mortem        |      3.89 |      9.37 |  **6.63** |    0.3149 |        91.74 |
| BBDM (Ours)           | Print & E-display  |     10.46 |     47.73 |     29.10 |    0.1782 |        62.93 |
| BBDM (Ours)           | Printed            |     16.82 |     40.30 |     28.56 |    0.1981 |        68.32 |
| BBDM (Ours)           | **Overall**        |     14.72 |     45.51 |     30.12 |    0.1835 |        76.89 |

*Best ACER per table bolded. DDPM and Conditional Diffusion rows are placeholders pending evaluation.*

---

## TABLE 3: Ablation Study 2 — Denoising Steps

Source: `iris_bbdm_pad/results/phase3_evaluation/ablation/ablation_denoising_steps.csv`  
(*) marks the selected operating point (100 steps). Best ACER bolded.

| Sample Steps | ACER (%)    | APCER (%) | BPCER (%) | EER (%)  | AUC    | Time (ms/img) |
|:------------:|------------:|----------:|----------:|---------:|-------:|--------------:|
|           10 |       32.15 |     36.48 |     27.38 |    34.47 | 0.7380 |         48.70 |
|           25 |       29.84 |     34.99 |     24.57 |    32.44 | 0.7608 |        120.16 |
|           50 |       28.23 |     32.57 |     23.18 |    29.52 | 0.7809 |        229.77 |
|        100\* |  **26.66**  |     33.61 |     20.92 |    27.79 | 0.7896 |        454.79 |
|          200 |       26.71 |     33.22 |     21.07 |    27.70 | 0.7945 |        875.51 |

*Steps 100 and 200 yield nearly identical ACER; 100 steps chosen as the operating point for 1.9× faster inference.*

---

## TABLE 4: Leave-One-Out Evaluation Results

Source: `iris_bbdm_pad/results/leave_one_out/bbdm_open_set_summary.csv`  
Threshold is optimised on the validation set for each attack type independently.  
The BBDM model was trained **only on bonafide images** (zero attack samples in training).

| Attack Left Out    | APCER (%) | BPCER (%) | ACER (%)  | Threshold |
|-------------------|----------:|----------:|----------:|----------:|
| Artifact           |      4.44 |     34.01 |     19.23 |    0.2199 |
| Contact Lens       |     13.13 |     55.49 |     34.31 |    0.1612 |
| E-display          |     18.23 |     53.61 |     35.92 |    0.1649 |
| Fake with Add On   |     28.36 |     18.02 |     23.19 |    0.2736 |
| Generated          |      6.39 |     43.61 |     25.00 |    0.1881 |
| Post-Mortem        |      3.89 |      9.37 |  **6.63** |    0.3149 |
| Print & E-display  |     10.46 |     47.73 |     29.10 |    0.1782 |
| Printed            |     16.82 |     40.30 |     28.56 |    0.1981 |
| **Overall (macro avg)** |  12.72 |     37.77 |     25.24 |    0.2124 |

*Best ACER bolded. Overall = macro-average across all 8 attack types.*

---

## File Discovery Summary

### Found
| File | Status | Used in |
|------|--------|---------|
| `iris_bbdm_pad/results/phase3_evaluation/metrics_summary.json` | Found | Table 1 (overall metrics reference) |
| `iris_bbdm_pad/results/phase3_evaluation/per_attack_metrics.csv` | Found | Table 1 (per-attack APCER) |
| `iris_bbdm_pad/results/phase3_evaluation/ablation/ablation_denoising_steps.csv` | Found | Table 3 |
| `iris_bbdm_pad/results/leave_one_out/bbdm_open_set_summary.csv` | Found | Tables 1, 2, 4 |
| `iris_bbdm_pad/results/leave_one_out/bbdm_open_set_detailed.csv` | Found | Table 4 (EER/AUC reference) |
| `iris_bbdm_pad/results/leave_one_out/combined_comparison.csv` | Found | Supervised baselines reference |

### Missing
| Expected File | Status | Impact |
|---------------|--------|--------|
| Standard DDPM evaluation metrics CSV | **Missing** — `iris_td/results/ddpm_run1/` has only checkpoints/logs | Table 2 has placeholder rows |
| Conditional Diffusion evaluation metrics | **Missing** — no such directory exists | Table 2 has placeholder rows |
| `iris_bbdm_pad/results/ablation_denoising_steps.csv` (root-level) | Not at root — found at `phase3_evaluation/ablation/` | Resolved |
| `iris_bbdm_pad/results/phase3_evaluation/per_attack_metrics.json` | Not found as JSON — available as CSV | Resolved |
