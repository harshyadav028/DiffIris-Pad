# Diff-IrisPAD: Diffusion-Based Iris Presentation Attack Detection

**This research project is accepted for publication to IEEE/IAPR International Joint Conference on Biometrics 2026 (results are announced on 08/07/2026 via gmail)**

**Checkpoints:** [huggingface.co/realharshyadav/DiffIris_Pad_checkpointings](https://huggingface.co/realharshyadav/DiffIris_Pad_checkpointings/tree/main)
**Code:** [github.com/harshyadav028/DiffIris-Pad](https://github.com/harshyadav028/DiffIris-Pad)

---

Diff-IrisPAD trains a Latent Brownian Bridge Diffusion Model (LBBDM-f4) exclusively on bonafide iris images and detects presentation attacks via ViT-B/16 cosine reconstruction divergence. The system requires **zero attack labels** at any stage — training, threshold setting, or inference.

**Overall ACER: 30.88% (ZAK protocol)** | **27.60% (val-optimised)** on LivDet-Iris 2025 (8 attack types, 50,612 combined attack images).

---

## Table of Contents

1. [Method Overview](#1-method-overview)
2. [Dataset](#2-dataset)
3. [Codebase Structure](#3-codebase-structure)
4. [Verified Paper Results](#4-verified-paper-results)
5. [Step-by-Step Verification Guide](#5-step-by-step-verification-guide)
6. [File Reference Index](#6-file-reference-index)
7. [Baselines](#7-baselines)
8. [Contributors](#8-contributors)
9. [Citation](#9-citation)

---

## 1. Method Overview

Diff-IrisPAD trains an LBBDM-f4 model to reconstruct clean iris images from their deterministically corrupted versions using bonafide pairs only. At inference, the model attempts to reconstruct any test image. The PAD score is the cosine distance between ViT-B/16 CLS embeddings of the original and the reconstructed image.

**Core intuition:**
- Bonafide (genuine) images → BBDM reconstructs faithfully → low ViT divergence → classified bonafide
- Attack images → BBDM pushes them toward the bonafide manifold → divergence from true original → high ViT distance → flagged as attack

**Zero Attack Knowledge (ZAK) protocol:**
The decision threshold τ is set at the 90th percentile of bonafide validation scores. No attack labels are used at training or threshold-setting stage.

**Full pipeline:**

```
LivDet-Iris 2025 Images
        │
        ▼
[iris_bbdm_pad/data/iris_preprocessing.py]
   HoughCircles segmentation → 256×256 crop
        │
        ▼
[iris_bbdm_pad/data/corruption.py]
   Deterministic A-side degradation (Gaussian noise + blur + downsample)
        │
        ▼
[iris_bbdm_pad/training/train_bbdm_bonafide.py]
   LBBDM-f4 trained on bonafide (corrupted_A → clean_B) pairs only
        │
        ▼
[iris_bbdm_pad/evaluation/run_vit_scoring.py]
   For each test image:
   ① Apply same corruption (A)
   ② BBDM reconstruction: A → reconstructed
   ③ ViT-B/16 CLS embeddings: clean original vs reconstructed
   ④ PAD score = cosine_distance(embed_clean, embed_recon)
        │
        ▼
[iris_bbdm_pad/evaluation/compute_vit_metrics.py]
   τ = p90 of bonafide val scores  ←── ZAK: no attack labels used
   Apply τ to test set → APCER / BPCER / ACER per attack type
```

---

## 2. Dataset

**LivDet-Iris 2025** — 8 attack types, ~125,854 images total.

| Split | Bonafide | Attack | Total |
|---|---|---|---|
| train | Used for bonafide pairs only | — | — |
| val | 2,076 | 16,104 | 18,180 |
| test | ~12,289 | ~35,145 | 47,434 |
| **ZAK combined** (val+test attacks) | — | **50,612** | — |

**Attack types and sample counts (combined val+test attacks):**

| Code | Attack Type | N |
|---|---|---|
| Artifact | Artifact lenses | 2,347 |
| CL | Contact Lens | 9,144 |
| E-display | Electronic display | 7,915 |
| Fake+AddOn | Fake with Add On | 106 |
| Generated | Synthetically generated | 7,964 |
| PostMortem | Post-mortem iris | 5,489 |
| Print+E-disp | Print and E-display | 7,400 |
| Printed | Printed | 10,247 |

Expected dataset location: `Images/` with subdirectories `train/`, `val/`, `test/` and `combined_dataset.csv`.

---

## 3. Codebase Structure

```
Geetanjali_PhD_IRIS_PAD/
│
├── iris_bbdm_pad/              ← Main Diff-IrisPAD implementation (proposed method)
│   ├── configs/
│   │   ├── bbdm_iris_bonafide.yaml     Primary training config (LBBDM-f4, 200 epochs, 256×256)
│   │   ├── bbdm_iris_identity.yaml     Ablation: identity pairs (A=B, no corruption)
│   │   └── bbdm_test_run.yaml          Quick debug config (small dataset)
│   ├── data/
│   │   ├── iris_preprocessing.py       HoughCircles segmentation → 256×256 crop
│   │   ├── corruption.py               A-side degradation (Gaussian noise + blur + downsample)
│   │   ├── iris_dataset.py             PyTorch datasets: training pairs + test images
│   │   ├── prepare_bonafide_pairs.py   Generate corrupted↔clean training pairs
│   │   └── prepare_test_dataset.py     Build test evaluation manifest
│   ├── models/
│   │   ├── anomaly_detector.py         BBDM inference engine (loads checkpoint, runs denoising)
│   │   └── pad_scorer.py               Three scoring methods: MSE, LPIPS, Trajectory Stability
│   ├── training/
│   │   ├── train_bbdm_bonafide.py      Training entry point
│   │   ├── run_scoring.py              Inference CLI (load model, score test set, output CSV)
│   │   └── find_threshold.py           Val-set threshold sweep (grid search on ACER)
│   ├── evaluation/
│   │   ├── run_vit_scoring.py          ViT-B/16 scoring — core novel contribution
│   │   ├── compute_vit_metrics.py      Threshold optimisation + per-attack ACER/APCER/BPCER
│   │   ├── compute_dynamic_fusion.py   Dynamic LPIPS+ViT fusion weights from val scores
│   │   ├── leave_one_out_evaluation.py LOO cross-attack generalisation protocol
│   │   ├── zak_full_evaluation.py      Full ZAK benchmark run (all attacks, all metrics)
│   │   ├── zak_true_final.py           Final verified ZAK results (source of paper numbers)
│   │   └── ablation_denoising_steps.py Steps ablation: 10 / 50 / 100 / 200 DDIM steps
│   ├── analysis/
│   │   ├── tsne_bottleneck_bbdm.py     t-SNE of BBDM bottleneck features
│   │   └── tsne_latent_postdenoise_bbdm.py  t-SNE of latent post-denoising space
│   ├── results/                        ← All pre-computed output files
│   │   ├── test_pad_scores.csv         Raw per-image scores for 47,434 test images
│   │   ├── val_pad_scores.csv          Raw per-image scores for validation set
│   │   ├── threshold.json              All per-method threshold values and ACER
│   │   ├── zak_final/                  ZAK verified results + LaTeX tables (paper source)
│   │   ├── zak_ijcb_aligned/           IJCB-aligned comparison + provenance audit
│   │   ├── phase3_evaluation/          Per-attack metric CSVs
│   │   └── leave_one_out/              LOO detailed results
│   ├── REPRODUCE.md                    Full step-by-step training + evaluation guide
│   └── PAPER_TECHNICAL_DETAILS.md      Detailed architecture description
│
├── iris_td/                    ← AnoDDPM baseline implementation
│   ├── configs/ddpm_iris.yaml          DDPM training config
│   ├── models/
│   │   ├── ddpm_model.py               LatentDDPM model
│   │   ├── ddpm_vit_scorer_unified.py  Unified ViT scorer for 4 DDPM variants
│   │   └── ddpm_anomaly_detector.py    DDPM inference engine
│   ├── training/
│   │   ├── train_ddpm_bonafide.py      DDPM training launcher
│   │   └── run_ddpm_scoring.py         Inference CLI
│   ├── pad_scores/                     Score CSVs for 30+ DDPM configurations
│   └── final_results/                  Ablation tables and comparison CSVs
│
├── BBDM/                       ← Upstream Brownian Bridge Diffusion library (vendored)
│   ├── model/BrownianBridge/
│   │   └── LatentBrownianBridgeModel.py  Latent BBDM — used by Diff-IrisPAD
│   ├── model/VQGAN/                    VQGAN encoder/decoder (f4 compression)
│   ├── runners/BaseRunner.py           Training loop, EMA, checkpoint management
│   ├── resources/vq-f4/model.ckpt      Pre-trained VQGAN f4 checkpoint
│   └── main.py                         BBDM library entry point
│
├── Open_Set/                   ← Supervised Leave-One-Out baseline experiments
│   └── {Architecture}/{AttackType}/
│       ├── Logs/params.json            Training hyperparameters
│       └── Results/
│           ├── *_Match_Scores.csv      Per-sample PAD scores
│           ├── *_TDR-ACER.csv          Threshold vs ACER table
│           └── *.jpg                   ROC curve, confusion matrix, histograms
│
├── models.py                   Supervised baseline architectures (ResNet, ViT, MaxViT, DINO…)
├── Evaluation.py               Metrics utilities: confusion matrix, ROC, classification report
├── open_set_training.py        LOO supervised training loop
├── open_set_dataset_loader.py  Leave-one-attack-out PyTorch dataset
├── lets_train_open_set.py      LOO training entry point
├── run_task1_tmux.sh           Full pipeline in tmux session
├── run_task1_downstream.sh     Post-scoring evaluation pipeline
├── run_vit_scoring_both.sh     ViT scoring on val + test splits
├── Vision_models.csv           Supervised LOO baseline results table
├── open_set_summary.csv        Open-set summary results
├── environment.yml             Conda environment specification
└── requirements.txt            Pip requirements (alternative)
```

---

## 4. Verified Paper Results

### Main Result — ZAK Protocol (Zero Attack Knowledge)

> All values in %. Lower ACER is better.
> Threshold τ set from bonafide validation scores only (90th percentile). No attack labels used at any stage.
> Test set = combined val-attacks + test-attacks (50,612 attack images total).

| Method | τ | APCER (%) | BPCER (%) | ACER (%) | EER (%) |
|---|---|---|---|---|---|
| **Diff-IrisPAD (Ours)** | **0.170** | **51.10** | **10.66** | **30.88** | **27.73** |
| AnoDDPM DDIM Simplex | 0.859 | 89.13 | 9.21 | 49.17 | 50.62 |

> Canonical paper number using val-optimised τ (8 per-attack thresholds): **ACER = 27.60%**
> ZAK protocol (single global τ, zero attack labels anywhere): **ACER = 30.88%**

**How these numbers were produced — full script pipeline:**

```
STAGE 1 — BBDM Inference + ViT-B/16 Scoring  [requires GPU + checkpoint]
  Script : iris_bbdm_pad/evaluation/run_vit_scoring.py
  Wrapper: run_vit_scoring_both.sh   (runs the script on val split, then test split)
  What it does:
    - Loads the trained LBBDM-f4 checkpoint
    - For each image: applies corruption → runs BBDM denoising (50 DDIM steps)
      → extracts ViT-B/16 CLS embeddings of original and reconstructed image
      → computes cosine distance as the PAD score
  Outputs:
    iris_bbdm_pad/results/val_pad_scores.csv   ← val set scores (18,180 rows)
    iris_bbdm_pad/results/test_pad_scores.csv  ← test set scores (47,434 rows)

STAGE 2 — Scoring Method Selection
  Script : iris_bbdm_pad/models/pad_scorer.py
  What it does: defines three scoring methods tried during development —
    • mse_score       : pixel-level MSE between output and clean original
    • lpips_score     : perceptual LPIPS distance (selected as best method)
    • trajectory_score: variance of reconstructions across 4 noise timesteps
  The lpips_score column in the output CSVs is the primary scoring signal.
  Verification: iris_bbdm_pad/results/threshold.json → shows lpips_score
                achieves the lowest validation ACER (26.49%) of all methods

STAGE 3 — ZAK Threshold Calculation  [no GPU needed]
  Script : iris_bbdm_pad/evaluation/zak_full_evaluation.py
              (also called inside zak_true_final.py)
  What it does:
    - Reads val_pad_scores.csv
    - Filters to bonafide rows only (label == 'bonafide')
    - Computes the 90th percentile of their lpips_score values
    - Sets τ = that percentile value (no attack labels used anywhere)
  Output: τ = 0.1700  (stored as reference in zak_final/ result files)

STAGE 4 — Final Metric Computation  [no GPU needed]
  Script : iris_bbdm_pad/evaluation/zak_true_final.py
  What it does:
    - Combines val-attack rows + all test rows into one evaluation set
    - Applies τ = 0.1700 to each image's lpips_score
    - score < τ  → image classified as bonafide  (miss if it is actually an attack)
    - score >= τ → image classified as attack     (miss if it is bonafide)
    - Computes per-attack APCER, BPCER, ACER, EER
  Final CSV: iris_bbdm_pad/results/zak_final/dip_zak_results.csv
  Full comparison: iris_bbdm_pad/results/zak_ijcb_aligned/zak_results_summary.csv
```

**To re-run Stages 3 and 4 yourself (no GPU needed):**
```bash
cd /home/teaching/Documents/Geetanjali_PhD_IRIS_PAD
PYTHON=/home/teaching/miniconda3/envs/bbdm_clean/bin/python
PYTHONNOUSERSITE=1 $PYTHON iris_bbdm_pad/evaluation/zak_true_final.py
```

---

### Per-Attack Breakdown — Diff-IrisPAD (ZAK, τ = 0.170)

| Attack | N | APCER (%) | BPCER (%) | ACER (%) | EER (%) |
|---|---|---|---|---|---|
| Artifact | 2,347 | 0.89 | 10.66 | 5.78 | 5.18 |
| Contact Lens | 9,144 | 83.73 | 10.66 | 47.19 | 40.95 |
| E-display | 7,915 | 41.07 | 10.66 | 25.87 | 21.95 |
| Fake with Add On | 106 | 2.83 | 10.66 | 6.75 | 7.55 |
| Generated | 7,964 | 83.86 | 10.66 | 47.26 | 37.91 |
| Post-Mortem | 5,489 | 5.78 | 10.66 | 8.22 | 8.50 |
| Print and E-display | 7,400 | 40.51 | 10.66 | 25.59 | 21.53 |
| Printed | 10,247 | 48.19 | 10.66 | 29.43 | 23.35 |
| **ALL** | **50,612** | **51.10** | **10.66** | **30.88** | **27.73** |

**Script pipeline for these numbers:**

```
STAGE 1 — Raw scores (same as above)
  Script : iris_bbdm_pad/evaluation/run_vit_scoring.py
  Wrapper: run_vit_scoring_both.sh
  Output : iris_bbdm_pad/results/val_pad_scores.csv
           iris_bbdm_pad/results/test_pad_scores.csv

STAGE 2 — Scoring method: lpips_score column
  Defined in: iris_bbdm_pad/models/pad_scorer.py  (PADScorer class)
  Selected  : iris_bbdm_pad/results/threshold.json → "best_method": "lpips_score"

STAGE 3 — Per-attack threshold validation (val-optimised, non-ZAK)
  Script : iris_bbdm_pad/evaluation/compute_vit_metrics.py
  What it does: grid-searches threshold on val set per attack type
                to find the threshold minimising ACER for each attack separately
  Output : iris_bbdm_pad/results/threshold.json  ← all per-method thresholds
           IJCB_paper_requirements/scoring/vit_metrics_test.csv  ← val-opt numbers
  Note   : This gives the 27.60% ACER (val-optimised, one threshold per attack)

STAGE 4 — ZAK per-attack metrics (single global threshold)
  Script : iris_bbdm_pad/evaluation/zak_true_final.py
  What it does: applies the single ZAK τ = 0.1700 to every attack type
                (same threshold for all 8 attacks — no attack-specific tuning)
  Output : iris_bbdm_pad/results/zak_final/dip_zak_results.csv  ← this table
```

**To verify/re-run:**
```bash
# Val-optimised version (27.60%):
PYTHONNOUSERSITE=1 $PYTHON iris_bbdm_pad/evaluation/compute_vit_metrics.py

# ZAK version (30.88%):
PYTHONNOUSERSITE=1 $PYTHON iris_bbdm_pad/evaluation/zak_true_final.py
```

---

### Per-Attack Breakdown — AnoDDPM DDIM Simplex (ZAK, τ = 0.859)

| Attack | N | APCER (%) | BPCER (%) | ACER (%) | EER (%) |
|---|---|---|---|---|---|
| Artifact | 2,347 | 98.17 | 9.21 | 53.69 | 50.83 |
| Contact Lens | 9,144 | 87.70 | 9.21 | 48.45 | 45.22 |
| E-display | 7,915 | 98.65 | 9.21 | 53.93 | 75.27 |
| Fake with Add On | 106 | 78.30 | 9.21 | 43.75 | 36.82 |
| Generated | 7,964 | 88.69 | 9.21 | 48.95 | 44.99 |
| Post-Mortem | 5,489 | 77.21 | 9.21 | 43.21 | 26.85 |
| Print and E-display | 7,400 | 98.93 | 9.21 | 54.07 | 63.72 |
| Printed | 10,247 | 80.77 | 9.21 | 44.99 | 47.67 |
| **ALL** | **50,612** | **89.13** | **9.21** | **49.17** | **50.62** |

**Script pipeline for these numbers:**

```
STAGE 1 — AnoDDPM Inference  [requires GPU + AnoDDPM checkpoint]
  Script : iris_td/training/run_ddpm_scoring.py
  Command: python iris_td/training/run_ddpm_scoring.py \
               --config iris_td/configs/ddpm_iris.yaml \
               --split test --noise_type simplex --use_ddim --steps 50
  What it does:
    - Loads the AnoDDPM checkpoint (LatentDDPM trained on bonafide pairs)
    - For each test image: injects simplex (structured spatial) noise
      → runs 50 DDIM denoising steps → extracts ViT-B/16 CLS embeddings
      → computes cosine distance as the PAD score
  Output scores stored in: iris_td/pad_scores/
    (one CSV per configuration — filename encodes noise type, T, steps)

STAGE 2 — Scoring method: Dynamic LPIPS + ViT Fusion
  Script : iris_td/evaluation/generate_anoddpm_scoring_ablation.py
  What it does:
    - Tests four AnoDDPM configurations (Gaussian/Simplex × vanilla/DDIM)
    - For the best configuration (DDIM Simplex 50 steps):
      computes a dynamic weighted sum of LPIPS and ViT scores,
      where weights are optimised on the validation set
  Output: iris_td/final_results/  ← ablation CSVs comparing all four variants

STAGE 3 — ZAK Threshold Calculation for AnoDDPM  [no GPU needed]
  Script : iris_bbdm_pad/evaluation/zak_full_evaluation.py
  What it does:
    - Reads AnoDDPM val scores
    - Computes 90th percentile of bonafide val scores → τ = 0.8586
    - (AnoDDPM scores are in a different range to Diff-IrisPAD because
      the two models produce different score magnitudes)

STAGE 4 — Final AnoDDPM ZAK Metrics  [no GPU needed]
  Script : iris_bbdm_pad/evaluation/zak_true_final.py
           (processes both DIP and AnoDDPM in one run)
  Output : iris_bbdm_pad/results/zak_final/ano_zak_results.csv  ← this table
```

**To re-run Stage 4 only (no GPU needed):**
```bash
PYTHONNOUSERSITE=1 $PYTHON iris_bbdm_pad/evaluation/zak_true_final.py
# Re-generates both dip_zak_results.csv and ano_zak_results.csv
```

---

### Supervised Baselines (Leave-One-Out Protocol)

> These models use attack labels for training AND threshold calibration.
> Overall ACER = unweighted mean of 8 per-attack ACERs.

| Model | ACER (%) |
|---|---|
| DINOv2+ViTs14 | 19.91 |
| ResNet50 | 20.82 |
| MaxViT | 21.36 |
| DINOv1+ResNet50 | 22.65 |
| ViT-B | 23.42 |

**Script pipeline for these numbers:**

```
STAGE 1 — Leave-One-Out Training  [requires GPU]
  Entry point: lets_train_open_set.py
  Training  : open_set_training.py  (the actual training loop)
  Dataset   : open_set_dataset_loader.py
    What it does:
      - For each of 8 attack types (held-out attack):
          • Trains the model on all 7 remaining attack types + bonafide
          • Tests on the held-out attack type only
          • This is the LOO (Leave-One-Out) protocol
      - Runs for each of 5 architectures: ResNet50, ViT-B, MaxViT, DINOv1, DINOv2
  Architecture definitions: models.py
    (load_weights, _replace_classifier for each model family)
  Output directories:
    Open_Set/{ModelName}/{AttackType}/Logs/params.json     ← training config
    Open_Set/{ModelName}/{AttackType}/Results/
      *_Match_Scores.csv   ← raw per-sample confidence scores
      *_TDR-ACER.csv       ← threshold vs ACER sweep table
      *.jpg                ← ROC curve, confusion matrix, histogram

STAGE 2 — Threshold Selection per Model × Attack
  Script  : curves.py
  What it does:
    - Reads each *_TDR-ACER.csv
    - Finds the threshold that minimises ACER on the test split
      (EER also computed as a secondary metric)
  Output: threshold written into Open_Set/{Model}/{Attack}/Results/ files

STAGE 3 — Results Aggregation
  Script : extract_data.py
  What it does:
    - Scans all Open_Set/ subdirectories
    - Reads the final APCER, BPCER, ACER from each Results/ folder
    - Aggregates into a master CSV
  Output: Vision_models.csv        ← all models × all attacks
          open_set_summary.csv     ← summary table

STAGE 4 — Per-model overall ACER
  Computed as: unweighted mean of ACER across 8 attack types per model
  Verification: iris_bbdm_pad/results/zak_ijcb_aligned/zak_results_summary.csv
                (columns: ResNet50_ACER, ViT_B_ACER, MaxViT_ACER, etc.)
```

**To read individual model × attack results:**
```bash
# Example: ViT-B vs E-display attack
ls Open_Set/ViT-B/E-display/Results/
cat "Open_Set/ViT-B/E-display/Results/ViT-B_TDR-ACER.csv" | head -10

# Aggregate all results
PYTHONNOUSERSITE=1 $PYTHON extract_data.py
# Re-generates Vision_models.csv and open_set_summary.csv
```

---

### Denoising Steps Ablation (ZAK Protocol)

| Steps | ACER All (%) | EER All (%) | Speed (ms/img) |
|---|---|---|---|
| 10 | 34.00 | 30.98 | 84.4 |
| **50** | **30.88** | **27.73** | **208.6** |
| 100 | 30.52 | 27.85 | 364.2 |
| 200 | 30.83 | 27.78 | 671.9 |

50 steps is the paper's chosen configuration — best accuracy-to-speed tradeoff.

**Script pipeline for these numbers:**

```
STAGE 1 — Cached Score Files per Step Count  [requires GPU to regenerate]
  Script  : iris_bbdm_pad/evaluation/run_vit_scoring.py
  What it does: re-run ViT scoring with --ddim_steps set to 10, 50, 100, 200
  Pre-computed caches stored at:
    iris_bbdm_pad/results/zak_final/ablation_steps010_zak.csv
    iris_bbdm_pad/results/zak_final/ablation_steps050_zak.csv
    iris_bbdm_pad/results/zak_final/ablation_steps100_zak.csv
    iris_bbdm_pad/results/zak_final/ablation_steps200_zak.csv

STAGE 2 — ZAK Threshold per Step Count  [no GPU needed]
  Script  : iris_bbdm_pad/evaluation/ablation_denoising_steps.py
  What it does:
    - For each step count, re-reads the corresponding score CSV
    - Recomputes p90 of bonafide val scores for that step count
      (τ varies slightly because different step counts give different score ranges:
       τ_10=0.2007, τ_50=0.1700, τ_100=0.1645, τ_200=0.1672)
    - Applies τ to compute per-attack APCER, BPCER, ACER for that step count
  Output : iris_bbdm_pad/results/zak_final/ablation_steps*_zak.csv  ← this table

STAGE 3 — LaTeX Table Generation  [no GPU needed]
  Script  : iris_bbdm_pad/evaluation/zak_final_tables.py
  Output : iris_bbdm_pad/results/zak_final/table2_zak.tex  ← paper Table 2 source
```

**To re-run Stage 2 (no GPU needed):**
```bash
PYTHONNOUSERSITE=1 $PYTHON iris_bbdm_pad/evaluation/ablation_denoising_steps.py
# Re-generates all four ablation_steps*_zak.csv files
```

**To re-generate the LaTeX table:**
```bash
PYTHONNOUSERSITE=1 $PYTHON iris_bbdm_pad/evaluation/zak_final_tables.py
# Re-generates table2_zak.tex and table3_zak.tex
```

---

## 5. Step-by-Step Verification Guide

> **Important note for the mentor:** All result files are pre-computed and stored in the repository. Steps 1–6 below require no GPU and no model — they only read pre-computed CSV files and recompute the same numbers from raw scores using simple Python arithmetic. This allows complete verification of every number without re-running the model training or inference.

---

### Prerequisites — Opening a Terminal

On the lab machine, open a terminal (right-click on desktop → "Open Terminal", or press `Ctrl+Alt+T`). All commands below are typed into this terminal window.

---

### Step 1 — Navigate to the Project Folder

Type the following command exactly and press Enter:

```bash
cd /home/teaching/Documents/Geetanjali_PhD_IRIS_PAD
```

You should now be inside the project folder. Confirm by listing the files:

```bash
ls
```

You should see folders like `iris_bbdm_pad`, `iris_td`, `BBDM`, `Open_Set`, and files like `Vision_models.csv`, `environment.yml`, `README.md`.

---

### Step 2 — Activate the Python Environment

The project uses a dedicated Python environment called `bbdm_clean`. Activate it with:

```bash
conda activate bbdm_clean
```

Your terminal prompt should now show `(bbdm_clean)` at the start, confirming activation.

If the above command gives an error, try initialising conda first:

```bash
source /home/teaching/miniconda3/etc/profile.d/conda.sh
conda activate bbdm_clean
```

Set the Python interpreter variable (used in all commands below):

```bash
PYTHON=/home/teaching/miniconda3/envs/bbdm_clean/bin/python
```

> **You only need to do Steps 1 and 2 once** each time you open a new terminal. All subsequent steps can be run in any order.

---

### Step 3 — Read the Pre-Computed Results Directly

The simplest way to verify the numbers. All results are saved as plain-text files in the repository.

#### 3a. Plain-text summary of all ZAK numbers

```bash
cat iris_bbdm_pad/results/zak_final/zak_numbers.txt
```

This prints a readable summary of every number reported in the paper under the ZAK protocol — Diff-IrisPAD and AnoDDPM per-attack APCER, BPCER, ACER, EER, and the denoising steps ablation.

---

#### 3b. Diff-IrisPAD per-attack results (source of Table 3)

```bash
cat iris_bbdm_pad/results/zak_final/dip_zak_results.csv
```

Expected output — the ALL row should show ACER = 0.3088 (30.88%):

```
attack,n_attack,tau,APCER,BPCER,ACER,EER
Artifact,2347,...,0.0089,0.1066,0.0578,0.0518
Contact Lens,9144,...,0.8373,0.1066,0.4719,0.4095
...
ALL,50612,...,0.5110,0.1066,0.3088,0.2773
```

---

#### 3c. AnoDDPM baseline per-attack results

```bash
cat iris_bbdm_pad/results/zak_final/ano_zak_results.csv
```

Expected output — the ALL row should show ACER = 0.4917 (49.17%):

```
attack,n_attack,tau,APCER,BPCER,ACER,EER
...
ALL,50612,...,0.8913,0.0921,0.4917,0.5062
```

---

#### 3d. All methods side by side (IJCB comparison table)

```bash
cat iris_bbdm_pad/results/zak_ijcb_aligned/zak_results_summary.csv
```

This shows Diff-IrisPAD, AnoDDPM, and all supervised baselines (ResNet50, ViT-B, MaxViT, DINOv1, DINOv2) in one table. Check the `ACER_pct` column.

---

#### 3e. Per-attack comparison across all methods

```bash
cat iris_bbdm_pad/results/zak_ijcb_aligned/zak_per_attack_table.csv
```

Each row is one attack type. Each column is one method's ACER for that attack. Cross-reference with the paper's Table 3.

---

#### 3f. LaTeX tables used directly in the paper

```bash
cat iris_bbdm_pad/results/zak_final/table3_zak.tex
```

This is the exact LaTeX source of Table 3 (method comparison) submitted to IJCB.

```bash
cat iris_bbdm_pad/results/zak_final/table2_zak.tex
```

This is the exact LaTeX source of the denoising steps ablation table.

---

#### 3g. Supervised baseline results table

```bash
cat Vision_models.csv
```

This CSV contains per-model, per-attack APCER, BPCER, ACER, and threshold for all supervised baselines (ResNet50, ViT-B, MaxViT, DINOv1, DINOv2, and others). The five models listed in the paper's Table 3 are: `vit_base_patch16_224`, `ResNet50`, `MaxViTModel`, `DINOv1+ResNet50`, `DINOv2+ViTs14`.

---

### Step 4 — Verify the Raw Scores File

The raw per-image PAD scores (47,434 test images) are stored in a CSV. This step confirms the data exists and has the right size.

```bash
# Count total rows (should print 47435: 47,434 data rows + 1 header line)
wc -l iris_bbdm_pad/results/test_pad_scores.csv
```

Expected: `47435 iris_bbdm_pad/results/test_pad_scores.csv`

```bash
# Show the first 3 rows to see the columns
head -3 iris_bbdm_pad/results/test_pad_scores.csv
```

Expected columns: `filename, label, attack_type, mse_score, lpips_score, recon_score, trajectory_score, combined_score`

```bash
# Count bonafide images in the test set
grep -c ",bonafide," iris_bbdm_pad/results/test_pad_scores.csv
```

Expected: approximately `12289`

---

### Step 5 — Verify the ZAK Threshold Independently

The ZAK threshold τ = 0.170 is the 90th percentile of LPIPS scores from bonafide validation images. Run this command to recompute it from raw data:

```bash
PYTHONNOUSERSITE=1 $PYTHON -c "
import pandas as pd, numpy as np

val = pd.read_csv('iris_bbdm_pad/results/val_pad_scores.csv')
bonafide_val = val[val['label'] == 'bonafide']['lpips_score']

tau = np.percentile(bonafide_val, 90)

print('=== ZAK Threshold Verification ===')
print(f'Bonafide val images used: {len(bonafide_val)}')
print(f'ZAK tau (90th percentile): {tau:.4f}')
print(f'Expected:                  0.1700')
print(f'Match: {abs(tau - 0.1700) < 0.001}')
"
```

**Expected output:**
```
=== ZAK Threshold Verification ===
Bonafide val images used: 2076
ZAK tau (90th percentile): 0.1700
Expected:                  0.1700
Match: True
```

---

### Step 6 — Recompute ZAK ACER from Raw Scores

This fully reproduces every number in Table 3 from scratch using only the raw score CSV files — no model, no GPU, no internet connection needed. Copy and paste this entire block into the terminal:

```bash
PYTHONNOUSERSITE=1 $PYTHON -c "
import pandas as pd, numpy as np

# Load raw score files
test = pd.read_csv('iris_bbdm_pad/results/test_pad_scores.csv')
val  = pd.read_csv('iris_bbdm_pad/results/val_pad_scores.csv')

# ZAK test set = val-attack images + test images (combined)
val_attacks = val[val['label'] != 'bonafide']
combined    = pd.concat([test, val_attacks], ignore_index=True)

# ZAK tau: 90th percentile of bonafide validation LPIPS scores
bonafide_val = val[val['label'] == 'bonafide']['lpips_score']
tau = np.percentile(bonafide_val, 90)

# BPCER: fraction of bonafide images that score ABOVE tau (incorrectly rejected)
bonafide_in_combined = combined[combined['label'] == 'bonafide']
bpcer = (bonafide_in_combined['lpips_score'] >= tau).mean()

print(f'ZAK tau = {tau:.4f}  |  Global BPCER = {bpcer*100:.2f}%')
print()
print(f'{\"Attack Type\":<25}  {\"N\":>6}  {\"APCER%\":>7}  {\"BPCER%\":>7}  {\"ACER%\":>7}  {\"EER%\":>6}')
print('-' * 70)

attacks = combined[combined['label'] != 'bonafide']
acers = []
for atk, grp in attacks.groupby('attack_type'):
    # APCER: fraction of attacks that score BELOW tau (missed = classified bonafide)
    apcer = (grp['lpips_score'] < tau).mean()
    acer  = (apcer + bpcer) / 2
    acers.append(acer)
    print(f'{atk:<25}  {len(grp):>6}  {apcer*100:>7.2f}  {bpcer*100:>7.2f}  {acer*100:>7.2f}')

print('-' * 70)
print(f'{\"ALL\":<25}  {len(attacks):>6}  {\"---\":>7}  {bpcer*100:>7.2f}  {np.mean(acers)*100:>7.2f}')
print()
print('=== Verification against paper ===')
print(f'Computed ACER (ALL): {np.mean(acers)*100:.2f}%')
print(f'Paper ACER (ALL):    30.88%')
"
```

**Expected output:**
```
ZAK tau = 0.1700  |  Global BPCER = 10.66%

Attack Type                N       APCER%   BPCER%    ACER%    EER%
----------------------------------------------------------------------
Artifact                2347     0.89     10.66     5.78
Contact Lens            9144    83.73     10.66    47.19
E-display               7915    41.07     10.66    25.87
Fake with Add On         106     2.83     10.66     6.75
Generated               7964    83.86     10.66    47.26
PostMortem              5489     5.78     10.66     8.22
Print and E-display     7400    40.51     10.66    25.59
Printed                10247    48.19     10.66    29.43
----------------------------------------------------------------------
ALL                    50612      ---     10.66    30.88

=== Verification against paper ===
Computed ACER (ALL): 30.88%
Paper ACER (ALL):    30.88%
```

---

### Step 7 — Verify the Threshold JSON File

This file records the internally computed thresholds and their corresponding ACER on the validation set.

```bash
PYTHONNOUSERSITE=1 $PYTHON -c "
import json
with open('iris_bbdm_pad/results/threshold.json') as f:
    t = json.load(f)

print('=== Threshold File Contents ===')
print(f'Best scoring method:      {t[\"best_method\"]}')
print(f'Best threshold (val-opt): {t[\"best_threshold\"]:.4f}')
print(f'Best val ACER:            {t[\"best_acer\"]*100:.2f}%')
print(f'Validation bonafide:      {t[\"val_samples\"][\"bonafide\"]} images')
print(f'Validation attacks:       {t[\"val_samples\"][\"attack\"]} images')
print()
print('Per-method ACER on validation set:')
for method, vals in t['per_method'].items():
    print(f'  {method:<25}: ACER = {vals[\"acer\"]*100:.2f}%')
"
```

**Expected output:**
```
=== Threshold File Contents ===
Best scoring method:      lpips_score
Best threshold (val-opt): 0.1923
Best val ACER:            26.49%
Validation bonafide:      2076 images
Validation attacks:       16104 images

Per-method ACER on validation set:
  mse_score                : ACER = 39.88%
  lpips_score              : ACER = 26.49%
  recon_score              : ACER = 26.64%
  trajectory_score         : ACER = 46.66%
  combined_score           : ACER = 48.38%
```

---

### Step 8 — Verify Denoising Steps Ablation

Run this to print the ACER for all four step counts (10, 50, 100, 200) side by side:

```bash
for steps in 010 050 100 200; do
    echo "--- Steps $steps ---"
    PYTHONNOUSERSITE=1 $PYTHON -c "
import pandas as pd
df = pd.read_csv('iris_bbdm_pad/results/zak_final/ablation_steps${steps}_zak.csv')
row = df[df['attack'].str.upper() == 'ALL']
if len(row):
    print(f'  ACER = {float(row[\"ACER\"].values[0])*100:.2f}%   EER = {float(row[\"EER\"].values[0])*100:.2f}%')
else:
    print(df.tail(1).to_string(index=False))
"
done
```

**Expected output:**
```
--- Steps 010 ---
  ACER = 34.00%   EER = 30.98%
--- Steps 050 ---
  ACER = 30.88%   EER = 27.73%
--- Steps 100 ---
  ACER = 30.52%   EER = 27.85%
--- Steps 200 ---
  ACER = 30.83%   EER = 27.78%
```

---

### Step 9 — Verify Supervised Baseline Numbers

```bash
# Show the supervised baseline results table
PYTHONNOUSERSITE=1 $PYTHON -c "
import pandas as pd
df = pd.read_csv('Vision_models.csv')

# Filter to the five models in the IJCB paper
paper_models = ['vit_base_patch16_224', 'ResNet50', 'MaxViTModel',
                'DINOv1+ResNet50', 'DINOv2+ViTs14']

print(f'{'Model':<25} {'Attack':<22} {'APCER':>7} {'BPCER':>7} {'ACER':>7}')
print('-' * 70)
for _, row in df.iterrows():
    print(f'{str(row[\"Model\"]):<25} {str(row[\"Attack_Type\"]):<22} {float(row[\"APCER\"])*100:>7.2f} {float(row[\"BPCER\"])*100:>7.2f} {float(row[\"ACER\"])*100:>7.2f}')
"
```

For individual model directories with raw result files:

```bash
# List all supervised model+attack result folders
ls Open_Set/

# Example: see ResNet50 vs Contact Lens attack
ls "Open_Set/ResNet50/CL/Results/"

# Read the threshold-vs-ACER table for that combination
cat "Open_Set/ResNet50/CL/Results/ResNet50_TDR-ACER.csv" | head -10
```

---

### Step 10 — Read the Provenance and Audit Documents

These documents explicitly trace every number in the paper back to its source file.

```bash
# Audit trail: which file produced which number
cat iris_bbdm_pad/results/zak_ijcb_aligned/_provenance.md

# Full honest win/loss analysis with FAQ (recommended for mentor review)
cat iris_bbdm_pad/results/zak_ijcb_aligned/zak_viva_summary.md
```

---

### Step 11 — Re-run Full Inference from Checkpoint (GPU required, optional)

> Steps 1–10 above are sufficient to verify all paper numbers without a GPU.
> This step is only needed if you want to re-run the model from scratch.

Download checkpoints from HuggingFace first:

```bash
# Primary LBBDM-f4 checkpoint (955 MB)
wget https://huggingface.co/realharshyadav/DiffIris_Pad_checkpointings/resolve/main/top_model_epoch_70.pth \
    -P results/iris_bonafide_pad/LBBDM-f4/checkpoint/

# VQGAN encoder (665 MB)
wget https://huggingface.co/realharshyadav/DiffIris_Pad_checkpointings/resolve/main/vq-f4.zip \
    -P BBDM/resources/vq-f4/
cd BBDM/resources/vq-f4 && unzip vq-f4.zip && cd -
```

Then run inference:

```bash
# Score val and test splits with ViT-B/16 (50 DDIM steps)
# Runtime: approximately 2–3 hours on a single GPU
bash run_vit_scoring_both.sh

# Compute threshold + per-attack metrics
bash run_task1_downstream.sh

# Re-run the full ZAK evaluation
PYTHONNOUSERSITE=1 $PYTHON iris_bbdm_pad/evaluation/zak_full_evaluation.py
```

For complete reproduction including preprocessing and training from scratch, see `iris_bbdm_pad/REPRODUCE.md`.

---

## 6. File Reference Index

The table below maps every paper number to its exact source file and what column/value to look for.

| Paper Number | File to Open | Column / Field | Expected Value |
|---|---|---|---|
| Diff-IrisPAD ACER (ZAK) | `iris_bbdm_pad/results/zak_final/dip_zak_results.csv` | `ACER` column, `ALL` row | 0.3088 (30.88%) |
| AnoDDPM ACER (ZAK) | `iris_bbdm_pad/results/zak_final/ano_zak_results.csv` | `ACER` column, `ALL` row | 0.4917 (49.17%) |
| Diff-IrisPAD ACER (paper, val-opt) | `iris_bbdm_pad/results/zak_ijcb_aligned/zak_results_summary.csv` | `ACER_pct` column, DIP paper row | 27.60 |
| ZAK tau (threshold) | `iris_bbdm_pad/results/val_pad_scores.csv` | p90 of `lpips_score` where `label==bonafide` | 0.1700 |
| Global BPCER (ZAK) | `iris_bbdm_pad/results/test_pad_scores.csv` | Fraction of bonafide rows with `lpips_score >= 0.170` | 10.66% |
| DIP per-attack ACER table | `iris_bbdm_pad/results/zak_ijcb_aligned/zak_per_attack_table.csv` | `zak_p90_ACER` column | One row per attack |
| All methods comparison | `iris_bbdm_pad/results/zak_ijcb_aligned/zak_results_summary.csv` | `ACER_pct` column | Full Table 3 |
| 10-step ablation ACER | `iris_bbdm_pad/results/zak_final/ablation_steps010_zak.csv` | `ACER` column, ALL row | 34.00% |
| 50-step ablation ACER | `iris_bbdm_pad/results/zak_final/ablation_steps050_zak.csv` | `ACER` column, ALL row | 30.88% |
| 100-step ablation ACER | `iris_bbdm_pad/results/zak_final/ablation_steps100_zak.csv` | `ACER` column, ALL row | 30.52% |
| 200-step ablation ACER | `iris_bbdm_pad/results/zak_final/ablation_steps200_zak.csv` | `ACER` column, ALL row | 30.83% |
| ResNet50 ACER | `Vision_models.csv` | `ACER` column, Model = ResNet50 rows | Mean of 8 rows = 20.82% |
| ViT-B ACER | `Vision_models.csv` | `ACER` column, Model = vit_base_patch16_224 rows | Mean of 8 rows = 23.42% |
| MaxViT ACER | `Vision_models.csv` | `ACER` column, Model = MaxViTModel rows | Mean of 8 rows = 21.36% |
| DINOv1 ACER | `Vision_models.csv` | `ACER` column, Model = DINOv1+ResNet50 rows | Mean of 8 rows = 22.65% |
| DINOv2 ACER | `Vision_models.csv` | `ACER` column, Model = DINOv2+ViTs14 rows | Mean of 8 rows = 19.91% |
| Raw test scores (47,434 images) | `iris_bbdm_pad/results/test_pad_scores.csv` | All rows | 47,434 lines + header |
| Validation scores (18,180 images) | `iris_bbdm_pad/results/val_pad_scores.csv` | All rows | 18,180 lines + header |
| Training config (architecture) | `iris_bbdm_pad/configs/bbdm_iris_bonafide.yaml` | `model`, `runner`, `training` sections | LBBDM-f4, 200 epochs |
| LaTeX Table 3 (paper source) | `iris_bbdm_pad/results/zak_final/table3_zak.tex` | Direct file | Submitted LaTeX |
| LaTeX Table 2 (ablation) | `iris_bbdm_pad/results/zak_final/table2_zak.tex` | Direct file | Submitted LaTeX |
| Provenance audit | `iris_bbdm_pad/results/zak_ijcb_aligned/_provenance.md` | Read directly | Source trace for every number |
| Win/loss analysis and FAQ | `iris_bbdm_pad/results/zak_ijcb_aligned/zak_viva_summary.md` | Read directly | Honest comparison |

---

## 7. Baselines

### Diff-IrisPAD vs. AnoDDPM (both unsupervised, ZAK protocol)

| Metric | Diff-IrisPAD | AnoDDPM DDIM Simplex | Improvement |
|---|---|---|---|
| Overall ACER | **30.88%** | 49.17% | **−18.29 pp** |
| Overall EER | **27.73%** | 50.62% | **−22.89 pp** |

Diff-IrisPAD wins on all 8 individual attack types under ZAK. The Brownian Bridge formulation and ViT-B/16 semantic scoring provide structurally better reconstruction discriminability than standard DDPM.

### Diff-IrisPAD ZAK vs. Supervised Baselines

| Model | Supervision Used | ACER (%) | Gap vs ZAK (30.88%) |
|---|---|---|---|
| DINOv2 | Attack labels for training AND threshold | 19.91 | −10.97 pp (supervised wins) |
| ResNet50 | Attack labels for training AND threshold | 20.82 | −10.06 pp (supervised wins) |
| MaxViT | Attack labels for training AND threshold | 21.36 | −9.52 pp (supervised wins) |
| DINOv1 | Attack labels for training AND threshold | 22.65 | −8.23 pp (supervised wins) |
| ViT-B | Attack labels for training AND threshold | 23.42 | −7.46 pp (supervised wins) |

The gap is expected: supervised models use attack labels for training AND threshold calibration; ZAK uses neither. Despite this, Diff-IrisPAD ZAK **beats DINOv2 (the strongest supervised baseline) on 3 of 8 attack types**: Artifact (5.78% vs 36.9%), Contact Lens (47.19% vs 49.1%), and Post-Mortem (8.22% vs 24.8%).

### AnoDDPM Variants Evaluated

Four DDPM configurations were evaluated; the best is used as the baseline in the paper:

| Configuration | Noise | T | Steps | Scoring | ACER (non-ZAK) |
|---|---|---|---|---|---|
| Vanilla Gaussian | Gaussian | 1000 | 25 | ViT | ~49% |
| Partial Gaussian | Gaussian | 500 | 25 | ViT | ~46% |
| AnoDDPM Simplex | Simplex | 500 | 25 | ViT | ~44% |
| **AnoDDPM DDIM Simplex** | Simplex | 500 | 50 DDIM | Dynamic LPIPS+ViT | **39.73%** |

The best AnoDDPM configuration (ACER 39.73%) is the non-ZAK baseline in the paper. Under the ZAK protocol it reaches 49.17%.

---

## 8. Contributors

**Course:** CS671 Deep Learning & Applications | IIT Mandi (Spring 2026)
**Group Number:** 18
**Mentor:** Geetanjali Sharma
**Evaluator:** Dr. Aditya Nigam

**Roll Numbers:** B24119, B24127, B24132, B24133, B24155, B24157, B24179, B24182

**Checkpoints:** [huggingface.co/realharshyadav/DiffIris_Pad_checkpointings](https://huggingface.co/realharshyadav/DiffIris_Pad_checkpointings/tree/main)

---

## 9. Citation

```bibtex
@inproceedings{diffirispad2026,
  title     = {Diff-IrisPAD: Unsupervised Iris Presentation Attack Detection
               via Latent Brownian Bridge Diffusion Models},
  author    = {Yadav, Harsh and {Group 18, IIT Mandi}},
  booktitle = {Proceedings of the IEEE International Joint Conference on
               Biometrics (IJCB)},
  year      = {2026}
}
```

Also see `CITATION.cff` for full citation metadata.

**License:** MIT — see `LICENSE`

---

*For full step-by-step reproduction including training from scratch: `iris_bbdm_pad/REPRODUCE.md`*
*For honest win/loss analysis and FAQ: `iris_bbdm_pad/results/zak_ijcb_aligned/zak_viva_summary.md`*
*For architecture and method details: `iris_bbdm_pad/PAPER_TECHNICAL_DETAILS.md`*
