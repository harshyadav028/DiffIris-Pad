# AnoDDPM Baseline — iris_td/

DDPM-based anomaly detection baseline for iris PAD. Implements four DDPM configurations evaluated in the paper, with the canonical result using AnoDDPM + DDIM Simplex (T=500, 50 steps) with Dynamic LPIPS+ViT fusion scoring.

**Baseline ACER: 0.397** on LivDet-Iris 2025 (Table 3, paper).

---

## Folder Structure

```
iris_td/
├── configs/
│   └── ddpm_iris.yaml              ← DDPM training config
├── models/
│   ├── ddpm_model.py               ← LatentDDPM model
│   ├── ddpm_pad_scorer.py          ← MSE/LPIPS scoring
│   ├── ddpm_vit_scorer.py          ← ViT-B/16 scoring
│   ├── ddpm_vit_scorer_unified.py  ← unified scorer for all 4 DDPM configs
│   └── ddpm_anomaly_detector.py    ← inference engine
├── training/
│   ├── train_ddpm_bonafide.py      ← training launcher
│   ├── run_ddpm_scoring.py         ← inference CLI
│   └── find_ddpm_threshold.py      ← val-set threshold sweep
├── evaluation/
│   ├── generate_anoddpm_scoring_ablation.py
│   └── generate_ddpm_scoring_ablation.py
├── scripts/
│   ├── create_labels.py
│   ├── verify_data.py
│   └── verify_pipeline.py
├── data/
│   └── bonafide_pairs/             ← training pairs (shared format with iris_bbdm_pad)
├── pad_scores/                     ← per-config inference score CSVs
├── final_results/                  ← ablation tables and comparison CSVs
└── results/
    └── ddpm_run1/DDPM/checkpoint/  ← training outputs + checkpoints
```

---

## The Four DDPM Configurations

| Config | Noise | T | Steps | Scoring | ACER |
|---|---|---|---|---|---|
| Vanilla Gaussian | Gaussian | 1000 | 25 | ViT | ~0.49 |
| Partial Gaussian | Gaussian | 500 | 25 | ViT | ~0.46 |
| AnoDDPM Simplex | Simplex | 500 | 25 | ViT | ~0.44 |
| **AnoDDPM DDIM Simplex** | **Simplex** | **500** | **50 DDIM** | **Dynamic LPIPS+ViT** | **0.397** |

The fourth configuration is the canonical Table 3 result.

---

## Quick Start

### Download checkpoint

```bash
mkdir -p iris_td/results/ddpm_run1/DDPM/checkpoint/
wget [HF_URL]/top_model_epoch_best.pth \
    -P iris_td/results/ddpm_run1/DDPM/checkpoint/
```

### Run AnoDDPM DDIM Simplex scoring

```bash
PYTHONNOUSERSITE=1 /home/teaching/miniconda3/envs/bbdm_clean/bin/python \
    iris_td/training/run_ddpm_scoring.py \
    --config iris_td/configs/ddpm_iris.yaml \
    --split test --noise_type simplex --use_ddim --steps 50
```

### Compute dynamic fusion metrics

```bash
PYTHONNOUSERSITE=1 /home/teaching/miniconda3/envs/bbdm_clean/bin/python \
    iris_td/evaluation/generate_anoddpm_scoring_ablation.py
```

Results: `iris_td/final_results/anoddpm_ddim_50steps_dynamic_combined_format.csv`

---

## Key Implementation Notes

- `ddpm_vit_scorer_unified.py` handles all 4 configurations via `noise_type` and `use_ddim` flags
- Simplex noise generation: `generate_simplex_noise(shape, device, scale, octaves, seed)` — OpenSimplex for structured spatial noise
- Dynamic fusion weights are computed per-sample from score reliability estimates (from [22] fingerphoto method)
- VQGAN encoder shared with `iris_bbdm_pad` — same `BBDM/resources/vq-f4/model.ckpt`
