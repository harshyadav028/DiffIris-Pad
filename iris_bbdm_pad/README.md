# Diff-IrisPAD — iris_bbdm_pad/

Unsupervised iris presentation attack detection via Latent Brownian Bridge Diffusion Model (LBBDM-f4).

Trains on bona fide iris images only. Detects attacks at inference time via ViT-B/16 cosine reconstruction divergence — no attack labels ever used.

**Overall ACER: 0.276** on LivDet-Iris 2025 (8 attack types, 125,854 images).

---

## Folder Structure

```
iris_bbdm_pad/
├── configs/
│   ├── bbdm_iris_bonafide.yaml    ← primary training config
│   ├── bbdm_iris_identity.yaml    ← identity-pair variant config
│   └── bbdm_test_run.yaml         ← short debug run config
├── data/
│   ├── iris_preprocessing.py      ← HoughCircles segmentation + 256×256 crop
│   ├── corruption.py              ← deterministic degradation pipeline (A-side)
│   ├── prepare_bonafide_pairs.py  ← generates A/B training pairs
│   ├── prepare_identity_pairs.py  ← A=B identity pairs (ablation)
│   ├── prepare_test_dataset.py    ← test set manifest generation
│   ├── iris_dataset.py            ← PyTorch Dataset classes
│   └── bonafide_pairs/            ← generated training pairs (A/ + B/)
├── models/
│   ├── anomaly_detector.py        ← BBDMAnomalyDetector inference engine
│   └── pad_scorer.py              ← MSE / LPIPS / ViT / fusion scoring
├── training/
│   ├── train_bbdm_bonafide.py     ← training launcher (entry point)
│   ├── train_bbdm_identity.py     ← identity-pair training variant
│   ├── run_scoring.py             ← inference CLI
│   └── find_threshold.py          ← val-set threshold sweep
├── evaluation/
│   ├── evaluate_pad.py
│   ├── leave_one_out_evaluation.py
│   ├── compute_vit_metrics.py
│   ├── compute_dynamic_fusion.py
│   ├── ablation_denoising_steps.py
│   ├── ablation_denoising_steps_per_attack.py
│   ├── generate_ijcb_tables.py
│   └── run_vit_scoring.py
├── analysis/
│   ├── tsne_bottleneck_bbdm.py
│   └── tsne_latent_postdenoise_bbdm.py
├── checkpoints/                   ← place downloaded .pth here (gitignored)
├── results/                       ← training + evaluation outputs
└── REPRODUCE.md                   ← step-by-step Table 3 reproduction
```

---

## Quick Start

### 1. Download checkpoint

```bash
mkdir -p results/iris_bonafide_pad/LBBDM-f4/checkpoint/
wget [HF_URL]/top_model_epoch_70.pth \
    -P results/iris_bonafide_pad/LBBDM-f4/checkpoint/
```

### 2. Run inference on test set

```bash
PYTHONNOUSERSITE=1 /home/teaching/miniconda3/envs/bbdm_clean/bin/python \
    iris_bbdm_pad/training/run_scoring.py \
    --config iris_bbdm_pad/configs/bbdm_iris_bonafide.yaml \
    --checkpoint results/iris_bonafide_pad/LBBDM-f4/checkpoint/top_model_epoch_70.pth \
    --data_dir iris_bbdm_pad/data/evaluation_sets/test/ \
    --output_csv results/test_pad_scores.csv
```

### 3. Compute metrics

```bash
bash run_task1_downstream.sh
```

See `REPRODUCE.md` for the complete step-by-step pipeline including preprocessing and training.

---

## Key Design Choices

| Decision | Choice | Why |
|---|---|---|
| Diffusion model | LBBDM-f4 (latent bridge) | 16× memory reduction vs pixel-space; bridge preserves source image |
| Scoring | ViT-B/16 cosine distance | Best on val set; semantic-level distance outperforms pixel MSE by 13 pp ACER |
| Training data | Bona fide only | Unsupervised — no attack labels needed |
| Corruption | Gaussian noise + blur + downsample | Provides non-trivial A→B learning objective |
| Inference steps | 50 DDIM steps | Best efficiency–accuracy tradeoff (ablation in `evaluation/ablation_denoising_steps.py`) |

---

## Config Notes

Three historical YAML pitfalls (documented in `configs/bbdm_iris_bonafide.yaml`):
- Use plain YAML lists `[1,2,3]`, NOT `!!python/tuple` — `yaml.safe_load` rejects the latter
- Top-level data key must be `data:`, not `Dataset:`
- `n_steps` must exceed `n_epochs × iters_per_epoch` (set to 410000 to be safe)
