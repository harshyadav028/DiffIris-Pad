# Phase 2: BBDM Training & Anomaly Detection

## What This Phase Does

Trains LBBDM-f4 on bona fide noisy-to-clean pairs, then uses the trained model
as an anomaly detector. Test images are corrupted identically to training pairs,
reconstructed by BBDM, and the reconstruction error (MSE + LPIPS) against the
clean original becomes the PAD score. A threshold optimised on the validation
set classifies images as bonafide or attack. Three complementary scores are
computed: reconstruction error, denoising trajectory stability, and a weighted
combination of the two.

## Prerequisites

- Phase 1 completed successfully:
  - `iris_bbdm_pad/data/bonafide_pairs/train/A/` — 14 028 corrupted bona fide images
  - `iris_bbdm_pad/data/bonafide_pairs/train/B/` — 14 028 clean bona fide images
  - `iris_bbdm_pad/data/bonafide_pairs/val/A+B/`  — 5 276 pairs
  - `iris_bbdm_pad/data/evaluation_sets/test/`    — 47 434 images with labels.csv
  - `iris_bbdm_pad/data/evaluation_sets/val/`     — 21 380 images with labels.csv
- VQGAN checkpoint at `BBDM/resources/vq-f4/model.ckpt` (756 MB)
- NVIDIA GPU with ≥ 16 GB VRAM (training validated on RTX A5000 25.4 GB)
- Conda environment `iris_pad` with all dependencies:
  ```bash
  conda activate iris_pad
  pip install lpips umap-learn pytorch-grad-cam
  ```

## Steps to Run

### Step 1: Train BBDM (~10–20 h on RTX A5000)

```bash
# From Geetanjali_PhD_IRIS_PAD/
PYTHONPATH=BBDM:$PYTHONPATH python iris_bbdm_pad/training/train_bbdm_bonafide.py \
    --config iris_bbdm_pad/configs/bbdm_iris_bonafide.yaml \
    --gpu_ids 0
```

To resume an interrupted run:
```bash
PYTHONPATH=BBDM:$PYTHONPATH python iris_bbdm_pad/training/train_bbdm_bonafide.py \
    --config iris_bbdm_pad/configs/bbdm_iris_bonafide.yaml \
    --gpu_ids 0 \
    --resume_model results/iris_bonafide_pad/LBBDM-f4/checkpoint/latest_model_XXX.pth \
    --resume_optim results/iris_bonafide_pad/LBBDM-f4/checkpoint/last_optim_sche_XXX.pth
```

The script validates all prerequisites, sets `PYTHONPATH` correctly, streams
logs to both the console and `iris_bbdm_pad/results/bbdm_training.log`, and
saves training metadata to `iris_bbdm_pad/results/bbdm_training_meta.json`.

### Step 2: Run complete scoring pipeline (recommended)

Replace `XXX` with the actual epoch number from the best checkpoint:

```bash
PYTHONPATH=BBDM:$PYTHONPATH python iris_bbdm_pad/training/run_scoring.py \
    --config iris_bbdm_pad/configs/bbdm_iris_bonafide.yaml \
    --checkpoint results/iris_bonafide_pad/LBBDM-f4/checkpoint/top_model_epoch_XXX.pth \
    --output_dir iris_bbdm_pad/results/
```

This runs Steps 3–5 below in sequence and generates all visualisations.
Add `--resume` to continue from a previous interrupted scoring run.

### Step 3: Score validation set only

```bash
PYTHONPATH=BBDM:$PYTHONPATH python iris_bbdm_pad/models/anomaly_detector.py \
    --config iris_bbdm_pad/configs/bbdm_iris_bonafide.yaml \
    --checkpoint results/iris_bonafide_pad/LBBDM-f4/checkpoint/top_model_epoch_XXX.pth \
    --test_dir iris_bbdm_pad/data/evaluation_sets/val/ \
    --output_dir iris_bbdm_pad/results/
```

Renames output to `val_pad_scores.csv` for the next step.

### Step 4: Find optimal threshold

```bash
python iris_bbdm_pad/training/find_threshold.py \
    --scores_csv iris_bbdm_pad/results/val_pad_scores.csv \
    --output iris_bbdm_pad/results/threshold.json
```

Sweeps thresholds for all 5 scoring methods, runs weight grid search for the
combined score, and saves `threshold.json` with the recommended method and
threshold.

### Step 5: Score test set

```bash
PYTHONPATH=BBDM:$PYTHONPATH python iris_bbdm_pad/models/anomaly_detector.py \
    --config iris_bbdm_pad/configs/bbdm_iris_bonafide.yaml \
    --checkpoint results/iris_bonafide_pad/LBBDM-f4/checkpoint/top_model_epoch_XXX.pth \
    --test_dir iris_bbdm_pad/data/evaluation_sets/test/ \
    --output_dir iris_bbdm_pad/results/
```

### Step 6: Explore results in the notebook

```bash
conda activate iris_pad
jupyter notebook iris_bbdm_pad/notebooks/02_training_monitor.ipynb
```

## Expected Outputs

```
iris_bbdm_pad/results/
├── bbdm_training.log                  # Full BBDM training stdout
├── bbdm_training_meta.json            # start/end time, GPU, checkpoint path
├── val_pad_scores.csv                 # Per-image scores + labels (validation)
├── test_pad_scores.csv                # Per-image scores + labels (test)
├── threshold.json                     # Optimal threshold per method + best
└── phase2_visualizations/
    ├── training_loss_curve.png
    ├── training_samples_grid.png
    ├── reconstruction_grid_bonafide.png
    ├── reconstruction_grid_attack.png
    ├── reconstruction_comparison.png
    ├── trajectory_stability_visualization.png
    ├── trajectory_variance_by_timestep.png
    ├── score_scatter.png
    ├── score_scatter_3way.png
    ├── threshold_method_comparison.png
    ├── threshold_acer_curve.png
    ├── threshold_score_distributions.png
    ├── threshold_far_frr.png
    ├── weight_optimization_heatmap.png
    └── paper/
        ├── reconstruction_error_heatmaps.png
        ├── diffusion_denoising_frames.png
        ├── tsne_latent_embeddings.png
        ├── umap_latent_embeddings.png
        ├── tsne_by_attack_type.png
        ├── gradcam_unet_attention.png
        └── per_attack_reconstruction_grids/
            ├── recon_grid_Live.png
            ├── recon_grid_Artifact.png
            ├── recon_grid_CL.png
            └── ... (one per attack type)

iris_bbdm_pad/checkpoints/
├── val_scoring_checkpoint.json        # Resume state for val scoring
└── test_scoring_checkpoint.json       # Resume state for test scoring

results/iris_bonafide_pad/LBBDM-f4/
├── checkpoint/
│   ├── top_model_epoch_XXX.pth        # Best model (lowest val loss)
│   ├── latest_model_XXX.pth           # Most recent model
│   ├── last_model.pth                 # Alias for latest
│   └── *_optim_sche_*.pth             # Optimizer/scheduler states
├── log/                               # TensorBoard event files
└── sample/                            # Per-epoch sample reconstructions
```

## Visualisations Generated

| File | Content |
|------|---------|
| `training_loss_curve.png` | Training L1 loss vs step |
| `training_samples_grid.png` | Corrupted → reconstruction → clean at last epoch |
| `reconstruction_grid_bonafide.png` | 4 bonafide: input / recon / clean |
| `reconstruction_grid_attack.png` | 4 attacks: input / recon / clean |
| `reconstruction_comparison.png` | 1 bonafide + 1 attack, annotated with all scores |
| `trajectory_stability_visualization.png` | 4 trajectory timesteps for 1 bonafide + 1 attack |
| `trajectory_variance_by_timestep.png` | MSE ± std vs timestep for 50 bonafide + 50 attack |
| `score_scatter.png` | recon_score vs trajectory_score scatter |
| `score_scatter_3way.png` | MSE/LPIPS, recon/traj, combined histogram |
| `threshold_method_comparison.png` | Bar chart of ACER per method |
| `threshold_acer_curve.png` | ACER vs threshold for best method |
| `threshold_score_distributions.png` | Score histograms with threshold line |
| `threshold_far_frr.png` | FAR + FRR with EER crossing |
| `weight_optimization_heatmap.png` | ACER vs w_recon for combined score |
| `paper/reconstruction_error_heatmaps.png` | Pixel-wise error heatmaps (jet colormap) |
| `paper/diffusion_denoising_frames.png` | 8-step denoising trajectory |
| `paper/tsne_latent_embeddings.png` | t-SNE of VQGAN latents, binary label |
| `paper/umap_latent_embeddings.png` | UMAP of VQGAN latents |
| `paper/tsne_by_attack_type.png` | t-SNE coloured by attack type |
| `paper/gradcam_unet_attention.png` | Grad-CAM heatmaps on UNet middle_block |
| `paper/per_attack_reconstruction_grids/*.png` | One 4×4 grid per attack type |

## How to Verify Success

```bash
# 1. Check training checkpoint exists
ls results/iris_bonafide_pad/LBBDM-f4/checkpoint/top_model_epoch_*.pth

# 2. Check val scores were created
wc -l iris_bbdm_pad/results/val_pad_scores.csv
head -3 iris_bbdm_pad/results/val_pad_scores.csv

# 3. Check threshold JSON is valid
python3 -c "import json; d=json.load(open('iris_bbdm_pad/results/threshold.json')); \
    print(f'Best method: {d[\"best_method\"]}, ACER: {d[\"best_acer\"]*100:.2f}%')"

# 4. Check test scores
wc -l iris_bbdm_pad/results/test_pad_scores.csv

# 5. Check all visualisations were generated
ls iris_bbdm_pad/results/phase2_visualizations/*.png
ls iris_bbdm_pad/results/phase2_visualizations/paper/*.png
```

## Resume After Interruption

**Training:**
```bash
# Find the latest checkpoint
ls results/iris_bonafide_pad/LBBDM-f4/checkpoint/

PYTHONPATH=BBDM:$PYTHONPATH python iris_bbdm_pad/training/train_bbdm_bonafide.py \
    --config iris_bbdm_pad/configs/bbdm_iris_bonafide.yaml \
    --gpu_ids 0 \
    --resume_model results/iris_bonafide_pad/LBBDM-f4/checkpoint/latest_model_XXX.pth \
    --resume_optim results/iris_bonafide_pad/LBBDM-f4/checkpoint/last_optim_sche_XXX.pth
```

**Scoring:**
```bash
# Add --resume to continue from scoring_checkpoint.json
PYTHONPATH=BBDM:$PYTHONPATH python iris_bbdm_pad/training/run_scoring.py \
    --checkpoint <path_to_best.pth> \
    --resume
```

The scoring pipeline saves progress every 50 batches to
`iris_bbdm_pad/checkpoints/{val,test}_scoring_checkpoint.json`.

## Known Issues

- **PYTHONPATH**: Always prefix commands with `PYTHONPATH=BBDM:$PYTHONPATH`.
  BBDM uses relative imports (`from datasets.custom import ...`) that require
  `BBDM/` to be on `sys.path`. Forgetting this causes `ModuleNotFoundError`.
- **n_steps**: The BBDM config uses `n_steps=410000` as the primary training
  terminator. If this is smaller than `n_epochs × iters_per_epoch`, training
  stops early. With 14 028 training pairs and batch_size=16, one epoch ≈ 877
  iterations; 200 epochs × 877 ≈ 175 400 steps, well within 410 000.
- **Checkpoint auto-detection**: BBDM's auto-detection of the best checkpoint
  is unreliable. Always pass `--resume_model` explicitly.
- **Trajectory scoring overhead**: Each image requires 4 additional forward
  passes for trajectory stability. On RTX A5000 with batch_size=32, expect
  ~5× slower scoring vs reconstruction-only. The scoring checkpoint saves
  progress every 50 batches to allow resume.
- **LPIPS device**: LPIPS (net=alex) is loaded once in PADScorer.__init__().
  Ensure the device argument matches the tensor device to avoid cross-device
  errors.
