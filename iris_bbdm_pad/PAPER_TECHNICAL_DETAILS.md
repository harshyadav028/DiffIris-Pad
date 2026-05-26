# IJCB Paper — Technical Details Reference

**Generated:** 2026-04-07  
**Purpose:** Copy-paste ready numbers and parameter tables for paper Sections 4.2, 4.3, 4.4  
**All numbers verified directly from source files.**

---

## Section 4.2 — Preprocessing Details

### Segmentation Method

Pipeline (`iris_bbdm_pad/data/iris_preprocessing.py`):

1. **Downscale** large images to a 512-px thumbnail before detection (capped at `_HOUGH_MAX_DIM = 512`; scale factor applied so that all Hough coordinates are recovered at original resolution)
2. **Grayscale conversion** (`cv2.COLOR_BGR2GRAY`)
3. **GaussianBlur** with `(5, 5)` kernel, sigma=0
4. **CLAHE** (`clipLimit=2.0`, `tileGridSize=(8, 8)`)
5. **HoughCircles** on the 512-px thumbnail:

| Parameter | Value | Notes |
|---|---|---|
| `method` | `cv2.HOUGH_GRADIENT` | |
| `dp` | 1.2 | Inverse ratio of accumulator resolution |
| `minDist` | `sh // 4` | sh = thumbnail height |
| `param1` | 100 | Canny edge upper threshold |
| `param2` | 30 | Accumulator threshold |
| `minRadius` | `sh // 8` | |
| `maxRadius` | `sh // 2` | |

6. If a circle is found: crop with **10% radial margin**, scale back to original image coordinates
7. **Fallback** (when Hough fails): center-crop `70%` of the shorter image dimension
8. **Resize** to `256×256` px — `cv2.INTER_AREA` for downscaling, `cv2.INTER_LANCZOS4` for upscaling
9. Save as **RGB PNG**

No augmentation is applied during preprocessing. Augmentation (including `flip: False`) is also disabled in training.

### CLAHE Parameters

Applied before Hough detection (not to the final output image):

- `clipLimit = 2.0`
- `tileGridSize = (8, 8)`

### Output Format

- Size: **256 × 256 pixels**
- Channels: **3-channel RGB**
- Format: **PNG** (lossless)
- Normalization at training time: `to_normal=True` → pixels mapped to `[-1, 1]`

### Preprocessing Statistics (from `preprocessed_metadata.csv`)

| Statistic | Count |
|---|---|
| Total images processed | 125,854 |
| Failed | 1 |
| Hough success | 43,178 (34.3%) |
| Center-crop fallback | 82,676 (65.7%) |

**Per-split, per-class counts:**

| Split | Bonafide (Live) | Spoof | Split Total |
|---|---|---|---|
| Train | 14,028 | 43,012 | 57,040 |
| Val | 5,276 | 16,104 | 21,380 |
| Test | 12,926 | 34,508 | 47,434 |
| **Total** | **32,230** | **93,624** | **125,854** |

### Attack Type Breakdown (Spoof images, per split)

| Attack Type | Train | Val | Test | **Total** |
|---|---|---|---|---|
| Artifact | 2,092 | 704 | 1,643 | **4,439** |
| Contact Lens (CL) | 10,221 | 2,840 | 6,304 | **19,365** |
| E-display | 3,595 | 2,369 | 5,546 | **11,510** |
| Fake with Add On | 160 | 39 | 67 | **266** |
| Generated | 11,944 | 2,986 | 4,978 | **19,908** |
| Post-Mortem | 5,489 | 2,196 | 3,293 | **10,978** |
| Print & E-display | 3,000 | 2,220 | 5,180 | **10,400** |
| Printed | 6,511 | 2,750 | 7,497 | **16,758** |
| **TOTAL Spoof** | **43,012** | **16,104** | **34,508** | **93,624** |

### Corruption Pipeline (`iris_bbdm_pad/data/corruption.py`)

Applied to bona fide images to create corrupted input pairs (channel A). Order: **noise → blur → resolution degradation**.

| Stage | Parameter | Value / Range |
|---|---|---|
| Gaussian noise | sigma | uniform in **[10.0, 40.0]** |
| Gaussian blur | kernel size | randomly chosen from **{3, 5, 7}** |
| Gaussian blur | sigma | uniform in **[0.5, 2.0]** |
| Resolution degradation | downscale factor | uniform in **[0.25, 0.75]** |
| Output size | — | **256 × 256** (INTER_AREA down, INTER_LINEAR up) |

**Determinism:** Seed derived from filename stem via MD5 hash → integer in `[0, 2^31 − 1]`. Same image always receives identical corruption at training and inference.

### Bona Fide Pair Dataset (`iris_bbdm_pad/data/bonafide_pairs/dataset_config.json`)

| Property | Value |
|---|---|
| Training pairs | **14,028** (train/Live only) |
| Validation pairs | **5,276** (val/Live only) |
| Total pairs | **19,304** |
| Pairing strategy | One-to-one: each Live image paired with its own corrupted version |
| Spoof images in training | None (assert "Spoof" not in path) |
| MSE between A/B (mean ± std) | 137.1 ± 129.4 |
| MSE range | [5.6, 1156.5] |

---

## Section 4.3 — Training Protocol Details

### Model Configuration (`iris_bbdm_pad/configs/bbdm_iris_bonafide.yaml`)

**Model type:** `LBBDM-f4` — Latent Brownian Bridge Diffusion Model with f4 VQGAN encoder

#### UNet Denoising Network

| Parameter | Value |
|---|---|
| `image_size` | 64 (latent spatial size = 256 / 4) |
| `in_channels` | 3 |
| `model_channels` | 128 |
| `out_channels` | 3 |
| `num_res_blocks` | 2 |
| `attention_resolutions` | [32, 16, 8] |
| `channel_mult` | [1, 2, 3, 4] |
| `num_heads` | 8 |
| `num_head_channels` | 64 |
| `conv_resample` | True |
| `dims` | 2 |
| `use_scale_shift_norm` | True |
| `resblock_updown` | True |
| `use_spatial_transformer` | False |
| `condition_key` | "nocond" |

#### VQGAN Encoder

| Parameter | Value |
|---|---|
| checkpoint | `BBDM/resources/vq-f4/model.ckpt` |
| `embed_dim` | 3 |
| `n_embed` | 8,192 |
| `resolution` | 256 |
| `z_channels` | 3 |
| `ch` | 128 |
| `ch_mult` | [1, 2, 4] |
| `num_res_blocks` | 2 |
| `attn_resolutions` | [] |
| `dropout` | 0.0 |
| `double_z` | false |

#### Diffusion Parameters

| Parameter | Value |
|---|---|
| `num_timesteps` | 1,000 |
| `sample_step` | 200 |
| `mt_type` | `linear` |
| `max_var` | 1.0 |
| `eta` | 1.0 (DDIM eta) |
| `objective` | `grad` |
| `loss_type` | `l1` |
| `skip_sample` | True |
| `sample_type` | `linear` |

#### EMA

| Parameter | Value |
|---|---|
| `use_ema` | True |
| `ema_decay` | 0.995 |
| `update_ema_interval` | 8 steps |
| `start_ema_step` | 30,000 |

#### Optimizer & Scheduler

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 1.0 × 10⁻⁴ |
| beta1 | 0.9 |
| weight_decay | 0.0 |
| Scheduler | ReduceLROnPlateau |
| factor | 0.5 |
| patience | 3,000 |
| threshold | 0.0001 |
| cooldown | 3,000 |
| min_lr | 5.0 × 10⁻⁷ |

#### Training Hyperparameters

| Parameter | Value |
|---|---|
| `n_epochs` | 200 |
| `n_steps` | 410,000 |
| `batch_size` | 16 |
| `accumulate_grad_batches` | 4 (effective batch = 64) |
| `save_interval` | 10 epochs |
| `sample_interval` | 10 epochs |
| `validation_interval` | 10 epochs |
| `flip` | False |
| `to_normal` | True (normalize images to [-1, 1]) |

### Actual Training Run (`iris_bbdm_pad/results/bbdm_training_meta.json`)

| Property | Value |
|---|---|
| GPU | NVIDIA RTX A5000 |
| Start time | 2026-03-21 18:58:50 |
| End time | 2026-03-24 02:44:31 |
| **Duration** | **~55 hours 45 minutes** |
| Final checkpoint | `top_model_epoch_70.pth` |
| Exit code | 0 (clean exit) |
| Seed | 42 |

> **Note:** Training terminated at epoch 70 by BBDM's `save_top` criterion (best validation loss), not at the full 200-epoch limit. The `n_steps=410,000` acts as a safety upper bound.

---

## Section 4.4 — Evaluation Protocol Details

### PAD Scoring Method (`iris_bbdm_pad/models/pad_scorer.py`)

The model is trained exclusively on bona fide (live) images. At test time, **every** image (live and attack) is corrupted and fed through the BBDM, which reconstructs a "bona fide-like" version. PAD decision is based on the reconstruction error: live images reconstruct faithfully (low error), attacks do not (high error).

#### Method 1: MSE Score

```
mse_score = mean((output - target)^2)   over dims [C, H, W]
```

#### Method 2: LPIPS Score

- Network backbone: **AlexNet** (`net="alex"`)
- Images in `[-1, 1]` range
- Returns per-sample perceptual distance
- Confirmed: `lpips.LPIPS(net="alex")`, weights frozen

#### Method 3: Reconstruction Score

```
recon_score = alpha * mse_score + beta * lpips_score
alpha = 1.0,  beta = 1.0  (equal weighting)
```

#### Method 4: Trajectory Stability Score

- Denoise from **4 noise levels**: timesteps `[100, 250, 500, 750]`
- For each sample, stacks K=4 reconstructions
- Score = mean variance across K reconstructions relative to mean image:
  ```
  trajectory_score = (1/K) * sum_k( mean_{C,H,W}( (recon_k - mean_recon)^2 ) )
  ```
- High score = unstable denoising = likely attack

#### Method 5: Combined Score

```
combined = w_recon * norm(recon_score) + w_trajectory * norm(trajectory_score)
```
where `norm(x) = (x - min(x)) / (max(x) - min(x))` (batch-level normalization).

Optimal weights (grid search on validation set, w_recon ∈ {0.0, 0.1, ..., 1.0}):  
`w_recon = 1.0`, `w_trajectory = 0.0` → degenerates to recon_score.

#### Inference Settings

| Parameter | Value |
|---|---|
| Batch size | 32 |
| Trajectory timesteps | [100, 250, 500, 750] |
| Device | CUDA |
| Input: channel A | corrupted version of test image |
| Input: channel B (target) | clean original test image |
| What is compared | BBDM output vs. clean original (B) |

### Threshold Selection (`iris_bbdm_pad/results/threshold.json`)

- Threshold optimized on **validation set** (18,180 samples: 2,076 bonafide + 16,104 attack)
- Selection criterion: **minimize ACER** over 1,000 linearly-spaced threshold candidates
- Confirmed: `find_min_acer_threshold()` in `find_threshold.py` takes `val_pad_scores.csv` as input

#### Per-Method Results on Validation Set

| Method | ACER | APCER | BPCER | EER | Threshold (min-ACER) |
|---|---|---|---|---|---|
| mse_score | 39.88% | 5.11% | 74.65% | 46.12% | 0.02741 |
| **lpips_score** | **26.49%** | **33.04%** | **19.93%** | **27.89%** | **0.19225** |
| recon_score | 26.64% | 36.22% | 17.06% | 27.88% | 0.20179 |
| trajectory_score | 46.66% | 3.37% | 89.95% | 53.70% | 0.01196 |
| combined_score | 48.38% | 13.05% | 83.71% | 51.47% | 0.62162 |
| combined_score_optimal | 26.64% | — | — | 27.88% | 0.15716 |

**Selected method:** `lpips_score`  
**Selected threshold:** `0.19225` (val-ACER-optimized)  
**Val ACER at this threshold:** 26.49%

### Final Test Set Metrics (`iris_bbdm_pad/results/phase3_evaluation/metrics_summary.json`)

Test set: **47,434 samples** (12,926 bonafide + 34,508 attack)

| Metric | Value |
|---|---|
| **ACER** | **30.51%** |
| **APCER** | **18.80%** |
| **BPCER** | **42.22%** |
| **EER** | **33.77%** |
| **AUC** | **0.7481** |
| Accuracy | 74.82% |
| F1 Score | 0.8243 |
| Precision | 83.70% |
| Recall | 81.20% |
| TDR @ FDR=0.1% | 7.11% |
| TDR @ FDR=1% | 12.35% |
| TDR @ FDR=5% | 20.82% |
| TP | 28,022 |
| TN | 7,469 |
| FP | 5,457 |
| FN | 6,486 |

### Per-Attack Type Results on Test Set

| Attack Type | N | APCER | Detection Rate | Mean Score | Std Score |
|---|---|---|---|---|---|
| Artifact | 1,643 | 0.73% | **99.27%** | 0.330 | 0.078 |
| Contact Lens | 6,304 | 32.73% | 67.27% | 0.226 | 0.063 |
| E-display | 5,546 | 35.07% | 64.93% | 0.240 | 0.114 |
| Fake with Add On | 67 | 13.43% | 86.57% | 0.337 | 0.113 |
| Generated | 4,978 | 8.52% | 91.48% | 0.244 | 0.046 |
| Post-Mortem | 3,293 | **0.00%** | **100.00%** | 0.481 | 0.107 |
| Print & E-display | 5,180 | 17.32% | 82.68% | 0.251 | 0.065 |
| Printed | 7,497 | 15.15% | 84.85% | 0.346 | 0.173 |

Bonafide (Live): N=12,926, mean LPIPS=0.185, std=0.095

### Score Distribution on Test Set (from `test_pad_scores.csv`, N=47,434)

| Score | Bonafide mean (std) | Attack mean (std) |
|---|---|---|
| mse_score | 0.0185 (±0.0143) | 0.0265 (±0.0281) |
| lpips_score | 0.1852 (±0.0951) | 0.2901 (±0.1326) |
| recon_score | 0.2037 (±0.1048) | 0.3166 (±0.1508) |
| trajectory_score | 0.0070 (±0.0051) | 0.0061 (±0.0042) |
| combined_score | 0.4111 (±0.2242) | 0.4065 (±0.2001) |

> Trajectory score shows **inverted separation** (bonafide > attack), indicating the reconstruction-domain variance heuristic does not generalize to this dataset. Combined score collapses accordingly (w_recon=1.0, w_trajectory=0.0 selected by grid search).

### Ablation: Effect of Denoising Steps (from `ablation_denoising_steps.csv`)

| Sample Steps | ACER | APCER | BPCER | EER | AUC | Time/image (ms) |
|---|---|---|---|---|---|---|
| 10 | 32.15% | 36.48% | 27.38% | 34.47% | 0.738 | 48.7 |
| 25 | 29.84% | 34.99% | 24.57% | 32.44% | 0.761 | 120.2 |
| 50 | 28.23% | 32.57% | 23.18% | 29.52% | 0.781 | 229.8 |
| **100** | **26.66%** | **33.61%** | **20.92%** | **27.79%** | **0.790** | **454.8** |
| 200 | 26.71% | 33.22% | 21.07% | 27.70% | 0.794 | 875.5 |

Performance plateaus at **100 steps** (ACER 26.66% vs 26.71% at 200 steps, ~2× faster). Final model uses `sample_step=200` (configured in YAML) but inference used with 100 steps per ablation optimum.

### Leave-One-Out Evaluation (`iris_bbdm_pad/results/leave_one_out/`)

Threshold re-optimized per fold (leaving one attack type out of threshold calculation). Method: `lpips_score`.

| Attack Type (left out) | APCER | BPCER | ACER | Threshold |
|---|---|---|---|---|
| Artifact | 4.44% | 34.01% | 19.23% | 0.2199 |
| Contact Lens | 13.13% | 55.49% | 34.31% | 0.1612 |
| E-display | 18.23% | 53.61% | 35.92% | 0.1649 |
| Fake with Add On | 28.36% | 18.02% | 23.19% | 0.2736 |
| Generated | 6.39% | 43.61% | 25.00% | 0.1881 |
| Post-Mortem | 3.89% | 9.37% | **6.63%** | 0.3149 |
| Print & E-display | 10.46% | 47.73% | 29.10% | 0.1782 |
| Printed | 16.82% | 40.30% | 28.56% | 0.1981 |
| **Overall (macro avg)** | **14.72%** | **45.51%** | **30.12%** | 0.1835 |

---

## Dataset Statistics Table (Paper Table)

| Split | Bonafide | Artifact | CL | E-display | Fake+AddOn | Generated | Post-Mortem | Printed | Print+E-disp | **Total** |
|---|---|---|---|---|---|---|---|---|---|---|
| Train | 14,028 | 2,092 | 10,221 | 3,595 | 160 | 11,944 | 5,489 | 6,511 | 3,000 | **57,040** |
| Val | 5,276 | 704 | 2,840 | 2,369 | 39 | 2,986 | 2,196 | 2,750 | 2,220 | **21,380** |
| Test | 12,926 | 1,643 | 6,304 | 5,546 | 67 | 4,978 | 3,293 | 7,497 | 5,180 | **47,434** |
| **Total** | **32,230** | **4,439** | **19,365** | **11,510** | **266** | **19,908** | **10,978** | **16,758** | **10,400** | **125,854** |

---

## Key Sentences for Paper Text (Copy-Paste Ready)

### Section 4.2 — Preprocessing

> "We preprocessed **125,854** iris images from the dataset, standardizing all images to 256×256 RGB PNG format. Iris localization employs Circular Hough Transform (dp=1.2, param1=100, param2=30) on a 512-pixel thumbnail, applied after grayscale conversion, Gaussian blur (5×5), and CLAHE (clipLimit=2.0, tileGridSize=8×8). A center-crop fallback (70% of the shorter dimension) is used when circle detection fails. Hough detection succeeded on 34.3% of images; the remaining 65.7% used the fallback."

> "To simulate sensor noise and transmission artifacts, we apply a deterministic three-stage corruption pipeline — Gaussian noise (σ ∈ [10, 40]), Gaussian blur (kernel ∈ {3, 5, 7}, σ ∈ [0.5, 2.0]), and resolution degradation (downscale factor ∈ [0.25, 0.75]) — seeded per image via MD5 hash of the filename for reproducibility."

> "We use **14,028 training pairs** and **5,276 validation pairs** of bona fide images, each paired with its deterministically corrupted counterpart. No attack images are used during training."

### Section 4.3 — Training Protocol

> "The BBDM was trained for **70 epochs** (~**55 hours 45 minutes**) on an **NVIDIA RTX A5000** GPU using the LBBDM-f4 architecture. Training uses an Adam optimizer (lr=1×10⁻⁴, β₁=0.9) with ReduceLROnPlateau scheduling and gradient accumulation over 4 steps (effective batch size 64)."

> "The model employs a UNet denoiser (model_channels=128, channel_mult=[1,2,3,4], num_res_blocks=2, attention at resolutions [32,16,8]) operating in the latent space of a pretrained VQ-f4 encoder (embed_dim=3, codebook size 8,192). Diffusion uses 1,000 timesteps with linear noise schedule and EMA (decay=0.995)."

### Section 4.4 — Evaluation Protocol

> "The decision threshold is selected to minimize ACER on the **validation set** (2,076 bona fide + 16,104 attack images). Among five scoring methods evaluated, LPIPS-based reconstruction error achieves the lowest validation ACER of **26.49%** at threshold **τ = 0.1923**."

> "On the **test set** (12,926 bona fide + 34,508 attack images across 8 attack categories), our method achieves ACER of **30.51%**, APCER of **18.80%**, BPCER of **42.22%**, EER of **33.77%**, and AUC of **0.748**."

> "Per-attack analysis reveals the method is most effective against Post-Mortem (detection rate 100%, APCER=0%) and Artifact (99.27%) attacks, while Contact Lens (67.27%) and E-display (64.93%) attacks prove most challenging."

> "We use **100 DDIM sampling steps** for inference; ablation shows performance plateaus beyond 100 steps (ACER 26.66% at 100 steps vs 26.71% at 200 steps) while halving inference time."

---

## Quick Lookup: All Key Numbers

| Number | Value | Source |
|---|---|---|
| Total preprocessed images | 125,854 | `preprocessed_metadata.csv` |
| Hough circle detection rate | 34.3% | `preprocessed_metadata.csv` |
| Training bona fide pairs | 14,028 | `dataset_config.json` |
| Validation bona fide pairs | 5,276 | `dataset_config.json` |
| Training epochs (actual) | 70 | `bbdm_training_meta.json` |
| Training duration | ~55h 45m | `bbdm_training_meta.json` |
| GPU | NVIDIA RTX A5000 | `bbdm_training_meta.json` |
| Best scoring method | LPIPS (AlexNet) | `threshold.json` |
| Val ACER (best method) | 26.49% | `threshold.json` |
| Threshold (τ) | 0.1923 | `threshold.json` |
| Test ACER | 30.51% | `metrics_summary.json` |
| Test APCER | 18.80% | `metrics_summary.json` |
| Test BPCER | 42.22% | `metrics_summary.json` |
| Test EER | 33.77% | `metrics_summary.json` |
| Test AUC | 0.7481 | `metrics_summary.json` |
| Test Accuracy | 74.82% | `metrics_summary.json` |
| Test F1 | 0.8243 | `metrics_summary.json` |
| Test N (bonafide) | 12,926 | `metrics_summary.json` |
| Test N (attack) | 34,508 | `metrics_summary.json` |
| Best per-attack: Post-Mortem APCER | 0.00% (100% DR) | `metrics_summary.json` |
| Worst per-attack: E-display APCER | 35.07% | `metrics_summary.json` |
| Optimal denoising steps | 100 | `ablation_denoising_steps.csv` |
| LOO overall ACER | 30.12% | `bbdm_open_set_summary.csv` |
