# f-AnoGAN Unsupervised Baseline — Comparison Summary

**Purpose:** Add a second unsupervised baseline (f-AnoGAN) under the *identical*
protocol as Diff-IrisPAD, to answer IJCB Reviewer #2/#3:
*"Is the improvement attributable to BBDM specifically, or to reconstruction-based
anomaly detection in general?"*

## Protocol (identical to Diff-IrisPAD paper Table 2)
- Training: bona-fide pairs only (same data, same degradation C(·), frozen VQ-f4).
- Threshold: ZAK p=90 — 90th percentile of bona-fide validation scores only.
- Test pool: 50,612 attacks (val+test) vs 12,926 bona-fide (test).
- Scoring: ViT-B/16 cosine distance (same scoring as paper Table 2).
- Metrics: ISO APCER / BPCER / ACER / EER; decision: attack if S > τ.

## Headline result (OverAll ACER, lower = better)

| Method | Scoring | OverAll ACER | OverAll EER |
|---|---|---|---|
| **Diff-IrisPAD (BBDM)** | ViT | **30.52** | 27.85 |
| AnoDDPM | ViT | 49.17 | — |
| **f-AnoGAN (ours)** | ViT (same) | **51.25** | 50.97 |
| f-AnoGAN | native GAN score | 43.82 | 45.22 |
| f-AnoGAN | recon (sMSE+LPIPS) | 47.53 | 44.09 |

## Conclusion for rebuttal

Under the identical ViT scoring protocol, the unsupervised methods rank:
**Diff-IrisPAD (30.52) ≪ AnoDDPM (49.17) ≈ f-AnoGAN (51.25).**

A second, architecturally different reconstruction-based unsupervised method
(f-AnoGAN, a WGAN-GP) performs no better than AnoDDPM. Only BBDM's Brownian-Bridge
reconstruction achieves the large improvement. Even at its best-case native scoring
(43.82), f-AnoGAN trails Diff-IrisPAD by 13+ ACER points. The improvement is
therefore attributable to BBDM specifically, not to reconstruction-based anomaly
detection in general.

## f-AnoGAN per-attack (vit_score, same scoring as paper)

| Attack | APCER | BPCER | ACER | EER |
|---|---|---|---|---|
| Artifact | 99.06 | 9.55 | 54.31 | 60.00 |
| Contact Lens | 98.34 | 9.55 | 53.95 | 56.48 |
| E-display | 87.96 | 9.55 | 48.76 | 41.20 |
| Fake W/AO | 99.06 | 9.55 | 54.31 | 68.18 |
| Generated | 98.93 | 9.55 | 54.24 | 60.63 |
| Post-Mortem | 93.39 | 9.55 | 51.47 | 46.87 |
| Print & ED | 84.43 | 9.55 | 46.99 | 40.05 |
| Printed | 91.77 | 9.55 | 50.66 | 52.54 |
| **OverAll** | **92.94** | **9.55** | **51.25** | **50.97** |

## Implementation
- Folder: `iris_anogan/` (independent; BBDM and iris_td untouched)
- Stage 1: WGAN-GP on bona-fide VQ-f4 latents (100 epochs)
- Stage 2: ziz Encoder inverting the Generator (100 epochs)
- Inference: single forward pass (encoder→generator→VQGAN decode), then score
- Tables: `iris_anogan/results/fanogan_run1/paper_table/` (csv + tex + md, 3 scorings)
