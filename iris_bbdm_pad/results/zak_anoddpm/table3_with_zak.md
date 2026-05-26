## Table 3 (Updated) — With Diff-IrisPAD ZAK Row Added

> **ZAK row note**: The `†` row shows the Zero-Attack-Knowledge threshold for Diff-IrisPAD —
> τ set at the 90th percentile of bona fide validation scores only.
> Zero attack labels are used at any stage (not in training, not in threshold calibration).

> **Why no AnoDDPM ZAK row**: The Table 3 AnoDDPM uses Dynamic LPIPS+ViT fusion scoring.
> No val dynamic-score file is cached. The ViT-only AnoDDPM variant (which has a cached val file)
> produces near-identical bona fide and attack score distributions
> (bona fide mean = 0.6038, attack mean = 0.6040), making ZAK meaningless for that variant
> (ZAK ACER ≈ 49% regardless of percentile). Applying ZAK to AnoDDPM requires re-running
> inference with dynamic scoring on the val set, which is outside scope of this analysis.

---

### AnoDDPM with DDIM Sampler — Dynamic LPIPS+ViT Fusion, 50 Steps

| Attack | τ | APCER | BPCER | ACER |
|---|---|---|---|---|
| Artifact | 0.5829 | 0.001 | 0.782 | 0.392 |
| Contact Lens | 0.6189 | 0.057 | 0.807 | 0.432 |
| E-display | 0.5071 | 0.156 | 0.292 | 0.224 |
| Fake W/AO | 0.6709 | 0.015 | 0.846 | 0.431 |
| Generated | 0.4565 | 0.100 | 0.406 | 0.253 |
| PostMortem | 0.4496 | 0.067 | 0.575 | 0.321 |
| Print & ED | 0.4496 | 0.128 | 0.425 | 0.277 |
| Printed | 0.3298 | 0.754 | 0.123 | 0.439 |
| **All** | **0.4476** | **0.364** | **0.431** | **0.397** |

---

### Diff-IrisPAD (Ours) — LBBDM-f4, 50 DDIM Steps, ViT Scoring

| Attack | τ | APCER | BPCER | ACER |
|---|---|---|---|---|
| Artifact | 0.198 | 0.018 | 0.072 | **0.045** |
| Contact Lens | 0.052 | 0.103 | 0.693 | 0.398 |
| E-display | 0.116 | 0.194 | 0.232 | 0.213 |
| Fake W/AO | 0.178 | 0.045 | 0.095 | **0.070** |
| Generated | 0.064 | 0.125 | 0.567 | 0.346 |
| PostMortem | 0.162 | 0.049 | 0.120 | **0.084** |
| Print & ED | 0.117 | 0.190 | 0.230 | 0.210 |
| Printed | 0.100 | 0.156 | 0.305 | 0.230 |
| **All** | **0.093** | **0.208** | **0.343** | **0.276** |

---

### Diff-IrisPAD ZAK† — ViT Scoring, τ = p90 of Bona Fide Val (τ = 0.1700)

*Zero attack labels used at any stage — neither in training nor in threshold calibration.*

| Attack | τ | APCER | BPCER | ACER |
|---|---|---|---|---|
| Artifact | 0.1700 | 0.005 | 0.107 | 0.056 |
| Contact Lens | 0.1700 | 0.810 | 0.107 | 0.459 |
| E-display | 0.1700 | 0.400 | 0.107 | 0.253 |
| Fake W/AO | 0.1700 | 0.045 | 0.107 | 0.076 |
| Generated | 0.1700 | 0.848 | 0.107 | 0.477 |
| PostMortem | 0.1700 | 0.063 | 0.107 | 0.085 |
| Print & ED | 0.1700 | 0.388 | 0.107 | 0.247 |
| Printed | 0.1700 | 0.476 | 0.107 | 0.291 |
| **All** | **0.1700** | **0.503** | **0.107** | **0.305** |

---

### ZAK Summary

| Method | Attack Labels Used | Overall ACER | vs AnoDDPM-Dynamic |
|---|---|---|---|
| AnoDDPM-Dynamic (Table 3) | Val attacks for per-attack τ | 39.73% | — |
| Diff-IrisPAD (Table 3) | Val attacks for per-attack τ | **27.60%** | −12.13 pp |
| Diff-IrisPAD ZAK† | **None** | **30.46%** | **−9.27 pp** |

† ZAK = Zero-Attack-Knowledge. τ = 0.1700 (p90 of bona fide val scores, Diff-IrisPAD).
Diff-IrisPAD ZAK outperforms AnoDDPM (which uses val attack labels for calibration) by 9.27 pp,
with zero attack labels used at any stage.
