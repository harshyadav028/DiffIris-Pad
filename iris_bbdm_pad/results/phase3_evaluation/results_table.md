# Iris PAD Evaluation Results

## Best Method Summary

| Metric | Value |
|--------|-------|
| Scoring method | LPIPS |
| Threshold | 0.192253 |
| Test samples (bonafide) | 12,926 |
| Test samples (attack) | 34,508 |
| ACER | 30.51% |
| APCER | 18.80% |
| BPCER | 42.22% |
| EER | 33.77% |
| AUC | 0.7481 |
| Accuracy | 74.82% |
| Precision | 83.70% |
| Recall | 81.20% |
| F1 | 0.8243 |
| TDR @ FDR=0.1% | 7.11% |
| TDR @ FDR=1.0% | 12.35% |
| TDR @ FDR=5.0% | 20.82% |

## Confusion Matrix

| | Predicted Bonafide | Predicted Attack |
|---|---|---|
| Actual Bonafide | 7,469 (TN) | 5,457 (FP) |
| Actual Attack   | 6,486 (FN) | 28,022 (TP) |

## Per-Attack APCER

| Attack Type | N Samples | APCER | Detection Rate | Mean Score |
|-------------|-----------|-------|----------------|------------|
| Artifact | 1,643 | 0.73% | 99.27% | 0.3304 |
| Contact Lens | 6,304 | 32.73% | 67.27% | 0.2265 |
| E-display | 5,546 | 35.07% | 64.93% | 0.2401 |
| Fake with Add On | 67 | 13.43% | 86.57% | 0.3368 |
| Generated | 4,978 | 8.52% | 91.48% | 0.2438 |
| Post-Mortem | 3,293 | 0.00% | 100.00% | 0.4809 |
| Print & E-display | 5,180 | 17.32% | 82.68% | 0.2505 |
| Printed | 7,497 | 15.15% | 84.85% | 0.3457 |

## All Methods Comparison

| Method | Threshold | ACER | APCER | BPCER | EER | AUC |
|--------|-----------|------|-------|-------|-----|-----|
| mse_score | 0.027411 | 44.08% | 71.71% | 16.44% | 43.81% | 0.6161 |
| lpips_score | 0.192253 | 30.51% | 18.80% | 42.22% | 33.77% | 0.7481 |
| recon_score | 0.201790 | 30.41% | 15.65% | 45.17% | 33.92% | 0.7408 |
| trajectory_score | 0.011959 | 50.76% | 90.65% | 10.88% | 53.74% | 0.4457 |
| combined_score | 0.621622 | 52.07% | 84.96% | 19.17% | 50.46% | 0.4970 |