# ZAK Per-Attack Threshold — Results Summary

## Four Methods Compared

| Method | τ Source | Attack Labels Used | Overall ACER (mean) |
|--------|----------|--------------------|---------------------|
| A: Current | ACER-min on full val | Yes (to evaluate ACER per candidate τ) | 19.95% |
| B: ZAK-Global | p90 of bona fide val | **None** | 24.29% |
| C: ZAK-PerAtk Full | p90 of bona fide val (per attack subset) | **None** | 24.29% |
| D: ZAK-PerAtk Partial | best percentile per attack from val | Yes (to pick percentile only) | 19.97% |

## Per-Attack ACER Table

| Attack | A: Current | B: ZAK-Global | C: ZAK-PA-Full | D: ZAK-PA-Partial | Best |
|--------|-----------|---------------|----------------|-------------------|------|
| Artifact | 4.47% | 5.57% | 5.57% | 4.75% | **A** |
| Contact Lens | 39.81% | 45.85% | 45.85% | 38.43% | **D** |
| E-display | 21.33% | 25.34% | 25.34% | 21.39% | **A** |
| Fake W/AO | 6.99% | 7.57% | 7.57% | 7.57% | **A** |
| Generated | 34.58% | 47.72% | 47.72% | 34.59% | **A** |
| PostMortem | 8.45% | 8.46% | 8.46% | 8.70% | **A** |
| Print & ED | 20.98% | 24.71% | 24.71% | 20.95% | **D** |
| Printed | 23.02% | 29.14% | 29.14% | 23.35% | **A** |
| **Mean** | **19.95%** | **24.29%** | **24.29%** | **19.97%** | |

## Chosen Percentiles for Method D (ZAK-PerAtk Partial)

| Attack | Chosen Percentile | Meaning |
|--------|-------------------|---------|
| Artifact | p92 | BPCER ≤ 8% by construction |
| Contact Lens | p50 | BPCER ≤ 50% by construction |
| E-display | p80 | BPCER ≤ 20% by construction |
| Fake W/AO | p90 | BPCER ≤ 10% by construction |
| Generated | p50 | BPCER ≤ 50% by construction |
| PostMortem | p92 | BPCER ≤ 8% by construction |
| Print & ED | p80 | BPCER ≤ 20% by construction |
| Printed | p70 | BPCER ≤ 30% by construction |

## Key Takeaways

1. **Cost of full ZAK** (B vs A): +4.34 pp (zero attack labels at any stage)
2. **Per-attack ZAK fully** (C vs B): +0.00 pp (same attack supervision level as B)
3. **Per-attack ZAK partial** (D vs A): +0.01 pp (uses only percentile selection from val attacks)
4. **Best ZAK method**: D with 19.97% mean ACER

Method C (ZAK-PerAtk Full) is the **purest** unsupervised variant:
same p90 rule applied per attack subset, zero attack labels ever used.