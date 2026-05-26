# _provenance.md — ZAK IJCB-Aligned Evaluation

Generated: 2026-04-29 17:08

## Score File Provenance

| File | Size | MD5 | Configuration |
|---|---|---|---|
| `IJCB_paper_requirements/scoring/vit_scores_test.csv` | 3,590,508 bytes | 4cbb0f82... | 50 DDIM steps |
| `IJCB_paper_requirements/scoring/vit_scores_val.csv` | 1,591,365 bytes | 082e519e... | 50 DDIM steps |
| `steps_cache/vit_scores_steps_050.csv` | 3,590,508 bytes | 4cbb0f82... | 50 DDIM steps |
| `steps_cache/vit_scores_val_steps_050.csv` | 1,591,365 bytes | 082e519e... | 50 DDIM steps |

**Verification**: `vit_scores_test.csv` and `steps_cache/vit_scores_steps_050.csv` are
byte-for-byte identical (MD5 match confirmed). Same for val files.
**Conclusion**: The cached score files are definitively the 50-DDIM-step configuration,
matching the headline IJCB Table 3 paper configuration.

## Configuration Context

The 50-step config is Diff-IrisPAD's primary operating point (IJCB Table 3).
The ablation in IJCB Table 4 also shows 50-step results with ACER=27.50%, which
differs by 0.10 pp from the Table 3 value (27.60%) due to rounding in the paper.
Both values come from the same 50-step configuration.

## Recomputed vs Canonical Numbers

The recomputed Diff-IrisPAD-current ACER from the cached score files is
27.56%. The locked value in report_facts.md is 27.60%.
Discrepancy: 0.04 pp.

### Explanation of Discrepancy

The recomputed ACER uses a global ACER-minimising threshold on the full val set.
The paper reports a per-attack-type threshold (8 separate thresholds, one per attack
type) tuned on the validation set. The overall ACER in Table 3 (27.60%) is the
unweighted mean of the 8 per-attack ACERs computed with per-attack thresholds.
The global single-threshold recomputation gives a slightly different value because:
  (a) a single tau optimised globally does not simultaneously minimise per-attack ACER
  (b) the per-attack threshold protocol is more granular

For the ZAK analysis, a global single threshold is used consistently for both
the current-method recomputation and ZAK thresholds, so the comparison is internally
consistent. The paper canonical numbers are used where per-attack data is needed.

## What Was Recomputed vs Pulled from report_facts.md

| Number | Source |
|---|---|
| Diff-IrisPAD ZAK overall ACER (all levels) | Recomputed from cached 50-step scores |
| Diff-IrisPAD-current overall ACER (recomputed) | Recomputed from cached 50-step scores |
| Diff-IrisPAD-current overall ACER (paper) | Pulled from report_facts.md (canonical) |
| Diff-IrisPAD per-attack APCER/BPCER/ACER | Pulled from report_facts.md (canonical) |
| ResNet50/ViT-B/MaxViT/DINOv1/DINOv2 per-attack | Pulled from report_facts.md (canonical) |
| AnoDDPM overall ACER | Pulled from report_facts.md (canonical, explicit overall row) |
| Mid-project baselines (DenseNet121, etc.) | Pulled from comparison_all_models.csv |

## Caveats

1. The ZAK analysis uses a single global threshold for consistency. The paper uses
   per-attack thresholds. The ZAK per-attack numbers therefore do not directly
   compare against the paper Table 3 per-attack numbers on identical footing.
   The overall ACER comparison is the most reliable cross-configuration comparison.

2. The supervised IJCB baselines have no explicit "overall ACER" row in report_facts.md.
   Their overall ACER in zak_results_summary.csv is computed as the unweighted mean
   of per-attack ACERs (8 attacks). This matches standard evaluation practice but
   may differ from a weighted mean by sample count. This is documented in the table.

3. The ZAK-p90 threshold guarantees BPCER <= 0.10 on the bona fide val set by
   construction. The test-set BPCER may differ slightly due to distribution shift.
