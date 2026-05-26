"""
iris_bbdm_pad/evaluation/compute_dynamic_fusion.py

Fuses LPIPS and ViT scores using the tanh-based dynamic weighting strategy.
Finds an optimal threshold on val set and evaluates per-attack on test.

Dynamic fusion rule:
  - Both > 0.5 (both agree: attack)  → tanh-weight toward higher confidence
  - Both < 0.5 (both agree: bonafide) → tanh-weight toward lower divergence
  - Disagreement                       → equal weight (0.5 each)

Usage:
    cd /home/teaching/Documents/Geetanjali_PhD_IRIS_PAD
    python iris_bbdm_pad/evaluation/compute_dynamic_fusion.py
"""

import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Reuse shared metric functions
sys.path.insert(0, "iris_bbdm_pad/evaluation")
from compute_vit_metrics import find_threshold, compute_per_attack

SCORING_DIR = Path("IJCB_paper_requirements/scoring")


def _tanh(x: float) -> float:
    return float(np.tanh(x))


def dynamic_fuse(lpips_norm: float, vit_norm: float) -> float:
    """Adaptive tanh-weighted fusion of min-max-normalised LPIPS and ViT scores.

    Both inputs should be in [0, 1] (0 = bonafide-like, 1 = attack-like).
    """
    s_l = float(lpips_norm)
    s_v = float(vit_norm)

    if s_l > 0.5 and s_v > 0.5:
        # Both indicate attack → weight toward the more extreme (higher) score
        lam1 = _tanh(s_l)
        lam2 = 1.0 - lam1
    elif s_l < 0.5 and s_v < 0.5:
        # Both indicate bonafide → weight toward the lower-divergence score
        lam1 = 1.0 - _tanh(s_l)
        lam2 = 1.0 - lam1
    else:
        # Disagreement → equal contribution
        lam1, lam2 = 0.5, 0.5

    return lam1 * s_l + lam2 * s_v


def minmax_normalise(series: pd.Series) -> pd.Series:
    mn, mx = series.min(), series.max()
    return (series - mn) / (mx - mn + 1e-8)


def build_merged(lpips_csv: str, vit_csv: str) -> pd.DataFrame:
    """Merge LPIPS and ViT score CSVs on filename.

    Deduplicates both CSVs on filename before merging (the original val scoring
    run had resume-induced duplicates for ~800 images).
    """
    lpips_df = pd.read_csv(lpips_csv)
    vit_df   = pd.read_csv(vit_csv)[["filename", "vit_score"]]

    # Deduplicate: keep first occurrence for both
    lpips_dup = lpips_df.duplicated("filename").sum()
    vit_dup   = vit_df.duplicated("filename").sum()
    if lpips_dup:
        print(f"  Deduplicating {lpips_dup} rows from {Path(lpips_csv).name}")
        lpips_df = lpips_df.drop_duplicates("filename", keep="first")
    if vit_dup:
        print(f"  Deduplicating {vit_dup} rows from {Path(vit_csv).name}")
        vit_df = vit_df.drop_duplicates("filename", keep="first")

    merged = lpips_df.merge(vit_df, on="filename", how="inner")
    print(f"  {Path(lpips_csv).name}: {len(lpips_df)} unique rows")
    print(f"  {Path(vit_csv).name}:   {len(vit_df)} unique rows")
    print(f"  Merged:          {len(merged)} rows")
    return merged


def add_fusion_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalised and fused score columns in-place."""
    df = df.copy()
    df["lpips_norm"]     = minmax_normalise(df["lpips_score"])
    df["vit_norm"]       = minmax_normalise(df["vit_score"])
    df["fixed_score"]    = 0.5 * df["lpips_norm"] + 0.5 * df["vit_norm"]
    df["dynamic_score"]  = df.apply(
        lambda r: dynamic_fuse(r["lpips_norm"], r["vit_norm"]), axis=1
    )
    return df


if __name__ == "__main__":
    print("=" * 60)
    print("Dynamic Fusion: LPIPS + ViT")
    print("=" * 60)

    # ── Load and merge ────────────────────────────────────────────
    print("\nLoading val set...")
    val_merged = build_merged(
        "iris_bbdm_pad/results/val_pad_scores.csv",
        str(SCORING_DIR / "vit_scores_val.csv"),
    )
    print("\nLoading test set...")
    test_merged = build_merged(
        "iris_bbdm_pad/results/test_pad_scores.csv",
        str(SCORING_DIR / "vit_scores_test.csv"),
    )

    # Normalise independently (each split normalised within itself)
    val_merged  = add_fusion_scores(val_merged)
    test_merged = add_fusion_scores(test_merged)

    # ── Fixed fusion threshold ────────────────────────────────────
    fixed_thresh, fixed_val_acer = find_threshold(val_merged, "fixed_score")
    print(f"\nFixed fusion   → threshold={fixed_thresh:.4f}  val ACER={fixed_val_acer:.2f}%")

    fixed_metrics = compute_per_attack(test_merged, "fixed_score", fixed_thresh)
    print("\nFixed fusion per-attack results (test set):")
    print(fixed_metrics.to_string(index=False))

    # ── Dynamic fusion threshold ──────────────────────────────────
    dyn_thresh, dyn_val_acer = find_threshold(val_merged, "dynamic_score")
    print(f"\nDynamic fusion → threshold={dyn_thresh:.4f}  val ACER={dyn_val_acer:.2f}%")

    dyn_metrics = compute_per_attack(test_merged, "dynamic_score", dyn_thresh)
    print("\nDynamic fusion per-attack results (test set):")
    print(dyn_metrics.to_string(index=False))

    # ── Save ─────────────────────────────────────────────────────
    fixed_metrics.to_csv(SCORING_DIR / "fixed_fusion_metrics_test.csv", index=False)
    dyn_metrics.to_csv(SCORING_DIR / "dynamic_fusion_metrics_test.csv", index=False)

    test_merged[["filename", "label", "attack_type",
                 "lpips_score", "vit_score",
                 "lpips_norm", "vit_norm",
                 "fixed_score", "dynamic_score"]].to_csv(
        SCORING_DIR / "all_scores_test.csv", index=False
    )

    # Save thresholds
    with open(SCORING_DIR / "fusion_thresholds.json", "w") as f:
        json.dump({
            "fixed_threshold":   fixed_thresh,
            "fixed_val_acer":    fixed_val_acer,
            "dynamic_threshold": dyn_thresh,
            "dynamic_val_acer":  dyn_val_acer,
        }, f, indent=2)

    print(f"\nSaved → {SCORING_DIR}/fixed_fusion_metrics_test.csv")
    print(f"Saved → {SCORING_DIR}/dynamic_fusion_metrics_test.csv")
    print(f"Saved → {SCORING_DIR}/all_scores_test.csv")
    print(f"Saved → {SCORING_DIR}/fusion_thresholds.json")
