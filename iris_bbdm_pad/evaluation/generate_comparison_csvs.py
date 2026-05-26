"""
iris_bbdm_pad/evaluation/generate_comparison_csvs.py

Generates 6 CSVs in the same format as ddpm_comparison_fixed.csv /
ddpm_comparison_global.csv for the three DiffIrisPAD scoring methods:

  • ViT-only
  • Fixed Fusion (LPIPS + ViT, equal weight)
  • Dynamic Fusion (LPIPS + ViT, tanh-adaptive weight)

Two CSV variants per method:
  _fixed   – per-attack-type threshold (each optimised separately on val)
  _global  – one global threshold (optimised on all attacks combined on val)

Output columns (fractions, not %):
    Model, Attack_Type, APCER, BPCER, ACER, Threshold, Accuracy

Usage:
    cd /home/teaching/Documents/Geetanjali_PhD_IRIS_PAD
    PYTHONNOUSERSITE=1 /home/teaching/miniconda3/envs/bbdm_clean/bin/python \\
        iris_bbdm_pad/evaluation/generate_comparison_csvs.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, "iris_bbdm_pad/evaluation")
from compute_dynamic_fusion import build_merged, add_fusion_scores

SCORING_DIR  = Path("IJCB_paper_requirements/scoring")
OUT_DIR      = Path("IJCB_paper_requirements/scoring")
N_CANDIDATES = 2000


# ── Threshold helpers ─────────────────────────────────────────────────────────

def _find_threshold(bon_scores: np.ndarray, atk_scores: np.ndarray) -> float:
    """Return threshold that minimises ACER (sweep N_CANDIDATES candidates)."""
    all_s = np.concatenate([bon_scores, atk_scores])
    lo, hi = np.percentile(all_s, 1), np.percentile(all_s, 99)
    best_t, best_acer = lo, float("inf")
    for t in np.linspace(lo, hi, N_CANDIDATES):
        bpcer = (bon_scores > t).mean()
        apcer = (atk_scores <= t).mean()
        acer  = (apcer + bpcer) / 2.0
        if acer < best_acer:
            best_acer, best_t = acer, float(t)
    return best_t


def _metrics(bon: np.ndarray, atk: np.ndarray, t: float) -> dict:
    """APCER, BPCER, ACER, Accuracy — all as fractions [0,1]."""
    bpcer = float((bon > t).mean())
    apcer = float((atk <= t).mean())
    acer  = (apcer + bpcer) / 2.0
    tp    = (atk > t).sum()
    tn    = (bon <= t).sum()
    acc   = (tp + tn) / (len(atk) + len(bon))
    return {"APCER": apcer, "BPCER": bpcer, "ACER": acer, "Accuracy": float(acc)}


# ── Main builder ──────────────────────────────────────────────────────────────

def build_comparison_csvs(
    model_name: str,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    score_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (fixed_df, global_df) — one row per attack type + NaN overall row.

    fixed_df : per-attack optimal threshold (val), evaluated on test
    global_df: one global threshold (val, all attacks), evaluated on test
    """
    val_bon  = val_df[val_df["label"] == "bonafide"][score_col].dropna().values
    val_atk  = val_df[val_df["label"] != "bonafide"].dropna(subset=[score_col])
    test_bon = test_df[test_df["label"] == "bonafide"][score_col].dropna().values
    test_atk = test_df[test_df["label"] != "bonafide"].dropna(subset=[score_col])

    # Global threshold: minimise ACER on val (all attacks combined)
    global_t = _find_threshold(val_bon, val_atk[score_col].values)

    attack_types = sorted(test_atk["attack_type"].unique())

    fixed_rows, global_rows = [], []

    for atk_type in attack_types:
        # val subset for this attack
        val_sub = val_atk[val_atk["attack_type"] == atk_type][score_col].values
        # test subset for this attack
        test_sub = test_atk[test_atk["attack_type"] == atk_type][score_col].values

        if len(val_sub) == 0:
            print(f"  WARNING: no val samples for {atk_type!r}, skipping")
            continue

        # per-attack threshold
        per_t = _find_threshold(val_bon, val_sub)
        fm = _metrics(test_bon, test_sub, per_t)
        fixed_rows.append({
            "Model": model_name, "Attack_Type": atk_type,
            **fm, "Threshold": per_t,
        })

        # global threshold applied to this attack
        gm = _metrics(test_bon, test_sub, global_t)
        global_rows.append({
            "Model": model_name, "Attack_Type": atk_type,
            **gm, "Threshold": global_t,
        })

    # Overall / NaN row — always uses global threshold (consistent with DDPM format)
    om = _metrics(test_bon, test_atk[score_col].values, global_t)
    for rows in (fixed_rows, global_rows):
        rows.append({
            "Model": model_name, "Attack_Type": float("nan"),
            **om, "Threshold": global_t,
        })

    col_order = ["Model", "Attack_Type", "APCER", "BPCER", "ACER", "Threshold", "Accuracy"]
    return (
        pd.DataFrame(fixed_rows)[col_order],
        pd.DataFrame(global_rows)[col_order],
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Generating DiffIrisPAD comparison CSVs")
    print("=" * 60)

    # ── 1. Load / build val scores ────────────────────────────────
    print("\n[1/4] Building val fusion scores...")
    val_merged = build_merged(
        "iris_bbdm_pad/results/val_pad_scores.csv",
        str(SCORING_DIR / "vit_scores_val.csv"),
    )
    val_merged = add_fusion_scores(val_merged)

    # ── 2. Load test scores (already has all columns) ─────────────
    print("\n[2/4] Loading test scores from all_scores_test.csv...")
    test_merged = pd.read_csv(SCORING_DIR / "all_scores_test.csv")
    print(f"  Test rows: {len(test_merged)}")
    print(f"  Columns: {test_merged.columns.tolist()}")

    # ── 3. Generate CSVs for each method ─────────────────────────
    print("\n[3/4] Computing thresholds and evaluating...")
    methods = [
        ("DiffIrisPAD_LPIPS",           "lpips_score"),
        ("DiffIrisPAD_ViT",             "vit_score"),
        ("DiffIrisPAD_Fixed_Fusion",    "fixed_score"),
        ("DiffIrisPAD_Dynamic_Fusion",  "dynamic_score"),
    ]

    all_fixed, all_global = [], []

    for model_name, score_col in methods:
        print(f"\n  {model_name} ({score_col})...")
        # Val df: need attack_type from the val data
        # For ViT: val_df = vit_scores_val.csv (has attack_type, label, vit_score)
        # For fusion: val_df = val_merged (has all columns including fixed/dynamic)
        if score_col == "vit_score":
            val_df = pd.read_csv(SCORING_DIR / "vit_scores_val.csv")
        elif score_col == "lpips_score":
            val_df = val_merged   # val_merged has lpips_score from val_pad_scores.csv
        else:
            val_df = val_merged

        fixed_df, global_df = build_comparison_csvs(
            model_name, val_df, test_merged, score_col
        )
        all_fixed.append(fixed_df)
        all_global.append(global_df)

        print(f"    Fixed  → {len(fixed_df)} rows "
              f"(global threshold from overall NaN row: "
              f"{global_df[global_df['Attack_Type'].isna()]['Threshold'].values[0]:.4f})")

    # ── 4. Save ───────────────────────────────────────────────────
    print("\n[4/4] Saving...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for rows, stem in zip(all_fixed, [m[0] for m in methods]):
        short = {
            "DiffIrisPAD_LPIPS":          "lpips",
            "DiffIrisPAD_ViT":            "vit",
            "DiffIrisPAD_Fixed_Fusion":   "fixed_fusion",
            "DiffIrisPAD_Dynamic_Fusion": "dynamic_fusion",
        }[stem]
        path = OUT_DIR / f"{short}_comparison_fixed.csv"
        rows.to_csv(path, index=False)
        print(f"  Saved → {path}")

    for rows, stem in zip(all_global, [m[0] for m in methods]):
        short = {
            "DiffIrisPAD_LPIPS":          "lpips",
            "DiffIrisPAD_ViT":            "vit",
            "DiffIrisPAD_Fixed_Fusion":   "fixed_fusion",
            "DiffIrisPAD_Dynamic_Fusion": "dynamic_fusion",
        }[stem]
        path = OUT_DIR / f"{short}_comparison_global.csv"
        rows.to_csv(path, index=False)
        print(f"  Saved → {path}")

    # ── Summary table ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Overall ACER summary (NaN row = all attacks pooled, test set):")
    print(f"{'Method':<30} {'Global τ':>10} {'Fixed ACER%':>12} {'Global ACER%':>13}")
    print("-" * 60)
    for (model_name, _), fd, gd in zip(methods, all_fixed, all_global):
        nan_f = fd[fd["Attack_Type"].isna()].iloc[0]
        nan_g = gd[gd["Attack_Type"].isna()].iloc[0]
        print(f"  {model_name:<28} {nan_g['Threshold']:>10.4f} "
              f"{nan_f['ACER']*100:>11.2f}% {nan_g['ACER']*100:>12.2f}%")
    print("=" * 60)
