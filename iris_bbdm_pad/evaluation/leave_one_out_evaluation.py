"""
Leave-One-Out Evaluation for BBDM Iris PAD.

For each of the 8 attack types:
1. Use ALL bonafide test images as the negative class
2. Use ONLY that attack type's images as the positive class
3. Compute: APCER, BPCER, ACER, EER, AUC, Accuracy
4. Find optimal threshold for THIS specific attack type

This mirrors Shubham's supervised open-set methodology:
- His models: trained on 7 attack types, tested on 1 held-out
- Our BBDM: trained on 0 attack types (bona fide only)
- Results are directly comparable: same test splits, same metrics

Output format matches open_set_summary.csv exactly.

Usage:
    python iris_bbdm_pad/evaluation/leave_one_out_evaluation.py \\
        --test_scores iris_bbdm_pad/results/test_pad_scores.csv \\
        --val_scores iris_bbdm_pad/results/val_pad_scores.csv \\
        --scoring_method lpips_score \\
        --output_dir iris_bbdm_pad/results/leave_one_out/

    # Compare against Shubham's results:
    python iris_bbdm_pad/evaluation/leave_one_out_evaluation.py \\
        --test_scores iris_bbdm_pad/results/test_pad_scores.csv \\
        --val_scores iris_bbdm_pad/results/val_pad_scores.csv \\
        --open_set_summary open_set_summary.csv \\
        --output_dir iris_bbdm_pad/results/leave_one_out/
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def compute_metrics(bonafide_scores: np.ndarray, attack_scores: np.ndarray, threshold: float) -> dict:
    """Compute PAD metrics at a given threshold.

    Convention:
        score > threshold  →  predicted ATTACK
        score <= threshold →  predicted BONAFIDE

    Args:
        bonafide_scores: PAD scores for bona fide samples (label = 0).
        attack_scores:   PAD scores for attack samples (label = 1).
        threshold:       Decision threshold.

    Returns:
        Dict with APCER, BPCER, ACER, Accuracy, TP, TN, FP, FN.
    """
    TN = int((bonafide_scores <= threshold).sum())   # correct bona fide
    FP = int((bonafide_scores > threshold).sum())    # bona fide misclassified as attack
    TP = int((attack_scores > threshold).sum())      # correct attack
    FN = int((attack_scores <= threshold).sum())     # attack misclassified as bona fide

    APCER  = FN / (FN + TP) if (FN + TP) > 0 else 0.0
    BPCER  = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    ACER   = (APCER + BPCER) / 2.0
    N      = TP + TN + FP + FN
    accuracy = (TP + TN) / N if N > 0 else 0.0

    return {
        "APCER": APCER, "BPCER": BPCER, "ACER": ACER,
        "Accuracy": accuracy, "TP": TP, "TN": TN, "FP": FP, "FN": FN,
    }


def find_best_threshold(
    bonafide_scores: np.ndarray,
    attack_scores: np.ndarray,
    n_thresholds: int = 1000,
) -> tuple[float, float]:
    """Find threshold that minimises ACER.

    Args:
        bonafide_scores: Scores for bona fide class.
        attack_scores:   Scores for attack class.
        n_thresholds:    Number of candidate thresholds (linearly spaced).

    Returns:
        (best_threshold, best_acer)
    """
    all_scores = np.concatenate([bonafide_scores, attack_scores])
    thresholds = np.linspace(all_scores.min(), all_scores.max(), n_thresholds)

    best_acer = 1.0
    best_threshold = thresholds[0]
    for t in thresholds:
        m = compute_metrics(bonafide_scores, attack_scores, t)
        if m["ACER"] < best_acer:
            best_acer = m["ACER"]
            best_threshold = float(t)
    return best_threshold, best_acer


def find_eer(bonafide_scores: np.ndarray, attack_scores: np.ndarray) -> tuple[float, float]:
    """Find Equal Error Rate (point where FPR ≈ FNR).

    Args:
        bonafide_scores: Scores for bona fide class (label = 0).
        attack_scores:   Scores for attack class (label = 1).

    Returns:
        (eer, eer_threshold)
    """
    labels = np.concatenate([
        np.zeros(len(bonafide_scores)),
        np.ones(len(attack_scores)),
    ])
    scores = np.concatenate([bonafide_scores, attack_scores])
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    eer_idx = int(np.argmin(np.abs(fpr - fnr)))
    return float(fpr[eer_idx]), float(thresholds[eer_idx])


# ---------------------------------------------------------------------------
# Leave-one-out evaluation
# ---------------------------------------------------------------------------

def evaluate_leave_one_out(
    test_df: pd.DataFrame,
    scoring_method: str = "lpips_score",
    model_name: str = "BBDM_LBBDM-f4",
    val_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """Run leave-one-out evaluation over all attack types.

    For each attack type:
    - Threshold found on val set (bonafide + that attack type from val_df)
    - Negative class: ALL bona fide test samples
    - Positive class: ONLY that attack type's test samples
    - Compute all metrics on test set using val-derived threshold

    If val_df is None, falls back to finding threshold on test data (in-sample).

    Args:
        test_df:        DataFrame with columns: label, attack_type, <scoring_method>.
        scoring_method: Column to use as PAD score.
        model_name:     Model name string for output CSV.
        val_df:         Validation DataFrame (same columns). Threshold found here.

    Returns:
        DataFrame with one row per attack type + one overall row.
    """
    bonafide_mask = test_df["label"] == "bonafide"
    bonafide_scores = test_df.loc[bonafide_mask, scoring_method].values
    logger.info(f"Bona fide test samples: {len(bonafide_scores)}")

    attack_df = test_df[test_df["label"] == "attack"]
    attack_types = sorted(attack_df["attack_type"].unique())
    logger.info(f"Attack types found: {attack_types}")

    # Val-set data for threshold finding
    if val_df is not None:
        val_bon_scores = val_df.loc[val_df["label"] == "bonafide", scoring_method].values
        val_attack_df  = val_df[val_df["label"] == "attack"]
        logger.info(f"Bona fide val samples: {len(val_bon_scores)}")
    else:
        logger.warning("No val_df provided — threshold found on test data (in-sample).")
        val_bon_scores = bonafide_scores
        val_attack_df  = attack_df

    rows = []
    for attack_type in attack_types:
        attack_scores = attack_df.loc[
            attack_df["attack_type"] == attack_type, scoring_method
        ].values

        if len(attack_scores) == 0:
            logger.warning(f"No samples for attack type '{attack_type}', skipping.")
            continue

        # Find threshold on val subset for this attack type
        val_atk_scores = val_attack_df.loc[
            val_attack_df["attack_type"] == attack_type, scoring_method
        ].values
        if len(val_atk_scores) == 0:
            logger.warning(f"No val samples for '{attack_type}' — using test for threshold.")
            val_atk_scores = attack_scores

        threshold, _ = find_best_threshold(val_bon_scores, val_atk_scores)
        metrics = compute_metrics(bonafide_scores, attack_scores, threshold)

        labels = np.concatenate([
            np.zeros(len(bonafide_scores)),
            np.ones(len(attack_scores)),
        ])
        scores_arr = np.concatenate([bonafide_scores, attack_scores])
        try:
            auc = float(roc_auc_score(labels, scores_arr))
        except ValueError:
            auc = float("nan")

        eer, _ = find_eer(bonafide_scores, attack_scores)

        rows.append({
            "Model":       model_name,
            "Attack_Type": attack_type,
            "APCER":       metrics["APCER"],
            "BPCER":       metrics["BPCER"],
            "ACER":        metrics["ACER"],
            "EER":         eer,
            "AUC":         auc,
            "Threshold":   threshold,
            "Accuracy":    metrics["Accuracy"],
            "N_bonafide":  len(bonafide_scores),
            "N_attack":    len(attack_scores),
        })
        logger.info(
            f"  {attack_type:<25} ACER={metrics['ACER']*100:5.2f}%  "
            f"AUC={auc:.4f}  N_attack={len(attack_scores)}"
        )

    # Overall (all attacks pooled) — threshold found on val, evaluated on test
    all_attack_scores     = attack_df[scoring_method].values
    val_all_attack_scores = val_attack_df[scoring_method].values
    overall_threshold, _  = find_best_threshold(val_bon_scores, val_all_attack_scores)
    overall_metrics = compute_metrics(bonafide_scores, all_attack_scores, overall_threshold)
    overall_labels = np.concatenate([
        np.zeros(len(bonafide_scores)),
        np.ones(len(all_attack_scores)),
    ])
    overall_scores_arr = np.concatenate([bonafide_scores, all_attack_scores])
    try:
        overall_auc = float(roc_auc_score(overall_labels, overall_scores_arr))
    except ValueError:
        overall_auc = float("nan")
    overall_eer, _ = find_eer(bonafide_scores, all_attack_scores)

    rows.append({
        "Model":       model_name,
        "Attack_Type": "",   # empty = overall (matches Shubham's format)
        "APCER":       overall_metrics["APCER"],
        "BPCER":       overall_metrics["BPCER"],
        "ACER":        overall_metrics["ACER"],
        "EER":         overall_eer,
        "AUC":         overall_auc,
        "Threshold":   overall_threshold,
        "Accuracy":    overall_metrics["Accuracy"],
        "N_bonafide":  len(bonafide_scores),
        "N_attack":    len(all_attack_scores),
    })
    logger.info(
        f"  {'Overall (pooled)':<25} ACER={overall_metrics['ACER']*100:5.2f}%  "
        f"AUC={overall_auc:.4f}  N_attack={len(all_attack_scores)}"
    )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def save_outputs(
    results_df: pd.DataFrame,
    output_dir: Path,
    suffix: str = "",
) -> None:
    """Save bbdm_open_set_summary and bbdm_open_set_detailed CSVs.

    Args:
        results_df:  Full results DataFrame (from evaluate_leave_one_out).
        output_dir:  Directory to save files.
        suffix:      Optional suffix for filenames (e.g. '_val').
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Summary CSV (matches Shubham's format exactly)
    summary_cols = ["Model", "Attack_Type", "APCER", "BPCER", "ACER", "Threshold", "Accuracy"]
    summary_df = results_df[summary_cols].copy()
    summary_path = output_dir / f"bbdm_open_set_summary{suffix}.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved summary → {summary_path}")

    # Detailed CSV (extended with EER, AUC, counts)
    detailed_path = output_dir / f"bbdm_open_set_detailed{suffix}.csv"
    results_df.to_csv(detailed_path, index=False)
    logger.info(f"Saved detailed → {detailed_path}")


def save_combined_comparison(
    bbdm_summary_df: pd.DataFrame,
    open_set_summary_path: Path,
    output_dir: Path,
) -> None:
    """Merge Shubham's open_set_summary.csv with our BBDM results.

    Args:
        bbdm_summary_df:       BBDM summary DataFrame (summary cols only).
        open_set_summary_path: Path to Shubham's open_set_summary.csv.
        output_dir:            Output directory.
    """
    if not open_set_summary_path.exists():
        logger.warning(f"open_set_summary.csv not found at {open_set_summary_path} — skipping combined CSV.")
        return

    shubham_df = pd.read_csv(open_set_summary_path)
    combined_df = pd.concat([shubham_df, bbdm_summary_df], ignore_index=True)
    out_path = output_dir / "combined_comparison.csv"
    combined_df.to_csv(out_path, index=False)
    logger.info(f"Saved combined comparison ({len(combined_df)} rows) → {out_path}")


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(
    results_df: pd.DataFrame,
    open_set_summary_path: Path,
    scoring_method: str,
) -> None:
    """Print comprehensive results table to console.

    Args:
        results_df:            Full BBDM results DataFrame.
        open_set_summary_path: Path to Shubham's CSV for comparison.
        scoring_method:        Scoring method name (for header).
    """
    sep = "═" * 103
    dash = "─" * 103

    print(f"\n{sep}")
    print(f"  BBDM Leave-One-Out Evaluation Results")
    print(f"  Model: LBBDM-f4 | Scoring: {scoring_method} | Training: Bona fide only (0 attacks)")
    print(f"{sep}\n")

    header = (
        f"  {'Attack Type':<20} │ {'APCER':>7} │ {'BPCER':>7} │ {'ACER':>7} │ "
        f"{'EER':>7} │ {'AUC':>6} │ {'Threshold':>9} │ {'N_attack':>8}"
    )
    print(header)
    print(f"  {'─'*20}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*6}─┼─{'─'*9}─┼─{'─'*8}")

    for _, row in results_df.iterrows():
        at = row["Attack_Type"] if row["Attack_Type"] else "Overall (pooled)"
        if at == "Overall (pooled)":
            print(f"  {'─'*20}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*6}─┼─{'─'*9}─┼─{'─'*8}")
        print(
            f"  {at:<20} │ {row['APCER']*100:>6.2f}% │ {row['BPCER']*100:>6.2f}% │ "
            f"{row['ACER']*100:>6.2f}% │ {row['EER']*100:>6.2f}% │ {row['AUC']:>6.4f} │ "
            f"{row['Threshold']:>9.4f} │ {int(row['N_attack']):>8}"
        )

    # Comparison with supervised models
    if open_set_summary_path.exists():
        print()
        shubham_df = pd.read_csv(open_set_summary_path)
        overall_sup = shubham_df[shubham_df["Attack_Type"].isna() | (shubham_df["Attack_Type"] == "")]
        if overall_sup.empty:
            # Compute mean ACER per model across all attack types
            overall_sup = (
                shubham_df.groupby("Model")["ACER"].mean()
                .reset_index()
                .rename(columns={"ACER": "ACER"})
            )
        else:
            overall_sup = overall_sup[["Model", "ACER"]].copy()
        overall_sup = overall_sup.sort_values("ACER").head(3)

        bbdm_overall_acer = results_df[results_df["Attack_Type"] == ""]["ACER"].values[0]

        print(f"  Comparison with supervised models (top 3 by overall ACER):")
        print(f"  {'─'*20}─┬─{'─'*12}─┬─{'─'*10}─┬─{'─'*20}")
        print(f"  {'Model':<20} │ {'Overall ACER':>12} │ {'Training':>10} │ {'Attacks in Train'}")
        print(f"  {'─'*20}─┼─{'─'*12}─┼─{'─'*10}─┼─{'─'*20}")
        for _, row in overall_sup.iterrows():
            print(f"  {str(row['Model']):<20} │ {row['ACER']*100:>11.2f}% │ {'Supervised':>10} │ 7/8 attack types")
        print(f"  {'BBDM (Ours)':<20} │ {bbdm_overall_acer*100:>11.2f}% │ {'Unsuperv.':>10} │ 0 attack types")
        print(f"  {'─'*20}─┴─{'─'*12}─┴─{'─'*10}─┴─{'─'*20}")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Leave-One-Out Evaluation for BBDM Iris PAD",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--test_scores",
        type=Path,
        default=Path("iris_bbdm_pad/results/test_pad_scores.csv"),
        help="Path to test_pad_scores.csv",
    )
    parser.add_argument(
        "--val_scores",
        type=Path,
        default=Path("iris_bbdm_pad/results/val_pad_scores.csv"),
        help="Path to val_pad_scores.csv (for val-based results)",
    )
    parser.add_argument(
        "--scoring_method",
        type=str,
        default="lpips_score",
        choices=["mse_score", "lpips_score", "recon_score", "trajectory_score", "combined_score"],
        help="PAD scoring column to use",
    )
    parser.add_argument(
        "--open_set_summary",
        type=Path,
        default=Path("open_set_summary.csv"),
        help="Path to Shubham's open_set_summary.csv for combined comparison",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("iris_bbdm_pad/results/leave_one_out"),
        help="Directory to write outputs",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="BBDM_LBBDM-f4",
        help="Model identifier string for CSV outputs",
    )
    args = parser.parse_args()

    # --- Load val scores for threshold finding ---
    val_df = None
    if args.val_scores.exists():
        logger.info(f"[1/4] Loading val scores from {args.val_scores}")
        val_df = pd.read_csv(args.val_scores)
        logger.info(f"  {len(val_df)} rows | bonafide={(val_df['label']=='bonafide').sum()} | attack={(val_df['label']=='attack').sum()}")
    else:
        logger.warning(f"Val scores not found at {args.val_scores} — threshold will use test data.")

    # --- Test set evaluation ---
    logger.info(f"[2/4] Loading test scores from {args.test_scores}")
    test_df = pd.read_csv(args.test_scores)
    logger.info(f"  {len(test_df)} rows | bonafide={(test_df['label']=='bonafide').sum()} | attack={(test_df['label']=='attack').sum()}")

    logger.info(f"[3/4] Running leave-one-out evaluation (scoring={args.scoring_method})")
    results_df = evaluate_leave_one_out(test_df, args.scoring_method, args.model_name, val_df=val_df)

    logger.info("[4/4] Saving outputs")
    save_outputs(results_df, args.output_dir)

    summary_cols = ["Model", "Attack_Type", "APCER", "BPCER", "ACER", "Threshold", "Accuracy"]
    save_combined_comparison(
        results_df[summary_cols].copy(),
        args.open_set_summary,
        args.output_dir,
    )

    # --- Print summary ---
    print_summary(results_df, args.open_set_summary, args.scoring_method)

    logger.info(f"[Done] All outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
