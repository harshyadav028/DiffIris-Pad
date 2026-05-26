"""
Comprehensive PAD evaluation on test set.

Computes: APCER, BPCER, ACER, EER, AUC, Accuracy, Precision, Recall, F1,
TDR@FDR (0.1%, 1%, 5%), per-attack-type APCER breakdown.

Usage:
    python iris_bbdm_pad/evaluation/evaluate_pad.py \
        --test_scores iris_bbdm_pad/results/test_pad_scores.csv \
        --threshold_json iris_bbdm_pad/results/threshold.json \
        --output_dir iris_bbdm_pad/results/phase3_evaluation/
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("evaluate_pad")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCORE_METHODS: List[str] = [
    "mse_score",
    "lpips_score",
    "recon_score",
    "trajectory_score",
    "combined_score",
]

# Human-readable display names for attack types
ATTACK_DISPLAY_NAMES: Dict[str, str] = {
    "Artifact": "Artifact",
    "CL": "Contact Lens",
    "E-display": "E-display",
    "Fake_with_Add_On": "Fake with Add On",
    "Fake with Add On": "Fake with Add On",
    "Generated": "Generated",
    "Post-Mortem": "Post-Mortem",
    "PostMortem": "Post-Mortem",
    "Printed": "Printed",
    "Print_E-display": "Print & E-display",
    "Print and E-display": "Print & E-display",
    "Print E-display": "Print & E-display",
}


# ---------------------------------------------------------------------------
# Metric computation helpers
# ---------------------------------------------------------------------------

def compute_apcer_bpcer_acer(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[float, float, float]:
    """Compute APCER, BPCER, ACER from binary predictions.

    Args:
        y_true: Ground-truth labels (0=bonafide, 1=attack).
        y_pred: Predicted binary labels (0=bonafide, 1=attack).

    Returns:
        Tuple of (apcer, bpcer, acer) as fractions in [0, 1].
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # APCER: fraction of attacks incorrectly classified as bonafide
    apcer = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    # BPCER: fraction of bonafide incorrectly classified as attack
    bpcer = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    acer = (apcer + bpcer) / 2.0

    return float(apcer), float(bpcer), float(acer)


def compute_eer(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> Tuple[float, float]:
    """Compute Equal Error Rate (EER) by sweeping thresholds.

    Args:
        y_true: Ground-truth binary labels (0=bonafide, 1=attack).
        scores: Continuous anomaly scores (higher = more likely attack).

    Returns:
        Tuple of (eer_value, eer_threshold) as fractions.
    """
    fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label=1)
    fnr = 1.0 - tpr  # FRR in biometrics
    # Find the index where |FAR - FRR| is minimized
    abs_diff = np.abs(fpr - fnr)
    eer_idx = int(np.argmin(abs_diff))
    eer_value = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    eer_threshold = float(thresholds[eer_idx])
    return eer_value, eer_threshold


def compute_tdr_at_fdr(
    y_true: np.ndarray,
    scores: np.ndarray,
    fdr_levels: Tuple[float, ...] = (0.001, 0.01, 0.05),
) -> Dict[float, float]:
    """Compute True Detection Rate (TDR) at specified False Detection Rates (FDR).

    FDR here corresponds to FPR on the ROC curve (bonafide wrongly flagged as attack).

    Args:
        y_true: Ground-truth binary labels (0=bonafide, 1=attack).
        scores: Continuous anomaly scores (higher = more likely attack).
        fdr_levels: Tuple of FDR values to query.

    Returns:
        Dict mapping each fdr_level to its corresponding TDR value.
    """
    fpr, tpr, _ = roc_curve(y_true, scores, pos_label=1)
    result: Dict[float, float] = {}
    for fdr in fdr_levels:
        # Find first index where fpr >= fdr, interpolate if needed
        indices = np.where(fpr >= fdr)[0]
        if len(indices) == 0:
            result[fdr] = float(tpr[-1])
        else:
            idx = indices[0]
            if idx == 0:
                result[fdr] = float(tpr[0])
            else:
                # Linear interpolation between (fpr[idx-1], tpr[idx-1]) and (fpr[idx], tpr[idx])
                x0, x1 = fpr[idx - 1], fpr[idx]
                y0, y1 = tpr[idx - 1], tpr[idx]
                if x1 - x0 > 1e-12:
                    tdr = y0 + (y1 - y0) * (fdr - x0) / (x1 - x0)
                else:
                    tdr = float(y1)
                result[fdr] = float(np.clip(tdr, 0.0, 1.0))
    return result


def compute_full_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    method_name: str,
    threshold_json_data: Dict,
) -> Dict:
    """Compute the complete set of PAD metrics for one scoring method.

    Args:
        y_true: Ground-truth binary labels (0=bonafide, 1=attack).
        scores: Continuous anomaly scores (higher = more likely attack).
        threshold: Decision threshold (score > threshold → attack).
        method_name: Name of the scoring method (for logging).
        threshold_json_data: Full threshold.json contents for EER lookup.

    Returns:
        Dict containing all computed metrics.
    """
    y_pred = (scores > threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    apcer, bpcer, acer = compute_apcer_bpcer_acer(y_true, y_pred)

    eer_value, eer_threshold = compute_eer(y_true, scores)
    auc = float(roc_auc_score(y_true, scores))
    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    tdr_at_fdr = compute_tdr_at_fdr(y_true, scores)

    logger.debug(
        "[%s] threshold=%.6f  ACER=%.4f  AUC=%.4f  EER=%.4f",
        method_name, threshold, acer, auc, eer_value,
    )

    return {
        "method": method_name,
        "threshold": float(threshold),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "apcer": apcer,
        "bpcer": bpcer,
        "acer": acer,
        "eer": eer_value,
        "eer_threshold": eer_threshold,
        "auc": auc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tdr_at_fdr_0001": tdr_at_fdr[0.001],
        "tdr_at_fdr_001": tdr_at_fdr[0.01],
        "tdr_at_fdr_005": tdr_at_fdr[0.05],
    }


def compute_per_attack_metrics(
    df: pd.DataFrame,
    score_col: str,
    threshold: float,
) -> pd.DataFrame:
    """Compute per-attack-type APCER, detection rate, and score statistics.

    Args:
        df: Full test DataFrame (must contain 'label', 'attack_type', score_col).
        score_col: Column name of the anomaly score to use.
        threshold: Decision threshold (score > threshold → attack).

    Returns:
        DataFrame indexed by attack_type with columns: apcer, detection_rate,
        mean_score, std_score, n_samples.
    """
    attack_df = df[df["label"] == "attack"].copy()
    records = []
    for attack_type, group in attack_df.groupby("attack_type"):
        n = len(group)
        scores = group[score_col].values
        preds = (scores > threshold).astype(int)
        # APCER for this type: fraction of attacks with score <= threshold (missed)
        missed = int(np.sum(preds == 0))
        apcer = missed / n if n > 0 else 0.0
        detection_rate = 1.0 - apcer
        records.append(
            {
                "attack_type": attack_type,
                "display_name": ATTACK_DISPLAY_NAMES.get(str(attack_type), str(attack_type)),
                "n_samples": n,
                "apcer": apcer,
                "detection_rate": detection_rate,
                "mean_score": float(np.mean(scores)),
                "std_score": float(np.std(scores)),
                "min_score": float(np.min(scores)),
                "max_score": float(np.max(scores)),
                "median_score": float(np.median(scores)),
            }
        )
    return pd.DataFrame(records).sort_values("attack_type").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def build_results_table_md(
    best_metrics: Dict,
    per_attack_df: pd.DataFrame,
    all_methods_df: pd.DataFrame,
    n_bonafide: int,
    n_attack: int,
) -> str:
    """Build a Markdown-formatted results table.

    Args:
        best_metrics: Metrics dict for the best scoring method.
        per_attack_df: Per-attack metrics DataFrame.
        all_methods_df: All-methods comparison DataFrame.
        n_bonafide: Count of bonafide test samples.
        n_attack: Count of attack test samples.

    Returns:
        Markdown string with all tables.
    """
    method_label = best_metrics["method"].replace("_score", "").upper()
    lines = [
        "# Iris PAD Evaluation Results",
        "",
        "## Best Method Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Scoring method | {method_label} |",
        f"| Threshold | {best_metrics['threshold']:.6f} |",
        f"| Test samples (bonafide) | {n_bonafide:,} |",
        f"| Test samples (attack) | {n_attack:,} |",
        f"| ACER | {best_metrics['acer']*100:.2f}% |",
        f"| APCER | {best_metrics['apcer']*100:.2f}% |",
        f"| BPCER | {best_metrics['bpcer']*100:.2f}% |",
        f"| EER | {best_metrics['eer']*100:.2f}% |",
        f"| AUC | {best_metrics['auc']:.4f} |",
        f"| Accuracy | {best_metrics['accuracy']*100:.2f}% |",
        f"| Precision | {best_metrics['precision']*100:.2f}% |",
        f"| Recall | {best_metrics['recall']*100:.2f}% |",
        f"| F1 | {best_metrics['f1']:.4f} |",
        f"| TDR @ FDR=0.1% | {best_metrics['tdr_at_fdr_0001']*100:.2f}% |",
        f"| TDR @ FDR=1.0% | {best_metrics['tdr_at_fdr_001']*100:.2f}% |",
        f"| TDR @ FDR=5.0% | {best_metrics['tdr_at_fdr_005']*100:.2f}% |",
        "",
        "## Confusion Matrix",
        "",
        f"| | Predicted Bonafide | Predicted Attack |",
        f"|---|---|---|",
        f"| Actual Bonafide | {best_metrics['tn']:,} (TN) | {best_metrics['fp']:,} (FP) |",
        f"| Actual Attack   | {best_metrics['fn']:,} (FN) | {best_metrics['tp']:,} (TP) |",
        "",
        "## Per-Attack APCER",
        "",
        "| Attack Type | N Samples | APCER | Detection Rate | Mean Score |",
        "|-------------|-----------|-------|----------------|------------|",
    ]
    for _, row in per_attack_df.iterrows():
        lines.append(
            f"| {row['display_name']} | {row['n_samples']:,} | "
            f"{row['apcer']*100:.2f}% | {row['detection_rate']*100:.2f}% | "
            f"{row['mean_score']:.4f} |"
        )

    lines += [
        "",
        "## All Methods Comparison",
        "",
        "| Method | Threshold | ACER | APCER | BPCER | EER | AUC |",
        "|--------|-----------|------|-------|-------|-----|-----|",
    ]
    for _, row in all_methods_df.iterrows():
        lines.append(
            f"| {row['method']} | {row['threshold']:.6f} | "
            f"{row['acer']*100:.2f}% | {row['apcer']*100:.2f}% | "
            f"{row['bpcer']*100:.2f}% | {row['eer']*100:.2f}% | "
            f"{row['auc']:.4f} |"
        )

    return "\n".join(lines)


def print_results_console(
    best_metrics: Dict,
    per_attack_df: pd.DataFrame,
    n_bonafide: int,
    n_attack: int,
) -> None:
    """Print the formatted evaluation summary to stdout.

    Args:
        best_metrics: Metrics dict for the best scoring method.
        per_attack_df: Per-attack metrics DataFrame.
        n_bonafide: Count of bonafide test samples.
        n_attack: Count of attack test samples.
    """
    method_label = best_metrics["method"].replace("_score", "").upper()
    sep = "═" * 62
    dash = "─" * 62

    # Console output uses plain print per spec
    print(sep)
    print(" Iris PAD Evaluation Results (BBDM Anomaly Detection)")
    print(sep)
    print(f" Scoring method:    {method_label}")
    print(f" Threshold:         {best_metrics['threshold']:.4f}")
    print(f" Test samples:      {n_bonafide:,} bonafide + {n_attack:,} attack")
    print(dash)
    print(f" ACER:              {best_metrics['acer']*100:.2f}%")
    print(f" APCER:             {best_metrics['apcer']*100:.2f}%    (attack miss rate)")
    print(f" BPCER:             {best_metrics['bpcer']*100:.2f}%    (bonafide rejection rate)")
    print(f" EER:               {best_metrics['eer']*100:.2f}%")
    print(f" AUC:               {best_metrics['auc']:.4f}")
    print(f" Accuracy:          {best_metrics['accuracy']*100:.2f}%")
    print(f" Precision:         {best_metrics['precision']*100:.2f}%")
    print(f" Recall:            {best_metrics['recall']*100:.2f}%")
    print(f" F1:                {best_metrics['f1']:.4f}")
    print(dash)
    print(f" TDR @ FDR=0.1%:   {best_metrics['tdr_at_fdr_0001']*100:.2f}%")
    print(f" TDR @ FDR=1.0%:   {best_metrics['tdr_at_fdr_001']*100:.2f}%")
    print(f" TDR @ FDR=5.0%:   {best_metrics['tdr_at_fdr_005']*100:.2f}%")
    print(dash)
    print(" Per-Attack APCER:")
    for _, row in per_attack_df.iterrows():
        name = row["display_name"]
        apcer_pct = row["apcer"] * 100
        print(f"   {name:<22}{apcer_pct:.2f}%")
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Comprehensive PAD evaluation on test set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--test_scores",
        type=Path,
        default=Path("iris_bbdm_pad/results/test_pad_scores.csv"),
        help="Path to test_pad_scores.csv",
    )
    parser.add_argument(
        "--threshold_json",
        type=Path,
        default=Path("iris_bbdm_pad/results/threshold.json"),
        help="Path to threshold.json",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("iris_bbdm_pad/results/phase3_evaluation"),
        help="Directory to write evaluation outputs",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser.parse_args()


def main() -> None:
    """Main evaluation entry point."""
    args = parse_args()
    logging.getLogger().setLevel(args.log_level)

    # ------------------------------------------------------------------
    # 1. Load inputs
    # ------------------------------------------------------------------
    logger.info("[1/6] Loading test scores from %s", args.test_scores)
    if not args.test_scores.exists():
        logger.error("Test scores file not found: %s", args.test_scores)
        sys.exit(1)
    df = pd.read_csv(args.test_scores)

    # Validate required columns
    required_cols = {"filename", "label", "attack_type"} | set(SCORE_METHODS)
    missing = required_cols - set(df.columns)
    if missing:
        logger.error("Missing columns in test CSV: %s", missing)
        sys.exit(1)

    logger.info("[1/6] Loaded %d rows", len(df))

    logger.info("[1/6] Loading threshold config from %s", args.threshold_json)
    if not args.threshold_json.exists():
        logger.error("Threshold JSON not found: %s", args.threshold_json)
        sys.exit(1)
    with open(args.threshold_json, "r") as fh:
        threshold_data = json.load(fh)

    best_method: str = threshold_data["best_method"]
    per_method_cfg: Dict = threshold_data["per_method"]

    # ------------------------------------------------------------------
    # 2. Prepare labels and score arrays
    # ------------------------------------------------------------------
    logger.info("[2/6] Encoding labels (bonafide=0, attack=1)")
    df["label_binary"] = (df["label"] == "attack").astype(int)
    y_true = df["label_binary"].values

    n_bonafide = int((y_true == 0).sum())
    n_attack = int((y_true == 1).sum())
    logger.info("[2/6] Bonafide: %d  |  Attack: %d", n_bonafide, n_attack)

    # ------------------------------------------------------------------
    # 3. Compute metrics for all 5 methods
    # ------------------------------------------------------------------
    logger.info("[3/6] Computing metrics for all %d scoring methods", len(SCORE_METHODS))
    all_method_metrics: List[Dict] = []
    for method in SCORE_METHODS:
        cfg = per_method_cfg.get(method, {})
        threshold = cfg.get("min_acer_threshold", 0.5)
        scores = df[method].values
        metrics = compute_full_metrics(y_true, scores, threshold, method, threshold_data)
        all_method_metrics.append(metrics)
        logger.info(
            "  [%s]  threshold=%.6f  ACER=%.4f  AUC=%.4f",
            method, threshold, metrics["acer"], metrics["auc"],
        )

    all_methods_df = pd.DataFrame(all_method_metrics)

    # ------------------------------------------------------------------
    # 4. Identify best method and compute detailed metrics
    # ------------------------------------------------------------------
    logger.info("[4/6] Best method: %s", best_method)
    best_threshold = float(threshold_data["best_threshold"])
    best_scores = df[best_method].values
    best_metrics = compute_full_metrics(
        y_true, best_scores, best_threshold, best_method, threshold_data
    )

    # ------------------------------------------------------------------
    # 5. Per-attack APCER breakdown
    # ------------------------------------------------------------------
    logger.info("[5/6] Computing per-attack APCER breakdown")
    per_attack_df = compute_per_attack_metrics(df, best_method, best_threshold)
    logger.info("[5/6] Attack types found: %s", per_attack_df["attack_type"].tolist())

    # ------------------------------------------------------------------
    # 6. Save outputs
    # ------------------------------------------------------------------
    logger.info("[6/6] Saving outputs to %s", args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # metrics_summary.json
    metrics_summary = {
        "best_method": best_method,
        "threshold": best_threshold,
        "n_bonafide": n_bonafide,
        "n_attack": n_attack,
        "metrics": best_metrics,
        "per_attack": per_attack_df.to_dict(orient="records"),
    }
    summary_json_path = args.output_dir / "metrics_summary.json"
    with open(summary_json_path, "w") as fh:
        json.dump(metrics_summary, fh, indent=2)
    logger.info("  Saved %s", summary_json_path)

    # metrics_summary.csv (flat)
    flat_row = {**best_metrics, "n_bonafide": n_bonafide, "n_attack": n_attack}
    summary_csv_path = args.output_dir / "metrics_summary.csv"
    pd.DataFrame([flat_row]).to_csv(summary_csv_path, index=False)
    logger.info("  Saved %s", summary_csv_path)

    # per_attack_metrics.csv
    per_attack_csv_path = args.output_dir / "per_attack_metrics.csv"
    per_attack_df.to_csv(per_attack_csv_path, index=False)
    logger.info("  Saved %s", per_attack_csv_path)

    # confusion_matrix.csv
    cm_path = args.output_dir / "confusion_matrix.csv"
    pd.DataFrame(
        [
            {
                "TP": best_metrics["tp"],
                "TN": best_metrics["tn"],
                "FP": best_metrics["fp"],
                "FN": best_metrics["fn"],
            }
        ]
    ).to_csv(cm_path, index=False)
    logger.info("  Saved %s", cm_path)

    # all_methods_comparison.csv
    all_methods_csv_path = args.output_dir / "all_methods_comparison.csv"
    all_methods_df.to_csv(all_methods_csv_path, index=False)
    logger.info("  Saved %s", all_methods_csv_path)

    # results_table.md
    md_content = build_results_table_md(
        best_metrics, per_attack_df, all_methods_df, n_bonafide, n_attack
    )
    md_path = args.output_dir / "results_table.md"
    md_path.write_text(md_content)
    logger.info("  Saved %s", md_path)

    logger.info(
        "[Step 6/6] Evaluation complete — saved 6 output files to %s",
        args.output_dir,
    )

    # ------------------------------------------------------------------
    # 7. Print formatted console summary
    # ------------------------------------------------------------------
    print_results_console(best_metrics, per_attack_df, n_bonafide, n_attack)


if __name__ == "__main__":
    main()
