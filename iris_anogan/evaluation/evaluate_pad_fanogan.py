"""
PAD evaluation for f-AnoGAN — iris dataset.

Computes: APCER, BPCER, ACER, EER, ROC-AUC, Accuracy, F1,
          TDR @ FDR (0.1%, 1%, 5%), per-attack APCER breakdown,
          DET curve.

Mirrors iris_bbdm_pad/evaluation/evaluate_pad.py and
iris_td's evaluation scripts — same metrics, same format,
different score column names (fanogan_score / residual_score / feature_score).

Usage
-----
  cd ~/Documents/Geetanjali_PhD_IRIS_PAD

  # Evaluate with fanogan_score (default)
  conda run --no-capture-output -n iris_pad python \\
      iris_anogan/evaluation/evaluate_pad_fanogan.py \\
      --scores   iris_anogan/results/fanogan_run1/pad_scores/test_scores.csv \\
      --output_dir iris_anogan/results/fanogan_run1/evaluation/

  # Use a pre-computed threshold from val set
  conda run --no-capture-output -n iris_pad python \\
      iris_anogan/evaluation/evaluate_pad_fanogan.py \\
      --scores     iris_anogan/results/fanogan_run1/pad_scores/test_scores.csv \\
      --val_scores iris_anogan/results/fanogan_run1/pad_scores/val_scores.csv \\
      --output_dir iris_anogan/results/fanogan_run1/evaluation/

  # Evaluate all score columns
  conda run --no-capture-output -n iris_pad python \\
      iris_anogan/evaluation/evaluate_pad_fanogan.py \\
      --scores iris_anogan/results/fanogan_run1/pad_scores/test_scores.csv \\
      --output_dir iris_anogan/results/fanogan_run1/evaluation/ \\
      --score_cols fanogan_score residual_score feature_score pixel_mse
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

try:
    from sklearn.metrics import (
        roc_auc_score, roc_curve, accuracy_score,
        precision_score, recall_score, f1_score,
    )
    _SKLEARN = True
except ImportError:
    log.warning("scikit-learn not found — some metrics unavailable. pip install scikit-learn")
    _SKLEARN = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL = True
except ImportError:
    log.warning("matplotlib not found — plots disabled. pip install matplotlib")
    _MPL = False


# ── Attack display names (mirrors iris_bbdm_pad) ──────────────────────────────

ATTACK_DISPLAY = {
    "Artifact":          "Artifact",
    "CL":                "Contact Lens",
    "E-display":         "E-display",
    "Fake_with_Add_On":  "Fake with Add On",
    "Fake with Add On":  "Fake with Add On",
    "Generated":         "Generated",
    "Post-Mortem":       "Post-Mortem",
    "PostMortem":        "Post-Mortem",
    "Printed":           "Printed",
    "Print_E-display":   "Print & E-display",
    "Print and E-display": "Print & E-display",
}

# Evaluate all score columns; vit_score matches Diff-IrisPAD paper Table 2 scoring
SCORE_COLS_DEFAULT = ["vit_score", "fanogan_score", "recon_score", "mse_score",
                      "lpips_score", "residual_score", "feature_score"]


# ---------------------------------------------------------------------------
# Core metric helpers
# ---------------------------------------------------------------------------

def compute_eer(fpr: np.ndarray, tpr: np.ndarray) -> Tuple[float, float]:
    """Return (EER, threshold_index) via FPR/FNR crossing."""
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    eer = float((fpr[idx] + fnr[idx]) / 2)
    return eer, idx


def tdr_at_fdr(fpr: np.ndarray, tpr: np.ndarray, target_fdr: float) -> float:
    """TDR (TPR) at a given FDR (FPR) operating point."""
    idx = np.searchsorted(fpr, target_fdr, side="right") - 1
    idx = max(0, min(idx, len(tpr) - 1))
    return float(tpr[idx])


def compute_pad_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: Optional[float] = None,
) -> dict:
    """Compute full PAD metric suite.

    Args:
        y_true    : binary array (1=attack, 0=bonafide)
        scores    : anomaly scores (higher = more anomalous)
        threshold : if None, use EER threshold from ROC curve

    Returns:
        dict with APCER, BPCER, ACER, EER, AUC, Acc, P, R, F1, TDR@FDR*, threshold
    """
    if not _SKLEARN:
        return {}

    auc = float(roc_auc_score(y_true, scores))
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    eer, eer_idx = compute_eer(fpr, tpr)

    if threshold is None:
        threshold = float(thresholds[eer_idx])

    # BBDM convention (matches iris_bbdm_pad/training/find_threshold.py exactly):
    #   score > threshold → predicted attack (positive=1)
    #   label 1 = attack, label 0 = bonafide
    y_pred = (scores > threshold).astype(int)

    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))

    # Matches find_threshold.py: apcer = fp/(tn+fp), bpcer = fn/(tp+fn)
    apcer = float(fp / max(tn + fp, 1))
    bpcer = float(fn / max(tp + fn, 1))
    acer  = (apcer + bpcer) / 2

    return {
        "APCER":    round(apcer * 100, 3),
        "BPCER":    round(bpcer * 100, 3),
        "ACER":     round(acer  * 100, 3),
        "EER":      round(eer   * 100, 3),
        "AUC":      round(auc   * 100, 3),
        "Accuracy": round(float(accuracy_score(y_true, y_pred)) * 100, 3),
        "Precision":round(float(precision_score(y_true, y_pred, zero_division=0)) * 100, 3),
        "Recall":   round(float(recall_score(y_true, y_pred, zero_division=0)) * 100, 3),
        "F1":       round(float(f1_score(y_true, y_pred, zero_division=0)) * 100, 3),
        "TDR@FDR0.1": round(tdr_at_fdr(fpr, tpr, 0.001) * 100, 3),
        "TDR@FDR1":   round(tdr_at_fdr(fpr, tpr, 0.01)  * 100, 3),
        "TDR@FDR5":   round(tdr_at_fdr(fpr, tpr, 0.05)  * 100, 3),
        "threshold":  round(threshold, 6),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "_fpr": fpr.tolist(), "_tpr": tpr.tolist(), "_thresholds": thresholds.tolist(),
    }


# ---------------------------------------------------------------------------
# Per-attack APCER breakdown
# ---------------------------------------------------------------------------

def per_attack_apcer(
    df: pd.DataFrame, score_col: str, threshold: float
) -> Dict[str, float]:
    """APCER per attack type using the BBDM convention (score > threshold = attack)."""
    attacks = df[df["label"] == "attack"]
    result  = {}
    for atype, grp in attacks.groupby("attack_type"):
        display = ATTACK_DISPLAY.get(str(atype), str(atype))
        scores  = grp[score_col].values
        # score > threshold → predicted attack; fraction NOT predicted = per-attack error
        err_rate = float(np.sum(scores <= threshold) / max(len(scores), 1))
        result[display] = round(err_rate * 100, 3)
    return result


# ---------------------------------------------------------------------------
# DET / ROC plots
# ---------------------------------------------------------------------------

def save_det_curve(fpr, fnr, out_path: Path, title: str = "DET Curve"):
    if not _MPL:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr * 100, fnr * 100, lw=2)
    ax.set_xlabel("FPR (%)")
    ax.set_ylabel("FNR (%)")
    ax.set_title(title)
    ax.grid(True, ls="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_roc_curve(fpr, tpr, auc: float, out_path: Path, title: str = "ROC Curve"):
    if not _MPL:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr * 100, tpr * 100, lw=2, label=f"AUC={auc:.1f}%")
    ax.plot([0, 100], [0, 100], "k--", lw=1)
    ax.set_xlabel("FPR (%)")
    ax.set_ylabel("TPR (%)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, ls="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(args):
    df = pd.read_csv(args.scores)
    log.info(f"Loaded {len(df)} rows from {args.scores}")
    log.info(f"Columns: {list(df.columns)}")

    if "label" not in df.columns:
        log.error("CSV must have a 'label' column (bonafide / attack)")
        sys.exit(1)

    y_true = (df["label"] == "attack").astype(int).values

    # Determine which score columns to evaluate
    score_cols = args.score_cols or SCORE_COLS_DEFAULT
    score_cols = [c for c in score_cols if c in df.columns]
    if not score_cols:
        log.error(f"None of {args.score_cols} found in CSV. Available: {list(df.columns)}")
        sys.exit(1)
    log.info(f"Evaluating score columns: {score_cols}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for col in score_cols:
        log.info(f"\n{'='*60}")
        log.info(f"Score: {col}")
        scores = df[col].values

        # Val threshold (if val_scores provided)
        val_threshold = None
        if args.val_scores and Path(args.val_scores).exists():
            df_val   = pd.read_csv(args.val_scores)
            y_val    = (df_val["label"] == "attack").astype(int).values
            sc_val   = df_val[col].values
            if _SKLEARN:
                fpr_v, tpr_v, thr_v = roc_curve(y_val, sc_val)
                _, eer_idx_v = compute_eer(fpr_v, tpr_v)
                val_threshold = float(thr_v[eer_idx_v])
                log.info(f"  Val EER threshold: {val_threshold:.6f}")

        metrics = compute_pad_metrics(y_true, scores, threshold=val_threshold)
        if not metrics:
            log.warning("sklearn unavailable — skipping metrics")
            continue

        # Per-attack breakdown
        if "attack_type" in df.columns:
            pa_apcer = per_attack_apcer(df, col, threshold=metrics["threshold"])
            metrics["per_attack_APCER"] = pa_apcer
        else:
            pa_apcer = {}

        all_results[col] = metrics

        # Console summary
        log.info(f"  APCER: {metrics['APCER']:.3f}%")
        log.info(f"  BPCER: {metrics['BPCER']:.3f}%")
        log.info(f"  ACER:  {metrics['ACER']:.3f}%")
        log.info(f"  EER:   {metrics['EER']:.3f}%")
        log.info(f"  AUC:   {metrics['AUC']:.3f}%")
        log.info(f"  TDR@FDR0.1%: {metrics['TDR@FDR0.1']:.3f}%")
        log.info(f"  TDR@FDR1%:   {metrics['TDR@FDR1']:.3f}%")
        log.info(f"  TDR@FDR5%:   {metrics['TDR@FDR5']:.3f}%")
        if pa_apcer:
            log.info("  Per-attack APCER:")
            for at, rate in sorted(pa_apcer.items()):
                log.info(f"    {at:25s}: {rate:.3f}%")

        # DET + ROC curves
        fpr_a = np.array(metrics.pop("_fpr"))
        tpr_a = np.array(metrics.pop("_tpr"))
        metrics.pop("_thresholds", None)
        fnr_a = 1 - tpr_a

        safe_col = col.replace("/", "_")
        save_det_curve(fpr_a, fnr_a, out_dir / f"det_{safe_col}.png",
                       title=f"DET — {col}")
        save_roc_curve(fpr_a, tpr_a, metrics["AUC"],
                       out_dir / f"roc_{safe_col}.png",
                       title=f"ROC — {col} (AUC={metrics['AUC']:.1f}%)")

    # Save JSON summary
    json_out = out_dir / "results.json"
    with open(json_out, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"\nFull results → {json_out}")

    # Save threshold JSON (compatible with iris_bbdm_pad find_threshold.py format)
    if all_results:
        primary_col = score_cols[0]
        thr_out = out_dir / "threshold.json"
        with open(thr_out, "w") as f:
            json.dump({
                "score_col": primary_col,
                "threshold": all_results[primary_col].get("threshold"),
                "method":    "EER",
            }, f, indent=2)
        log.info(f"Threshold → {thr_out}")

    # Print comparison table
    log.info("\n" + "="*70)
    log.info(f"{'Score column':<25} {'APCER%':>8} {'BPCER%':>8} {'ACER%':>8} {'EER%':>7} {'AUC%':>7}")
    log.info("-"*70)
    for col, m in all_results.items():
        log.info(
            f"{col:<25} {m['APCER']:>8.3f} {m['BPCER']:>8.3f} "
            f"{m['ACER']:>8.3f} {m['EER']:>7.3f} {m['AUC']:>7.3f}"
        )
    log.info("="*70)


def parse_args():
    parser = argparse.ArgumentParser(
        description="f-AnoGAN PAD evaluation — APCER/BPCER/ACER/EER/AUC"
    )
    parser.add_argument("--scores",
        default="iris_anogan/results/fanogan_run1/pad_scores/test_scores.csv",
        help="PAD scores CSV from fanogan_anomaly_detector.py")
    parser.add_argument("--val_scores",
        default="iris_anogan/results/fanogan_run1/pad_scores/val_scores.csv",
        help="Validation scores CSV (ACER-minimizing threshold; same protocol as BBDM)")
    parser.add_argument("--output_dir",
        default="iris_anogan/results/fanogan_run1/evaluation/",
        help="Directory for metrics JSON + DET/ROC plots")
    parser.add_argument("--score_cols", nargs="+", default=None,
        help="Score columns to evaluate (default: fanogan_score residual_score feature_score pixel_mse)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
