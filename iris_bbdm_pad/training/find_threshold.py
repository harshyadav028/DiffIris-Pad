"""
Find optimal PAD score threshold using validation set.

For EACH scoring method (mse, lpips, recon_score, trajectory_score, combined_score):
1. Minimize ACER: threshold where (APCER + BPCER) / 2 is lowest
2. EER point: threshold where FAR == FRR

Additionally optimizes combined score weights (w_recon, w_trajectory) via grid search.

Usage:
    python iris_bbdm_pad/training/find_threshold.py \
        --scores_csv iris_bbdm_pad/results/val_pad_scores.csv \
        --output iris_bbdm_pad/results/threshold.json
"""

import argparse
import datetime
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_metrics_at_threshold(
    scores: np.ndarray, labels: np.ndarray, threshold: float
) -> Dict[str, float]:
    """Compute TP, FP, TN, FN, APCER, BPCER, ACER at a given threshold.

    Convention: score > threshold -> predicted attack (positive).
                label 1 = attack, label 0 = bonafide.

    Args:
        scores: (N,) PAD scores.
        labels: (N,) binary labels (1=attack, 0=bonafide).
        threshold: Decision threshold.

    Returns:
        Dict with tp, fp, tn, fn, apcer, bpcer, acer.
    """
    preds = (scores > threshold).astype(int)
    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))

    apcer = fp / max(tn + fp, 1)  # Attack Presentation Classification Error Rate
    bpcer = fn / max(tp + fn, 1)  # Bonafide Presentation Classification Error Rate
    acer = (apcer + bpcer) / 2.0

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "apcer": apcer, "bpcer": bpcer, "acer": acer,
    }


def find_min_acer_threshold(
    scores: np.ndarray, labels: np.ndarray, n_thresholds: int = 1000
) -> Tuple[float, float]:
    """Find threshold minimizing ACER.

    Args:
        scores: (N,) PAD scores.
        labels: (N,) binary labels.
        n_thresholds: Number of threshold candidates to try.

    Returns:
        (best_threshold, best_acer).
    """
    thresholds = np.linspace(scores.min(), scores.max(), n_thresholds)
    best_thr = thresholds[0]
    best_acer = float("inf")
    for thr in thresholds:
        m = compute_metrics_at_threshold(scores, labels, thr)
        if m["acer"] < best_acer:
            best_acer = m["acer"]
            best_thr = thr
    return float(best_thr), float(best_acer)


def find_eer_threshold(
    scores: np.ndarray, labels: np.ndarray, n_thresholds: int = 1000
) -> Tuple[float, float]:
    """Find threshold where |FAR - FRR| is minimized (EER point).

    Args:
        scores: (N,) PAD scores.
        labels: (N,) binary labels.
        n_thresholds: Number of threshold candidates.

    Returns:
        (eer_threshold, eer_rate).
    """
    thresholds = np.linspace(scores.min(), scores.max(), n_thresholds)
    best_thr = thresholds[0]
    best_diff = float("inf")
    best_eer = float("inf")
    for thr in thresholds:
        preds = (scores > thr).astype(int)
        # FAR: FP / (FP + TN) = false accept rate (bonafide rejected as genuine)
        fp = np.sum((preds == 1) & (labels == 0))
        tn = np.sum((preds == 0) & (labels == 0))
        # FRR: FN / (FN + TP) = false reject rate (attack accepted as genuine)
        fn = np.sum((preds == 0) & (labels == 1))
        tp = np.sum((preds == 1) & (labels == 1))
        far = fp / max(fp + tn, 1)
        frr = fn / max(fn + tp, 1)
        diff = abs(far - frr)
        if diff < best_diff:
            best_diff = diff
            best_thr = thr
            best_eer = (far + frr) / 2.0
    return float(best_thr), float(best_eer)


def optimize_combined_weights(
    df: pd.DataFrame, n_thresholds: int = 1000
) -> Dict[str, float]:
    """Grid search for optimal w_recon and w_trajectory.

    Tests all w_recon in [0.0, 0.1, ..., 1.0], w_trajectory = 1 - w_recon.
    Selects weights minimizing ACER on validation set.

    Args:
        df: DataFrame with 'recon_score', 'trajectory_score', 'label' columns.
        n_thresholds: Threshold sweep resolution.

    Returns:
        Dict with 'w_recon', 'w_trajectory', 'best_acer', 'acers_by_weight'.
    """
    labels = (df["label"] == "attack").astype(int).values
    recon = df["recon_score"].values.astype(np.float64)
    traj = df["trajectory_score"].values.astype(np.float64)

    def global_norm(x: np.ndarray) -> np.ndarray:
        mn, mx = x.min(), x.max()
        if abs(mx - mn) < 1e-10:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)

    recon_n = global_norm(recon)
    traj_n = global_norm(traj)

    weight_grid = np.round(np.linspace(0.0, 1.0, 11), 2)
    best_acer = float("inf")
    best_w_recon = 0.5
    acers_by_weight = {}

    for w_r in weight_grid:
        w_t = round(1.0 - w_r, 2)
        combined = w_r * recon_n + w_t * traj_n
        _, acer = find_min_acer_threshold(combined, labels, n_thresholds)
        acers_by_weight[float(w_r)] = float(acer)
        if acer < best_acer:
            best_acer = acer
            best_w_recon = w_r

    return {
        "w_recon": float(best_w_recon),
        "w_trajectory": float(round(1.0 - best_w_recon, 2)),
        "best_acer": float(best_acer),
        "acers_by_weight": acers_by_weight,
    }


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def save_threshold_visualizations(
    df: pd.DataFrame,
    results: Dict,
    best_method: str,
    best_scores: np.ndarray,
    labels: np.ndarray,
    vis_dir: Path,
) -> None:
    """Save all threshold-related visualizations.

    Args:
        df: Full scores DataFrame.
        results: Threshold analysis results dict.
        best_method: Name of the best scoring method.
        best_scores: (N,) best method scores.
        labels: (N,) binary labels.
        vis_dir: Output directory.
    """
    vis_dir.mkdir(parents=True, exist_ok=True)
    best_thr = results["recommended"]["threshold"]

    # 1. Method comparison bar chart
    methods = list(results["per_method"].keys())
    acers = [results["per_method"][m]["acer"] * 100 for m in methods]
    colors = ["#10b981" if m == best_method else "#6b7280" for m in methods]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(methods, acers, color=colors, edgecolor="white", linewidth=0.5)
    ax.bar_label(bars, fmt="%.2f%%", padding=3, fontsize=9)
    ax.set_ylabel("ACER (%)")
    ax.set_title("ACER Comparison Across PAD Scoring Methods")
    ax.set_ylim(0, max(acers) * 1.2)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", alpha=0.3)
    # Highlight best
    best_idx = methods.index(best_method)
    bars[best_idx].set_edgecolor("#10b981")
    bars[best_idx].set_linewidth(2)
    ax.text(best_idx, acers[best_idx] + 0.5, "BEST", ha="center", fontsize=8, color="#10b981")
    fig.tight_layout()
    fig.savefig(vis_dir / "threshold_method_comparison.png", dpi=300)
    plt.close(fig)
    logger.info("[Step] Saved threshold_method_comparison.png")

    # 2. ACER curve
    thresholds = np.linspace(best_scores.min(), best_scores.max(), 1000)
    acer_vals = []
    for thr in thresholds:
        m = compute_metrics_at_threshold(best_scores, labels, thr)
        acer_vals.append(m["acer"] * 100)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, acer_vals, linewidth=2, color="#2563eb")
    ax.axvline(x=best_thr, color="#ef4444", linestyle="--", label=f"Threshold={best_thr:.4f}")
    ax.set_xlabel(f"Threshold ({best_method})")
    ax.set_ylabel("ACER (%)")
    ax.set_title(f"ACER vs Threshold — {best_method}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(vis_dir / "threshold_acer_curve.png", dpi=300)
    plt.close(fig)
    logger.info("[Step] Saved threshold_acer_curve.png")

    # 3. Score distributions
    bf_scores = best_scores[labels == 0]
    atk_scores = best_scores[labels == 1]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(bf_scores, bins=60, alpha=0.6, color="#3b82f6", density=True, label="Bona Fide")
    ax.hist(atk_scores, bins=60, alpha=0.6, color="#ef4444", density=True, label="Attack")
    ax.axvline(x=best_thr, color="black", linestyle="--",
               linewidth=1.5, label=f"Threshold={best_thr:.4f}")
    ax.set_xlabel(f"PAD Score ({best_method})")
    ax.set_ylabel("Density")
    ax.set_title(f"PAD Score Distributions — {best_method}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(vis_dir / "threshold_score_distributions.png", dpi=300)
    plt.close(fig)
    logger.info("[Step] Saved threshold_score_distributions.png")

    # 4. FAR / FRR curves
    far_vals, frr_vals = [], []
    for thr in thresholds:
        preds = (best_scores > thr).astype(int)
        fp = np.sum((preds == 1) & (labels == 0))
        tn = np.sum((preds == 0) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))
        tp = np.sum((preds == 1) & (labels == 1))
        far_vals.append(fp / max(fp + tn, 1))
        frr_vals.append(fn / max(fn + tp, 1))

    eer_thr = results["per_method"][best_method].get("eer_threshold", best_thr)
    eer_val = results["per_method"][best_method].get("eer", 0)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, far_vals, linewidth=2, color="#ef4444", label="FAR")
    ax.plot(thresholds, frr_vals, linewidth=2, color="#3b82f6", label="FRR")
    ax.axvline(x=eer_thr, color="black", linestyle="--",
               label=f"EER={eer_val*100:.2f}% @ thr={eer_thr:.4f}")
    ax.set_xlabel(f"Threshold ({best_method})")
    ax.set_ylabel("Rate")
    ax.set_title(f"FAR / FRR Curves — {best_method}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(vis_dir / "threshold_far_frr.png", dpi=300)
    plt.close(fig)
    logger.info("[Step] Saved threshold_far_frr.png")

    # 5. Weight optimization heatmap (line plot)
    if "optimal_weights" in results and "acers_by_weight" in results.get("optimal_weights", {}):
        wdata = results["optimal_weights"]["acers_by_weight"]
        ws = sorted(wdata.keys())
        acer_w = [wdata[w] * 100 for w in ws]
        best_w = results["optimal_weights"]["w_recon"]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(ws, acer_w, marker="o", linewidth=2, color="#7c3aed")
        ax.axvline(x=best_w, color="#ef4444", linestyle="--",
                   label=f"Optimal w_recon={best_w:.1f}")
        ax.set_xlabel("w_recon (w_trajectory = 1 - w_recon)")
        ax.set_ylabel("Min ACER (%)")
        ax.set_title("Combined Score ACER vs Weight w_recon")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(vis_dir / "weight_optimization_heatmap.png", dpi=300)
        plt.close(fig)
        logger.info("[Step] Saved weight_optimization_heatmap.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    scores_csv = Path(args.scores_csv)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading validation scores from {scores_csv}")
    df = pd.read_csv(scores_csv)

    if "label" not in df.columns:
        raise ValueError("CSV must have a 'label' column with 'bonafide'/'attack' values.")

    labels = (df["label"] == "attack").astype(int).values
    n_bonafide = int((labels == 0).sum())
    n_attack = int((labels == 1).sum())
    logger.info(f"Validation set: {n_bonafide} bonafide, {n_attack} attack")

    score_cols = ["mse_score", "lpips_score", "recon_score", "trajectory_score", "combined_score"]
    # Only keep columns that exist
    available_cols = [c for c in score_cols if c in df.columns]

    per_method: Dict[str, Dict] = {}
    for col in available_cols:
        scores = df[col].values.astype(np.float64)
        best_thr, best_acer = find_min_acer_threshold(scores, labels)
        eer_thr, eer_rate = find_eer_threshold(scores, labels)
        m = compute_metrics_at_threshold(scores, labels, best_thr)
        per_method[col] = {
            "min_acer_threshold": float(best_thr),
            "acer": float(best_acer),
            "apcer": float(m["apcer"]),
            "bpcer": float(m["bpcer"]),
            "eer_threshold": float(eer_thr),
            "eer": float(eer_rate),
        }
        logger.info(
            f"{col:25s}: ACER={best_acer*100:.2f}%  EER={eer_rate*100:.2f}%  thr={best_thr:.4f}"
        )

    # Optimize combined weights
    opt_weights: Dict = {}
    if "recon_score" in df.columns and "trajectory_score" in df.columns:
        logger.info("Running combined weight grid search...")
        opt_weights = optimize_combined_weights(df)
        logger.info(
            f"Optimal weights: w_recon={opt_weights['w_recon']:.1f}, "
            f"w_trajectory={opt_weights['w_trajectory']:.1f}, "
            f"ACER={opt_weights['best_acer']*100:.2f}%"
        )

        # Recompute combined score with optimal weights and find threshold
        recon_n = (df["recon_score"].values - df["recon_score"].min()) / \
                  max(df["recon_score"].max() - df["recon_score"].min(), 1e-10)
        traj_n = (df["trajectory_score"].values - df["trajectory_score"].min()) / \
                 max(df["trajectory_score"].max() - df["trajectory_score"].min(), 1e-10)
        opt_combined = opt_weights["w_recon"] * recon_n + opt_weights["w_trajectory"] * traj_n
        opt_thr, opt_acer = find_min_acer_threshold(opt_combined, labels)
        opt_eer_thr, opt_eer = find_eer_threshold(opt_combined, labels)

        per_method["combined_score_optimal"] = {
            "min_acer_threshold": float(opt_thr),
            "acer": float(opt_acer),
            "eer_threshold": float(opt_eer_thr),
            "eer": float(opt_eer),
            "w_recon": opt_weights["w_recon"],
            "w_trajectory": opt_weights["w_trajectory"],
        }
        logger.info(
            f"{'combined_score_optimal':25s}: ACER={opt_acer*100:.2f}%  "
            f"EER={opt_eer*100:.2f}%  thr={opt_thr:.4f}"
        )

    # Select best method
    best_method = min(per_method, key=lambda m: per_method[m]["acer"])
    best_threshold = per_method[best_method]["min_acer_threshold"]
    best_acer = per_method[best_method]["acer"]

    logger.info(f"\nBest method: {best_method}  ACER={best_acer*100:.2f}%  thr={best_threshold:.4f}")

    results = {
        "per_method": per_method,
        "optimal_weights": opt_weights,
        "best_method": best_method,
        "best_threshold": float(best_threshold),
        "best_acer": float(best_acer),
        "recommended": {
            "method": best_method,
            "threshold": float(best_threshold),
            "w_recon": opt_weights.get("w_recon", 0.5),
            "w_trajectory": opt_weights.get("w_trajectory", 0.5),
        },
        "val_samples": {"bonafide": n_bonafide, "attack": n_attack},
        "computed_at": datetime.datetime.now().isoformat(),
    }

    output_path.write_text(json.dumps(results, indent=2))
    logger.info(f"Threshold results saved → {output_path}")

    # Visualizations
    vis_dir = PROJECT_ROOT / "iris_bbdm_pad" / "results" / "phase2_visualizations"
    # Use the primary best method (not the optimal recomputed one) for plots
    primary_method = min(
        {k: v for k, v in per_method.items() if k != "combined_score_optimal"},
        key=lambda m: per_method[m]["acer"],
        default=best_method,
    )
    best_scores = df[primary_method].values.astype(np.float64) if primary_method in df.columns else (
        df["recon_score"].values.astype(np.float64)
    )
    save_threshold_visualizations(df, results, primary_method, best_scores, labels, vis_dir)

    logger.info(
        f"\n{'='*50}\n"
        f"  Best method: {best_method}\n"
        f"  Best threshold: {best_threshold:.4f}\n"
        f"  Val ACER: {best_acer*100:.2f}%\n"
        f"  Val EER:  {per_method[best_method]['eer']*100:.2f}%\n"
        f"{'='*50}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find optimal PAD threshold from validation set scores."
    )
    parser.add_argument(
        "--scores_csv",
        type=str,
        default="iris_bbdm_pad/results/val_pad_scores.csv",
        help="Path to validation PAD scores CSV.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="iris_bbdm_pad/results/threshold.json",
        help="Output path for threshold JSON.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
