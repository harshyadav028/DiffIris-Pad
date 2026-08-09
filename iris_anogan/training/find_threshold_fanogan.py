"""
Find optimal PAD threshold from f-AnoGAN validation scores.

Direct mirror of iris_bbdm_pad/training/find_threshold.py — same algorithm,
same threshold logic (score > threshold → attack), same APCER/BPCER convention,
applied to f-AnoGAN score columns instead of BBDM score columns.

This guarantees the threshold selection strategy is identical between
Diff-IrisPAD (BBDM) and f-AnoGAN, enabling a fair comparison in Table 2.

For EACH scoring method (fanogan_score, recon_score, mse_score, lpips_score, ...):
  1. ACER-minimising threshold  (same as BBDM)
  2. EER threshold              (same as BBDM)

Additionally supports Zero-Attack-Knowledge (ZAK) threshold:
  threshold = N-th percentile of bonafide-only validation scores
  (matches iris_bbdm_pad/evaluation/zero_attack_threshold.py)

Usage
-----
  cd ~/Documents/Geetanjali_PhD_IRIS_PAD

  # Standard (ACER-minimising, mirrors BBDM)
  conda run --no-capture-output -n iris_pad python \\
      iris_anogan/training/find_threshold_fanogan.py \\
      --scores_csv iris_anogan/results/fanogan_run1/pad_scores/val_scores.csv \\
      --output     iris_anogan/results/fanogan_run1/threshold.json

  # ZAK threshold (bonafide-only percentile)
  conda run --no-capture-output -n iris_pad python \\
      iris_anogan/training/find_threshold_fanogan.py \\
      --scores_csv iris_anogan/results/fanogan_run1/pad_scores/val_scores.csv \\
      --output     iris_anogan/results/fanogan_run1/threshold.json \\
      --zak_percentiles 90 95 97 99
"""

import argparse
import datetime
import json
import logging
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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL = True
except ImportError:
    _MPL = False

# Score columns produced by fanogan_anomaly_detector.py
# recon_score = mse_score + lpips_score  (same formula as BBDM recon_score)
SCORE_COLS = [
    "vit_score",       # ViT-B/16 cosine distance — SAME scoring as Diff-IrisPAD paper Table 2
    "fanogan_score",   # native f-AnoGAN score (latent residual + feature)
    "recon_score",     # pixel MSE + LPIPS — directly comparable to BBDM recon
    "mse_score",       # pixel MSE only
    "lpips_score",     # LPIPS only
    "residual_score",  # latent residual only
    "feature_score",   # discriminator feature score only
]


# ---------------------------------------------------------------------------
# Core helpers — identical to iris_bbdm_pad/training/find_threshold.py
# ---------------------------------------------------------------------------

def compute_metrics_at_threshold(
    scores: np.ndarray, labels: np.ndarray, threshold: float
) -> Dict[str, float]:
    """BBDM convention: score > threshold → predicted attack.

    Matches iris_bbdm_pad/training/find_threshold.py line-for-line.
    label 1 = attack, label 0 = bonafide.
    """
    preds = (scores > threshold).astype(int)
    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))

    apcer = fp / max(tn + fp, 1)
    bpcer = fn / max(tp + fn, 1)
    acer  = (apcer + bpcer) / 2.0

    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "apcer": apcer, "bpcer": bpcer, "acer": acer}


def find_min_acer_threshold(
    scores: np.ndarray, labels: np.ndarray, n_thresholds: int = 1000
) -> Tuple[float, float]:
    thresholds = np.linspace(scores.min(), scores.max(), n_thresholds)
    best_thr, best_acer = thresholds[0], float("inf")
    for thr in thresholds:
        m = compute_metrics_at_threshold(scores, labels, thr)
        if m["acer"] < best_acer:
            best_acer = m["acer"]
            best_thr  = thr
    return float(best_thr), float(best_acer)


def find_eer_threshold(
    scores: np.ndarray, labels: np.ndarray, n_thresholds: int = 1000
) -> Tuple[float, float]:
    thresholds = np.linspace(scores.min(), scores.max(), n_thresholds)
    best_thr, best_diff, best_eer = thresholds[0], float("inf"), float("inf")
    for thr in thresholds:
        preds = (scores > thr).astype(int)
        fp = np.sum((preds == 1) & (labels == 0))
        tn = np.sum((preds == 0) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))
        tp = np.sum((preds == 1) & (labels == 1))
        far = fp / max(fp + tn, 1)
        frr = fn / max(fn + tp, 1)
        diff = abs(far - frr)
        if diff < best_diff:
            best_diff = diff
            best_thr  = thr
            best_eer  = (far + frr) / 2.0
    return float(best_thr), float(best_eer)


# ---------------------------------------------------------------------------
# Zero-Attack-Knowledge threshold (bonafide percentile)
# Mirrors iris_bbdm_pad/evaluation/zero_attack_threshold.py
# ---------------------------------------------------------------------------

def find_zak_thresholds(
    scores: np.ndarray,
    labels: np.ndarray,
    percentiles: List[int] = (90, 95, 97, 99),
) -> Dict[str, dict]:
    """Set threshold from N-th percentile of bonafide val scores only."""
    bf_scores = scores[labels == 0]
    results = {}
    for p in percentiles:
        tau = float(np.percentile(bf_scores, p))
        m   = compute_metrics_at_threshold(scores, labels, tau)
        results[f"zak_p{p}"] = {
            "threshold": tau,
            "acer":  m["acer"],
            "apcer": m["apcer"],
            "bpcer": m["bpcer"],
        }
        log.info(f"  ZAK p{p:02d}: τ={tau:.4f} | ACER={m['acer']*100:.2f}% | "
                 f"APCER={m['apcer']*100:.2f}% | BPCER={m['bpcer']*100:.2f}%")
    return results


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _save_score_distribution(
    scores: np.ndarray, labels: np.ndarray,
    threshold: float, col: str, out_path: Path
):
    if not _MPL:
        return
    bf  = scores[labels == 0]
    atk = scores[labels == 1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(bf,  bins=60, alpha=0.6, density=True, color="#3b82f6", label="Bona fide")
    ax.hist(atk, bins=60, alpha=0.6, density=True, color="#ef4444", label="Attack")
    ax.axvline(threshold, color="black", linestyle="--", lw=1.5,
               label=f"τ={threshold:.4f}")
    ax.set_xlabel(f"PAD Score ({col})")
    ax.set_ylabel("Density")
    ax.set_title(f"Val Score Distributions — {col}")
    ax.legend()
    ax.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    scores_csv = Path(args.scores_csv)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading val scores from {scores_csv}")
    df = pd.read_csv(scores_csv)

    if "label" not in df.columns:
        raise ValueError("CSV must have a 'label' column with 'bonafide'/'attack' values.")

    labels    = (df["label"] == "attack").astype(int).values
    n_bf      = int((labels == 0).sum())
    n_atk     = int((labels == 1).sum())
    log.info(f"Val set: {n_bf} bonafide, {n_atk} attack")

    available = [c for c in SCORE_COLS if c in df.columns]
    log.info(f"Score columns: {available}")

    per_method: Dict[str, dict] = {}
    for col in available:
        scores   = df[col].values.astype(np.float64)
        best_thr, best_acer = find_min_acer_threshold(scores, labels)
        eer_thr,  eer_rate  = find_eer_threshold(scores, labels)
        m = compute_metrics_at_threshold(scores, labels, best_thr)
        per_method[col] = {
            "min_acer_threshold": float(best_thr),
            "acer":          float(best_acer),
            "apcer":         float(m["apcer"]),
            "bpcer":         float(m["bpcer"]),
            "eer_threshold": float(eer_thr),
            "eer":           float(eer_rate),
        }
        log.info(
            f"{col:25s}: ACER={best_acer*100:.2f}% | EER={eer_rate*100:.2f}% | "
            f"τ={best_thr:.4f}"
        )

    # ZAK thresholds (bonafide-only percentile) — for ZAK experiment
    zak_results: Dict[str, dict] = {}
    if args.zak_percentiles and available:
        primary_col = available[0]
        primary_scores = df[primary_col].values.astype(np.float64)
        log.info(f"\nZero-Attack-Knowledge thresholds ({primary_col}):")
        zak_results = find_zak_thresholds(
            primary_scores, labels,
            percentiles=args.zak_percentiles
        )

    # Select best method (lowest val ACER)
    best_method    = min(per_method, key=lambda m: per_method[m]["acer"])
    best_threshold = per_method[best_method]["min_acer_threshold"]
    best_acer      = per_method[best_method]["acer"]

    log.info(f"\nBest: {best_method} | ACER={best_acer*100:.2f}% | τ={best_threshold:.4f}")

    results = {
        "per_method":  per_method,
        "zak":         zak_results,
        "best_method": best_method,
        "best_threshold": float(best_threshold),
        "best_acer":   float(best_acer),
        "recommended": {
            "method":    best_method,
            "threshold": float(best_threshold),
        },
        "val_samples": {"bonafide": n_bf, "attack": n_atk},
        "computed_at": datetime.datetime.now().isoformat(),
        "note": (
            "Threshold convention: score > threshold → attack. "
            "Identical to iris_bbdm_pad/training/find_threshold.py."
        ),
    }

    output_path.write_text(json.dumps(results, indent=2))
    log.info(f"Threshold saved → {output_path}")

    # Score distribution plots
    vis_dir = output_path.parent / "threshold_vis"
    vis_dir.mkdir(parents=True, exist_ok=True)
    for col in available[:3]:
        _save_score_distribution(
            df[col].values.astype(np.float64), labels,
            per_method[col]["min_acer_threshold"], col,
            vis_dir / f"dist_{col}.png",
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Find f-AnoGAN PAD threshold (mirrors BBDM find_threshold.py)"
    )
    parser.add_argument(
        "--scores_csv",
        default="iris_anogan/results/fanogan_run1/pad_scores/val_scores.csv",
        help="Validation PAD scores CSV from fanogan_anomaly_detector.py",
    )
    parser.add_argument(
        "--output",
        default="iris_anogan/results/fanogan_run1/threshold.json",
        help="Output path for threshold JSON",
    )
    parser.add_argument(
        "--zak_percentiles", nargs="+", type=int,
        default=[90, 95, 97, 99],
        help="Percentile thresholds for ZAK experiment (bonafide-only val scores)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
