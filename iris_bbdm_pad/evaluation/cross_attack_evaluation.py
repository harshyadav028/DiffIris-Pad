"""
Cross-attack evaluation: detailed analysis per attack type.

For each of the 8 attack types:
- Compute APCER, detection rate, score statistics
- Identify which attacks are hardest/easiest to detect
- Compute score separation from bonafide distribution

Usage:
    python iris_bbdm_pad/evaluation/cross_attack_evaluation.py \
        --test_scores iris_bbdm_pad/results/test_pad_scores.csv \
        --threshold_json iris_bbdm_pad/results/threshold.json \
        --output_dir iris_bbdm_pad/results/phase3_evaluation/
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("cross_attack_eval")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
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
# Core computation
# ---------------------------------------------------------------------------

def pooled_std(
    mean_a: float,
    std_a: float,
    n_a: int,
    mean_b: float,
    std_b: float,
    n_b: int,
) -> float:
    """Compute pooled standard deviation for two groups.

    Args:
        mean_a: Mean of group A (unused in formula but kept for clarity).
        std_a: Standard deviation of group A.
        n_a: Sample count for group A.
        mean_b: Mean of group B (unused in formula but kept for clarity).
        std_b: Standard deviation of group B.
        n_b: Sample count for group B.

    Returns:
        Pooled standard deviation as a float. Returns NaN if total n <= 2.
    """
    if n_a + n_b <= 2:
        return float("nan")
    numerator = (n_a - 1) * std_a**2 + (n_b - 1) * std_b**2
    denominator = n_a + n_b - 2
    variance_pooled = numerator / denominator
    return math.sqrt(max(variance_pooled, 1e-12))


def score_separation_d(
    attack_scores: np.ndarray,
    bonafide_scores: np.ndarray,
) -> float:
    """Compute Cohen's d-style score separation between attack and bonafide distributions.

    A positive value indicates attack scores are higher on average (correct direction).

    Args:
        attack_scores: 1-D array of anomaly scores for attack samples.
        bonafide_scores: 1-D array of anomaly scores for bonafide samples.

    Returns:
        Score separation as (mean_attack - mean_bonafide) / pooled_std.
        Returns NaN if either array is empty.
    """
    if len(attack_scores) == 0 or len(bonafide_scores) == 0:
        return float("nan")

    mean_a = float(np.mean(attack_scores))
    std_a = float(np.std(attack_scores, ddof=1)) if len(attack_scores) > 1 else 0.0
    n_a = len(attack_scores)

    mean_b = float(np.mean(bonafide_scores))
    std_b = float(np.std(bonafide_scores, ddof=1)) if len(bonafide_scores) > 1 else 0.0
    n_b = len(bonafide_scores)

    p_std = pooled_std(mean_a, std_a, n_a, mean_b, std_b, n_b)
    if math.isnan(p_std) or p_std < 1e-12:
        return float("nan")

    return (mean_a - mean_b) / p_std


def analyze_per_attack(
    df: pd.DataFrame,
    score_col: str,
    threshold: float,
    bonafide_scores: np.ndarray,
) -> pd.DataFrame:
    """Compute per-attack-type metrics including score separation from bonafide.

    Args:
        df: Full test DataFrame containing 'label', 'attack_type', and score_col.
        score_col: Column name of the anomaly score to use.
        threshold: Decision threshold (score > threshold → detected as attack).
        bonafide_scores: 1-D array of bonafide anomaly scores (for separation calc).

    Returns:
        DataFrame with one row per attack type and columns:
        attack_type, display_name, n_samples, apcer, detection_rate,
        mean_score, std_score, min_score, max_score, median_score,
        score_separation, difficulty_rank.
        Sorted by difficulty_rank ascending (hardest = rank 1).
    """
    attack_df = df[df["label"] == "attack"].copy()
    records: List[Dict] = []

    for attack_type, group in attack_df.groupby("attack_type"):
        scores = group[score_col].values
        n = len(scores)

        # APCER: fraction of attacks with score <= threshold (missed/not flagged)
        n_missed = int(np.sum(scores <= threshold))
        apcer = n_missed / n if n > 0 else 0.0
        detection_rate = 1.0 - apcer

        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores, ddof=1)) if n > 1 else 0.0
        min_score = float(np.min(scores))
        max_score = float(np.max(scores))
        median_score = float(np.median(scores))

        separation = score_separation_d(scores, bonafide_scores)

        records.append(
            {
                "attack_type": str(attack_type),
                "display_name": ATTACK_DISPLAY_NAMES.get(str(attack_type), str(attack_type)),
                "n_samples": n,
                "apcer": apcer,
                "detection_rate": detection_rate,
                "mean_score": mean_score,
                "std_score": std_score,
                "min_score": min_score,
                "max_score": max_score,
                "median_score": median_score,
                "score_separation": separation,
            }
        )

    result_df = pd.DataFrame(records)

    # Rank by difficulty: highest APCER = hardest to detect = rank 1
    result_df = result_df.sort_values("apcer", ascending=False).reset_index(drop=True)
    result_df["difficulty_rank"] = result_df.index + 1

    return result_df


def compute_bonafide_stats(bonafide_scores: np.ndarray) -> Dict:
    """Compute summary statistics for bonafide score distribution.

    Args:
        bonafide_scores: 1-D array of anomaly scores for bonafide samples.

    Returns:
        Dict with mean, std, min, max, median, n_samples.
    """
    n = len(bonafide_scores)
    return {
        "n_samples": n,
        "mean": float(np.mean(bonafide_scores)) if n > 0 else float("nan"),
        "std": float(np.std(bonafide_scores, ddof=1)) if n > 1 else 0.0,
        "min": float(np.min(bonafide_scores)) if n > 0 else float("nan"),
        "max": float(np.max(bonafide_scores)) if n > 0 else float("nan"),
        "median": float(np.median(bonafide_scores)) if n > 0 else float("nan"),
    }


def compute_overall_separation(
    df: pd.DataFrame,
    score_col: str,
    bonafide_scores: np.ndarray,
) -> float:
    """Compute score separation for all attacks combined vs all bonafide.

    Args:
        df: Full test DataFrame.
        score_col: Column name of the anomaly score.
        bonafide_scores: 1-D array of bonafide anomaly scores.

    Returns:
        Cohen's d-style separation (all attack vs all bonafide).
    """
    all_attack_scores = df[df["label"] == "attack"][score_col].values
    return score_separation_d(all_attack_scores, bonafide_scores)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_summary(
    result_df: pd.DataFrame,
    bonafide_stats: Dict,
    overall_separation: float,
    best_method: str,
    threshold: float,
) -> None:
    """Print a console summary of cross-attack analysis, sorted hardest-first.

    Args:
        result_df: Per-attack metrics DataFrame (already sorted by difficulty).
        bonafide_stats: Dict of bonafide score distribution statistics.
        overall_separation: Cohen's d for all attacks vs bonafide.
        best_method: Name of the scoring method used.
        threshold: Decision threshold applied.
    """
    method_label = best_method.replace("_score", "").upper()
    sep = "═" * 70
    dash = "─" * 70

    print(sep)
    print(" Cross-Attack Analysis — Sorted by Difficulty (Hardest First)")
    print(sep)
    print(f" Method: {method_label}   Threshold: {threshold:.6f}")
    print(
        f" Bonafide distribution — mean: {bonafide_stats['mean']:.4f}  "
        f"std: {bonafide_stats['std']:.4f}  "
        f"n: {bonafide_stats['n_samples']:,}"
    )
    print(f" Overall score separation (all attacks vs bonafide): {overall_separation:.4f}")
    print(dash)
    header = (
        f"  {'Rank':<5} {'Attack Type':<24} {'N':>6}  "
        f"{'APCER':>8}  {'DetRate':>8}  {'MeanScore':>10}  {'Separation':>11}"
    )
    print(header)
    print(dash)
    for _, row in result_df.iterrows():
        sep_str = (
            f"{row['score_separation']:>11.4f}"
            if not math.isnan(row["score_separation"])
            else f"{'N/A':>11}"
        )
        print(
            f"  {int(row['difficulty_rank']):<5} "
            f"{row['display_name']:<24} "
            f"{row['n_samples']:>6,}  "
            f"{row['apcer']*100:>7.2f}%  "
            f"{row['detection_rate']*100:>7.2f}%  "
            f"{row['mean_score']:>10.4f}  "
            f"{sep_str}"
        )
    print(dash)
    easiest = result_df.iloc[-1]
    hardest = result_df.iloc[0]
    print(
        f"  Hardest to detect: {hardest['display_name']} "
        f"(APCER={hardest['apcer']*100:.2f}%)"
    )
    print(
        f"  Easiest to detect: {easiest['display_name']} "
        f"(APCER={easiest['apcer']*100:.2f}%)"
    )
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Cross-attack detailed analysis for iris PAD.",
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
        help="Directory to write output files",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser.parse_args()


def main() -> None:
    """Main cross-attack evaluation entry point."""
    args = parse_args()
    logging.getLogger().setLevel(args.log_level)

    # ------------------------------------------------------------------
    # 1. Load inputs
    # ------------------------------------------------------------------
    logger.info("[1/5] Loading test scores from %s", args.test_scores)
    if not args.test_scores.exists():
        logger.error("Test scores file not found: %s", args.test_scores)
        sys.exit(1)
    df = pd.read_csv(args.test_scores)
    logger.info("[1/5] Loaded %d rows", len(df))

    logger.info("[1/5] Loading threshold config from %s", args.threshold_json)
    if not args.threshold_json.exists():
        logger.error("Threshold JSON not found: %s", args.threshold_json)
        sys.exit(1)
    with open(args.threshold_json, "r") as fh:
        threshold_data = json.load(fh)

    best_method: str = threshold_data["best_method"]
    best_threshold: float = float(threshold_data["best_threshold"])
    logger.info(
        "[1/5] Best method: %s  threshold: %.6f", best_method, best_threshold
    )

    # Validate score column exists
    if best_method not in df.columns:
        logger.error(
            "Score column '%s' not found in CSV. Available: %s",
            best_method, df.columns.tolist(),
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Compute bonafide baseline stats
    # ------------------------------------------------------------------
    logger.info("[2/5] Computing bonafide baseline statistics")
    bonafide_scores = df[df["label"] == "bonafide"][best_method].values
    bonafide_stats = compute_bonafide_stats(bonafide_scores)
    logger.info(
        "[2/5] Bonafide — n=%d  mean=%.4f  std=%.4f",
        bonafide_stats["n_samples"],
        bonafide_stats["mean"],
        bonafide_stats["std"],
    )

    # ------------------------------------------------------------------
    # 3. Per-attack analysis
    # ------------------------------------------------------------------
    logger.info("[3/5] Analyzing per-attack-type metrics")
    result_df = analyze_per_attack(df, best_method, best_threshold, bonafide_scores)
    n_attack_types = len(result_df)
    logger.info(
        "[3/5] Found %d distinct attack types: %s",
        n_attack_types,
        result_df["attack_type"].tolist(),
    )

    # ------------------------------------------------------------------
    # 4. Overall separation
    # ------------------------------------------------------------------
    logger.info("[4/5] Computing overall score separation")
    overall_sep = compute_overall_separation(df, best_method, bonafide_scores)
    logger.info("[4/5] Overall score separation (Cohen's d): %.4f", overall_sep)

    # ------------------------------------------------------------------
    # 5. Save outputs
    # ------------------------------------------------------------------
    logger.info("[5/5] Saving outputs to %s", args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # cross_attack_detailed.csv
    # Save the final column set as specified
    output_cols = [
        "attack_type",
        "n_samples",
        "apcer",
        "detection_rate",
        "mean_score",
        "std_score",
        "min_score",
        "max_score",
        "median_score",
        "score_separation",
        "difficulty_rank",
    ]
    csv_path = args.output_dir / "cross_attack_detailed.csv"
    result_df[output_cols].to_csv(csv_path, index=False)
    logger.info("  Saved %s", csv_path)

    # cross_attack_detailed.json
    json_payload = {
        "best_method": best_method,
        "threshold": best_threshold,
        "bonafide_stats": bonafide_stats,
        "overall_score_separation": overall_sep,
        "per_attack": result_df[output_cols].to_dict(orient="records"),
        "hardest_attack": result_df.iloc[0]["attack_type"],
        "easiest_attack": result_df.iloc[-1]["attack_type"],
    }
    json_path = args.output_dir / "cross_attack_detailed.json"
    with open(json_path, "w") as fh:
        json.dump(json_payload, fh, indent=2)
    logger.info("  Saved %s", json_path)

    logger.info(
        "[Step 5/5] Cross-attack analysis complete — %d attack types analysed, "
        "outputs saved to %s",
        n_attack_types,
        args.output_dir,
    )

    # ------------------------------------------------------------------
    # 6. Print console summary
    # ------------------------------------------------------------------
    print_summary(result_df, bonafide_stats, overall_sep, best_method, best_threshold)


if __name__ == "__main__":
    main()
