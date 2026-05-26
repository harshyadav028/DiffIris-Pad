"""
Run complete PAD scoring pipeline:
1. Score validation set -> val_pad_scores.csv
2. Find optimal threshold -> threshold.json
3. Score test set -> test_pad_scores.csv
4. Generate all Phase 2 visualizations

Usage:
    python iris_bbdm_pad/training/run_scoring.py \
        --config iris_bbdm_pad/configs/bbdm_iris_bonafide.yaml \
        --checkpoint results/iris_bonafide_pad/LBBDM-f4/checkpoint/top_model_epoch_XXX.pth \
        --output_dir iris_bbdm_pad/results/
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Add BBDM to sys.path for BBDM imports in anomaly_detector
BBDM_ROOT = PROJECT_ROOT / "BBDM"
if str(BBDM_ROOT) not in sys.path:
    sys.path.insert(0, str(BBDM_ROOT))


def run_scoring_pipeline(args: argparse.Namespace) -> None:
    """Orchestrate the complete scoring pipeline.

    Args:
        args: Parsed CLI arguments.
    """
    # Imports after sys.path is set
    from iris_bbdm_pad.models.anomaly_detector import BBDMAnomalyDetector
    from iris_bbdm_pad.training.find_threshold import (
        compute_metrics_at_threshold,
        find_min_acer_threshold,
        find_eer_threshold,
        optimize_combined_weights,
        save_threshold_visualizations,
    )

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "phase2_visualizations"
    paper_dir = vis_dir / "paper"

    val_dir = PROJECT_ROOT / args.val_dir
    test_dir = PROJECT_ROOT / args.test_dir
    val_csv = output_dir / "val_pad_scores.csv"
    test_csv = output_dir / "test_pad_scores.csv"
    threshold_json = output_dir / "threshold.json"

    logger.info("=" * 60)
    logger.info("Phase 2 Scoring Pipeline")
    logger.info("=" * 60)

    # ── Load detector ──────────────────────────────────────────────────────
    logger.info(f"[Step 1/5] Loading detector from {args.checkpoint}")
    detector = BBDMAnomalyDetector(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        trajectory_timesteps=args.trajectory_timesteps,
    )

    # ── Score validation set ───────────────────────────────────────────────
    logger.info(f"[Step 2/5] Scoring validation set: {val_dir}")
    val_ckpt = PROJECT_ROOT / "iris_bbdm_pad" / "checkpoints" / "val_scoring_checkpoint.json"
    val_df = detector.process_dataset(
        data_dir=val_dir,
        output_csv=val_csv,
        batch_size=args.batch_size,
        resume=args.resume,
        checkpoint_path=val_ckpt,
    )
    logger.info(f"  Validation scores saved → {val_csv}")

    # ── Find threshold ─────────────────────────────────────────────────────
    logger.info("[Step 3/5] Finding optimal threshold...")
    labels = (val_df["label"] == "attack").astype(int).values

    score_cols = ["mse_score", "lpips_score", "recon_score", "trajectory_score", "combined_score"]
    available_cols = [c for c in score_cols if c in val_df.columns]

    per_method = {}
    for col in available_cols:
        scores = val_df[col].values.astype(float)
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
            f"  {col:25s}: ACER={best_acer*100:.2f}%  EER={eer_rate*100:.2f}%  thr={best_thr:.4f}"
        )

    opt_weights = {}
    if "recon_score" in val_df.columns and "trajectory_score" in val_df.columns:
        opt_weights = optimize_combined_weights(val_df)
        logger.info(
            f"  Optimal weights: w_recon={opt_weights['w_recon']:.1f}, "
            f"w_trajectory={opt_weights['w_trajectory']:.1f}, "
            f"ACER={opt_weights['best_acer']*100:.2f}%"
        )
        import numpy as np
        recon_n = (val_df["recon_score"].values - val_df["recon_score"].min()) / \
                  max(val_df["recon_score"].max() - val_df["recon_score"].min(), 1e-10)
        traj_n = (val_df["trajectory_score"].values - val_df["trajectory_score"].min()) / \
                 max(val_df["trajectory_score"].max() - val_df["trajectory_score"].min(), 1e-10)
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

    best_method = min(per_method, key=lambda m: per_method[m]["acer"])
    best_threshold = per_method[best_method]["min_acer_threshold"]
    best_acer = per_method[best_method]["acer"]
    best_eer = per_method[best_method]["eer"]

    import datetime
    threshold_results = {
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
        "val_samples": {
            "bonafide": int((labels == 0).sum()),
            "attack": int((labels == 1).sum()),
        },
        "computed_at": datetime.datetime.now().isoformat(),
    }
    threshold_json.write_text(json.dumps(threshold_results, indent=2))
    logger.info(f"  Threshold saved → {threshold_json}")

    # ── Score test set ─────────────────────────────────────────────────────
    logger.info(f"[Step 4/5] Scoring test set: {test_dir}")
    test_ckpt = PROJECT_ROOT / "iris_bbdm_pad" / "checkpoints" / "test_scoring_checkpoint.json"
    test_df = detector.process_dataset(
        data_dir=test_dir,
        output_csv=test_csv,
        batch_size=args.batch_size,
        resume=args.resume,
        checkpoint_path=test_ckpt,
    )
    logger.info(f"  Test scores saved → {test_csv}")

    # ── Generate all visualizations ────────────────────────────────────────
    logger.info("[Step 5/5] Generating visualizations...")

    # Threshold visualizations (using val set)
    primary_method = min(
        {k: v for k, v in per_method.items() if k != "combined_score_optimal"},
        key=lambda m: per_method[m]["acer"],
        default=best_method,
    )
    import numpy as np
    best_scores = val_df[primary_method].values.astype(float) if primary_method in val_df.columns \
        else val_df["recon_score"].values.astype(float)
    save_threshold_visualizations(
        val_df, threshold_results, primary_method, best_scores, labels, vis_dir
    )

    # Reconstruction grids (using test set)
    detector.save_reconstruction_grids(test_dir, test_df, vis_dir)
    detector.save_trajectory_visualizations(test_dir, test_df, vis_dir)
    detector.save_score_scatter(test_df, vis_dir)

    # Paper figures (using test set)
    detector.save_paper_figures(test_dir, test_df, paper_dir)

    # ── Summary ────────────────────────────────────────────────────────────
    n_val_bf = int((val_df["label"] == "bonafide").sum())
    n_val_atk = int((val_df["label"] == "attack").sum())
    n_test_bf = int((test_df["label"] == "bonafide").sum())
    n_test_atk = int((test_df["label"] == "attack").sum())

    print("\n" + "═" * 51)
    print("  Phase 2 Scoring Complete")
    print("═" * 51)
    print(f"  Val set:  {n_val_bf:5d} bonafide, {n_val_atk:5d} attack")
    print(f"  Test set: {n_test_bf:5d} bonafide, {n_test_atk:5d} attack")
    print()
    print(f"  Best method:          {best_method}")
    print(f"  Threshold (min ACER): {best_threshold:.4f}")
    print(f"  Val ACER:             {best_acer * 100:.2f}%")
    print(f"  Val EER:              {best_eer * 100:.2f}%")
    print()
    def _fmt(p: Path) -> str:
        try:
            return str(p.relative_to(PROJECT_ROOT))
        except ValueError:
            return str(p)

    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Scores:     {_fmt(val_csv)}")
    print(f"              {_fmt(test_csv)}")
    print(f"  Threshold:  {_fmt(threshold_json)}")
    print(f"  Figures:    {_fmt(vis_dir)}/")
    print("═" * 51 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Complete Phase 2 PAD scoring pipeline."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="iris_bbdm_pad/configs/bbdm_iris_bonafide.yaml",
        help="BBDM YAML config path.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained LBBDM checkpoint (.pth).",
    )
    parser.add_argument(
        "--val_dir",
        type=str,
        default="iris_bbdm_pad/data/evaluation_sets/val/",
        help="Validation set directory (images/, labels.csv).",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default="iris_bbdm_pad/data/evaluation_sets/test/",
        help="Test set directory (images/, labels.csv).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="iris_bbdm_pad/results/",
        help="Output directory for all results.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="'cuda' or 'cpu'.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for scoring.",
    )
    parser.add_argument(
        "--trajectory_timesteps",
        type=int,
        nargs="+",
        default=[100, 250, 500, 750],
        help="Timesteps for trajectory stability scoring.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from scoring checkpoints.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_scoring_pipeline(args)
