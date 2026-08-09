"""
CLI wrapper for f-AnoGAN scoring — thin shell around fanogan_anomaly_detector.

Provides convenient presets for val and test splits, matching the interface
of iris_td/training/run_ddpm_scoring.py and iris_bbdm_pad/training/run_scoring.py.

Usage
-----
  cd ~/Documents/Geetanjali_PhD_IRIS_PAD

  # Score test set
  conda run --no-capture-output -n iris_pad python \\
      iris_anogan/training/run_fanogan_scoring.py --split test

  # Score val set
  conda run --no-capture-output -n iris_pad python \\
      iris_anogan/training/run_fanogan_scoring.py --split val

  # Custom kappa / batch size
  conda run --no-capture-output -n iris_pad python \\
      iris_anogan/training/run_fanogan_scoring.py \\
      --split test --kappa 0.5 --batch_size 64

  # Dry-run (first 200 images)
  conda run --no-capture-output -n iris_pad python \\
      iris_anogan/training/run_fanogan_scoring.py \\
      --split test --max_images 200
"""

import argparse
import subprocess
import sys
from pathlib import Path

# ── Defaults ──────────────────────────────────────────────────────────────────

CONFIG    = "iris_anogan/configs/fanogan_iris.yaml"
S1_CKPT   = "iris_anogan/results/fanogan_run1/stage1/checkpoint/best.pth"
S2_CKPT   = "iris_anogan/results/fanogan_run1/stage2/checkpoint/best.pth"

SPLIT_DEFAULTS = {
    # Use iris_bbdm_pad evaluation sets (same 47434 test images as iris_td)
    # — keeps the comparison against BBDM on identical data
    "test": {
        "test_dir":   "iris_bbdm_pad/data/evaluation_sets/test/",
        "labels":     "iris_bbdm_pad/data/evaluation_sets/test/labels.csv",
        "output_csv": "iris_anogan/results/fanogan_run1/pad_scores/test_scores.csv",
    },
    "val": {
        "test_dir":   "iris_bbdm_pad/data/evaluation_sets/val/",
        "labels":     "iris_bbdm_pad/data/evaluation_sets/val/labels.csv",
        "output_csv": "iris_anogan/results/fanogan_run1/pad_scores/val_scores.csv",
    },
}

DETECTOR = str(
    Path(__file__).resolve().parents[1] / "models" / "fanogan_anomaly_detector.py"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run f-AnoGAN PAD scoring on iris evaluation sets"
    )
    parser.add_argument("--split", choices=["test", "val"], default="test")
    parser.add_argument("--config",     default=CONFIG)
    parser.add_argument("--s1_ckpt",    default=S1_CKPT)
    parser.add_argument("--s2_ckpt",    default=S2_CKPT)
    parser.add_argument("--output_csv", default=None,
        help="Override output CSV path")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--kappa",      type=float, default=1.0)
    parser.add_argument("--no_lpips",   action="store_true")
    parser.add_argument("--no_pixel",   action="store_true")
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--resume", action="store_true",
        help="Resume interrupted scoring — skip already-scored images, append to CSV")
    return parser.parse_args()


def main():
    args    = parse_args()
    split_d = SPLIT_DEFAULTS[args.split]

    output_csv = args.output_csv or split_d["output_csv"]

    cmd = [
        sys.executable, DETECTOR,
        "--config",      args.config,
        "--s1_ckpt",     args.s1_ckpt,
        "--s2_ckpt",     args.s2_ckpt,
        "--test_dir",    split_d["test_dir"],
        "--labels",      split_d["labels"],
        "--output_csv",  output_csv,
        "--batch_size",  str(args.batch_size),
        "--kappa",       str(args.kappa),
        "--split",       args.split,
    ]
    if args.no_lpips:
        cmd.append("--no_lpips")
    if args.no_pixel:
        cmd.append("--no_pixel")
    if args.resume:
        cmd.append("--resume")
    if args.max_images:
        cmd.extend(["--max_images", str(args.max_images)])

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
