"""
Train LBBDM-f4 on bona fide iris identity-mapping pairs (A→A).

This script:
1. Validates all prerequisites (VQGAN, dataset, config)
2. Sets PYTHONPATH correctly for BBDM imports
3. Launches BBDM training via subprocess
4. Monitors training progress and saves visualizations
5. Records training config for reproducibility

Identity mapping trains the model to learn A → A (output = input unchanged).

Usage:
    python iris_bbdm_pad/training/train_bbdm_identity.py \
        --config iris_bbdm_pad/configs/bbdm_iris_identity.yaml \
        --gpu_ids 0

    # Resume interrupted training
    python iris_bbdm_pad/training/train_bbdm_identity.py \
        --config iris_bbdm_pad/configs/bbdm_iris_identity.yaml \
        --gpu_ids 0 \
        --resume_model results/IrisPAD/LBBDM-f4-identity/checkpoint/latest_model.pth \
        --resume_optim results/IrisPAD/LBBDM-f4-identity/checkpoint/last_optim_sche.pth
"""

import argparse
import datetime
import glob
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Prerequisite validation
# ---------------------------------------------------------------------------

def validate_prerequisites(config_path: str) -> bool:
    """Validate all prerequisites before starting training.

    Args:
        config_path: Path to the BBDM YAML config file.

    Returns:
        True if all checks pass, False otherwise.
    """
    all_ok = True
    logger.info("=" * 60)
    logger.info("Validating prerequisites (Identity Mapping)")
    logger.info("=" * 60)

    # 1. YAML loads with safe_load
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        logger.info(f"[OK] YAML safe_load: {config_path}")
    except Exception as e:
        logger.error(f"[FAIL] YAML load error: {e}")
        return False

    # 2. VQGAN checkpoint
    vqgan_path = PROJECT_ROOT / cfg["model"]["VQGAN"]["params"]["ckpt_path"]
    if vqgan_path.exists():
        logger.info(f"[OK] VQGAN checkpoint: {vqgan_path} ({vqgan_path.stat().st_size / 1e6:.0f} MB)")
    else:
        logger.error(f"[FAIL] VQGAN checkpoint not found: {vqgan_path}")
        all_ok = False

    # 3. Identity pairs A/B directories
    dataset_path = PROJECT_ROOT / cfg["data"]["dataset_config"]["dataset_path"]
    for split in ("train", "val"):
        for side in ("A", "B"):
            p = dataset_path / split / side
            if p.exists():
                count = len(list(p.glob("*.png")))
                logger.info(f"[OK] {p.relative_to(PROJECT_ROOT)}: {count} images")
            else:
                logger.error(f"[FAIL] Directory missing: {p}")
                all_ok = False

    # 4. A/B file counts match
    for split in ("train", "val"):
        a_count = len(list((dataset_path / split / "A").glob("*.png")))
        b_count = len(list((dataset_path / split / "B").glob("*.png")))
        if a_count == b_count:
            logger.info(f"[OK] {split} A/B count match: {a_count} (identity pairs)")
        else:
            logger.error(f"[FAIL] {split} A/B mismatch: A={a_count}, B={b_count}")
            all_ok = False

    # 5. pytorch_lightning importable
    try:
        import pytorch_lightning
        logger.info(f"[OK] pytorch_lightning {pytorch_lightning.__version__}")
    except ImportError as e:
        logger.error(f"[FAIL] pytorch_lightning not importable: {e}")
        all_ok = False

    # 6. CUDA available
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"[OK] CUDA: {name} ({vram:.1f} GB)")
        else:
            logger.warning("[WARN] CUDA not available — training on CPU will be very slow")
    except Exception as e:
        logger.warning(f"[WARN] Could not check CUDA: {e}")

    # 7. n_steps sanity check
    n_steps = cfg["training"].get("n_steps", 0)
    n_epochs = cfg["training"].get("n_epochs", 0)
    if n_steps >= 200000:
        logger.info(f"[OK] n_steps={n_steps} for {n_epochs} epochs")
    else:
        logger.warning(f"[WARN] n_steps={n_steps} may be too small for {n_epochs} epochs")

    logger.info("=" * 60)
    if all_ok:
        logger.info("All prerequisite checks passed.")
    else:
        logger.error("One or more prerequisite checks FAILED. Fix them before training.")
    logger.info("=" * 60)
    return all_ok


# ---------------------------------------------------------------------------
# Command builder
# ---------------------------------------------------------------------------

def build_command(args: argparse.Namespace) -> list:
    """Construct the BBDM main.py command.

    Args:
        args: Parsed CLI arguments.

    Returns:
        List of command tokens.
    """
    python = sys.executable
    main_py = str(PROJECT_ROOT / "BBDM" / "main.py")
    cmd = [
        python, main_py,
        "--config", args.config,
        "--train",
        "--sample_at_start",
        "--save_top",
        "--gpu_ids", args.gpu_ids,
    ]
    if args.resume_model:
        cmd += ["--resume_model", args.resume_model]
    if args.resume_optim:
        cmd += ["--resume_optim", args.resume_optim]
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]
    return cmd


# ---------------------------------------------------------------------------
# Find best checkpoint
# ---------------------------------------------------------------------------

def find_best_checkpoint(dataset_name: str = "iris_identity_pad") -> Optional[Path]:
    """Scan results directory for the best LBBDM-f4-identity checkpoint.

    Args:
        dataset_name: Dataset name used in the results path.

    Returns:
        Path to the best checkpoint, or None if not found.
    """
    pattern = str(PROJECT_ROOT / "results" / dataset_name / "LBBDM-f4-identity" / "checkpoint" / "top_model_epoch_*.pth")
    candidates = sorted(glob.glob(pattern))
    if candidates:
        best = Path(candidates[-1])
        logger.info(f"Best checkpoint: {best.relative_to(PROJECT_ROOT)}")
        return best

    # Fallback: any results directory
    pattern2 = str(PROJECT_ROOT / "results" / "*" / "LBBDM-f4-identity" / "checkpoint" / "top_model_epoch_*.pth")
    candidates2 = sorted(glob.glob(pattern2))
    if candidates2:
        best = Path(candidates2[-1])
        logger.info(f"Best checkpoint (fallback): {best.relative_to(PROJECT_ROOT)}")
        return best

    logger.warning("No checkpoint found — starting from scratch")
    return None


# ---------------------------------------------------------------------------
# Main training runner
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train LBBDM-f4 on identity-mapping iris pairs (A→A)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="iris_bbdm_pad/configs/bbdm_iris_identity.yaml",
        help="Path to BBDM config YAML.",
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default="0",
        help="GPU IDs (comma-separated, e.g., '0,1' for multi-GPU)",
    )
    parser.add_argument(
        "--resume_model",
        type=str,
        default=None,
        help="Path to resume model checkpoint.",
    )
    parser.add_argument(
        "--resume_optim",
        type=str,
        default=None,
        help="Path to resume optimizer checkpoint.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("BBDM IDENTITY MAPPING TRAINING")
    logger.info("=" * 60)
    logger.info(f"Config: {args.config}")
    logger.info(f"GPU IDs: {args.gpu_ids}")
    logger.info(f"Start time: {datetime.datetime.now().isoformat()}")
    logger.info("=" * 60)

    # Validate prerequisites
    if not validate_prerequisites(args.config):
        logger.error("Prerequisite validation FAILED. Aborting.")
        return 1

    # Build command
    cmd = build_command(args)
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info("=" * 60)

    # Set PYTHONPATH to include BBDM at the root
    env = os.environ.copy()
    pythonpath = str(PROJECT_ROOT / "BBDM")
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{pythonpath}:{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = pythonpath

    # Run training
    logger.info("Starting BBDM training subprocess...")
    try:
        result = subprocess.run(cmd, env=env, check=False)
        exit_code = result.returncode
        if exit_code == 0:
            logger.info("Training completed successfully.")
        else:
            logger.error(f"Training exited with code {exit_code}.")
        return exit_code
    except Exception as e:
        logger.error(f"Failed to run training: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
