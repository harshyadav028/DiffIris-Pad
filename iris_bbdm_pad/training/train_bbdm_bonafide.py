"""
Train LBBDM-f4 on bona fide iris noisy-to-clean pairs.

This script:
1. Validates all prerequisites (VQGAN, dataset, config)
2. Sets PYTHONPATH correctly for BBDM imports
3. Launches BBDM training via subprocess
4. Monitors training progress and saves visualizations
5. Records training config for reproducibility

Usage:
    python iris_bbdm_pad/training/train_bbdm_bonafide.py \
        --config iris_bbdm_pad/configs/bbdm_iris_bonafide.yaml \
        --gpu_ids 0

    # Resume interrupted training
    python iris_bbdm_pad/training/train_bbdm_bonafide.py \
        --config iris_bbdm_pad/configs/bbdm_iris_bonafide.yaml \
        --gpu_ids 0 \
        --resume_model results/IrisPAD/LBBDM-f4/checkpoint/latest_model.pth \
        --resume_optim results/IrisPAD/LBBDM-f4/checkpoint/last_optim_sche.pth
"""

import argparse
import datetime
import glob
import importlib
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
    logger.info("Validating prerequisites")
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

    # 3. Bonafide pairs A/B directories
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
            logger.info(f"[OK] {split} A/B count match: {a_count}")
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
    if n_steps >= 400000:
        logger.info(f"[OK] n_steps={n_steps} (>= 400000)")
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

def find_best_checkpoint(dataset_name: str = "iris_bonafide_pad") -> Optional[Path]:
    """Scan results directory for the best LBBDM-f4 checkpoint.

    Args:
        dataset_name: Dataset name used in the results path.

    Returns:
        Path to the best checkpoint, or None if not found.
    """
    pattern = str(PROJECT_ROOT / "results" / dataset_name / "LBBDM-f4" / "checkpoint" / "top_model_epoch_*.pth")
    candidates = sorted(glob.glob(pattern))
    if candidates:
        best = Path(candidates[-1])
        logger.info(f"Best checkpoint: {best.relative_to(PROJECT_ROOT)}")
        return best

    # Fallback: any results directory
    pattern2 = str(PROJECT_ROOT / "results" / "*" / "LBBDM-f4" / "checkpoint" / "top_model_epoch_*.pth")
    candidates2 = sorted(glob.glob(pattern2))
    if candidates2:
        best = Path(candidates2[-1])
        logger.info(f"Best checkpoint: {best.relative_to(PROJECT_ROOT)}")
        return best

    logger.warning("No top_model_epoch_*.pth checkpoint found.")
    return None


# ---------------------------------------------------------------------------
# Post-training visualizations
# ---------------------------------------------------------------------------

def save_training_loss_curve(log_dir: Path, vis_dir: Path) -> None:
    """Parse TensorBoard-style JSON logs and plot training loss.

    Args:
        log_dir: Directory containing BBDM log files.
        vis_dir: Output directory for the PNG.
    """
    vis_dir.mkdir(parents=True, exist_ok=True)
    loss_files = list(log_dir.glob("*.json")) + list(log_dir.glob("events.out.tfevents.*"))

    # Try to parse any JSON loss files
    steps, losses = [], []
    for jf in sorted(log_dir.glob("*.json")):
        try:
            data = json.loads(jf.read_text())
            if isinstance(data, dict) and "loss" in data:
                steps.append(data.get("step", len(steps)))
                losses.append(float(data["loss"]))
        except Exception:
            pass

    if not losses:
        logger.warning("No JSON loss data found — skipping loss curve plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, losses, linewidth=1.2, color="#2563eb")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss (L1)")
    ax.set_title("BBDM Training Loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = vis_dir / "training_loss_curve.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    logger.info(f"[Step] Saved training loss curve → {out.relative_to(PROJECT_ROOT)}")


def save_training_samples_grid(sample_dir: Path, vis_dir: Path) -> None:
    """Collect BBDM sample images and arrange into a grid.

    Looks for the latest epoch's skip_sample.png, condition.png, ground_truth.png
    saved by BBDMRunner.sample().

    Args:
        sample_dir: Path to BBDM sample directory.
        vis_dir: Output directory for the PNG.
    """
    from PIL import Image

    vis_dir.mkdir(parents=True, exist_ok=True)

    # Gather train_sample subdirs (one per epoch)
    sample_subdirs = sorted(sample_dir.glob("train_sample"))
    if not sample_subdirs:
        sample_subdirs = sorted(sample_dir.rglob("skip_sample.png"))
        if not sample_subdirs:
            logger.warning("No training sample images found — skipping grid.")
            return
        # Use parent directories
        sample_subdirs = [p.parent for p in sample_subdirs[-3:]]
    else:
        sample_subdirs = sample_subdirs[-1:]  # latest

    rows = []
    for sd in sample_subdirs:
        cond_p = sd / "condition.png"
        out_p = sd / "skip_sample.png"
        gt_p = sd / "ground_truth.png"
        if cond_p.exists() and out_p.exists() and gt_p.exists():
            cond = np.array(Image.open(cond_p))
            out = np.array(Image.open(out_p))
            gt = np.array(Image.open(gt_p))
            rows.append((cond, out, gt))

    if not rows:
        logger.warning("Could not load sample images — skipping grid.")
        return

    n = min(3, len(rows))
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    for r, (cond, out, gt) in enumerate(rows[:n]):
        axes[r, 0].imshow(cond)
        axes[r, 0].set_title("Corrupted input (A)")
        axes[r, 1].imshow(out)
        axes[r, 1].set_title("BBDM reconstruction")
        axes[r, 2].imshow(gt)
        axes[r, 2].set_title("Clean target (B)")
        for c in range(3):
            axes[r, c].axis("off")
    fig.suptitle("Training Sample Reconstructions", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out_path = vis_dir / "training_samples_grid.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    logger.info(f"[Step] Saved training samples grid → {out_path.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Main training runner
# ---------------------------------------------------------------------------

def run_training(args: argparse.Namespace) -> int:
    """Launch BBDM training subprocess with correct PYTHONPATH.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code of the subprocess.
    """
    results_dir = PROJECT_ROOT / "iris_bbdm_pad" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "bbdm_training.log"
    meta_path = results_dir / "bbdm_training_meta.json"

    # Build environment
    env = os.environ.copy()
    bbdm_path = str((PROJECT_ROOT / "BBDM").resolve())
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = bbdm_path + (os.pathsep + existing_pp if existing_pp else "")
    logger.info(f"PYTHONPATH set: {env['PYTHONPATH'][:80]}...")

    cmd = build_command(args)
    logger.info("Command: " + " ".join(cmd))

    # Detect GPU name for metadata
    gpu_name = "unknown"
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        pass

    start_time = datetime.datetime.now()
    meta = {
        "config_path": args.config,
        "start_time": start_time.isoformat(),
        "end_time": None,
        "gpu": gpu_name,
        "final_checkpoint": None,
        "command": " ".join(cmd),
        "exit_code": None,
    }

    exit_code = 0
    try:
        with open(log_path, "w") as log_f:
            logger.info(f"Training log → {log_path.relative_to(PROJECT_ROOT)}")
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=str(PROJECT_ROOT),
                text=True,
                bufsize=1,
            )
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log_f.write(line)
            process.wait()
            exit_code = process.returncode

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
        if process.poll() is None:
            process.terminate()
        exit_code = -1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        exit_code = -2

    end_time = datetime.datetime.now()
    duration = end_time - start_time
    logger.info(f"Training {'completed' if exit_code == 0 else 'stopped'} in {duration}. Exit code: {exit_code}")

    # Find best checkpoint
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    dataset_name = cfg["data"].get("dataset_name", "iris_bonafide_pad")
    best_ckpt = find_best_checkpoint(dataset_name)

    meta["end_time"] = end_time.isoformat()
    meta["final_checkpoint"] = str(best_ckpt) if best_ckpt else None
    meta["exit_code"] = exit_code
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info(f"Training metadata saved → {meta_path.relative_to(PROJECT_ROOT)}")

    # Post-training visualizations
    vis_dir = results_dir / "phase2_visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    bbdm_results = PROJECT_ROOT / "results" / dataset_name / "LBBDM-f4"
    log_dir = bbdm_results / "log"
    sample_dir = bbdm_results / "sample"

    if log_dir.exists():
        save_training_loss_curve(log_dir, vis_dir)
    if sample_dir.exists():
        save_training_samples_grid(sample_dir, vis_dir)

    return exit_code


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LBBDM-f4 on bona fide iris noisy-to-clean pairs."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="iris_bbdm_pad/configs/bbdm_iris_bonafide.yaml",
        help="Path to BBDM YAML config file.",
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default="0",
        help="GPU IDs (e.g. '0' or '0,1'). Use '-1' for CPU.",
    )
    parser.add_argument(
        "--resume_model",
        type=str,
        default=None,
        help="Path to model checkpoint to resume from.",
    )
    parser.add_argument(
        "--resume_optim",
        type=str,
        default=None,
        help="Path to optimizer/scheduler checkpoint to resume from.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--skip_validation",
        action="store_true",
        help="Skip prerequisite validation (not recommended).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.skip_validation:
        ok = validate_prerequisites(args.config)
        if not ok:
            logger.error("Prerequisites failed. Aborting.")
            sys.exit(1)

    exit_code = run_training(args)
    sys.exit(exit_code)
