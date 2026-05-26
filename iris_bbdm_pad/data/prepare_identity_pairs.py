"""Phase 2 — Create Identity-to-Identity Bona Fide Training Pairs.

Reads preprocessed bona fide (Live) images and creates aligned pairs:
    A/ = clean original copy
    B/ = clean original copy (identical to A)

ONLY bona fide images are used. Spoof images are NEVER included.
An assertion guards against accidental inclusion of Spoof paths.

This is for training BBDM on identity mapping (A→A) instead of noisy-to-clean (A→A_noisy).

Output structure:
    identity_pairs/{train,val}/A/   <- clean originals
    identity_pairs/{train,val}/B/   <- clean originals (same as A)

Usage:
    python iris_bbdm_pad/data/prepare_identity_pairs.py \
        --source_dir iris_bbdm_pad/data/preprocessed/ \
        --output_dir iris_bbdm_pad/data/identity_pairs/ \
        --workers 4
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Worker function
# ---------------------------------------------------------------------------

def process_identity_pair(args: Tuple) -> Dict:
    """Create one A/B pair: copy the source to both A and B directories.

    Args:
        args: (src_str, out_a_str, out_b_str)

    Returns:
        Dict with success flag.
    """
    src_str, out_a_str, out_b_str = args
    src = Path(src_str)
    out_a = Path(out_a_str)
    out_b = Path(out_b_str)

    result = {
        "src": src_str,
        "out_a": out_a_str,
        "out_b": out_b_str,
        "success": False,
        "error": None,
    }

    try:
        # Safety check: NEVER include Spoof images
        assert "Spoof" not in str(src), f"ASSERTION FAILED: Spoof path detected: {src}"

        # Load clean image
        clean_arr = np.array(Image.open(src).convert("RGB"))

        # Ensure target_size (256×256)
        target_size = 256
        if clean_arr.shape[0] != target_size or clean_arr.shape[1] != target_size:
            clean_pil = Image.fromarray(clean_arr).resize((target_size, target_size), Image.LANCZOS)
            clean_arr = np.array(clean_pil)

        # Create directories
        out_a.parent.mkdir(parents=True, exist_ok=True)
        out_b.parent.mkdir(parents=True, exist_ok=True)

        # Copy clean image to both A and B (identity mapping)
        Image.fromarray(clean_arr).save(out_a, format="PNG")
        Image.fromarray(clean_arr).save(out_b, format="PNG")

        result["success"] = True

    except Exception as e:
        result["error"] = str(e)

    return result


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def save_identity_samples(results: List[Dict], vis_dir: Path, n: int = 5) -> None:
    """Save N rows × 2 cols: [Original A | Original B] (should look identical)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    success = [r for r in results if r["success"]]
    samples = success[:n]

    fig, axes = plt.subplots(len(samples), 2, figsize=(12, 4 * len(samples)))
    if len(samples) == 1:
        axes = axes[np.newaxis, :]

    axes[0, 0].set_title("Original (A)", fontsize=11, fontweight="bold")
    axes[0, 1].set_title("Copy (B)", fontsize=11, fontweight="bold")

    for row, res in enumerate(samples):
        try:
            img_a = np.array(Image.open(res["out_a"]))
            img_b = np.array(Image.open(res["out_b"]))

            axes[row, 0].imshow(img_a)
            axes[row, 1].imshow(img_b)
            axes[row, 0].set_ylabel(f"MSE=0.0", fontsize=8, rotation=0, ha="right", va="center")
        except Exception as e:
            for col in range(2):
                axes[row, col].set_title(f"Error: {e}", fontsize=6)
        for col in range(2):
            axes[row, col].axis("off")

    plt.suptitle("Identity Samples — Identical Pairs (A=B)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = vis_dir / "identity_samples.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"[Visualization] Saved identity samples → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def prepare_identity_pairs(
    source_dir: Path,
    output_dir: Path,
    workers: int,
    vis_dir: Path,
) -> None:
    """Create identity training pairs from preprocessed bona fide images.

    Args:
        source_dir: Root of preprocessed images (contains train/val/test/{Live,Spoof}).
        output_dir: Output root for identity_pairs/{train,val}/{A,B}.
        workers: Parallel workers.
        vis_dir: Directory for visualization outputs.
    """
    vis_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for split in ("train", "val"):
        live_dir = source_dir / split / "Live"
        if not live_dir.exists():
            logger.warning(f"Live directory not found: {live_dir} — skipping {split}")
            continue

        # Collect all PNG files (preprocessed output is always .png)
        srcs = sorted(live_dir.rglob("*.png"))
        if not srcs:
            logger.warning(f"No PNG files found in {live_dir}")
            continue

        logger.info(f"[{split}] Found {len(srcs)} bona fide images")

        # Build task list
        tasks = []
        for src in srcs:
            # Safety check at task-build time as well
            assert "Spoof" not in str(src), f"ASSERTION FAILED: Spoof path in Live dir: {src}"
            filename = src.name  # preserve original filename for matched A/B
            out_a = output_dir / split / "A" / filename
            out_b = output_dir / split / "B" / filename
            tasks.append((str(src), str(out_a), str(out_b)))

        logger.info(f"[{split}] Creating {len(tasks)} identity A/B pairs with {workers} workers")
        split_results = []

        with multiprocessing.Pool(workers) as pool:
            for res in tqdm(pool.imap_unordered(process_identity_pair, tasks), total=len(tasks), desc=f"Pairs [{split}]"):
                split_results.append(res)
                all_results.append(res)

        success = sum(1 for r in split_results if r["success"])
        logger.info(f"[{split}] {success}/{len(split_results)} pairs created successfully")

        # Per-split output dir
        out_dir_split = output_dir / split
        out_dir_split.mkdir(parents=True, exist_ok=True)

    # --- Global visualizations ---
    if all_results:
        save_identity_samples(all_results, vis_dir, n=5)

    # Save dataset_config.json (no corruption config for identity mapping)
    success_count = sum(1 for r in all_results if r["success"])
    config_data = {
        "pair_type": "identity",
        "description": "Identity mapping (A=B) — clean original to itself",
        "pair_counts": {
            "train": sum(1 for r in all_results if r["success"] and "/train/" in r["out_a"]),
            "val": sum(1 for r in all_results if r["success"] and "/val/" in r["out_a"]),
            "total": success_count,
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    config_path = output_dir / "dataset_config.json"
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)
    logger.info(f"[Config] Saved dataset_config.json → {config_path}")

    # Summary
    logger.info("=" * 60)
    logger.info("IDENTITY PAIR CREATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Total pairs created : {success_count}")
    logger.info(f"  Failed              : {len(all_results) - success_count}")
    logger.info(f"  Pair type           : Identity (A=B)")
    logger.info(f"  Output directory    : {output_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create identity (A=B) bona fide training pairs.")
    p.add_argument("--source_dir", type=Path, default=Path("iris_bbdm_pad/data/preprocessed/"), help="Preprocessed images root.")
    p.add_argument("--output_dir", type=Path, default=Path("iris_bbdm_pad/data/identity_pairs/"), help="Output directory for pairs.")
    p.add_argument("--workers", type=int, default=4, help="Parallel worker processes.")
    p.add_argument("--vis_dir", type=Path, default=Path("iris_bbdm_pad/results/phase2_visualizations/"), help="Visualization output directory.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    np.random.seed(42)

    prepare_identity_pairs(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        workers=args.workers,
        vis_dir=args.vis_dir,
    )
