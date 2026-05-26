"""Phase 1 — Create Noisy-to-Clean Bona Fide Training Pairs.

Reads preprocessed bona fide (Live) images and creates aligned pairs:
    A/ = corrupted (noisy) version
    B/ = clean original (copy)

ONLY bona fide images are used. Spoof images are NEVER included.
An assertion guards against accidental inclusion of Spoof paths.

Output structure:
    bonafide_pairs/{train,val}/A/   <- corrupted images
    bonafide_pairs/{train,val}/B/   <- clean originals

Usage:
    python iris_bbdm_pad/data/prepare_bonafide_pairs.py \
        --source_dir iris_bbdm_pad/data/preprocessed/ \
        --output_dir iris_bbdm_pad/data/bonafide_pairs/ \
        --workers 4
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import random
import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

from corruption import (
    apply_corruption_pipeline,
    apply_corruption_at_intensity,
    filename_to_seed,
    get_default_corruption_config,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Worker function
# ---------------------------------------------------------------------------

def process_pair(args: Tuple) -> Dict:
    """Create one A/B pair: corrupt the source, copy the clean version.

    Args:
        args: (src_str, out_a_str, out_b_str, corruption_config)

    Returns:
        Dict with success flag and MSE between corrupted/clean.
    """
    src_str, out_a_str, out_b_str, corruption_config = args
    src = Path(src_str)
    out_a = Path(out_a_str)
    out_b = Path(out_b_str)

    result = {
        "src": src_str,
        "out_a": out_a_str,
        "out_b": out_b_str,
        "success": False,
        "mse": None,
        "error": None,
    }

    try:
        # Safety check: NEVER include Spoof images
        assert "Spoof" not in str(src), f"ASSERTION FAILED: Spoof path detected: {src}"

        clean_arr = np.array(Image.open(src).convert("RGB"))

        seed = filename_to_seed(src.stem)
        corrupted_arr = apply_corruption_pipeline(
            clean_arr,
            seed=seed,
            noise_sigma_range=tuple(corruption_config["noise_sigma_range"]),
            blur_kernel_choices=tuple(corruption_config["blur_kernel_choices"]),
            blur_sigma_range=tuple(corruption_config["blur_sigma_range"]),
            downscale_range=tuple(corruption_config["downscale_range"]),
            target_size=corruption_config["target_size"],
        )

        # Resize clean to target_size to ensure A/B shapes match
        target_size = corruption_config["target_size"]
        if clean_arr.shape[0] != target_size or clean_arr.shape[1] != target_size:
            clean_pil = Image.fromarray(clean_arr).resize((target_size, target_size), Image.LANCZOS)
            clean_arr = np.array(clean_pil)

        out_a.parent.mkdir(parents=True, exist_ok=True)
        out_b.parent.mkdir(parents=True, exist_ok=True)

        Image.fromarray(corrupted_arr).save(out_a, format="PNG")
        Image.fromarray(clean_arr).save(out_b, format="PNG")

        mse = float(np.mean((corrupted_arr.astype(np.float32) - clean_arr.astype(np.float32)) ** 2))
        result["success"] = True
        result["mse"] = mse

    except Exception as e:
        result["error"] = str(e)

    return result


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def save_corruption_samples(results: List[Dict], vis_dir: Path, n: int = 5) -> None:
    """Save N rows × 3 cols: [Corrupted | Clean | Difference]."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    success = [r for r in results if r["success"]]
    samples = random.sample(success, min(n, len(success)))

    fig, axes = plt.subplots(len(samples), 3, figsize=(12, 4 * len(samples)))
    if len(samples) == 1:
        axes = axes[np.newaxis, :]

    axes[0, 0].set_title("Corrupted (A)", fontsize=11, fontweight="bold")
    axes[0, 1].set_title("Clean (B)", fontsize=11, fontweight="bold")
    axes[0, 2].set_title("Absolute Difference", fontsize=11, fontweight="bold")

    for row, res in enumerate(samples):
        try:
            corrupted = np.array(Image.open(res["out_a"]))
            clean = np.array(Image.open(res["out_b"]))
            diff = np.abs(corrupted.astype(np.float32) - clean.astype(np.float32))
            diff_norm = (diff / diff.max() * 255).astype(np.uint8) if diff.max() > 0 else diff.astype(np.uint8)

            axes[row, 0].imshow(corrupted)
            axes[row, 1].imshow(clean)
            axes[row, 2].imshow(diff_norm)
            axes[row, 0].set_ylabel(f"MSE={res['mse']:.1f}", fontsize=8, rotation=0, ha="right", va="center")
        except Exception as e:
            for col in range(3):
                axes[row, col].set_title(f"Error: {e}", fontsize=6)
        for col in range(3):
            axes[row, col].axis("off")

    plt.suptitle("Corruption Samples — Noisy-to-Clean Pairs", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = vis_dir / "corruption_samples.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"[Visualization] Saved corruption samples → {out_path}")


def save_corruption_intensity_spectrum(sample_src: Path, vis_dir: Path, target_size: int = 256) -> None:
    """Show ONE image corrupted at 5 intensity levels (very mild → very strong)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    intensities = [0.05, 0.25, 0.50, 0.75, 0.95]
    labels = ["Very Mild", "Mild", "Moderate", "Strong", "Very Strong"]
    cfg = get_default_corruption_config()

    try:
        clean = np.array(Image.open(sample_src).convert("RGB"))
        seed = filename_to_seed(sample_src.stem)

        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        for ax, intensity, label in zip(axes, intensities, labels):
            corrupted = apply_corruption_at_intensity(clean, intensity, seed, target_size=target_size)
            # Sample params to annotate
            import numpy as _np
            rng = _np.random.RandomState(seed)
            lo_n, hi_n = cfg["noise_sigma_range"]
            lo_b, hi_b = cfg["blur_sigma_range"]
            lo_d, hi_d = cfg["downscale_range"]
            noise_sigma = lo_n + intensity * (hi_n - lo_n)
            blur_sigma = lo_b + intensity * (hi_b - lo_b)
            scale = hi_d - intensity * (hi_d - lo_d)
            ax.imshow(corrupted)
            ax.set_title(f"{label}\nσ_noise={noise_sigma:.1f}\nσ_blur={blur_sigma:.2f}\nscale={scale:.2f}", fontsize=8)
            ax.axis("off")

        plt.suptitle("Corruption Intensity Spectrum — Same Image at 5 Levels", fontsize=13, fontweight="bold")
        plt.tight_layout()
        out_path = vis_dir / "corruption_intensity_spectrum.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"[Visualization] Saved intensity spectrum → {out_path}")
    except Exception as e:
        logger.warning(f"Intensity spectrum vis failed: {e}")


def save_corruption_per_attack_comparison(
    clean_samples: List[Path],
    vis_dir: Path,
    target_size: int = 256,
) -> None:
    """2×4 grid: top row = 4 clean bona fide, bottom row = their corruptions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    samples = random.sample(clean_samples, min(4, len(clean_samples)))
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    axes[0, 0].set_ylabel("Clean (B)", fontsize=10, rotation=0, ha="right", va="center")
    axes[1, 0].set_ylabel("Corrupted (A)", fontsize=10, rotation=0, ha="right", va="center")

    for col, src in enumerate(samples):
        try:
            clean = np.array(Image.open(src).convert("RGB"))
            seed = filename_to_seed(src.stem)
            corrupted = apply_corruption_pipeline(clean, seed=seed, target_size=target_size)
            if clean.shape[0] != target_size:
                clean = np.array(Image.fromarray(clean).resize((target_size, target_size), Image.LANCZOS))
            axes[0, col].imshow(clean)
            axes[0, col].set_title(src.name[:20], fontsize=7)
            axes[1, col].imshow(corrupted)
            mse = float(np.mean((corrupted.astype(float) - clean.astype(float)) ** 2))
            axes[1, col].set_title(f"MSE={mse:.1f}", fontsize=7)
        except Exception as e:
            axes[0, col].set_title(f"Error: {e}", fontsize=6)
        axes[0, col].axis("off")
        axes[1, col].axis("off")

    for col in range(len(samples), 4):
        axes[0, col].axis("off")
        axes[1, col].axis("off")

    plt.suptitle("Clean vs Corrupted — Bona Fide Iris Images", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = vis_dir / "corruption_per_attack_comparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"[Visualization] Saved clean/corrupted comparison → {out_path}")


def save_pair_histogram(results: List[Dict], vis_dir: Path) -> None:
    """Histogram of MSE values across all A/B pairs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mse_values = [r["mse"] for r in results if r["success"] and r["mse"] is not None]
    if not mse_values:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(mse_values, bins=50, edgecolor="black", alpha=0.75, color="steelblue")
    ax.set_xlabel("MSE (Corrupted vs Clean)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Distribution of Corruption Magnitude (N={len(mse_values)} pairs)", fontsize=13, fontweight="bold")
    ax.axvline(np.mean(mse_values), color="red", linestyle="--", label=f"Mean={np.mean(mse_values):.1f}")
    ax.axvline(np.median(mse_values), color="orange", linestyle="--", label=f"Median={np.median(mse_values):.1f}")
    ax.legend(fontsize=10)

    out_path = vis_dir / "pair_histogram.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"[Visualization] Saved MSE histogram → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def prepare_pairs(
    source_dir: Path,
    output_dir: Path,
    workers: int,
    vis_dir: Path,
) -> None:
    """Create noisy-to-clean training pairs from preprocessed bona fide images.

    Args:
        source_dir: Root of preprocessed images (contains train/val/test/{Live,Spoof}).
        output_dir: Output root for bonafide_pairs/{train,val}/{A,B}.
        workers: Parallel workers.
        vis_dir: Directory for visualization outputs.
    """
    corruption_config = get_default_corruption_config()
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
            tasks.append((str(src), str(out_a), str(out_b), corruption_config))

        logger.info(f"[{split}] Creating {len(tasks)} A/B pairs with {workers} workers")
        split_results = []

        with multiprocessing.Pool(workers) as pool:
            for res in tqdm(pool.imap_unordered(process_pair, tasks), total=len(tasks), desc=f"Pairs [{split}]"):
                split_results.append(res)
                all_results.append(res)

        success = sum(1 for r in split_results if r["success"])
        logger.info(f"[{split}] {success}/{len(split_results)} pairs created successfully")

        # Per-split corruption samples visualization (save to output dir)
        out_dir_split = output_dir / split
        out_dir_split.mkdir(parents=True, exist_ok=True)

    # --- Global visualizations ---
    if all_results:
        # corruption_samples.png → in bonafide_pairs/ root (as per spec)
        save_corruption_samples(all_results, output_dir)

        # phase1_visualizations/
        save_pair_histogram(all_results, vis_dir)

        # Pick a clean source sample for spectrum + comparison
        successful = [r for r in all_results if r["success"]]
        if successful:
            sample_b_path = Path(successful[0]["out_b"])
            save_corruption_intensity_spectrum(sample_b_path, vis_dir, target_size=corruption_config["target_size"])

            clean_b_paths = [Path(r["out_b"]) for r in successful[:20]]
            save_corruption_per_attack_comparison(clean_b_paths, vis_dir, target_size=corruption_config["target_size"])

    # Save dataset_config.json
    success_count = sum(1 for r in all_results if r["success"])
    mse_values = [r["mse"] for r in all_results if r["success"] and r["mse"] is not None]
    config_data = {
        "corruption_config": corruption_config,
        "pair_counts": {
            "train": sum(1 for r in all_results if r["success"] and "/train/" in r["out_a"]),
            "val": sum(1 for r in all_results if r["success"] and "/val/" in r["out_a"]),
            "total": success_count,
        },
        "mse_stats": {
            "mean": float(np.mean(mse_values)) if mse_values else 0.0,
            "std": float(np.std(mse_values)) if mse_values else 0.0,
            "min": float(np.min(mse_values)) if mse_values else 0.0,
            "max": float(np.max(mse_values)) if mse_values else 0.0,
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    config_path = output_dir / "dataset_config.json"
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)
    logger.info(f"[Config] Saved dataset_config.json → {config_path}")

    # Summary
    logger.info("=" * 60)
    logger.info("PAIR CREATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Total pairs created : {success_count}")
    logger.info(f"  Failed              : {len(all_results) - success_count}")
    if mse_values:
        logger.info(f"  MSE mean ± std      : {np.mean(mse_values):.2f} ± {np.std(mse_values):.2f}")
    logger.info(f"  Output directory    : {output_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create noisy-to-clean bona fide training pairs.")
    p.add_argument("--source_dir", type=Path, default=Path("iris_bbdm_pad/data/preprocessed/"), help="Preprocessed images root.")
    p.add_argument("--output_dir", type=Path, default=Path("iris_bbdm_pad/data/bonafide_pairs/"), help="Output directory for pairs.")
    p.add_argument("--workers", type=int, default=4, help="Parallel worker processes.")
    p.add_argument("--vis_dir", type=Path, default=Path("iris_bbdm_pad/results/phase1_visualizations/"), help="Visualization output directory.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    np.random.seed(42)
    random.seed(42)

    prepare_pairs(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        workers=args.workers,
        vis_dir=args.vis_dir,
    )
