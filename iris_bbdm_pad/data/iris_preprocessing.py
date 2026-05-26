"""Phase 1 — Iris Image Preprocessing.

Segments, normalizes, and standardizes raw iris images to 256×256 RGB PNG.
Mirrors input directory structure exactly under the output directory.
Supports resume via checkpoint JSON.

Usage:
    python iris_bbdm_pad/data/iris_preprocessing.py \
        --input_dir Images/ --output_dir iris_bbdm_pad/data/preprocessed/ \
        --image_size 256 --workers 4 --max_images 50
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import multiprocessing
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tiff", ".tif"}
MIN_IMAGE_SIZE = 50  # pixels — skip images smaller than this


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

_HOUGH_MAX_DIM = 512  # Hough runtime scales with image area; cap at this size


def segment_iris(image: np.ndarray) -> Tuple[np.ndarray, bool, Optional[int]]:
    """Segment the iris region from an image.

    Args:
        image: BGR image array (H, W, 3).

    Returns:
        Tuple of (cropped_rgb_array, hough_success, iris_radius).
        If Hough fails, falls back to center-crop 70% with hough_success=False.
    """
    h, w = image.shape[:2]

    # SPEED FIX: Downscale large images before Hough detection.
    # HoughCircles runtime scales with image area; a 1300×1000 image is ~10×
    # slower than a 400×300 one.  Detect on a 512-px thumbnail, then scale
    # the circle coordinates back to the original image.
    scale = 1.0
    if max(h, w) > _HOUGH_MAX_DIM:
        scale = _HOUGH_MAX_DIM / max(h, w)
        small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        small = image

    sh, sw = small.shape[:2]

    # Step 1: Grayscale (on small image)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    # Step 2: GaussianBlur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 3: CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)

    # Step 4: Hough Circle Detection on small image
    circles = cv2.HoughCircles(
        enhanced,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=sh // 4,
        param1=100,
        param2=30,
        minRadius=sh // 8,
        maxRadius=sh // 2,
    )

    if circles is not None:
        # Scale circle coordinates back to original image size
        best = circles[0, 0]
        cx = int(best[0] / scale)
        cy = int(best[1] / scale)
        r = int(best[2] / scale)
        iris_radius = r

        # Bounding box + 10% margin on original image
        margin = int(r * 0.10)
        x1 = max(0, cx - r - margin)
        y1 = max(0, cy - r - margin)
        x2 = min(w, cx + r + margin)
        y2 = min(h, cy + r + margin)

        cropped = image[y1:y2, x1:x2]
        return cropped, True, iris_radius
    else:
        # Center-crop fallback (70% of shorter dimension)
        crop_size = int(min(h, w) * 0.70)
        cx, cy = w // 2, h // 2
        half = crop_size // 2
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(w, cx + half)
        y2 = min(h, cy + half)
        cropped = image[y1:y2, x1:x2]
        return cropped, False, None


def get_intermediate_steps(image: np.ndarray) -> Dict[str, np.ndarray]:
    """Return intermediate pipeline stages for visualization.

    Args:
        image: BGR image array.

    Returns:
        Dict of named intermediate arrays (all BGR for display).
    """
    h, w = image.shape[:2]

    # Mirror the same downscale used in segment_iris for consistency
    scale = 1.0
    if max(h, w) > _HOUGH_MAX_DIM:
        scale = _HOUGH_MAX_DIM / max(h, w)
        small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        small = image
    sh, sw = small.shape[:2]

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)

    circles = cv2.HoughCircles(
        enhanced,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=sh // 4,
        param1=100,
        param2=30,
        minRadius=sh // 8,
        maxRadius=sh // 2,
    )

    # Draw circles on a copy for visualization
    hough_vis = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    if circles is not None:
        for c in np.round(circles[0, :]).astype(int):
            cv2.circle(hough_vis, (c[0], c[1]), c[2], (0, 255, 0), 2)
            cv2.circle(hough_vis, (c[0], c[1]), 2, (0, 0, 255), 3)

    return {
        "gray": cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
        "clahe": cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR),
        "hough": hough_vis,
    }


# ---------------------------------------------------------------------------
# Label extraction
# ---------------------------------------------------------------------------

def extract_label_info(image_path: Path, input_dir: Path) -> Tuple[str, str, str]:
    """Extract split, label, and attack_type from file path.

    Args:
        image_path: Absolute path to the raw image.
        input_dir: Root input directory.

    Returns:
        (split, label, attack_type) e.g. ('train', 'Live', 'Live') or
        ('train', 'Spoof', 'Printed').
    """
    rel = image_path.relative_to(input_dir)
    parts = rel.parts  # e.g. ('train', 'Live', 'file.bmp') or ('train', 'Spoof', 'Printed', 'file.bmp')

    split = parts[0]  # train / val / test

    if "Live" in parts:
        return split, "Live", "Live"

    if "Spoof" in parts:
        spoof_idx = list(parts).index("Spoof")
        if spoof_idx + 1 < len(parts) - 1:
            attack_type = parts[spoof_idx + 1]
        else:
            attack_type = "Unknown"
        return split, "Spoof", attack_type

    return split, "Unknown", "Unknown"


# ---------------------------------------------------------------------------
# Per-image worker
# ---------------------------------------------------------------------------

def process_single_image(args: Tuple) -> Dict:
    """Process one image: load → segment → resize → save PNG.

    Args:
        args: (src_path_str, out_path_str, image_size, input_dir_str)

    Returns:
        Dict with result info.
    """
    src_str, out_str, image_size, input_dir_str = args
    src = Path(src_str)
    out = Path(out_str)
    input_dir = Path(input_dir_str)

    result = {
        "original_path": src_str,
        "preprocessed_path": out_str,
        "success": False,
        "seg_success": False,
        "iris_radius": None,
        "error": None,
    }

    # Skip if output already exists — avoids redundant re-processing when the
    # checkpoint is stale/missing (e.g. after a SIGTERM).
    if out.exists():
        result["success"] = True
        result["error"] = "skipped_existing"
        return result

    try:
        # Load image
        img_bgr = cv2.imread(src_str, cv2.IMREAD_COLOR)
        if img_bgr is None:
            # Try PIL fallback for TIFF and unusual formats
            pil_img = Image.open(src_str).convert("RGB")
            img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        if img_bgr is None:
            result["error"] = "Cannot load image"
            return result

        h, w = img_bgr.shape[:2]
        if h < MIN_IMAGE_SIZE or w < MIN_IMAGE_SIZE:
            result["error"] = f"Image too small: {w}×{h}"
            return result

        # Segment
        cropped, seg_success, iris_radius = segment_iris(img_bgr)

        if cropped.size == 0:
            result["error"] = "Segmentation produced empty crop"
            return result

        # Resize to target — INTER_AREA is faster for downscaling, INTER_LANCZOS4 for upscaling
        ch, cw = cropped.shape[:2]
        interp = cv2.INTER_AREA if (ch > image_size or cw > image_size) else cv2.INTER_LANCZOS4
        resized = cv2.resize(cropped, (image_size, image_size), interpolation=interp)

        # Ensure 3-channel RGB
        if len(resized.shape) == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
        elif resized.shape[2] == 4:
            resized = resized[:, :, :3]

        # Convert BGR → RGB and save as PNG
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(out, format="PNG")

        result["success"] = True
        result["seg_success"] = seg_success
        result["iris_radius"] = iris_radius

    except Exception as e:
        result["error"] = str(e)

    return result


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def save_preprocessing_pipeline(
    samples: List[Path],
    input_dir: Path,
    output_dir: Path,
    image_size: int,
    vis_dir: Path,
) -> None:
    """Save 5-row × 7-col pipeline visualization.

    Columns: Original | Gray | CLAHE | Hough circles | Cropped | Resized | Final RGB
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(5, len(samples))
    if n == 0:
        return

    fig, axes = plt.subplots(n, 7, figsize=(28, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Original", "Grayscale", "CLAHE", "Hough Circles", "Cropped", "Resized 256×256", "Final RGB"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=10, fontweight="bold")

    for row, src in enumerate(samples[:n]):
        try:
            img_bgr = cv2.imread(str(src), cv2.IMREAD_COLOR)
            if img_bgr is None:
                pil_img = Image.open(src).convert("RGB")
                img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            intermediates = get_intermediate_steps(img_bgr)
            cropped, seg_success, _ = segment_iris(img_bgr)
            resized = cv2.resize(cropped, (image_size, image_size), interpolation=cv2.INTER_LANCZOS4)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

            steps = [
                cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                cv2.cvtColor(intermediates["gray"], cv2.COLOR_BGR2RGB),
                cv2.cvtColor(intermediates["clahe"], cv2.COLOR_BGR2RGB),
                cv2.cvtColor(intermediates["hough"], cv2.COLOR_BGR2RGB),
                cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB) if len(cropped.shape) == 3 else np.stack([cropped] * 3, -1),
                cv2.cvtColor(resized, cv2.COLOR_BGR2RGB),
                rgb,
            ]
            for col, step in enumerate(steps):
                axes[row, col].imshow(step)
                axes[row, col].axis("off")
            method = "Hough" if seg_success else "center-crop"
            axes[row, 0].set_ylabel(f"{src.name[:20]}\n({method})", fontsize=7, rotation=0, ha="right", va="center")
        except Exception as e:
            logger.warning(f"Pipeline vis failed for {src}: {e}")

    plt.suptitle("Iris Preprocessing Pipeline", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path = vis_dir / "preprocessing_pipeline.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"[Visualization] Saved preprocessing pipeline → {out_path}")


def save_segmentation_results(
    preprocessed_files: List[Dict],
    vis_dir: Path,
) -> None:
    """Save 4×5 grid of 20 random preprocessed images."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    samples = random.sample(preprocessed_files, min(20, len(preprocessed_files)))
    n_rows, n_cols = 4, 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 16))

    for idx, (ax, info) in enumerate(zip(axes.flat, samples)):
        try:
            img = np.array(Image.open(info["preprocessed_path"]))
            ax.imshow(img)
            method = "Hough" if info["seg_success"] else "center-crop"
            radius = info["iris_radius"] if info["iris_radius"] else "N/A"
            ax.set_title(f"{method}\nr={radius}", fontsize=7)
        except Exception:
            ax.set_title("Error", fontsize=7)
        ax.axis("off")

    for idx in range(len(samples), n_rows * n_cols):
        axes.flat[idx].axis("off")

    plt.suptitle("Segmentation Results — 20 Random Samples", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = vis_dir / "segmentation_results.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"[Visualization] Saved segmentation results → {out_path}")


def save_format_comparison(
    raw_samples_by_ext: Dict[str, Path],
    preprocessed_meta: List[Dict],
    vis_dir: Path,
) -> None:
    """Save format comparison: one image per original format, before/after."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Build a lookup from original_path to preprocessed_path
    lookup = {d["original_path"]: d for d in preprocessed_meta if d["success"]}

    rows = []
    for ext, src in raw_samples_by_ext.items():
        if str(src) in lookup:
            rows.append((ext, src, lookup[str(src)]["preprocessed_path"]))

    if not rows:
        return

    fig, axes = plt.subplots(len(rows), 2, figsize=(8, 4 * len(rows)))
    if len(rows) == 1:
        axes = axes[np.newaxis, :]

    axes[0, 0].set_title("Original (raw)", fontsize=11, fontweight="bold")
    axes[0, 1].set_title("Preprocessed (256×256 RGB)", fontsize=11, fontweight="bold")

    for row, (ext, src, pre_path) in enumerate(rows):
        try:
            raw_img = np.array(Image.open(src).convert("RGB"))
            pre_img = np.array(Image.open(pre_path))
            axes[row, 0].imshow(raw_img)
            axes[row, 0].set_ylabel(f"{ext.upper()} format", fontsize=9, rotation=0, ha="right", va="center")
            axes[row, 1].imshow(pre_img)
        except Exception as e:
            axes[row, 0].set_ylabel(f"{ext} — Error: {e}", fontsize=7)
        axes[row, 0].axis("off")
        axes[row, 1].axis("off")

    plt.suptitle("Format Comparison: Before vs After Preprocessing", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = vis_dir / "format_comparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"[Visualization] Saved format comparison → {out_path}")


# ---------------------------------------------------------------------------
# Main preprocessing pipeline
# ---------------------------------------------------------------------------

def collect_images(input_dir: Path, max_images: int = 0) -> List[Path]:
    """Collect all supported image files from input_dir.

    Uses a single rglob('*') pass and filters by extension — much faster than
    one rglob per extension on large trees (avoids 12 full directory traversals).
    """
    all_images = [
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    all_images.sort()
    if max_images > 0:
        all_images = all_images[:max_images]
    return all_images


def build_output_path(src: Path, input_dir: Path, output_dir: Path) -> Path:
    """Mirror the input structure under output_dir, changing extension to .png."""
    rel = src.relative_to(input_dir)
    return output_dir / rel.parent / (src.stem + ".png")


def load_checkpoint(checkpoint_path: Path) -> Dict:
    """Load checkpoint JSON if it exists."""
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            return json.load(f)
    return {"completed_files": [], "failed_files": [], "total_processed": 0}


def save_checkpoint(checkpoint_path: Path, data: Dict) -> None:
    """Save checkpoint JSON atomically."""
    tmp = checkpoint_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.rename(checkpoint_path)


def preprocess_images(
    input_dir: Path,
    output_dir: Path,
    image_size: int,
    workers: int,
    max_images: int,
    resume: bool,
    checkpoint_path: Path,
    vis_dir: Path,
) -> List[Dict]:
    """Main preprocessing loop.

    Args:
        input_dir: Root directory containing raw images.
        output_dir: Root directory for preprocessed PNGs.
        image_size: Target image size (pixels).
        workers: Number of parallel worker processes.
        max_images: Max images to process (0 = all).
        resume: If True, skip already-processed files.
        checkpoint_path: Path to checkpoint JSON.
        vis_dir: Directory for visualization outputs.

    Returns:
        List of result dicts for all processed images.
    """
    logger.info(f"[Step 1/7] Collecting images from {input_dir}")
    all_images = collect_images(input_dir, max_images)
    logger.info(f"Found {len(all_images)} images")

    checkpoint = load_checkpoint(checkpoint_path) if resume else {}
    completed_set = set(checkpoint.get("completed_files", []))

    if resume and completed_set:
        remaining = [p for p in all_images if str(p) not in completed_set]
        logger.info(f"[Resume] Skipping {len(completed_set)} already-processed files, {len(remaining)} remaining")
    else:
        remaining = all_images

    if not remaining:
        logger.info("All images already processed.")
        return []

    # Build task list
    tasks = []
    for src in remaining:
        out = build_output_path(src, input_dir, output_dir)
        tasks.append((str(src), str(out), image_size, str(input_dir)))

    logger.info(f"[Step 2/7] Processing {len(tasks)} images with {workers} workers")

    results = []
    all_results = []

    # Collect raw samples by extension for format comparison visualization
    raw_by_ext: Dict[str, Path] = {}

    with multiprocessing.Pool(workers) as pool:
        for res in tqdm(pool.imap_unordered(process_single_image, tasks), total=len(tasks), desc="Preprocessing"):
            all_results.append(res)
            src_path = Path(res["original_path"])
            ext = src_path.suffix.lower()
            if ext not in raw_by_ext and res["success"]:
                raw_by_ext[ext] = src_path

            if res["success"]:
                completed_set.add(res["original_path"])
                results.append(res)

    # Enrich results with label info
    enriched = []
    for res in all_results:
        src = Path(res["original_path"])
        out = Path(res["preprocessed_path"])
        split, label, attack_type = extract_label_info(src, input_dir)
        res["label"] = label
        res["attack_type"] = attack_type
        res["split"] = split
        enriched.append(res)

    # Save checkpoint
    checkpoint_data = {
        "completed_files": list(completed_set),
        "failed_files": [r["original_path"] for r in all_results if not r["success"]],
        "total_processed": len(completed_set),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(checkpoint_path, checkpoint_data)
    logger.info(f"[Step 3/7] Checkpoint saved → {checkpoint_path}")

    # Save metadata CSV
    csv_path = output_dir / "preprocessed_metadata.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = ["original_path", "preprocessed_path", "label", "attack_type", "split", "seg_success", "iris_radius", "error"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for res in enriched:
            writer.writerow({k: res.get(k, "") for k in fieldnames})
    logger.info(f"[Step 4/7] Metadata CSV saved → {csv_path}")

    # ------- Visualizations -------
    logger.info("[Step 5/7] Generating visualizations")
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Pipeline visualization — pick 5 raw samples (prefer variety)
    successful_srcs = [Path(r["original_path"]) for r in all_results if r["success"]]
    if successful_srcs:
        pipeline_samples = successful_srcs[:5]
        save_preprocessing_pipeline(pipeline_samples, input_dir, output_dir, image_size, vis_dir)
        logger.info(f"[Step 5/7] Pipeline visualization saved → {vis_dir}/preprocessing_pipeline.png")

    # Segmentation results visualization
    successful_results = [r for r in enriched if r["success"]]
    if successful_results:
        save_segmentation_results(successful_results, vis_dir)
        logger.info(f"[Step 5/7] Segmentation results saved → {vis_dir}/segmentation_results.png")

    # Format comparison
    if raw_by_ext:
        save_format_comparison(raw_by_ext, enriched, vis_dir)
        logger.info(f"[Step 5/7] Format comparison saved → {vis_dir}/format_comparison.png")

    # ------- Summary -------
    logger.info("[Step 6/7] Computing statistics")
    success_count = sum(1 for r in enriched if r["success"])
    fail_count = len(enriched) - success_count
    hough_count = sum(1 for r in enriched if r.get("seg_success"))
    fallback_count = sum(1 for r in enriched if r["success"] and not r.get("seg_success"))

    logger.info("=" * 60)
    logger.info("PREPROCESSING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Total processed : {len(enriched)}")
    logger.info(f"  Succeeded       : {success_count}")
    logger.info(f"  Failed          : {fail_count}")
    logger.info(f"  Hough success   : {hough_count} ({100*hough_count/max(success_count,1):.1f}%)")
    logger.info(f"  Center-crop     : {fallback_count} ({100*fallback_count/max(success_count,1):.1f}%)")

    # Per-split, per-class breakdown
    from collections import Counter
    split_class = Counter()
    for r in enriched:
        if r["success"]:
            split_class[(r["split"], r["label"])] += 1

    logger.info("\n  Per-split, per-class counts:")
    for (split, label), count in sorted(split_class.items()):
        logger.info(f"    {split:10s} {label:10s} : {count}")

    logger.info(f"\n[Step 7/7] All done. Metadata CSV at {csv_path}")
    return enriched


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess raw iris images for BBDM PAD.")
    p.add_argument("--input_dir", type=Path, default=Path("Images/"), help="Root directory of raw images.")
    p.add_argument("--output_dir", type=Path, default=Path("iris_bbdm_pad/data/preprocessed/"), help="Output directory for preprocessed PNGs.")
    p.add_argument("--image_size", type=int, default=256, help="Output image size (pixels).")
    p.add_argument("--workers", type=int, default=4, help="Number of parallel worker processes.")
    p.add_argument("--max_images", type=int, default=0, help="Max images to process (0=all, N=first N for dry run).")
    p.add_argument("--resume", action="store_true", help="Resume from checkpoint, skipping already-processed files.")
    p.add_argument("--vis_dir", type=Path, default=Path("iris_bbdm_pad/results/phase1_visualizations/"), help="Directory for visualization outputs.")
    p.add_argument("--checkpoint_dir", type=Path, default=Path("iris_bbdm_pad/checkpoints/"), help="Directory for checkpoint JSON.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    np.random.seed(42)
    random.seed(42)

    checkpoint_path = args.checkpoint_dir / "preprocessed_checkpoint.json"

    results = preprocess_images(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        image_size=args.image_size,
        workers=args.workers,
        max_images=args.max_images,
        resume=args.resume,
        checkpoint_path=checkpoint_path,
        vis_dir=args.vis_dir,
    )

    success = sum(1 for r in results if r.get("success"))
    logger.info(f"Finished: {success}/{len(results)} images preprocessed successfully.")
