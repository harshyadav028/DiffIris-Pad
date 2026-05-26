"""Phase 1 — Prepare Evaluation Datasets (Test & Val).

Copies preprocessed images (both Live and Spoof) into flat evaluation directories
with a labels.csv for BBDM PAD scoring.

Output structure:
    evaluation_sets/{test,val}/images/   <- flat directory of PNGs
    evaluation_sets/{test,val}/labels.csv <- filename,label,attack_type

Filename collision handling: prefix with attack_type.
    Live/img001.png        → Live_img001.png
    Spoof/Printed/img.png  → Printed_img.png
    Spoof/CL/img.png       → CL_img.png

CSV labels:
    label = "bonafide" for Live, "attack" for Spoof
    attack_type = "Live" for Live, else the attack subdirectory name

Usage:
    python iris_bbdm_pad/data/prepare_test_dataset.py \
        --source_dir iris_bbdm_pad/data/preprocessed/ \
        --output_dir iris_bbdm_pad/data/evaluation_sets/
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
from collections import Counter, defaultdict
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
# Helpers
# ---------------------------------------------------------------------------

def extract_label_info(path: Path, split_dir: Path) -> Tuple[str, str]:
    """Extract label and attack_type from a preprocessed image path.

    Args:
        path: Path to the preprocessed image.
        split_dir: Root split directory (e.g. preprocessed/test/).

    Returns:
        (label, attack_type): label is "bonafide" or "attack".
    """
    rel = path.relative_to(split_dir)
    parts = rel.parts

    if "Live" in parts:
        return "bonafide", "Live"

    if "Spoof" in parts:
        spoof_idx = list(parts).index("Spoof")
        if spoof_idx + 1 < len(parts) - 1:
            attack_type = parts[spoof_idx + 1]
        else:
            attack_type = "Unknown"
        return "attack", attack_type

    return "unknown", "Unknown"


def build_prefixed_filename(path: Path, split_dir: Path) -> str:
    """Generate a collision-free filename by prefixing with the attack type.

    Args:
        path: Path to the preprocessed image.
        split_dir: Root split directory.

    Returns:
        Prefixed filename string, e.g. "Live_img001.png" or "Printed_img001.png".
    """
    _, attack_type = extract_label_info(path, split_dir)
    # Replace spaces in attack type for safe filenames
    safe_type = attack_type.replace(" ", "_")
    return f"{safe_type}_{path.name}"


def copy_images_for_split(
    source_split_dir: Path,
    output_split_dir: Path,
) -> List[Dict]:
    """Copy all images for one split into a flat images/ directory.

    Args:
        source_split_dir: e.g. preprocessed/test/
        output_split_dir: e.g. evaluation_sets/test/

    Returns:
        List of dicts with filename, label, attack_type, source_path.
    """
    images_out = output_split_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    # Collect all PNGs
    all_pngs = sorted(source_split_dir.rglob("*.png"))
    if not all_pngs:
        logger.warning(f"No PNGs found in {source_split_dir}")
        return []

    records = []
    filename_counts: Counter = Counter()

    for src in tqdm(all_pngs, desc=f"Copying [{source_split_dir.name}]"):
        label, attack_type = extract_label_info(src, source_split_dir)
        safe_type = attack_type.replace(" ", "_")
        base_name = f"{safe_type}_{src.name}"

        # Handle duplicate filenames by appending counter
        if filename_counts[base_name] > 0:
            stem = Path(base_name).stem
            suffix = Path(base_name).suffix
            base_name = f"{stem}_{filename_counts[base_name]}{suffix}"
        filename_counts[base_name] += 1

        dst = images_out / base_name

        try:
            import shutil
            shutil.copy2(src, dst)
            records.append({
                "filename": base_name,
                "label": label,
                "attack_type": attack_type,
                "source_path": str(src),
            })
        except Exception as e:
            logger.warning(f"Failed to copy {src} → {dst}: {e}")

    return records


def save_labels_csv(records: List[Dict], output_split_dir: Path) -> None:
    """Save labels.csv for a split.

    Args:
        records: List of dicts with filename, label, attack_type.
        output_split_dir: Split output directory.
    """
    csv_path = output_split_dir / "labels.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "label", "attack_type"])
        writer.writeheader()
        for rec in records:
            writer.writerow({k: rec[k] for k in ["filename", "label", "attack_type"]})
    logger.info(f"Saved labels.csv → {csv_path} ({len(records)} entries)")


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def save_test_set_distribution(
    test_records: List[Dict],
    val_records: List[Dict],
    vis_dir: Path,
) -> None:
    """Horizontal bar chart: sample count per attack type for test and val."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    def count_by_type(records: List[Dict]) -> Dict[str, int]:
        counter: Counter = Counter()
        for r in records:
            counter[r["attack_type"]] += 1
        return dict(sorted(counter.items()))

    test_counts = count_by_type(test_records)
    val_counts = count_by_type(val_records)

    all_types = sorted(set(list(test_counts.keys()) + list(val_counts.keys())))

    test_vals = [test_counts.get(t, 0) for t in all_types]
    val_vals = [val_counts.get(t, 0) for t in all_types]

    x = np.arange(len(all_types))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, max(6, len(all_types) * 0.7)))
    bars1 = ax.barh(x + width / 2, test_vals, width, label="Test", color="steelblue", alpha=0.8)
    bars2 = ax.barh(x - width / 2, val_vals, width, label="Val", color="coral", alpha=0.8)

    ax.set_yticks(x)
    ax.set_yticklabels(all_types, fontsize=10)
    ax.set_xlabel("Sample Count", fontsize=11)
    ax.set_title("Evaluation Set Distribution by Attack Type", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.bar_label(bars1, fontsize=8, padding=3)
    ax.bar_label(bars2, fontsize=8, padding=3)

    plt.tight_layout()
    vis_dir.mkdir(parents=True, exist_ok=True)
    out_path = vis_dir / "test_set_distribution.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"[Visualization] Saved distribution chart → {out_path}")


def save_attack_type_samples(
    test_records: List[Dict],
    images_dir: Path,
    vis_dir: Path,
) -> None:
    """3×3 grid: one random sample per attack type + bona fide."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Group by attack_type
    by_type: Dict[str, List[Dict]] = defaultdict(list)
    for rec in test_records:
        by_type[rec["attack_type"]].append(rec)

    # Sample one from each type
    samples = []
    for attack_type in sorted(by_type.keys()):
        picked = random.choice(by_type[attack_type])
        samples.append(picked)

    # Build 3×3 grid (up to 9)
    n = min(9, len(samples))
    n_rows, n_cols = 3, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15))

    for idx, ax in enumerate(axes.flat):
        if idx < n:
            rec = samples[idx]
            img_path = images_dir / rec["filename"]
            try:
                img = np.array(Image.open(img_path))
                ax.imshow(img)
                ax.set_title(f"{rec['attack_type']}\n{rec['label']}\n{rec['filename'][:20]}", fontsize=7)
            except Exception as e:
                ax.set_title(f"Error: {e}", fontsize=6)
        ax.axis("off")

    plt.suptitle("Attack Type Samples — Evaluation Set", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = vis_dir / "attack_type_samples.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"[Visualization] Saved attack type samples → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def prepare_evaluation_sets(
    source_dir: Path,
    output_dir: Path,
    vis_dir: Path,
) -> None:
    """Prepare test and val evaluation sets.

    Args:
        source_dir: Root of preprocessed images.
        output_dir: Output root for evaluation_sets/.
        vis_dir: Visualization output directory.
    """
    all_records: Dict[str, List[Dict]] = {}

    for split in ("test", "val"):
        split_src = source_dir / split
        if not split_src.exists():
            logger.warning(f"Split directory not found: {split_src} — skipping")
            continue

        split_out = output_dir / split
        logger.info(f"[Step 1/4] Processing {split} split from {split_src}")

        records = copy_images_for_split(split_src, split_out)
        all_records[split] = records

        save_labels_csv(records, split_out)

        # Print class distribution
        label_counter: Counter = Counter(r["label"] for r in records)
        type_counter: Counter = Counter(r["attack_type"] for r in records)
        logger.info(f"[{split}] Class distribution: {dict(label_counter)}")
        logger.info(f"[{split}] Attack type distribution:")
        for t, c in sorted(type_counter.items()):
            logger.info(f"    {t:30s}: {c}")

    # Visualizations
    logger.info("[Step 2/4] Generating visualizations")
    vis_dir.mkdir(parents=True, exist_ok=True)

    test_records = all_records.get("test", [])
    val_records = all_records.get("val", [])

    if test_records or val_records:
        save_test_set_distribution(test_records, val_records, vis_dir)

    if test_records:
        test_images_dir = output_dir / "test" / "images"
        save_attack_type_samples(test_records, test_images_dir, vis_dir)

    # Summary
    logger.info("[Step 3/4] Summary")
    logger.info("=" * 60)
    logger.info("EVALUATION SET SUMMARY")
    logger.info("=" * 60)
    for split, records in all_records.items():
        bonafide = sum(1 for r in records if r["label"] == "bonafide")
        attack = sum(1 for r in records if r["label"] == "attack")
        logger.info(f"  {split:5s}: {len(records):5d} total | {bonafide:5d} bonafide | {attack:5d} attack")

    logger.info(f"[Step 4/4] Done. Evaluation sets saved to {output_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare evaluation datasets for iris PAD.")
    p.add_argument("--source_dir", type=Path, default=Path("iris_bbdm_pad/data/preprocessed/"), help="Root of preprocessed images.")
    p.add_argument("--output_dir", type=Path, default=Path("iris_bbdm_pad/data/evaluation_sets/"), help="Output directory for evaluation sets.")
    p.add_argument("--vis_dir", type=Path, default=Path("iris_bbdm_pad/results/phase1_visualizations/"), help="Visualization output directory.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    np.random.seed(42)
    random.seed(42)

    prepare_evaluation_sets(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        vis_dir=args.vis_dir,
    )
