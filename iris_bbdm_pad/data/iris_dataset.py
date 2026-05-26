"""Phase 1 — PyTorch Dataset Classes for Iris PAD.

Two dataset classes:
1. IrisBonafidePairDataset — for BBDM training (paired A/B with augmentation)
2. IrisTestDataset         — for PAD inference (on-the-fly corruption)

Both normalize to [-1, 1] range (as expected by BBDM/VQGAN).

Usage:
    from iris_bbdm_pad.data.iris_dataset import IrisBonafidePairDataset, IrisTestDataset
"""

from __future__ import annotations

import csv
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)

# Normalization to [-1, 1] (required by BBDM/VQGAN)
NORMALIZE_MEAN = [0.5, 0.5, 0.5]
NORMALIZE_STD = [0.5, 0.5, 0.5]


def _to_tensor_normalized(pil_image: Image.Image) -> torch.Tensor:
    """Convert PIL Image (RGB) to normalized tensor in [-1, 1].

    Args:
        pil_image: RGB PIL Image.

    Returns:
        Float tensor (3, H, W) in [-1, 1].
    """
    t = transforms.ToTensor()(pil_image)  # [0, 1]
    t = transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)(t)  # [-1, 1]
    return t


def _load_image_as_rgb_pil(path: Path, size: int = 256) -> Image.Image:
    """Load an image, convert to RGB, resize to size×size.

    Args:
        path: Path to the image file.
        size: Target spatial size (square).

    Returns:
        RGB PIL Image resized to size×size.
    """
    img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.LANCZOS)
    return img


# ---------------------------------------------------------------------------
# Dataset 1: Bona fide pairs (BBDM training)
# ---------------------------------------------------------------------------

class IrisBonafidePairDataset(Dataset):
    """Paired dataset of noisy (A) and clean (B) bona fide iris images.

    Used for BBDM training. Returns synchronized augmented pairs.
    A = corrupted bona fide (condition input)
    B = clean bona fide (target output)

    Args:
        pairs_dir: Root directory containing {split}/A/ and {split}/B/.
        split: 'train' or 'val'.
        image_size: Spatial size (default 256).
        flip: Apply random horizontal flip during training (synchronized for A and B).
    """

    def __init__(
        self,
        pairs_dir: str | Path,
        split: str = "train",
        image_size: int = 256,
        flip: Optional[bool] = None,
    ) -> None:
        super().__init__()
        self.pairs_dir = Path(pairs_dir)
        self.split = split
        self.image_size = image_size
        # Default: flip during training only
        self.flip = flip if flip is not None else (split == "train")

        self.dir_a = self.pairs_dir / split / "A"
        self.dir_b = self.pairs_dir / split / "B"

        if not self.dir_a.exists():
            raise FileNotFoundError(f"A directory not found: {self.dir_a}")
        if not self.dir_b.exists():
            raise FileNotFoundError(f"B directory not found: {self.dir_b}")

        # Collect filenames — rely on sorted order for alignment
        a_files = sorted(self.dir_a.glob("*.png"))
        b_files = sorted(self.dir_b.glob("*.png"))

        # Validate alignment
        a_names = {f.name for f in a_files}
        b_names = {f.name for f in b_files}
        if a_names != b_names:
            missing_a = b_names - a_names
            missing_b = a_names - b_names
            if missing_a:
                logger.warning(f"Files in B but not A: {list(missing_a)[:5]}")
            if missing_b:
                logger.warning(f"Files in A but not B: {list(missing_b)[:5]}")

        # Keep only matched pairs
        common = sorted(a_names & b_names)
        self.filenames = common

        logger.info(f"IrisBonafidePairDataset [{split}]: {len(self.filenames)} pairs "
                    f"(flip={self.flip}) from {self.pairs_dir}")

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        """Return a training sample.

        Returns:
            Dict with:
                'A': tensor (3, H, W) in [-1, 1] — corrupted (condition)
                'B': tensor (3, H, W) in [-1, 1] — clean (target)
                'path_A': str — path to A image
                'path_B': str — path to B image
        """
        fname = self.filenames[idx]
        path_a = self.dir_a / fname
        path_b = self.dir_b / fname

        img_a = _load_image_as_rgb_pil(path_a, self.image_size)
        img_b = _load_image_as_rgb_pil(path_b, self.image_size)

        # Synchronized random horizontal flip (same decision for A and B)
        if self.flip and random.random() > 0.5:
            img_a = img_a.transpose(Image.FLIP_LEFT_RIGHT)
            img_b = img_b.transpose(Image.FLIP_LEFT_RIGHT)

        tensor_a = _to_tensor_normalized(img_a)
        tensor_b = _to_tensor_normalized(img_b)

        return {
            "A": tensor_a,
            "B": tensor_b,
            "path_A": str(path_a),
            "path_B": str(path_b),
        }


# ---------------------------------------------------------------------------
# Dataset 2: Test/Val evaluation dataset (inference)
# ---------------------------------------------------------------------------

class IrisTestDataset(Dataset):
    """Evaluation dataset with on-the-fly corruption for PAD scoring.

    Loads clean images from evaluation_sets/{split}/images/,
    applies the same deterministic corruption as training (seeded by filename),
    and returns paired tensors for BBDM reconstruction + PAD scoring.

    A = corrupted (BBDM input)
    B = clean (compared against BBDM output to compute PAD score)

    Args:
        eval_dir: Root evaluation directory (evaluation_sets/).
        split: 'test' or 'val'.
        image_size: Spatial size (default 256).
        corruption_config_path: Path to dataset_config.json with corruption params.
            Defaults to iris_bbdm_pad/data/bonafide_pairs/dataset_config.json.
    """

    def __init__(
        self,
        eval_dir: str | Path,
        split: str = "test",
        image_size: int = 256,
        corruption_config_path: Optional[str | Path] = None,
    ) -> None:
        super().__init__()
        self.eval_dir = Path(eval_dir)
        self.split = split
        self.image_size = image_size

        self.images_dir = self.eval_dir / split / "images"
        self.labels_csv = self.eval_dir / split / "labels.csv"

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.labels_csv.exists():
            raise FileNotFoundError(f"Labels CSV not found: {self.labels_csv}")

        # Load corruption config
        if corruption_config_path is None:
            corruption_config_path = Path("iris_bbdm_pad/data/bonafide_pairs/dataset_config.json")
        corruption_config_path = Path(corruption_config_path)

        if corruption_config_path.exists():
            with open(corruption_config_path) as f:
                self.corruption_config = json.load(f)["corruption_config"]
        else:
            logger.warning(f"Corruption config not found at {corruption_config_path}, using defaults.")
            # Import here to avoid circular import at module level
            import sys
            sys.path.insert(0, str(Path(__file__).parent))
            from corruption import get_default_corruption_config
            self.corruption_config = get_default_corruption_config()

        # Load labels
        self.records = self._load_labels()
        logger.info(f"IrisTestDataset [{split}]: {len(self.records)} images from {self.images_dir}")

    def _load_labels(self) -> List[Dict]:
        """Load labels.csv into a list of dicts."""
        records = []
        with open(self.labels_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path = self.images_dir / row["filename"]
                if img_path.exists():
                    records.append({
                        "filename": row["filename"],
                        "label": row["label"],
                        "attack_type": row["attack_type"],
                        "path": img_path,
                    })
                else:
                    logger.warning(f"Image not found, skipping: {img_path}")
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        """Return an evaluation sample.

        Returns:
            Dict with:
                'A': tensor (3, H, W) in [-1, 1] — corrupted (BBDM input)
                'B': tensor (3, H, W) in [-1, 1] — clean original (for PAD scoring)
                'label': str — 'bonafide' or 'attack'
                'attack_type': str — attack type or 'Live'
                'filename': str — image filename
        """
        rec = self.records[idx]
        path = rec["path"]

        # Load clean image
        clean_pil = _load_image_as_rgb_pil(path, self.image_size)
        clean_arr = np.array(clean_pil)

        # Apply deterministic corruption (seeded by filename stem)
        # Import at call time to allow this module to be imported before corruption.py is on sys.path
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from corruption import apply_corruption_pipeline, filename_to_seed

        seed = filename_to_seed(Path(rec["filename"]).stem)
        corrupted_arr = apply_corruption_pipeline(
            clean_arr,
            seed=seed,
            noise_sigma_range=tuple(self.corruption_config["noise_sigma_range"]),
            blur_kernel_choices=tuple(self.corruption_config["blur_kernel_choices"]),
            blur_sigma_range=tuple(self.corruption_config["blur_sigma_range"]),
            downscale_range=tuple(self.corruption_config["downscale_range"]),
            target_size=self.image_size,
        )

        corrupted_pil = Image.fromarray(corrupted_arr.astype(np.uint8))
        clean_pil_resized = Image.fromarray(clean_arr)

        tensor_a = _to_tensor_normalized(corrupted_pil)
        tensor_b = _to_tensor_normalized(clean_pil_resized)

        return {
            "A": tensor_a,
            "B": tensor_b,
            "label": rec["label"],
            "attack_type": rec["attack_type"],
            "filename": rec["filename"],
        }


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    pairs_dir = Path("iris_bbdm_pad/data/bonafide_pairs")
    eval_dir = Path("iris_bbdm_pad/data/evaluation_sets")

    print("\n--- IrisBonafidePairDataset ---")
    if pairs_dir.exists():
        ds_train = IrisBonafidePairDataset(pairs_dir, split="train")
        sample = ds_train[0]
        print(f"  Pairs: {len(ds_train)}")
        print(f"  A shape: {sample['A'].shape}, range [{sample['A'].min():.2f}, {sample['A'].max():.2f}]")
        print(f"  B shape: {sample['B'].shape}, range [{sample['B'].min():.2f}, {sample['B'].max():.2f}]")
        assert sample["A"].shape == (3, 256, 256), f"Bad shape: {sample['A'].shape}"
        assert -1.1 <= sample["A"].min() <= 1.1, f"Out of range: {sample['A'].min()}"
        print("  OK")
    else:
        print(f"  Pairs directory not found: {pairs_dir}")

    print("\n--- IrisTestDataset ---")
    if eval_dir.exists():
        ds_test = IrisTestDataset(eval_dir, split="test")
        sample = ds_test[0]
        print(f"  Images: {len(ds_test)}")
        print(f"  A shape: {sample['A'].shape}")
        print(f"  B shape: {sample['B'].shape}")
        print(f"  label: {sample['label']}, attack_type: {sample['attack_type']}")
        print("  OK")
    else:
        print(f"  Evaluation directory not found: {eval_dir}")
