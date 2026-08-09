"""
Iris dataset loaders for f-AnoGAN training and inference.

Training (bonafide only):
    BonafideLatentDataset  — loads bonafide images from iris_td/data/bonafide_pairs
    and on-the-fly encodes them to VQGAN latents via a frozen VQ-f4 model.
    Only the B/ (clean) images are used, identical to iris_td and iris_bbdm_pad.

Inference (all classes):
    EvalDataset — loads all images (bonafide + attacks) from a flat evaluation
    directory alongside an optional labels.csv, consistent with iris_td.

Usage
-----
Both dataset classes are standalone — no dependency on BBDM datasets.
The VQGAN model is passed in from the caller (already loaded + frozen).
"""

import csv
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

log = logging.getLogger(__name__)

# Normalisation used by BBDM / iris_td across the whole project
_NORM = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

TRANSFORM_256 = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    _NORM,
])


# ---------------------------------------------------------------------------
# Training dataset — bonafide B/ images
# ---------------------------------------------------------------------------

class BonafideImageDataset(Dataset):
    """Returns 256×256 normalised tensors from bonafide_pairs/{stage}/B/.

    Designed to mirror iris_td's use of CustomAlignedDataset but without
    requiring BBDM's dataset infrastructure.

    Args:
        bonafide_pairs_root : path to iris_td/data/bonafide_pairs  (or iris_anogan equivalent)
        stage               : "train" or "val"
        transform           : torchvision transform (default: Resize+ToTensor+Normalize)
    """

    def __init__(
        self,
        bonafide_pairs_root: str,
        stage: str = "train",
        transform=None,
    ):
        root = Path(bonafide_pairs_root) / stage / "B"
        if not root.exists():
            raise FileNotFoundError(
                f"Expected bonafide B/ dir at {root}. "
                "Point bonafide_pairs_root to iris_td/data/bonafide_pairs."
            )

        self.images: List[Path] = sorted(root.rglob("*.png")) + sorted(root.rglob("*.jpg"))
        if not self.images:
            raise RuntimeError(f"No images found in {root}")

        self.transform = transform or TRANSFORM_256
        log.info(f"BonafideImageDataset [{stage}]: {len(self.images)} images from {root}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.images[idx]).convert("RGB")
        return self.transform(img)


# ---------------------------------------------------------------------------
# Inference / evaluation dataset — all classes
# ---------------------------------------------------------------------------

class EvalDataset(Dataset):
    """Loads all PNG images from an evaluation directory.

    Returns (image_tensor, filepath_str, label_str) tuples.

    Label resolution order:
      1. labels.csv  (if provided)
      2. Path heuristic: 'Spoof' / 'spoof' in path → 'attack', else 'bonafide'
    """

    def __init__(
        self,
        image_dir: str,
        labels_csv: Optional[str] = None,
        transform=None,
        max_images: Optional[int] = None,
    ):
        self.transform = transform or TRANSFORM_256

        # Load labels CSV
        labels: dict = {}
        if labels_csv and Path(labels_csv).exists():
            with open(labels_csv) as f:
                for row in csv.DictReader(f):
                    labels[row.get("filename", "")] = row.get("label", "unknown")
            log.info(f"Labels loaded: {len(labels)} entries from {labels_csv}")

        # Collect images
        img_dir = Path(image_dir)
        all_imgs: List[Tuple[Path, str]] = []
        seen = set()

        for subfolder in ["images", "Live", "Spoof", ""]:
            folder = img_dir / subfolder if subfolder else img_dir
            if not folder.exists():
                continue
            for p in sorted(folder.rglob("*.png")):
                key = str(p)
                if key in seen:
                    continue
                seen.add(key)
                lbl = labels.get(p.name)
                if lbl is None:
                    lbl = "attack" if ("Spoof" in key or "spoof" in key) else "bonafide"
                all_imgs.append((p, lbl))

        if max_images:
            all_imgs = all_imgs[:max_images]

        self.images = all_imgs
        n_bf  = sum(1 for _, l in all_imgs if l == "bonafide")
        n_atk = sum(1 for _, l in all_imgs if l == "attack")
        log.info(f"EvalDataset: {len(all_imgs)} images | bonafide={n_bf} | attack={n_atk}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_path, label = self.images[idx]
        x = self.transform(Image.open(img_path).convert("RGB"))
        return x, str(img_path), label


# ---------------------------------------------------------------------------
# Attack-type extraction (mirrors iris_td / iris_bbdm_pad)
# ---------------------------------------------------------------------------

_ATTACK_TYPES = [
    "Artifact", "CL", "E-display",
    "Fake_with_Add_On", "Generated",
    "Post-Mortem", "Printed", "Print_E-display",
]


def extract_attack_type(img_path: str, label: str) -> str:
    """Return 'Live' for bonafide or infer attack type from path."""
    if label == "bonafide":
        return "Live"
    p = img_path.lower()
    for atype in _ATTACK_TYPES:
        if atype.lower() in p:
            return atype
    return "unknown"
