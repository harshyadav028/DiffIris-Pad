"""BBDM UNet Bottleneck t-SNE Visualization (9-Class Iris PAD).

Extracts activations from the UNet middle block (bottleneck) using a single
forward pass per image — NOT the full 200-step denoising loop, making it ~200x
faster while still capturing the model's learned feature space.

GPU-safe: runs with torch.no_grad(), moves tensors to CPU immediately after
capture, and falls back to CPU if CUDA OOM is detected.

Usage (from project root):
    conda activate bbdm
    python iris_bbdm_pad/analysis/tsne_bottleneck_bbdm.py \\
        --config iris_bbdm_pad/configs/bbdm_iris_bonafide.yaml \\
        --checkpoint results/iris_bonafide_pad/LBBDM-f4/checkpoint/top_model_epoch_70.pth \\
        --test_dir iris_bbdm_pad/data/evaluation_sets/test/ \\
        --output_dir iris_bbdm_pad/results/tsne_bottleneck/ \\
        --n_per_class 100 \\
        --device cuda
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import yaml
from PIL import Image
from sklearn.manifold import TSNE
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project-root bootstrap — same pattern as anomaly_detector.py
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]   # …/Geetanjali_PhD_IRIS_PAD
BBDM_ROOT = PROJECT_ROOT / "BBDM"
if str(BBDM_ROOT) not in sys.path:
    sys.path.insert(0, str(BBDM_ROOT))

from model.BrownianBridge.LatentBrownianBridgeModel import LatentBrownianBridgeModel

# Define dict2namespace inline — avoids importing BBDM/utils.py which has
# runner dependencies (runners/DiffusionBasedModelRunners/) not present here.
import argparse as _argparse


def dict2namespace(config):
    ns = _argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            value = dict2namespace(value)
        setattr(ns, key, value)
    return ns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 9-class colour palette (fixed, reproducible)
# ---------------------------------------------------------------------------
CLASS_COLORS: Dict[str, str] = {
    "Live":                "#2ecc71",   # green — Bona Fide
    "Printed":             "#e74c3c",   # red
    "CL":                  "#3498db",   # blue
    "E-display":           "#f39c12",   # orange
    "Print and E-display": "#9b59b6",   # purple
    "Generated":           "#1abc9c",   # teal
    "PostMortem":          "#e91e63",   # pink
    "Artifact":            "#795548",   # brown
    "Fake with Add On":    "#607d8b",   # grey-blue
}

# Friendly display names for the legend
CLASS_DISPLAY: Dict[str, str] = {
    "Live":                "Bona Fide (Live)",
    "CL":                  "Contact Lens (CL)",
    "Print and E-display": "Print + E-display",
    "PostMortem":          "Post-Mortem",
    "Fake with Add On":    "Fake + Add-On",
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_bbdm_model(
    config_path: str,
    checkpoint_path: str,
    device: torch.device,
) -> LatentBrownianBridgeModel:
    """Load LatentBrownianBridgeModel from YAML config + checkpoint.

    Prefers EMA weights when available (better inference quality).
    Loads weights on CPU first to avoid fragmenting GPU memory.
    """
    with open(config_path) as f:
        cfg = dict2namespace(yaml.safe_load(f))

    logger.info("Instantiating LatentBrownianBridgeModel …")
    model = LatentBrownianBridgeModel(cfg.model)

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    logger.info(f"Loading checkpoint: {ckpt_path.name}")
    states = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    if "ema" in states and states["ema"]:
        logger.info("Using EMA weights.")
        model_state = model.state_dict()
        for name in model_state:
            if name in states["ema"]:
                model_state[name] = states["ema"][name]
        model.load_state_dict(model_state, strict=False)
    elif "model" in states:
        logger.info("Using model weights.")
        model.load_state_dict(states["model"], strict=False)
    else:
        model.load_state_dict(states, strict=False)

    epoch = states.get("epoch", "?")
    logger.info(f"Checkpoint loaded (epoch={epoch}). Moving model to {device} …")
    model.eval()
    model.to(device)
    return model


# ---------------------------------------------------------------------------
# Balanced subset sampling
# ---------------------------------------------------------------------------

def sample_balanced_subset(
    labels_csv: Path,
    n_per_class: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Return a balanced DataFrame with up to n_per_class rows per attack_type."""
    df = pd.read_csv(labels_csv)
    rng = np.random.default_rng(seed)

    chunks: List[pd.DataFrame] = []
    for cls, grp in df.groupby("attack_type"):
        n = min(n_per_class, len(grp))
        idx = rng.choice(len(grp), size=n, replace=False)
        chunks.append(grp.iloc[idx])

    subset = pd.concat(chunks, ignore_index=True)
    counts = subset["attack_type"].value_counts().to_dict()
    logger.info(f"Balanced subset: {len(subset)} images total")
    for cls in sorted(counts):
        logger.info(f"  {cls:<25s}: {counts[cls]}")
    return subset


# ---------------------------------------------------------------------------
# Feature extraction via middle-block hook
# ---------------------------------------------------------------------------

def extract_bottleneck_features(
    model: LatentBrownianBridgeModel,
    subset_df: pd.DataFrame,
    images_dir: Path,
    batch_size: int,
    timestep: int,
    device: torch.device,
    image_size: int = 256,
) -> np.ndarray:
    """Extract global-average-pooled UNet middle-block features.

    Strategy (single forward pass per image, ~200x faster than denoising loop):
      1. VQGAN-encode image → latent z  [B, 3, 64, 64]
      2. Hook model.denoise_fn.middle_block output  [B, 512, 8, 8]
      3. Run ONE UNet forward at fixed timestep
      4. Global-avg-pool → [B, 512]
      5. Move to CPU immediately

    Args:
        model: Loaded LatentBrownianBridgeModel in eval mode.
        subset_df: Balanced subset DataFrame (columns: filename, attack_type).
        images_dir: Directory containing flat image files.
        batch_size: Images per GPU batch.
        timestep: Fixed diffusion timestep for the single UNet forward pass.
        device: torch.device.
        image_size: Resize target (must match training, default 256).

    Returns:
        (N, 512) numpy array of bottleneck features.
    """
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Lambda(lambda x: (x - 0.5) * 2.0),  # [0,1] → [-1,1]
    ])

    # ── Register hook on model.denoise_fn.middle_block ──────────────────────
    captured: List[torch.Tensor] = []

    def _hook(module, inp, out):
        # out shape: [B, 512, 8, 8] — move to CPU immediately to save VRAM
        captured.append(out.detach().cpu())

    hook_handle = model.denoise_fn.middle_block.register_forward_hook(_hook)

    filenames = subset_df["filename"].tolist()
    all_features: List[np.ndarray] = []

    try:
        for batch_start in tqdm(range(0, len(filenames), batch_size),
                                desc="Extracting bottleneck features"):
            batch_fnames = filenames[batch_start: batch_start + batch_size]

            # Load images
            imgs: List[torch.Tensor] = []
            for fname in batch_fnames:
                img_path = images_dir / fname
                if not img_path.exists():
                    # Try with parent as-is (filename might include subdir)
                    img_path = images_dir / Path(fname).name
                try:
                    img = Image.open(img_path).convert("RGB")
                    imgs.append(transform(img))
                except Exception as exc:
                    logger.warning(f"Skipping {fname}: {exc}")

            if not imgs:
                continue

            batch = torch.stack(imgs).to(device)   # [B, 3, H, W]

            with torch.no_grad():
                # Step 1: encode to VQ latent space
                try:
                    z = model.encode(batch, cond=False)   # [B, 3, 64, 64]
                except Exception:
                    # Some VQGAN wrappers may expose encode differently
                    z, _, _ = model.vqgan.encode(batch)

                # Step 2: single UNet forward pass at fixed timestep
                # Hook captures middle_block output automatically
                t = torch.full((z.shape[0],), timestep,
                               dtype=torch.long, device=device)
                _ = model.denoise_fn(z, timesteps=t, context=None)

            # Step 3: pool captured features [B, 512, 8, 8] → [B, 512]
            if captured:
                feat = captured.pop()                        # [B, 512, 8, 8]
                feat = F.adaptive_avg_pool2d(feat, 1)        # [B, 512, 1, 1]
                feat = feat.squeeze(-1).squeeze(-1)          # [B, 512]
                all_features.append(feat.numpy())

            # Free GPU memory from this batch
            del batch, z, t
            if device.type == "cuda":
                torch.cuda.empty_cache()

    finally:
        hook_handle.remove()

    if not all_features:
        raise RuntimeError("No features extracted — check image paths and model.")

    features = np.concatenate(all_features, axis=0)
    logger.info(f"Feature matrix shape: {features.shape}")
    return features


# ---------------------------------------------------------------------------
# t-SNE
# ---------------------------------------------------------------------------

def compute_tsne(
    features: np.ndarray,
    perplexity: int = 30,
    n_iter: int = 1000,
    random_state: int = 42,
) -> np.ndarray:
    """Fit t-SNE and return 2-D coordinates."""
    # Adjust perplexity if dataset is very small
    effective_perplexity = min(perplexity, (features.shape[0] - 1) // 3)
    logger.info(
        f"Running t-SNE: N={features.shape[0]}, D={features.shape[1]}, "
        f"perplexity={effective_perplexity}, n_iter={n_iter}"
    )
    tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        max_iter=n_iter,
        random_state=random_state,
        verbose=1,
    )
    coords = tsne.fit_transform(features)
    logger.info("t-SNE complete.")
    return coords


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_tsne_9class(
    coords: np.ndarray,
    attack_types: np.ndarray,
    output_path: Path,
) -> None:
    """Scatter plot coloured by the 9 specific class labels."""
    fig, ax = plt.subplots(figsize=(14, 11))

    ordered_classes = [
        "Live",
        "Printed",
        "CL",
        "E-display",
        "Print and E-display",
        "Generated",
        "PostMortem",
        "Artifact",
        "Fake with Add On",
    ]

    for cls in ordered_classes:
        mask = attack_types == cls
        if mask.sum() == 0:
            continue
        color = CLASS_COLORS.get(cls, "#333333")
        display = CLASS_DISPLAY.get(cls, cls)
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=color,
            label=f"{display}  (n={mask.sum()})",
            alpha=0.7,
            s=60,
            edgecolors="black",
            linewidth=0.3,
        )

    ax.set_xlabel("t-SNE Dimension 1", fontsize=13)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=13)
    ax.set_title(
        "BBDM UNet Bottleneck Features — t-SNE (9 Classes)\n"
        "Single forward pass at mid-bridge timestep · Global-Avg-Pool of Middle Block",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=10, loc="best", framealpha=0.85)
    ax.grid(alpha=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Separation metrics
# ---------------------------------------------------------------------------

def print_separation_metrics(
    coords: np.ndarray,
    attack_types: np.ndarray,
) -> None:
    """Print centroid distance from Bona Fide to each attack class."""
    live_mask = attack_types == "Live"
    if live_mask.sum() == 0:
        logger.warning("No 'Live' samples found — skipping separation metrics.")
        return

    live_center = coords[live_mask].mean(axis=0)
    live_spread = coords[live_mask].std(axis=0).mean()

    logger.info("=" * 65)
    logger.info("BOTTLENECK SEPARATION METRICS  (centroid distance from Bona Fide)")
    logger.info("=" * 65)
    logger.info(f"  {'Class':<25s}  {'Distance':>10s}  {'n':>6s}")
    logger.info(f"  {'-'*25}  {'-'*10}  {'-'*6}")

    attacks = [c for c in np.unique(attack_types) if c != "Live"]
    distances = []
    for cls in sorted(attacks):
        mask = attack_types == cls
        if mask.sum() == 0:
            continue
        center = coords[mask].mean(axis=0)
        dist = np.linalg.norm(live_center - center)
        distances.append((cls, dist, mask.sum()))

    # Sort descending by distance
    for cls, dist, n in sorted(distances, key=lambda x: -x[1]):
        logger.info(f"  {cls:<25s}  {dist:>10.2f}  {n:>6d}")

    logger.info(f"\n  Bona Fide spread (std): {live_spread:.2f}")
    logger.info("=" * 65)
    logger.info("  Interpretation: larger distance = more separable from live irises")
    logger.info("=" * 65)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BBDM UNet bottleneck t-SNE visualization (9-class iris PAD)"
    )
    p.add_argument("--config",
                   default="iris_bbdm_pad/configs/bbdm_iris_bonafide.yaml",
                   help="Path to BBDM YAML config")
    p.add_argument("--checkpoint",
                   default="results/iris_bonafide_pad/LBBDM-f4/checkpoint/top_model_epoch_70.pth",
                   help="Path to trained BBDM checkpoint (.pth)")
    p.add_argument("--test_dir",
                   default="iris_bbdm_pad/data/evaluation_sets/test/",
                   help="Test directory containing images/ and labels.csv")
    p.add_argument("--output_dir",
                   default="iris_bbdm_pad/results/tsne_bottleneck/",
                   help="Directory for output PNG")
    p.add_argument("--n_per_class", type=int, default=100,
                   help="Max images per class in balanced subset (67 for Fake+Addon)")
    p.add_argument("--timestep", type=int, default=500,
                   help="Fixed diffusion timestep for single UNet forward pass")
    p.add_argument("--batch_size", type=int, default=8,
                   help="Batch size for GPU inference")
    p.add_argument("--device", default="cuda",
                   help="'cuda' or 'cpu'")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Resolve paths relative to cwd (project root)
    cwd = Path.cwd()
    config_path   = cwd / args.config
    ckpt_path     = cwd / args.checkpoint
    test_dir      = cwd / args.test_dir
    output_dir    = cwd / args.output_dir
    images_dir    = test_dir / "images"
    labels_csv    = test_dir / "labels.csv"

    for p in (config_path, ckpt_path, images_dir, labels_csv):
        if not p.exists():
            raise FileNotFoundError(f"Required path not found: {p}")

    # Device selection with OOM fallback
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        batch_size = args.batch_size
    else:
        device = torch.device("cpu")
        batch_size = max(1, args.batch_size // 2)
        logger.info("Using CPU (CUDA unavailable or not requested)")

    logger.info(f"Device: {device} | Batch size: {batch_size}")
    if device.type == "cuda":
        free_mb = (torch.cuda.get_device_properties(0).total_memory -
                   torch.cuda.memory_allocated(0)) // (1024 ** 2)
        logger.info(f"GPU free memory: ~{free_mb} MiB")

    # ── Step 1: Load model ──────────────────────────────────────────────────
    model = load_bbdm_model(str(config_path), str(ckpt_path), device)

    # ── Step 2: Balanced subset ─────────────────────────────────────────────
    subset_df = sample_balanced_subset(labels_csv, args.n_per_class, args.seed)
    attack_types = subset_df["attack_type"].values

    # ── Step 3: Extract bottleneck features ─────────────────────────────────
    try:
        features = extract_bottleneck_features(
            model, subset_df, images_dir,
            batch_size=batch_size,
            timestep=args.timestep,
            device=device,
        )
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower() and device.type == "cuda":
            logger.warning("CUDA OOM — falling back to CPU with batch_size=4")
            torch.cuda.empty_cache()
            model.to("cpu")
            features = extract_bottleneck_features(
                model, subset_df, images_dir,
                batch_size=4,
                timestep=args.timestep,
                device=torch.device("cpu"),
            )
        else:
            raise

    # Keep only rows where a feature was successfully extracted
    # (some images may have been skipped due to missing files)
    n_extracted = features.shape[0]
    if n_extracted < len(subset_df):
        logger.warning(
            f"Extracted {n_extracted} features for {len(subset_df)} samples "
            "(some images were skipped)."
        )
        attack_types = attack_types[:n_extracted]

    # ── Step 4: t-SNE ───────────────────────────────────────────────────────
    coords = compute_tsne(
        features,
        perplexity=30,
        n_iter=1000,
        random_state=args.seed,
    )

    # ── Step 5: Plot ────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    output_png = output_dir / "bbdm_tsne_9class.png"
    plot_tsne_9class(coords, attack_types, output_png)

    # ── Step 6: Separation metrics ──────────────────────────────────────────
    print_separation_metrics(coords, attack_types)

    logger.info("Done.")


if __name__ == "__main__":
    main()
