"""
BBDM-based anomaly detector for iris PAD.

Loads a trained LBBDM-f4 model and computes three types of PAD scores:

1. Reconstruction Error: Corrupt -> BBDM reconstruct -> compare output vs clean original
2. Trajectory Stability: Denoise from 4 different noise levels -> measure variance
3. Combined: Weighted sum of (1) and (2), weights from validation grid search

Usage:
    python iris_bbdm_pad/models/anomaly_detector.py \
        --config iris_bbdm_pad/configs/bbdm_iris_bonafide.yaml \
        --checkpoint results/iris_bonafide_pad/LBBDM-f4/checkpoint/top_model_epoch_XXX.pth \
        --test_dir iris_bbdm_pad/data/evaluation_sets/test/ \
        --output_dir iris_bbdm_pad/results/ \
        --device cuda \
        --batch_size 32 \
        --trajectory_timesteps 100 250 500 750
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import yaml
from PIL import Image
from tqdm import tqdm

# ── Project-root bootstrap ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BBDM_ROOT = PROJECT_ROOT / "BBDM"
if str(BBDM_ROOT) not in sys.path:
    sys.path.insert(0, str(BBDM_ROOT))

# BBDM imports (require BBDM on sys.path)
from model.BrownianBridge.LatentBrownianBridgeModel import LatentBrownianBridgeModel
from model.utils import extract
from utils import dict2namespace

from iris_bbdm_pad.models.pad_scorer import PADScorer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed_all(42)


# ---------------------------------------------------------------------------
# Image dataset for scoring
# ---------------------------------------------------------------------------

class IrisScoringDataset(torch.utils.data.Dataset):
    """Dataset for scoring — loads (corrupted, clean, metadata) triples.

    Expects:
        root/A/<image>.png  — corrupted images
        root/B/<image>.png  — clean originals
        root/labels.csv     — filename, label, attack_type columns

    Args:
        root: Root directory containing A/, B/, labels.csv.
        image_size: Target image size.
    """

    EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

    def __init__(self, root: Path, image_size: int = 256) -> None:
        self.root = Path(root)
        self.image_size = image_size

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Lambda(lambda x: (x - 0.5) * 2.0),  # [0,1] -> [-1,1]
        ])

        # Load labels
        labels_csv = self.root / "labels.csv"
        if labels_csv.exists():
            self.df = pd.read_csv(labels_csv)
        else:
            raise FileNotFoundError(f"labels.csv not found in {self.root}")

        # Verify A/B directories
        a_dir = self.root / "A"
        b_dir = self.root / "B"
        if not a_dir.exists() or not b_dir.exists():
            raise FileNotFoundError(f"A/ or B/ directory missing in {self.root}")

        # Build sample list
        self.samples: List[Dict] = []
        for _, row in self.df.iterrows():
            fname = row["filename"]
            stem = Path(fname).stem
            # Try multiple extensions
            a_path = self._find_image(a_dir, stem)
            b_path = self._find_image(b_dir, stem)
            if a_path and b_path:
                self.samples.append({
                    "filename": fname,
                    "a_path": a_path,
                    "b_path": b_path,
                    "label": str(row.get("label", "unknown")),
                    "attack_type": str(row.get("attack_type", row.get("class", "unknown"))),
                })

        logger.info(f"Loaded {len(self.samples)} samples from {self.root}")

    def _find_image(self, directory: Path, stem: str) -> Optional[Path]:
        for ext in self.EXTENSIONS:
            p = directory / (stem + ext)
            if p.exists():
                return p
        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        s = self.samples[idx]
        a_img = Image.open(s["a_path"]).convert("RGB")
        b_img = Image.open(s["b_path"]).convert("RGB")
        return self.transform(a_img), self.transform(b_img), {
            "filename": s["filename"],
            "label": s["label"],
            "attack_type": s["attack_type"],
        }


# ---------------------------------------------------------------------------
# Core anomaly detector
# ---------------------------------------------------------------------------

class BBDMAnomalyDetector:
    """BBDM-based anomaly detector for iris PAD.

    Loads a trained LBBDM-f4 checkpoint and provides methods for reconstruction
    and trajectory-based PAD scoring.

    Args:
        config_path: Path to the BBDM YAML config file.
        checkpoint_path: Path to the trained LBBDM checkpoint (.pth).
        device: 'cuda' or 'cpu'.
        trajectory_timesteps: Timesteps for trajectory stability scoring.
        w_recon: Weight for reconstruction score in combined score.
        w_trajectory: Weight for trajectory score in combined score.
    """

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str = "cuda",
        trajectory_timesteps: List[int] = None,
        w_recon: float = 0.5,
        w_trajectory: float = 0.5,
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.trajectory_timesteps = trajectory_timesteps or [100, 250, 500, 750]

        logger.info(f"Loading model on {self.device}")
        self.model = self._load_model(config_path, checkpoint_path)
        self.model.eval()
        self.model.to(self.device)

        self.scorer = PADScorer(
            device=str(self.device),
            w_recon=w_recon,
            w_trajectory=w_trajectory,
        )

        logger.info(f"Detector ready. Trajectory timesteps: {self.trajectory_timesteps}")

    def _load_model(self, config_path: str, checkpoint_path: str) -> LatentBrownianBridgeModel:
        """Load LatentBrownianBridgeModel from config and checkpoint.

        Args:
            config_path: YAML config path.
            checkpoint_path: Checkpoint .pth path.

        Returns:
            Loaded model in eval mode.
        """
        with open(config_path) as f:
            cfg_dict = yaml.safe_load(f)
        cfg = dict2namespace(cfg_dict)

        # Build the model
        model = LatentBrownianBridgeModel(cfg.model)

        # Load checkpoint
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        logger.info(f"Loading checkpoint: {ckpt_path.name}")
        states = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

        # Handle EMA weights (preferred for inference)
        if "ema" in states:
            logger.info("Using EMA weights for inference.")
            ema_state = states["ema"]
            # EMA shadow is a dict of param_name -> tensor
            model_state = model.state_dict()
            for name, param in model_state.items():
                if name in ema_state:
                    model_state[name] = ema_state[name]
            model.load_state_dict(model_state, strict=False)
        elif "model" in states:
            model.load_state_dict(states["model"], strict=False)
        else:
            model.load_state_dict(states, strict=False)

        epoch = states.get("epoch", "?")
        step = states.get("step", "?")
        logger.info(f"Checkpoint loaded: epoch={epoch}, step={step}")

        return model

    # -----------------------------------------------------------------------
    # Reconstruction
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def reconstruct(self, corrupted_images: torch.Tensor) -> torch.Tensor:
        """Run full BBDM reconstruction (corrupted -> clean).

        Args:
            corrupted_images: (B, C, H, W) in [-1, 1] — the corrupted input (A).

        Returns:
            (B, C, H, W) in [-1, 1] — reconstructed images.
        """
        corrupted_images = corrupted_images.to(self.device)
        return self.model.sample(corrupted_images, clip_denoised=False)

    @torch.no_grad()
    def reconstruct_from_timestep(
        self, corrupted_images: torch.Tensor, timestep: int
    ) -> torch.Tensor:
        """Reconstruct with bridge noise added at a specific timestep.

        Adds bridge-scale noise to the corrupted latent at timestep t, then
        runs full denoising from that perturbed state. This probes the model's
        stability at different noise levels:
        - Bona fide: stable (similar results across timesteps)
        - Attack: unstable (varying results because model is uncertain)

        Args:
            corrupted_images: (B, C, H, W) in [-1, 1].
            timestep: Bridge timestep (0 to num_timesteps-1).

        Returns:
            (B, C, H, W) in [-1, 1] — reconstructed images from this noise level.
        """
        corrupted_images = corrupted_images.to(self.device)
        b = corrupted_images.shape[0]

        # Encode corrupted -> latent (this is y in the bridge)
        y_latent = self.model.encode(corrupted_images, cond=True)

        # Get bridge variance at this timestep
        t_tensor = torch.full((b,), timestep, device=self.device, dtype=torch.long)
        t_clamped = t_tensor.clamp(0, self.model.num_timesteps - 1)
        var_t = extract(self.model.variance_t, t_clamped, y_latent.shape)
        sigma_t = torch.sqrt(var_t)

        # Perturb starting point from y by bridge-scale noise
        noise = torch.randn_like(y_latent)
        x_start = y_latent + sigma_t * noise

        # Run full reverse denoising from perturbed starting point
        context = self.model.get_cond_stage_context(corrupted_images)
        img = x_start
        for i in range(len(self.model.steps)):
            img, _ = self.model.p_sample(
                x_t=img,
                y=y_latent,
                context=context,
                i=i,
                clip_denoised=False,
            )

        # Decode latent -> image
        return self.model.decode(img, cond=False)

    @torch.no_grad()
    def compute_trajectory_reconstructions(
        self, corrupted_images: torch.Tensor
    ) -> List[torch.Tensor]:
        """Compute reconstructions from multiple trajectory timesteps.

        Args:
            corrupted_images: (B, C, H, W) in [-1, 1].

        Returns:
            List of len(trajectory_timesteps) tensors, each (B, C, H, W).
        """
        results = []
        for t in self.trajectory_timesteps:
            rec = self.reconstruct_from_timestep(corrupted_images, t)
            results.append(rec.cpu())
        return results

    # -----------------------------------------------------------------------
    # Dataset processing
    # -----------------------------------------------------------------------

    def process_dataset(
        self,
        data_dir: Path,
        output_csv: Path,
        batch_size: int = 32,
        resume: bool = False,
        checkpoint_path: Optional[Path] = None,
    ) -> pd.DataFrame:
        """Score all images in a dataset directory.

        Args:
            data_dir: Directory with A/, B/, labels.csv.
            output_csv: Output path for scores CSV.
            batch_size: Number of images per batch.
            resume: Whether to resume from a previous run.
            checkpoint_path: Path for scoring checkpoint file.

        Returns:
            DataFrame with all scores.
        """
        # Auto-detect dataset format:
        # New format: data_dir/images/ + labels.csv  (evaluation_sets structure)
        # Old format: data_dir/A/ + data_dir/B/ + labels.csv
        if (data_dir / "images").exists():
            from iris_bbdm_pad.data.iris_dataset import IrisTestDataset  # noqa: PLC0415
            dataset = IrisTestDataset(
                eval_dir=data_dir.parent,
                split=data_dir.name,
            )
        else:
            dataset = IrisScoringDataset(data_dir)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        ckpt_file = checkpoint_path or (
            PROJECT_ROOT / "iris_bbdm_pad" / "checkpoints" / "scoring_checkpoint.json"
        )
        ckpt_file.parent.mkdir(parents=True, exist_ok=True)

        # Resume support
        processed_files: set = set()
        all_rows: List[Dict] = []
        if resume and ckpt_file.exists():
            with open(ckpt_file) as f:
                ckpt_data = json.load(f)
            processed_files = set(ckpt_data.get("processed_files", []))
            logger.info(f"Resuming: {len(processed_files)} files already processed.")

        # Open CSV in append mode
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        csv_mode = "a" if resume and output_csv.exists() else "w"
        csv_f = open(output_csv, csv_mode, newline="")
        fieldnames = [
            "filename", "label", "attack_type",
            "mse_score", "lpips_score", "recon_score",
            "trajectory_score", "combined_score",
        ]
        writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
        if csv_mode == "w":
            writer.writeheader()

        start_time = time.time()
        total_batches = len(loader)
        batch_count = 0

        pbar = tqdm(loader, total=total_batches, desc="Scoring images")
        for batch_idx, batch in enumerate(pbar):
            # IrisTestDataset returns a dict; IrisScoringDataset returns a tuple
            if isinstance(batch, dict):
                corrupted = batch["A"]
                clean = batch["B"]
                filenames = list(batch["filename"])
                meta = {"label": list(batch["label"]), "attack_type": list(batch["attack_type"])}
            else:
                corrupted, clean, meta = batch
                filenames = list(meta["filename"])

            # Skip already-processed files (resume)
            if resume and all(fn in processed_files for fn in filenames):
                continue

            corrupted = corrupted.to(self.device)
            clean = clean.to(self.device)

            # Primary reconstruction
            primary_recon = self.reconstruct(corrupted)

            # Trajectory reconstructions
            traj_recons = self.compute_trajectory_reconstructions(corrupted)

            # Score batch
            scores = self.scorer.score_batch(
                primary_recon.cpu(),
                clean.cpu(),
                [t.cpu() for t in traj_recons],
            )

            # Write rows
            b = corrupted.shape[0]
            for i in range(b):
                fn = filenames[i]
                row = {
                    "filename": fn,
                    "label": meta["label"][i],
                    "attack_type": meta["attack_type"][i],
                    "mse_score": float(scores["mse"][i]),
                    "lpips_score": float(scores["lpips"][i]),
                    "recon_score": float(scores["recon_score"][i]),
                    "trajectory_score": float(scores.get("trajectory_score", [0] * b)[i]),
                    "combined_score": float(scores.get("combined_score", [0] * b)[i]),
                }
                writer.writerow(row)
                all_rows.append(row)
                processed_files.add(fn)

            csv_f.flush()
            batch_count += 1

            # Save scoring checkpoint every 50 batches
            if batch_count % 50 == 0:
                ckpt_data = {
                    "processed_files": list(processed_files),
                    "last_batch": batch_idx,
                    "total_scored": len(processed_files),
                }
                ckpt_file.write_text(json.dumps(ckpt_data, indent=2))

            # ETA
            elapsed = time.time() - start_time
            rate = batch_count / elapsed if elapsed > 0 else 1
            eta = (total_batches - batch_idx) / rate
            pbar.set_postfix({
                "scored": len(processed_files),
                "ETA": f"{eta/60:.1f}m",
            })

        csv_f.close()

        # Final checkpoint
        ckpt_file.write_text(json.dumps({
            "processed_files": list(processed_files),
            "total_scored": len(processed_files),
            "completed": True,
        }, indent=2))

        logger.info(f"Scoring complete: {len(processed_files)} images → {output_csv}")
        return pd.DataFrame(all_rows)

    # -----------------------------------------------------------------------
    # Visualizations
    # -----------------------------------------------------------------------

    def save_reconstruction_grids(
        self,
        data_dir: Path,
        scores_df: pd.DataFrame,
        vis_dir: Path,
        n_samples: int = 4,
    ) -> None:
        """Save reconstruction grids for bonafide and attack images.

        Args:
            data_dir: Directory with A/, B/ images.
            scores_df: DataFrame with scores and filenames.
            vis_dir: Output directory.
            n_samples: Number of samples per grid.
        """
        vis_dir.mkdir(parents=True, exist_ok=True)

        bonafide = scores_df[scores_df["label"] == "bonafide"].head(n_samples)
        attacks = scores_df[scores_df["label"] == "attack"]
        attack_types = attacks["attack_type"].unique()
        attack_sample = attacks.groupby("attack_type").head(1).head(n_samples)

        for subset_name, subset in [("bonafide", bonafide), ("attack", attack_sample)]:
            if len(subset) == 0:
                continue
            n = len(subset)
            fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
            if n == 1:
                axes = axes[np.newaxis, :]
            for r, (_, row) in enumerate(subset.iterrows()):
                fname = row["filename"]
                pair = self._load_image_pair(data_dir, fname)
                if pair is None:
                    continue
                corrupted_t, clean_t = pair
                recon_t = self.reconstruct(corrupted_t.unsqueeze(0).to(self.device))
                corrupted_np = self._tensor_to_np(corrupted_t)
                clean_np = self._tensor_to_np(clean_t)
                recon_np = self._tensor_to_np(recon_t.squeeze(0).cpu())

                axes[r, 0].imshow(corrupted_np)
                axes[r, 0].set_title("Corrupted (input)")
                axes[r, 1].imshow(recon_np)
                axes[r, 1].set_title(
                    f"BBDM recon\nrecon={row['recon_score']:.4f} traj={row['trajectory_score']:.4f}"
                )
                axes[r, 2].imshow(clean_np)
                axes[r, 2].set_title(f"Clean (target)\n{row['attack_type']}")
                for c in range(3):
                    axes[r, c].axis("off")

            title = "Bona Fide Reconstructions" if subset_name == "bonafide" else "Attack Reconstructions"
            fig.suptitle(title, fontsize=14, fontweight="bold")
            fig.tight_layout()
            out_path = vis_dir / f"reconstruction_grid_{subset_name}.png"
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"[Step] Saved {out_path.name}")

        # Comparison grid: 2 rows (1 bonafide, 1 attack)
        self._save_reconstruction_comparison(data_dir, scores_df, vis_dir)

    def _save_reconstruction_comparison(
        self, data_dir: Path, scores_df: pd.DataFrame, vis_dir: Path
    ) -> None:
        """2-row comparison: bonafide vs attack, annotated with all scores."""
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        pairs = [
            (scores_df[scores_df["label"] == "bonafide"].iloc[0], 0, "Bona Fide"),
            (scores_df[scores_df["label"] == "attack"].iloc[0], 1, "Attack"),
        ]
        for row_data, r, label in pairs:
            fname = row_data["filename"]
            pair = self._load_image_pair(data_dir, fname)
            if pair is None:
                continue
            corrupted_t, clean_t = pair
            recon_t = self.reconstruct(corrupted_t.unsqueeze(0).to(self.device))
            axes[r, 0].imshow(self._tensor_to_np(corrupted_t))
            axes[r, 0].set_title(f"{label}\nCorrupted Input")
            axes[r, 1].imshow(self._tensor_to_np(recon_t.squeeze(0).cpu()))
            axes[r, 1].set_title(
                f"BBDM Reconstruction\nrecon={row_data['recon_score']:.4f} "
                f"traj={row_data['trajectory_score']:.4f} "
                f"comb={row_data['combined_score']:.4f}"
            )
            axes[r, 2].imshow(self._tensor_to_np(clean_t))
            axes[r, 2].set_title("Clean Original")
            for c in range(3):
                axes[r, c].axis("off")

        fig.suptitle("Reconstruction Comparison: Bona Fide vs Attack", fontsize=14, fontweight="bold")
        fig.tight_layout()
        out_path = vis_dir / "reconstruction_comparison.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"[Step] Saved {out_path.name}")

    def save_trajectory_visualizations(
        self, data_dir: Path, scores_df: pd.DataFrame, vis_dir: Path
    ) -> None:
        """Save trajectory stability visualizations.

        Args:
            data_dir: Directory with A/, B/ images.
            scores_df: DataFrame with scores.
            vis_dir: Output directory.
        """
        vis_dir.mkdir(parents=True, exist_ok=True)

        bonafide_row = scores_df[scores_df["label"] == "bonafide"].iloc[0]
        attack_row = scores_df[scores_df["label"] == "attack"].iloc[0]

        n_t = len(self.trajectory_timesteps)
        fig, axes = plt.subplots(2, n_t, figsize=(4 * n_t, 8))

        for r, (row_data, label) in enumerate([
            (bonafide_row, "Bona Fide"),
            (attack_row, "Attack"),
        ]):
            fname = row_data["filename"]
            pair = self._load_image_pair(data_dir, fname)
            if pair is None:
                continue
            corrupted_t = pair[0].unsqueeze(0).to(self.device)
            traj_recons = self.compute_trajectory_reconstructions(corrupted_t)
            traj_var = float(row_data.get("trajectory_score", 0))

            for c, (t, rec) in enumerate(zip(self.trajectory_timesteps, traj_recons)):
                axes[r, c].imshow(self._tensor_to_np(rec.squeeze(0)))
                axes[r, c].set_title(f"t={t}")
                axes[r, c].axis("off")
            axes[r, 0].set_ylabel(f"{label}\nvar={traj_var:.4f}", fontsize=10)

        fig.suptitle(
            "Trajectory Stability: Bona Fide (stable) vs Attack (unstable)",
            fontsize=13, fontweight="bold"
        )
        fig.tight_layout()
        out_path = vis_dir / "trajectory_stability_visualization.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"[Step] Saved {out_path.name}")

        # Trajectory variance by timestep (50 per class)
        self._save_trajectory_variance_plot(data_dir, scores_df, vis_dir, n=50)

    def _save_trajectory_variance_plot(
        self, data_dir: Path, scores_df: pd.DataFrame, vis_dir: Path, n: int = 50
    ) -> None:
        """Plot reconstruction MSE vs timestep for bonafide and attack.

        Args:
            data_dir: Directory with A/, B/ images.
            scores_df: DataFrame with scores.
            vis_dir: Output directory.
            n: Number of samples per class.
        """
        bonafide_rows = scores_df[scores_df["label"] == "bonafide"].head(n)
        attack_rows = scores_df[scores_df["label"] == "attack"].head(n)

        results = {"bonafide": [], "attack": []}

        for cls, rows in [("bonafide", bonafide_rows), ("attack", attack_rows)]:
            for _, row_data in rows.iterrows():
                fname = row_data["filename"]
                pair = self._load_image_pair(data_dir, fname)
                if pair is None:
                    continue
                corrupted_t = pair[0].unsqueeze(0).to(self.device)
                clean_t = pair[1]
                mse_by_t = []
                for t in self.trajectory_timesteps:
                    rec = self.reconstruct_from_timestep(corrupted_t, t)
                    mse = float(F.mse_loss(rec.squeeze(0).cpu(), clean_t))
                    mse_by_t.append(mse)
                results[cls].append(mse_by_t)

        fig, ax = plt.subplots(figsize=(8, 5))
        for cls, color, label in [
            ("bonafide", "#3b82f6", "Bona Fide"),
            ("attack", "#ef4444", "Attack"),
        ]:
            if not results[cls]:
                continue
            arr = np.array(results[cls])
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)
            ax.plot(self.trajectory_timesteps, mean, color=color, label=label, linewidth=2)
            ax.fill_between(
                self.trajectory_timesteps,
                mean - std, mean + std,
                color=color, alpha=0.2,
            )

        ax.set_xlabel("Trajectory Timestep")
        ax.set_ylabel("MSE from Clean Original")
        ax.set_title("Reconstruction MSE vs Trajectory Timestep")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_path = vis_dir / "trajectory_variance_by_timestep.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        logger.info(f"[Step] Saved {out_path.name}")

    def save_score_scatter(
        self, scores_df: pd.DataFrame, vis_dir: Path
    ) -> None:
        """Save scatter plots of PAD scores.

        Args:
            scores_df: DataFrame with all scores.
            vis_dir: Output directory.
        """
        vis_dir.mkdir(parents=True, exist_ok=True)

        bonafide = scores_df[scores_df["label"] == "bonafide"]
        attack = scores_df[scores_df["label"] == "attack"]

        # 1. recon_score vs trajectory_score
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(bonafide["recon_score"], bonafide["trajectory_score"],
                   c="#3b82f6", alpha=0.4, s=10, label="Bona Fide")
        ax.scatter(attack["recon_score"], attack["trajectory_score"],
                   c="#ef4444", alpha=0.4, s=10, label="Attack")
        ax.set_xlabel("Reconstruction Score")
        ax.set_ylabel("Trajectory Score")
        ax.set_title("PAD Score: Reconstruction vs Trajectory")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = vis_dir / "score_scatter.png"
        fig.savefig(out, dpi=300)
        plt.close(fig)
        logger.info(f"[Step] Saved {out.name}")

        # 2. Three-way scatter plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        ax = axes[0]
        ax.scatter(bonafide["mse_score"], bonafide["lpips_score"],
                   c="#3b82f6", alpha=0.3, s=8, label="Bona Fide")
        ax.scatter(attack["mse_score"], attack["lpips_score"],
                   c="#ef4444", alpha=0.3, s=8, label="Attack")
        ax.set_xlabel("MSE")
        ax.set_ylabel("LPIPS")
        ax.set_title("MSE vs LPIPS")
        ax.legend(markerscale=2)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.scatter(bonafide["recon_score"], bonafide["trajectory_score"],
                   c="#3b82f6", alpha=0.3, s=8, label="Bona Fide")
        ax.scatter(attack["recon_score"], attack["trajectory_score"],
                   c="#ef4444", alpha=0.3, s=8, label="Attack")
        ax.set_xlabel("Reconstruction Score")
        ax.set_ylabel("Trajectory Score")
        ax.set_title("Recon vs Trajectory Score")
        ax.legend(markerscale=2)
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        ax.hist(bonafide["combined_score"], bins=50, alpha=0.6, color="#3b82f6",
                label="Bona Fide", density=True)
        ax.hist(attack["combined_score"], bins=50, alpha=0.6, color="#ef4444",
                label="Attack", density=True)
        ax.set_xlabel("Combined Score")
        ax.set_ylabel("Density")
        ax.set_title("Combined Score Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.suptitle("PAD Score Analysis", fontsize=14, fontweight="bold")
        fig.tight_layout()
        out = vis_dir / "score_scatter_3way.png"
        fig.savefig(out, dpi=300)
        plt.close(fig)
        logger.info(f"[Step] Saved {out.name}")

    def save_paper_figures(
        self, data_dir: Path, scores_df: pd.DataFrame, paper_dir: Path
    ) -> None:
        """Generate publication-quality figures.

        Args:
            data_dir: Directory with A/, B/ images.
            scores_df: DataFrame with all scores.
            paper_dir: Output directory for paper figures.
        """
        paper_dir.mkdir(parents=True, exist_ok=True)

        self._save_reconstruction_error_heatmaps(data_dir, scores_df, paper_dir)
        self._save_diffusion_denoising_frames(data_dir, scores_df, paper_dir)
        self._save_per_attack_grids(data_dir, scores_df, paper_dir)
        self._save_tsne_umap(data_dir, scores_df, paper_dir)
        self._save_gradcam(data_dir, scores_df, paper_dir)

    def _save_reconstruction_error_heatmaps(
        self, data_dir: Path, scores_df: pd.DataFrame, paper_dir: Path
    ) -> None:
        """Save pixel-wise error heatmaps for 3 bonafide + 3 attack images."""
        rows_bf = scores_df[scores_df["label"] == "bonafide"].head(3)
        rows_atk = scores_df[scores_df["label"] == "attack"].head(3)
        all_rows = list(rows_bf.iterrows()) + list(rows_atk.iterrows())
        n = len(all_rows)

        fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
        if n == 1:
            axes = axes[np.newaxis, :]

        col_titles = ["Clean Original", "BBDM Reconstruction", "Error Heatmap", "Overlay"]
        for c, title in enumerate(col_titles):
            axes[0, c].set_title(title, fontsize=11, fontweight="bold")

        for r, (_, row_data) in enumerate(all_rows):
            fname = row_data["filename"]
            pair = self._load_image_pair(data_dir, fname)
            if pair is None:
                continue

            corrupted_t = pair[0].unsqueeze(0).to(self.device)
            clean_t = pair[1]
            recon_t = self.reconstruct(corrupted_t).squeeze(0).cpu()

            clean_np = self._tensor_to_np(clean_t)
            recon_np = self._tensor_to_np(recon_t)
            error = np.abs(recon_np.astype(float) - clean_np.astype(float)).mean(axis=2)
            error_norm = (error - error.min()) / (error.max() - error.min() + 1e-8)

            heatmap = cm.jet(error_norm)[:, :, :3]
            overlay = 0.5 * (clean_np / 255.0) + 0.5 * heatmap

            label = "Bona Fide" if row_data["label"] == "bonafide" else f"Attack ({row_data['attack_type']})"
            axes[r, 0].imshow(clean_np)
            axes[r, 0].set_ylabel(f"{label}\nrecon={row_data['recon_score']:.4f}", fontsize=8)
            axes[r, 1].imshow(recon_np)
            axes[r, 2].imshow(heatmap)
            axes[r, 3].imshow(np.clip(overlay, 0, 1))
            for c in range(4):
                axes[r, c].axis("off")

        fig.suptitle("Reconstruction Error Heatmaps", fontsize=14, fontweight="bold")
        fig.tight_layout()
        out = paper_dir / "reconstruction_error_heatmaps.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"[Step] Saved {out.name}")

    def _save_diffusion_denoising_frames(
        self, data_dir: Path, scores_df: pd.DataFrame, paper_dir: Path
    ) -> None:
        """Show 8-step denoising trajectory for 1 bonafide + 1 attack image."""
        n_steps_vis = 8
        bonafide_row = scores_df[scores_df["label"] == "bonafide"].iloc[0]
        attack_row = scores_df[scores_df["label"] == "attack"].iloc[0]

        fig, axes = plt.subplots(2, n_steps_vis, figsize=(4 * n_steps_vis, 8))

        for r, (row_data, label) in enumerate([
            (bonafide_row, "Bona Fide"),
            (attack_row, "Attack"),
        ]):
            fname = row_data["filename"]
            pair = self._load_image_pair(data_dir, fname)
            if pair is None:
                continue

            corrupted_t = pair[0].to(self.device)
            x_cond = corrupted_t.unsqueeze(0)

            # Run p_sample_loop with sample_mid_step=True
            out_samples, _ = self.model.sample(x_cond, clip_denoised=False, sample_mid_step=True)

            total = len(out_samples)
            indices = np.linspace(0, total - 1, n_steps_vis, dtype=int)

            for c, idx in enumerate(indices):
                frame = out_samples[idx].squeeze(0)
                frame_np = self._tensor_to_np(frame.cpu())
                t_val = int(self.model.steps[min(idx, len(self.model.steps) - 1)])
                axes[r, c].imshow(frame_np)
                axes[r, c].set_title(f"T={t_val}", fontsize=8)
                axes[r, c].axis("off")
            axes[r, 0].set_ylabel(label, fontsize=11, fontweight="bold")

        fig.suptitle("Diffusion Denoising Trajectory", fontsize=14, fontweight="bold")
        fig.tight_layout()
        out = paper_dir / "diffusion_denoising_frames.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"[Step] Saved {out.name}")

    def _save_per_attack_grids(
        self, data_dir: Path, scores_df: pd.DataFrame, paper_dir: Path
    ) -> None:
        """Save one reconstruction grid per attack type."""
        grids_dir = paper_dir / "per_attack_reconstruction_grids"
        grids_dir.mkdir(parents=True, exist_ok=True)

        attack_types = ["Live"] + list(scores_df["attack_type"].unique())
        for attack_type in attack_types:
            if attack_type == "Live":
                subset = scores_df[scores_df["label"] == "bonafide"].head(4)
            else:
                subset = scores_df[scores_df["attack_type"] == attack_type].head(4)

            if len(subset) == 0:
                continue

            n = len(subset)
            fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
            if n == 1:
                axes = axes[np.newaxis, :]

            for c, title in enumerate(["Corrupted", "Reconstruction", "Clean", "Error Heatmap"]):
                axes[0, c].set_title(title, fontweight="bold")

            for r, (_, row_data) in enumerate(subset.iterrows()):
                fname = row_data["filename"]
                pair = self._load_image_pair(data_dir, fname)
                if pair is None:
                    continue

                corrupted_t = pair[0].unsqueeze(0).to(self.device)
                clean_t = pair[1]
                recon_t = self.reconstruct(corrupted_t).squeeze(0).cpu()

                clean_np = self._tensor_to_np(clean_t)
                recon_np = self._tensor_to_np(recon_t)
                error = np.abs(recon_np.astype(float) - clean_np.astype(float)).mean(axis=2)
                error_n = (error - error.min()) / (error.max() - error.min() + 1e-8)
                heatmap = cm.jet(error_n)[:, :, :3]

                axes[r, 0].imshow(self._tensor_to_np(corrupted_t.squeeze(0).cpu()))
                axes[r, 1].imshow(recon_np)
                axes[r, 2].imshow(clean_np)
                axes[r, 3].imshow(heatmap)
                axes[r, 0].set_ylabel(
                    f"recon={row_data['recon_score']:.3f}\ntraj={row_data['trajectory_score']:.3f}",
                    fontsize=8
                )
                for c in range(4):
                    axes[r, c].axis("off")

            safe_name = attack_type.replace("/", "_").replace(" ", "_")
            fig.suptitle(f"Reconstruction Grid: {attack_type}", fontsize=13, fontweight="bold")
            fig.tight_layout()
            out = grids_dir / f"recon_grid_{safe_name}.png"
            fig.savefig(out, dpi=300, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"[Step] Saved {out.name}")

    def _save_tsne_umap(
        self, data_dir: Path, scores_df: pd.DataFrame, paper_dir: Path, n: int = 500
    ) -> None:
        """Extract VQGAN latent embeddings and save t-SNE + UMAP plots."""
        try:
            from sklearn.manifold import TSNE
        except ImportError:
            logger.warning("scikit-learn not installed — skipping t-SNE/UMAP plots.")
            return

        bonafide = scores_df[scores_df["label"] == "bonafide"].head(n)
        attack = scores_df[scores_df["label"] == "attack"].head(n)
        combined = pd.concat([bonafide, attack]).reset_index(drop=True)

        embeddings = []
        labels_vis = []
        attack_types_vis = []

        logger.info(f"Extracting VQGAN latents for {len(combined)} images...")
        for _, row_data in tqdm(combined.iterrows(), total=len(combined), desc="Extracting latents"):
            fname = row_data["filename"]
            pair = self._load_image_pair(data_dir, fname)
            if pair is None:
                continue
            img_t = pair[0].unsqueeze(0).to(self.device)
            with torch.no_grad():
                latent = self.model.encode(img_t, cond=True)
            embeddings.append(latent.squeeze(0).cpu().numpy().flatten())
            labels_vis.append(row_data["label"])
            attack_types_vis.append(row_data["attack_type"])

        if len(embeddings) < 10:
            logger.warning("Not enough embeddings for t-SNE — skipping.")
            return

        embeddings_arr = np.array(embeddings)
        labels_arr = np.array(labels_vis)
        attack_types_arr = np.array(attack_types_vis)

        logger.info("Running t-SNE...")
        tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
        tsne_emb = tsne.fit_transform(embeddings_arr)

        # Binary label plot
        fig, ax = plt.subplots(figsize=(8, 7))
        mask_bf = labels_arr == "bonafide"
        ax.scatter(tsne_emb[mask_bf, 0], tsne_emb[mask_bf, 1],
                   c="#3b82f6", s=8, alpha=0.6, label="Bona Fide")
        ax.scatter(tsne_emb[~mask_bf, 0], tsne_emb[~mask_bf, 1],
                   c="#ef4444", s=8, alpha=0.6, label="Attack")
        ax.set_title("t-SNE of VQGAN Latent Embeddings")
        ax.legend()
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(paper_dir / "tsne_latent_embeddings.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("[Step] Saved tsne_latent_embeddings.png")

        # By attack type
        unique_types = list(np.unique(attack_types_arr))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_types)))
        fig, ax = plt.subplots(figsize=(9, 7))
        for i, atype in enumerate(unique_types):
            mask = attack_types_arr == atype
            ax.scatter(tsne_emb[mask, 0], tsne_emb[mask, 1],
                       s=8, alpha=0.6, color=colors[i], label=atype)
        ax.set_title("t-SNE by Attack Type")
        ax.legend(markerscale=2, fontsize=8, loc="best")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(paper_dir / "tsne_by_attack_type.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("[Step] Saved tsne_by_attack_type.png")

        # UMAP
        try:
            import umap
            logger.info("Running UMAP...")
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
            umap_emb = reducer.fit_transform(embeddings_arr)
            fig, ax = plt.subplots(figsize=(8, 7))
            ax.scatter(umap_emb[mask_bf, 0], umap_emb[mask_bf, 1],
                       c="#3b82f6", s=8, alpha=0.6, label="Bona Fide")
            ax.scatter(umap_emb[~mask_bf, 0], umap_emb[~mask_bf, 1],
                       c="#ef4444", s=8, alpha=0.6, label="Attack")
            ax.set_title("UMAP of VQGAN Latent Embeddings")
            ax.legend()
            ax.axis("off")
            fig.tight_layout()
            fig.savefig(paper_dir / "umap_latent_embeddings.png", dpi=300, bbox_inches="tight")
            plt.close(fig)
            logger.info("[Step] Saved umap_latent_embeddings.png")
        except ImportError:
            logger.warning("umap-learn not installed — skipping UMAP.")

    def _save_gradcam(
        self, data_dir: Path, scores_df: pd.DataFrame, paper_dir: Path
    ) -> None:
        """Implement Grad-CAM for UNet middle_block and save overlays.

        Hooks the middle_block of the BBDM UNet to compute Grad-CAM for
        3 bonafide and 3 attack images.
        """
        rows_bf = scores_df[scores_df["label"] == "bonafide"].head(3)
        rows_atk = scores_df[scores_df["label"] == "attack"].head(3)
        all_rows = list(rows_bf.iterrows()) + list(rows_atk.iterrows())
        n = len(all_rows)

        fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
        if n == 1:
            axes = axes[np.newaxis, :]

        for c, title in enumerate(["Original", "Grad-CAM Heatmap", "Overlay"]):
            axes[0, c].set_title(title, fontweight="bold")

        unet = self.model.denoise_fn

        for r, (_, row_data) in enumerate(all_rows):
            fname = row_data["filename"]
            pair = self._load_image_pair(data_dir, fname)
            if pair is None:
                continue

            img_t = pair[0].unsqueeze(0).to(self.device)
            img_np = self._tensor_to_np(img_t.squeeze(0).cpu())

            # Forward hooks for activations and gradients
            activations = {}
            gradients = {}

            def forward_hook(module, inp, out):
                activations["middle_block"] = out

            def backward_hook(module, grad_in, grad_out):
                gradients["middle_block"] = grad_out[0]

            fwd_h = unet.middle_block.register_forward_hook(forward_hook)
            bwd_h = unet.middle_block.register_full_backward_hook(backward_hook)

            try:
                # Encode corrupted image to latent
                x_cond_latent = self.model.encode(img_t, cond=True)

                # Forward pass at a mid-level timestep
                t_val = self.model.num_timesteps // 2
                t_tensor = torch.full((1,), t_val, device=self.device, dtype=torch.long)

                # Enable gradients temporarily
                x_cond_latent_req = x_cond_latent.detach().requires_grad_(True)
                unet.zero_grad()

                with torch.enable_grad():
                    pred = unet(x_cond_latent_req, timesteps=t_tensor, context=None)
                    # Loss: reconstruction quality
                    loss = pred.pow(2).mean()
                    loss.backward()

                act = activations["middle_block"].detach()  # (1, C, H', W')
                grad = gradients["middle_block"].detach()   # (1, C, H', W')

                # Grad-CAM: ReLU(mean(grad * act, dim=channel))
                weights = grad.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)
                cam = (weights * act).sum(dim=1, keepdim=True)  # (1, 1, H', W')
                cam = F.relu(cam)

                # Upscale to original resolution
                cam_up = F.interpolate(cam, size=img_t.shape[2:], mode="bilinear", align_corners=False)
                cam_np = cam_up.squeeze().cpu().numpy()
                cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)

                heatmap = cm.jet(cam_np)[:, :, :3]
                overlay = np.clip(0.6 * (img_np / 255.0) + 0.4 * heatmap, 0, 1)

                label = "Bona Fide" if row_data["label"] == "bonafide" else f"Attack ({row_data['attack_type']})"
                axes[r, 0].imshow(img_np)
                axes[r, 0].set_ylabel(label, fontsize=8)
                axes[r, 1].imshow(heatmap)
                axes[r, 2].imshow(overlay)

            except Exception as e:
                logger.warning(f"Grad-CAM failed for {fname}: {e}")
            finally:
                fwd_h.remove()
                bwd_h.remove()

            for c in range(3):
                axes[r, c].axis("off")

        fig.suptitle("Grad-CAM: UNet Middle Block Attention", fontsize=14, fontweight="bold")
        fig.tight_layout()
        out = paper_dir / "gradcam_unet_attention.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"[Step] Saved {out.name}")

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------

    @staticmethod
    def _find_image(directory: Path, stem: str) -> Optional[Path]:
        for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"]:
            p = directory / (stem + ext)
            if p.exists():
                return p
        return None

    @staticmethod
    def _load_tensor(path: Path, image_size: int = 256) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Lambda(lambda x: (x - 0.5) * 2.0),
        ])
        return transform(img)

    @staticmethod
    def _load_tensor_from_pil(pil_img: Image.Image, image_size: int = 256) -> torch.Tensor:
        """Convert PIL Image to normalized tensor in [-1, 1]."""
        if pil_img.size != (image_size, image_size):
            pil_img = pil_img.resize((image_size, image_size), Image.LANCZOS)
        transform = T.Compose([
            T.ToTensor(),
            T.Lambda(lambda x: (x - 0.5) * 2.0),
        ])
        return transform(pil_img)

    def _load_image_pair(
        self, data_dir: Path, filename: str, image_size: int = 256
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Load (corrupted, clean) tensor pair, handling both directory formats.

        Supports:
        - New format: ``data_dir/images/<filename>`` with on-the-fly corruption
        - Old format: ``data_dir/A/<stem>.png`` and ``data_dir/B/<stem>.png``

        Args:
            data_dir: Split-level directory.
            filename: Image filename (e.g. ``iris_001.png``).
            image_size: Square output size.

        Returns:
            ``(corrupted_t, clean_t)`` each ``(C, H, W)`` in ``[-1, 1]``,
            or ``None`` if the image cannot be found.
        """
        images_dir = data_dir / "images"
        if images_dir.exists():
            # ── New format: flat images/ with on-the-fly corruption ───────
            img_path = images_dir / filename
            if not img_path.exists():
                stem = Path(filename).stem
                for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"):
                    candidate = images_dir / (stem + ext)
                    if candidate.exists():
                        img_path = candidate
                        break
                else:
                    logger.warning(f"Image not found in {images_dir}: {filename}")
                    return None

            # Load corruption config (cached on first use)
            if not hasattr(self, "_corruption_config"):
                cfg_path = (
                    PROJECT_ROOT / "iris_bbdm_pad" / "data"
                    / "bonafide_pairs" / "dataset_config.json"
                )
                if cfg_path.exists():
                    with open(cfg_path) as _f:
                        self._corruption_config = json.load(_f)["corruption_config"]
                else:
                    self._corruption_config = None

            # Load clean image
            clean_pil = Image.open(img_path).convert("RGB")
            if clean_pil.size != (image_size, image_size):
                clean_pil = clean_pil.resize((image_size, image_size), Image.LANCZOS)
            clean_arr = np.array(clean_pil)

            # Apply deterministic corruption (seeded by filename stem)
            data_pkg = str(PROJECT_ROOT / "iris_bbdm_pad" / "data")
            if data_pkg not in sys.path:
                sys.path.insert(0, data_pkg)
            from corruption import apply_corruption_pipeline, filename_to_seed  # noqa: PLC0415

            seed = filename_to_seed(Path(img_path.name).stem)
            kw: Dict = {"target_size": image_size}
            if self._corruption_config:
                cfg = self._corruption_config
                kw.update({
                    "noise_sigma_range": tuple(cfg["noise_sigma_range"]),
                    "blur_kernel_choices": tuple(cfg["blur_kernel_choices"]),
                    "blur_sigma_range": tuple(cfg["blur_sigma_range"]),
                    "downscale_range": tuple(cfg["downscale_range"]),
                })
            corrupted_arr = apply_corruption_pipeline(clean_arr, seed=seed, **kw)

            corrupted_t = self._load_tensor_from_pil(
                Image.fromarray(corrupted_arr.astype(np.uint8)), image_size
            )
            clean_t = self._load_tensor_from_pil(
                Image.fromarray(clean_arr), image_size
            )
            return corrupted_t, clean_t

        else:
            # ── Old format: A/ and B/ subdirectories ─────────────────────
            stem = Path(filename).stem
            a_path = self._find_image(data_dir / "A", stem)
            b_path = self._find_image(data_dir / "B", stem)
            if a_path is None or b_path is None:
                return None
            return self._load_tensor(a_path), self._load_tensor(b_path)

    @staticmethod
    def _tensor_to_np(t: torch.Tensor) -> np.ndarray:
        """Convert [-1,1] tensor (C,H,W) to uint8 numpy (H,W,3)."""
        img = ((t.clamp(-1, 1) + 1) / 2 * 255).byte().permute(1, 2, 0).numpy()
        return img


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BBDM anomaly detector for iris PAD — computes PAD scores."
    )
    parser.add_argument("--config", type=str,
                        default="iris_bbdm_pad/configs/bbdm_iris_bonafide.yaml",
                        help="BBDM YAML config path.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained LBBDM checkpoint (.pth).")
    parser.add_argument("--test_dir", type=str,
                        default="iris_bbdm_pad/data/evaluation_sets/test/",
                        help="Test set directory with A/, B/, labels.csv.")
    parser.add_argument("--output_dir", type=str,
                        default="iris_bbdm_pad/results/",
                        help="Output directory for scores and visualizations.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="'cuda' or 'cpu'.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for scoring.")
    parser.add_argument("--trajectory_timesteps", type=int, nargs="+",
                        default=[100, 250, 500, 750],
                        help="Timesteps for trajectory stability scoring.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from scoring checkpoint if it exists.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    vis_dir = output_dir / "phase2_visualizations"
    paper_dir = vis_dir / "paper"

    detector = BBDMAnomalyDetector(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        trajectory_timesteps=args.trajectory_timesteps,
    )

    test_dir = PROJECT_ROOT / args.test_dir
    output_csv = output_dir / "test_pad_scores.csv"
    ckpt_file = PROJECT_ROOT / "iris_bbdm_pad" / "checkpoints" / "scoring_checkpoint.json"

    logger.info(f"[Step 1/4] Scoring test set: {test_dir}")
    scores_df = detector.process_dataset(
        data_dir=test_dir,
        output_csv=output_csv,
        batch_size=args.batch_size,
        resume=args.resume,
        checkpoint_path=ckpt_file,
    )

    logger.info("[Step 2/4] Saving reconstruction grids...")
    detector.save_reconstruction_grids(test_dir, scores_df, vis_dir)

    logger.info("[Step 3/4] Saving trajectory visualizations...")
    detector.save_trajectory_visualizations(test_dir, scores_df, vis_dir)
    detector.save_score_scatter(scores_df, vis_dir)

    logger.info("[Step 4/4] Saving paper figures...")
    detector.save_paper_figures(test_dir, scores_df, paper_dir)

    n_bf = (scores_df["label"] == "bonafide").sum()
    n_atk = (scores_df["label"] == "attack").sum()
    logger.info(
        f"\nScoring complete.\n"
        f"  Test: {n_bf} bonafide, {n_atk} attack\n"
        f"  Scores: {output_csv}\n"
        f"  Visualizations: {vis_dir}"
    )
