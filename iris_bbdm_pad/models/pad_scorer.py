"""
PAD score computation for iris anomaly detection.

Three complementary scoring methods:

1. RECONSTRUCTION ERROR (MSE + LPIPS):
   Compare BBDM output vs clean original.
   Genuine: low error (faithful reconstruction). Attack: high error (distorted toward bonafide).

2. DENOISING TRAJECTORY STABILITY:
   Denoise the same image from multiple different noise levels (timesteps).
   Genuine: all reconstructions look similar (model is confident) -> low variance.
   Attack: reconstructions vary wildly (model is uncertain) -> high variance.

3. COMBINED SCORE:
   Weighted sum: w1 * recon_error + w2 * trajectory_stability
   Weights optimized on validation set via grid search.

Higher score = more likely attack (anomaly) for ALL methods.
"""

import logging
from typing import Dict, List, Optional

import lpips
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class PADScorer:
    """Multi-method PAD scorer for iris anomaly detection.

    Args:
        device: 'cuda' or 'cpu'.
        alpha: Weight for MSE within reconstruction error.
        beta: Weight for LPIPS within reconstruction error.
        w_recon: Weight for reconstruction error in combined score.
        w_trajectory: Weight for trajectory stability in combined score.
    """

    def __init__(
        self,
        device: str = "cuda",
        alpha: float = 1.0,
        beta: float = 1.0,
        w_recon: float = 0.5,
        w_trajectory: float = 0.5,
    ) -> None:
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.w_recon = w_recon
        self.w_trajectory = w_trajectory

        logger.info(f"Loading LPIPS (net=alex) on {device}...")
        self.lpips_fn = lpips.LPIPS(net="alex").to(device)
        self.lpips_fn.eval()
        for p in self.lpips_fn.parameters():
            p.requires_grad_(False)

    # -----------------------------------------------------------------------
    # Method 1: Reconstruction Error
    # -----------------------------------------------------------------------

    def compute_mse(
        self, output: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute per-sample MSE.

        Args:
            output: BBDM reconstruction (B, C, H, W).
            target: Clean original (B, C, H, W).

        Returns:
            (B,) per-sample MSE values.
        """
        return F.mse_loss(output, target, reduction="none").mean(dim=[1, 2, 3])

    def compute_lpips(
        self, output: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute per-sample LPIPS distance.

        Args:
            output: BBDM reconstruction (B, C, H, W) in [-1, 1].
            target: Clean original (B, C, H, W) in [-1, 1].

        Returns:
            (B,) per-sample LPIPS values.
        """
        with torch.no_grad():
            dist = self.lpips_fn(
                output.to(self.device), target.to(self.device)
            )
            # lpips returns (B, 1, 1, 1) — squeeze to (B,)
            return dist.view(-1).cpu()

    def compute_reconstruction_score(
        self, output: torch.Tensor, target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute reconstruction error score: alpha*MSE + beta*LPIPS.

        Args:
            output: BBDM reconstruction (B, C, H, W) in [-1, 1].
            target: Clean original (B, C, H, W) in [-1, 1].

        Returns:
            Dict with keys 'mse', 'lpips', 'recon_score', each (B,).
        """
        mse = self.compute_mse(output, target)
        lp = self.compute_lpips(output, target)
        recon_score = self.alpha * mse + self.beta * lp
        return {"mse": mse, "lpips": lp, "recon_score": recon_score}

    # -----------------------------------------------------------------------
    # Method 2: Denoising Trajectory Stability
    # -----------------------------------------------------------------------

    def compute_trajectory_stability(
        self, reconstructions: List[torch.Tensor]
    ) -> torch.Tensor:
        """Measure variance across multiple reconstructions from different noise levels.

        For each sample, stacks K reconstructions and computes mean pairwise MSE
        relative to the mean image. High value = unstable denoising = likely attack.

        Args:
            reconstructions: List of K tensors, each (B, C, H, W).

        Returns:
            (B,) trajectory stability score (mean variance across K reconstructions).
        """
        if len(reconstructions) < 2:
            b = reconstructions[0].shape[0]
            return torch.zeros(b, device=reconstructions[0].device)

        # Stack to (K, B, C, H, W)
        stacked = torch.stack(reconstructions, dim=0)  # (K, B, C, H, W)
        k = stacked.shape[0]

        # Mean image across K: (B, C, H, W)
        mean_img = stacked.mean(dim=0)

        # Variance: mean MSE(each_recon, mean_img) over K and spatial dims
        variance = torch.zeros(stacked.shape[1], device=stacked.device)  # (B,)
        for i in range(k):
            diff = (stacked[i] - mean_img) ** 2  # (B, C, H, W)
            variance += diff.mean(dim=[1, 2, 3])  # (B,)
        variance /= k

        return variance  # (B,)

    # -----------------------------------------------------------------------
    # Method 3: Combined Score
    # -----------------------------------------------------------------------

    def compute_combined_score(
        self,
        recon_score: torch.Tensor,
        trajectory_score: torch.Tensor,
    ) -> torch.Tensor:
        """Compute weighted combination of reconstruction and trajectory scores.

        Both scores are min-max normalized to [0, 1] within the batch before
        combining, so they contribute equally regardless of natural scale.

        Args:
            recon_score: (B,) reconstruction error scores.
            trajectory_score: (B,) trajectory stability scores.

        Returns:
            (B,) combined score = w_recon * norm(recon) + w_trajectory * norm(traj).
        """
        def norm(x: torch.Tensor) -> torch.Tensor:
            mn, mx = x.min(), x.max()
            if (mx - mn).abs() < 1e-8:
                return torch.zeros_like(x)
            return (x - mn) / (mx - mn)

        combined = self.w_recon * norm(recon_score) + self.w_trajectory * norm(trajectory_score)
        return combined

    # -----------------------------------------------------------------------
    # Full batch scoring
    # -----------------------------------------------------------------------

    def score_batch(
        self,
        bbdm_output: torch.Tensor,
        clean_original: torch.Tensor,
        trajectory_reconstructions: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, np.ndarray]:
        """Compute ALL scores for a batch.

        Args:
            bbdm_output: Primary reconstruction from standard denoising (B, C, H, W).
            clean_original: Clean original images (B, C, H, W).
            trajectory_reconstructions: Optional list of K tensors, each (B, C, H, W),
                from denoising at different noise levels. If None, trajectory and
                combined scores are not computed.

        Returns:
            Dict with numpy arrays:
                'mse', 'lpips', 'recon_score', and optionally
                'trajectory_score', 'combined_score'.
        """
        with torch.no_grad():
            recon_dict = self.compute_reconstruction_score(bbdm_output, clean_original)
            result = {k: v.cpu().numpy() for k, v in recon_dict.items()}

            if trajectory_reconstructions is not None:
                traj = self.compute_trajectory_stability(trajectory_reconstructions)
                combined = self.compute_combined_score(
                    recon_dict["recon_score"], traj
                )
                result["trajectory_score"] = traj.cpu().numpy()
                result["combined_score"] = combined.cpu().numpy()

        return result

    # -----------------------------------------------------------------------
    # Weight optimization
    # -----------------------------------------------------------------------

    @staticmethod
    def optimize_weights(val_scores_csv: str) -> Dict[str, float]:
        """Grid search for optimal w_recon and w_trajectory on validation set.

        Tests all combinations: w_recon in [0.0, 0.1, ..., 1.0], w_trajectory = 1 - w_recon.
        Selects weights that minimize ACER on validation set.

        Args:
            val_scores_csv: Path to validation scores CSV with columns including
                'recon_score', 'trajectory_score', 'label' (bonafide/attack).

        Returns:
            Dict with 'w_recon', 'w_trajectory', 'best_acer'.
        """
        df = pd.read_csv(val_scores_csv)
        labels = (df["label"] == "attack").astype(int).values
        recon = df["recon_score"].values.astype(np.float32)
        traj = df["trajectory_score"].values.astype(np.float32)

        # Normalize each component to [0, 1] globally
        def global_norm(x: np.ndarray) -> np.ndarray:
            mn, mx = x.min(), x.max()
            if abs(mx - mn) < 1e-8:
                return np.zeros_like(x)
            return (x - mn) / (mx - mn)

        recon_n = global_norm(recon)
        traj_n = global_norm(traj)

        best_acer = float("inf")
        best_w_recon = 0.5

        weights = np.linspace(0.0, 1.0, 11)
        for w_r in weights:
            w_t = 1.0 - w_r
            combined = w_r * recon_n + w_t * traj_n

            # Sweep 1000 thresholds
            thresholds = np.linspace(combined.min(), combined.max(), 1000)
            for thr in thresholds:
                preds = (combined > thr).astype(int)
                tp = np.sum((preds == 1) & (labels == 1))
                fp = np.sum((preds == 1) & (labels == 0))
                tn = np.sum((preds == 0) & (labels == 0))
                fn = np.sum((preds == 0) & (labels == 1))
                apcer = fp / max(tn + fp, 1)
                bpcer = fn / max(tp + fn, 1)
                acer = (apcer + bpcer) / 2.0
                if acer < best_acer:
                    best_acer = acer
                    best_w_recon = w_r

        return {
            "w_recon": float(best_w_recon),
            "w_trajectory": float(1.0 - best_w_recon),
            "best_acer": float(best_acer),
        }
