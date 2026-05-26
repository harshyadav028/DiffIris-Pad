"""
LatentDDPM — Standard DDPM running in VQ-f4 latent space.

Encodes 256×256 images to 64×64 latents via a frozen VQ-f4 VQGAN,
runs the standard linear-beta DDPM noise schedule in latent space,
and decodes reconstructions back to pixel space.

Used as the fair comparison baseline against BBDM in the iris PAD ablation.
"""

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Resolve BBDM/ root so we can import UNetModel and VQModel without modifying
# any BBDM files.  iris_td/ models do NOT live inside BBDM/ — we add BBDM/
# to sys.path only to resolve their internal relative imports.
_bbdm_root = str(Path(__file__).resolve().parents[2] / "BBDM")
if _bbdm_root not in sys.path:
    sys.path.insert(0, _bbdm_root)

from model.BrownianBridge.base.modules.diffusionmodules.openaimodel import UNetModel  # noqa: E402
from model.VQGAN.vqgan import VQModel  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dict_to_ns(d: dict) -> argparse.Namespace:
    """Recursively convert nested dict to argparse.Namespace (mirrors BBDM utils.dict2namespace)."""
    ns = argparse.Namespace()
    for k, v in d.items():
        setattr(ns, k, _dict_to_ns(v) if isinstance(v, dict) else v)
    return ns


def _make_beta_schedule(
    schedule: str,
    num_timesteps: int,
    beta_start: float,
    beta_end: float,
) -> torch.Tensor:
    """Return a 1-D tensor of betas for the requested schedule."""
    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float64)
    if schedule == "cosine":
        # Nichol & Dhariwal 2021
        s = 0.008
        steps = num_timesteps + 1
        x = torch.linspace(0, num_timesteps, steps)
        alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return betas.clamp(0, 0.9999).float()
    raise ValueError(f"Unknown beta schedule: {schedule}")


def _extract(a: torch.Tensor, t: torch.Tensor, x_shape: Tuple) -> torch.Tensor:
    """Gather timestep-indexed values and broadcast to x_shape."""
    b = t.shape[0]
    out = a.gather(-1, t.to(a.device))
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def _disabled_train(self, mode: bool = True):  # noqa: ANN001
    """Prevent VQGAN from switching modes."""
    return self


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class LatentDDPM(nn.Module):
    """Standard DDPM operating in VQ-f4 latent space.

    Args:
        unet_config:  Dict of UNet kwargs (image_size, in_channels, …).
        vqgan_config: Dict of VQModel kwargs (ckpt_path, embed_dim, ddconfig, …).
        ddpm_config:  Dict with num_timesteps, beta_schedule, beta_start, beta_end.
    """

    def __init__(
        self,
        unet_config: dict,
        vqgan_config: dict,
        ddpm_config: dict,
    ) -> None:
        super().__init__()

        # ---- VQGAN (frozen) ----
        vq_ns = _dict_to_ns(vqgan_config)
        self.vqgan = VQModel(**vars(vq_ns)).eval()
        self.vqgan.train = _disabled_train.__get__(self.vqgan, type(self.vqgan))  # type: ignore[assignment]
        for p in self.vqgan.parameters():
            p.requires_grad_(False)
        logger.info("VQGAN loaded and frozen.")

        # ---- UNet ----
        self.unet = UNetModel(**unet_config)

        # ---- DDPM schedule (double precision, stored as float32 buffers) ----
        T = ddpm_config.get("num_timesteps", 1000)
        self.num_timesteps = T

        betas = _make_beta_schedule(
            schedule=ddpm_config.get("beta_schedule", "linear"),
            num_timesteps=T,
            beta_start=ddpm_config.get("beta_start", 1e-4),
            beta_end=ddpm_config.get("beta_end", 0.02),
        ).float()

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # Forward diffusion coefficients
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt())

        # Reverse diffusion coefficients (DDPM)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            posterior_variance.clamp(min=1e-20).log(),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * alphas_cumprod_prev.sqrt() / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * alphas.sqrt() / (1.0 - alphas_cumprod),
        )

    # -----------------------------------------------------------------------
    # Encode / Decode
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def encode(self, x: torch.Tensor, chunk_size: int = 8) -> torch.Tensor:
        """Encode pixel images to VQGAN latents in small chunks.

        Chunking mirrors decode() to keep peak memory bounded when
        the GPU is partially occupied by other processes.

        Args:
            x: (B, 3, 256, 256) in [-1, 1].
            chunk_size: images encoded at once (default=8)

        Returns:
            z: (B, 3, 64, 64) latent tensor (pre-quantisation, post quant_conv).
        """
        results = []
        for i in range(0, x.shape[0], chunk_size):
            x_chunk = x[i : i + chunk_size]
            z_chunk = self.vqgan.encoder(x_chunk)
            z_chunk = self.vqgan.quant_conv(z_chunk)
            results.append(z_chunk)
        return torch.cat(results, dim=0)

    @torch.no_grad()
    def decode(self, z: torch.Tensor,
               chunk_size: int = 8) -> torch.Tensor:
        """
        Decode latents to pixels in small chunks.
        VQGAN quantize needs large memory burst per batch.
        Chunking to 8 keeps peak memory safe regardless
        of inference batch size.

        Args:
            z: (B, 3, 64, 64)
            chunk_size: images decoded at once (default=8)

        Returns:
            x_rec: (B, 3, 256, 256)
        """
        results = []
        for i in range(0, z.shape[0], chunk_size):
            z_chunk   = z[i : i + chunk_size]
            z_q, _, _ = self.vqgan.quantize(z_chunk)
            x_chunk   = self.vqgan.decode(z_q)
            results.append(x_chunk)
        return torch.cat(results, dim=0)

    # -----------------------------------------------------------------------
    # Forward diffusion
    # -----------------------------------------------------------------------

    def q_sample(
        self,
        z_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add noise at timestep t (reparameterisation trick).

        Args:
            z_0: (B, C, H, W) clean latent.
            t:   (B,) integer timesteps.
            noise: Optional pre-sampled Gaussian noise (same shape as z_0).

        Returns:
            z_t:  (B, C, H, W) noisy latent.
            noise: (B, C, H, W) the added noise.
        """
        if noise is None:
            noise = torch.randn_like(z_0)
        sqrt_ac = _extract(self.sqrt_alphas_cumprod, t, z_0.shape)
        sqrt_omc = _extract(self.sqrt_one_minus_alphas_cumprod, t, z_0.shape)
        z_t = sqrt_ac * z_0 + sqrt_omc * noise
        return z_t, noise

    # -----------------------------------------------------------------------
    # UNet (noise prediction)
    # -----------------------------------------------------------------------

    def predict_noise(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict the noise added at timestep t.

        Args:
            z_t:     (B, C, H, W) noisy latent.
            t:       (B,) integer timesteps.
            context: Optional conditioning (None for unconditional).

        Returns:
            eps_pred: (B, C, H, W) predicted noise.
        """
        return self.unet(z_t, t, context=context)

    # -----------------------------------------------------------------------
    # Reverse diffusion — DDPM stochastic step
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def p_sample(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        """One DDPM reverse step: z_t → z_{t-1}.

        Args:
            z_t:  (B, C, H, W) noisy latent at timestep t.
            t:    (B,) integer timesteps (scalar tensor works too).

        Returns:
            z_prev: (B, C, H, W) latent at t-1.
        """
        if t.dim() == 0:
            t = t.expand(z_t.shape[0])

        eps = self.predict_noise(z_t, t)
        # Reconstruct z_0 estimate
        sqrt_ac = _extract(self.sqrt_alphas_cumprod, t, z_t.shape)
        sqrt_omc = _extract(self.sqrt_one_minus_alphas_cumprod, t, z_t.shape)
        z0_pred = (z_t - sqrt_omc * eps) / sqrt_ac

        if clip_denoised:
            z0_pred = z0_pred.clamp(-1.0, 1.0)

        # Posterior mean
        c1 = _extract(self.posterior_mean_coef1, t, z_t.shape)
        c2 = _extract(self.posterior_mean_coef2, t, z_t.shape)
        mean = c1 * z0_pred + c2 * z_t

        # Posterior variance (zero noise at t=0)
        log_var = _extract(self.posterior_log_variance_clipped, t, z_t.shape)
        noise = torch.randn_like(z_t)
        nonzero = (t > 0).float().view(-1, *([1] * (z_t.dim() - 1)))
        z_prev = mean + nonzero * (0.5 * log_var).exp() * noise
        return z_prev

    # -----------------------------------------------------------------------
    # Reverse diffusion — DDIM deterministic step
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def ddim_sample(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        t_prev: Optional[torch.Tensor] = None,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """One DDIM reverse step (Song et al. 2020).

        Args:
            z_t:    (B, C, H, W) noisy latent at timestep t.
            t:      (B,) or scalar integer timestep.
            t_prev: (B,) or scalar timestep for previous step (default: t-1).
            eta:    Stochasticity (0 = deterministic DDIM, 1 = DDPM).

        Returns:
            z_prev: (B, C, H, W) latent at t_prev.
        """
        if t.dim() == 0:
            t = t.expand(z_t.shape[0])
        if t_prev is None:
            t_prev = (t - 1).clamp(min=0)
        elif t_prev.dim() == 0:
            t_prev = t_prev.expand(z_t.shape[0])

        eps = self.predict_noise(z_t, t)

        ac_t = _extract(self.alphas_cumprod, t, z_t.shape)
        ac_prev = _extract(self.alphas_cumprod, t_prev, z_t.shape)

        # Predict x0
        sqrt_ac_t = ac_t.sqrt()
        z0_pred = (z_t - (1.0 - ac_t).sqrt() * eps) / sqrt_ac_t

        # Direction pointing to z_t
        sigma = eta * ((1.0 - ac_prev) / (1.0 - ac_t) * (1.0 - ac_t / ac_prev)).sqrt()
        dir_zt = (1.0 - ac_prev - sigma ** 2).clamp(min=0).sqrt() * eps

        noise = sigma * torch.randn_like(z_t) if eta > 0 else 0.0
        z_prev = ac_prev.sqrt() * z0_pred + dir_zt + noise
        return z_prev

    # -----------------------------------------------------------------------
    # Full DDPM / DDIM sampling loop
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        shape: Tuple,
        device: torch.device,
        use_ddim: bool = True,
        ddim_steps: int = 100,
        eta: float = 0.0,
        clip_denoised: bool = True,
        return_latent: bool = False,
    ) -> torch.Tensor:
        """Unconditional generation: noise → pixel image.

        Args:
            shape:  (B, C, H, W) latent shape (e.g. (1, 3, 64, 64)).
            device: Target device.
            use_ddim: If True, use DDIM; otherwise full DDPM.
            ddim_steps: Number of reverse steps when using DDIM.
            eta:    DDIM stochasticity.
            clip_denoised: Clip z0 estimates to [-1, 1].
            return_latent: If True, also return the final latent z_0.

        Returns:
            x_rec: (B, 3, 256, 256) reconstructed image, or
                   (x_rec, z_0) if return_latent is True.
        """
        z = torch.randn(shape, device=device)

        if use_ddim:
            timesteps = torch.linspace(
                self.num_timesteps - 1, 0, ddim_steps + 1, dtype=torch.long, device=device
            )
            for i in range(len(timesteps) - 1):
                t = timesteps[i].expand(shape[0])
                t_prev = timesteps[i + 1].expand(shape[0])
                z = self.ddim_sample(z, t, t_prev=t_prev, eta=eta)
        else:
            for t_val in reversed(range(self.num_timesteps)):
                t = torch.full((shape[0],), t_val, device=device, dtype=torch.long)
                z = self.p_sample(z, t, clip_denoised=clip_denoised)

        x_rec = self.decode(z)
        if return_latent:
            return x_rec, z
        return x_rec
