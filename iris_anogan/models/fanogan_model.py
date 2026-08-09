"""
f-AnoGAN model components for iris PAD.

Architecture operates in VQGAN latent space (3 × 64 × 64), exactly like
iris_td operates DDPM in the same space.  The frozen VQ-f4 encoder compresses
256×256 iris images to 64×64 latents; the GAN and Encoder work entirely in
that compressed space — no BBDM files are modified.

Components
----------
Generator   : z (z_dim,) → (latent_channels, 64, 64)
Discriminator: (latent_channels, 64, 64) → scalar + intermediate features
Encoder      : (latent_channels, 64, 64) → z (z_dim,)

GAN training  (Stage 1) : WGAN-GP on bonafide VQGAN latents
Encoder training (Stage 2): "ziz" scheme — E trained to invert G,
    loss = ||E(G(z)) − z||²  +  κ · ||f_D(G(z)) − f_D(G(E(G(z))))||²

References
----------
Schlegl et al., "f-AnoGAN: Fast unsupervised anomaly detection with
generative adversarial networks," MedIA 2019.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Spectral norm helpers
# ---------------------------------------------------------------------------

def _sn(layer: nn.Module) -> nn.Module:
    return nn.utils.spectral_norm(layer)


# ---------------------------------------------------------------------------
# Generator  z → latent (3, 64, 64)
# ---------------------------------------------------------------------------

class Generator(nn.Module):
    """DCGAN-style generator that maps z → VQ-latent (C, 64, 64).

    Architecture mirrors the BBDM/iris_td latent resolution:
      z (z_dim,) → FC → 4×4 feature map → ×4 upsample → 64×64
    """

    def __init__(self, z_dim: int = 128, latent_channels: int = 3, base_channels: int = 512):
        super().__init__()
        self.z_dim = z_dim
        C = base_channels   # 512

        # Project z to (C, 4, 4) feature volume
        self.fc = nn.Linear(z_dim, C * 4 * 4)

        # 4→8→16→32→64  (4 upsampling stages)
        self.deconv = nn.Sequential(
            # (C, 4, 4) → (C//2, 8, 8)
            nn.ConvTranspose2d(C,      C // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(C // 2),
            nn.ReLU(inplace=True),
            # (C//2, 8, 8) → (C//4, 16, 16)
            nn.ConvTranspose2d(C // 2, C // 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(C // 4),
            nn.ReLU(inplace=True),
            # (C//4, 16, 16) → (C//8, 32, 32)
            nn.ConvTranspose2d(C // 4, C // 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(C // 8),
            nn.ReLU(inplace=True),
            # (C//8, 32, 32) → (latent_channels, 64, 64)
            nn.ConvTranspose2d(C // 8, latent_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(z.size(0), -1, 4, 4)
        return self.deconv(h)


# ---------------------------------------------------------------------------
# Discriminator  latent (C, 64, 64) → scalar  +  intermediate features
# ---------------------------------------------------------------------------

class Discriminator(nn.Module):
    """DCGAN-style discriminator with spectral norm for WGAN-GP.

    Returns both the scalar score and the intermediate feature map
    (used for discriminator feature matching loss in Encoder training).
    """

    def __init__(self, latent_channels: int = 3, base_channels: int = 64):
        super().__init__()
        C = base_channels   # 64

        # Feature extractor (returns (C*8, 4, 4) tensor)
        self.features = nn.Sequential(
            # (3, 64, 64) → (C, 32, 32)
            _sn(nn.Conv2d(latent_channels, C,     4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # → (C*2, 16, 16)
            _sn(nn.Conv2d(C,     C * 2, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # → (C*4, 8, 8)
            _sn(nn.Conv2d(C * 2, C * 4, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # → (C*8, 4, 4)
            _sn(nn.Conv2d(C * 4, C * 8, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Scalar head
        self.score = _sn(nn.Conv2d(C * 8, 1, 4, 1, 0, bias=False))

    def forward(self, x: torch.Tensor):
        feat = self.features(x)
        score = self.score(feat).view(x.size(0))
        return score, feat


# ---------------------------------------------------------------------------
# Encoder  latent (C, 64, 64) → z (z_dim,)
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """Encoder that maps VQ-latent back to GAN z-space.

    Trained in Stage 2 with the "ziz" scheme:
      1. Sample z ~ N(0,I), generate G(z)
      2. Encode z_hat = E(G(z))
      3. Minimize ||z_hat − z||² + κ·feature_loss
    """

    def __init__(self, z_dim: int = 128, latent_channels: int = 3, base_channels: int = 64):
        super().__init__()
        C = base_channels

        self.conv = nn.Sequential(
            # (3, 64, 64) → (C, 32, 32)
            _sn(nn.Conv2d(latent_channels, C,     4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # → (C*2, 16, 16)
            _sn(nn.Conv2d(C,     C * 2, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # → (C*4, 8, 8)
            _sn(nn.Conv2d(C * 2, C * 4, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # → (C*8, 4, 4)
            _sn(nn.Conv2d(C * 4, C * 8, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc = nn.Linear(C * 8 * 4 * 4, z_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x).view(x.size(0), -1)
        return self.fc(h)


# ---------------------------------------------------------------------------
# WGAN-GP gradient penalty
# ---------------------------------------------------------------------------

def gradient_penalty(
    disc: Discriminator,
    real: torch.Tensor,
    fake: torch.Tensor,
    device: torch.device,
    lambda_gp: float = 10.0,
) -> torch.Tensor:
    """Compute WGAN-GP gradient penalty between real and fake latents."""
    B = real.size(0)
    alpha = torch.rand(B, 1, 1, 1, device=device)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)

    score_interp, _ = disc(interpolated)
    grad = torch.autograd.grad(
        outputs=score_interp,
        inputs=interpolated,
        grad_outputs=torch.ones_like(score_interp),
        create_graph=True,
        retain_graph=True,
    )[0]
    grad_norm = grad.view(B, -1).norm(2, dim=1)
    return lambda_gp * ((grad_norm - 1.0) ** 2).mean()


# ---------------------------------------------------------------------------
# f-AnoGAN anomaly score (inference only)
# ---------------------------------------------------------------------------

def anomaly_score(
    disc: Discriminator,
    encoder: Encoder,
    generator: Generator,
    z_latent: torch.Tensor,
    kappa: float = 1.0,
) -> torch.Tensor:
    """Compute f-AnoGAN anomaly score for a batch of VQ-latents.

    Score = residual_loss + κ · feature_loss  (per image, shape (B,))

    residual_loss : ||z_latent − G(E(z_latent))||² (MSE per image)
    feature_loss  : ||f_D(z_latent) − f_D(G(E(z_latent)))||² (per image)
    """
    z_hat = encoder(z_latent)
    z_recon = generator(z_hat)

    residual = F.mse_loss(z_recon, z_latent, reduction="none").mean(dim=[1, 2, 3])

    _, feat_real = disc(z_latent)
    _, feat_recon = disc(z_recon)
    feature = F.mse_loss(feat_recon, feat_real.detach(), reduction="none").mean(dim=[1, 2, 3])

    return residual + kappa * feature
