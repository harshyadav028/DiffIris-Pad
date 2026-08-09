"""
f-AnoGAN two-stage training on bonafide iris images.

Stage 1 — WGAN-GP (GAN on VQGAN latents)
    Trains Generator + Discriminator on VQGAN-encoded bonafide images.
    Checkpoints: iris_anogan/results/fanogan_run1/stage1/checkpoint/

Stage 2 — Encoder (ziz scheme, Schlegl et al. 2019)
    Trains Encoder to invert Generator using "z→image→z" loss.
    Requires a completed Stage 1 checkpoint.
    Checkpoints: iris_anogan/results/fanogan_run1/stage2/checkpoint/

The VQGAN is borrowed from BBDM/resources/vq-f4/model.ckpt (frozen, never
modified).  All code lives in iris_anogan/ — no BBDM files are touched.

Usage
-----
  cd ~/Documents/Geetanjali_PhD_IRIS_PAD

  # Stage 1 — GAN
  conda run --no-capture-output -n iris_pad python \\
      iris_anogan/training/train_fanogan_bonafide.py \\
      --config iris_anogan/configs/fanogan_iris.yaml \\
      --stage 1

  # Stage 1 — resume
  conda run --no-capture-output -n iris_pad python \\
      iris_anogan/training/train_fanogan_bonafide.py \\
      --config iris_anogan/configs/fanogan_iris.yaml \\
      --stage 1 --resume auto

  # Stage 2 — Encoder  (requires Stage 1 best checkpoint)
  conda run --no-capture-output -n iris_pad python \\
      iris_anogan/training/train_fanogan_bonafide.py \\
      --config iris_anogan/configs/fanogan_iris.yaml \\
      --stage 2

  # Run inside tmux to survive disconnection
  tmux new -s fanogan_train
  conda run --no-capture-output -n iris_pad python \\
      iris_anogan/training/train_fanogan_bonafide.py \\
      --config iris_anogan/configs/fanogan_iris.yaml --stage 1
  Ctrl+B D  (detach)
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from tqdm import tqdm

# ── Project paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parents[2]
IRIS_ANOGAN    = Path(__file__).resolve().parents[1]
BBDM_ROOT      = PROJECT_ROOT / "BBDM"

for _p in [str(BBDM_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model.VQGAN.vqgan import VQModel  # noqa: E402 — from BBDM (read-only)

# iris_anogan imports
sys.path.insert(0, str(IRIS_ANOGAN))
from models.fanogan_model import Generator, Discriminator, Encoder, gradient_penalty  # noqa: E402
from data.iris_dataset import BonafideImageDataset  # noqa: E402

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# VQGAN loader (identical pattern to iris_td/models/ddpm_model.py)
# ---------------------------------------------------------------------------

def _dict_to_ns(d: dict):
    import argparse as _ap
    ns = _ap.Namespace()
    for k, v in d.items():
        setattr(ns, k, _dict_to_ns(v) if isinstance(v, dict) else v)
    return ns


def load_vqgan(cfg: dict, device: torch.device) -> VQModel:
    """Load frozen VQ-f4 model from BBDM checkpoint."""
    vq_ns = _dict_to_ns(cfg["vqgan"])
    vqgan = VQModel(**vars(vq_ns)).to(device).eval()
    for p in vqgan.parameters():
        p.requires_grad_(False)

    def _disabled_train(self, mode=True):
        return self
    vqgan.train = _disabled_train.__get__(vqgan, type(vqgan))

    log.info(f"VQGAN loaded from {cfg['vqgan']['ckpt_path']} (frozen)")
    return vqgan


@torch.no_grad()
def encode_batch(vqgan: VQModel, x: torch.Tensor) -> torch.Tensor:
    """256×256 pixels → 64×64 VQ-latents (pre-quantisation)."""
    z = vqgan.encoder(x)
    z = vqgan.quant_conv(z)
    return z


@torch.no_grad()
def decode_batch(vqgan: VQModel, z: torch.Tensor) -> torch.Tensor:
    """64×64 VQ-latents → 256×256 pixel images."""
    z_q, _, _ = vqgan.quantize(z)
    return vqgan.decode(z_q)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_stage1(path: Path, gen, disc, opt_g, opt_d, epoch, best_val_loss):
    torch.save({
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "generator": gen.state_dict(),
        "discriminator": disc.state_dict(),
        "opt_g": opt_g.state_dict(),
        "opt_d": opt_d.state_dict(),
    }, path)


def load_stage1(path: Path, gen, disc, opt_g, opt_d, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    gen.load_state_dict(ckpt["generator"])
    disc.load_state_dict(ckpt["discriminator"])
    opt_g.load_state_dict(ckpt["opt_g"])
    opt_d.load_state_dict(ckpt["opt_d"])
    epoch = ckpt.get("epoch", 0)
    best  = ckpt.get("best_val_loss", float("inf"))
    log.info(f"Stage 1 resumed from epoch {epoch}")
    return epoch, best


def save_stage2(path: Path, enc, opt_e, epoch, best_val_loss):
    torch.save({
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "encoder": enc.state_dict(),
        "opt_e": opt_e.state_dict(),
    }, path)


def load_stage2(path: Path, enc, opt_e, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    enc.load_state_dict(ckpt["encoder"])
    opt_e.load_state_dict(ckpt["opt_e"])
    epoch = ckpt.get("epoch", 0)
    best  = ckpt.get("best_val_loss", float("inf"))
    log.info(f"Stage 2 resumed from epoch {epoch}")
    return epoch, best


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def make_loaders(cfg: dict):
    root = cfg["data"]["bonafide_pairs_root"]
    train_ds = BonafideImageDataset(root, stage="train")
    val_ds   = BonafideImageDataset(root, stage="val")

    n_cpu = min(8, os.cpu_count() or 4)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["data"]["train"]["batch_size"],
        shuffle=True,
        num_workers=n_cpu,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["data"]["val"]["batch_size"],
        shuffle=False,
        num_workers=n_cpu,
        pin_memory=True,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Stage 1 — WGAN-GP GAN training
# ---------------------------------------------------------------------------

def train_stage1(cfg: dict, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Stage 1 — WGAN-GP | device={device}")
    if device.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")

    mc   = cfg["model"]
    s1   = cfg["stage1"]
    gp_w = mc["lambda_gp"]

    result_dir = Path(cfg["result_path"]) / "stage1"
    ckpt_dir   = result_dir / "checkpoint"
    log_dir    = result_dir / "log"
    sample_dir = result_dir / "samples"
    for d in [ckpt_dir, log_dir, sample_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Models
    vqgan = load_vqgan(cfg, device)
    gen   = Generator(mc["z_dim"], mc["latent_channels"], mc["gen_base_channels"]).to(device)
    disc  = Discriminator(mc["latent_channels"], mc["disc_base_channels"]).to(device)

    log.info(f"Generator params:     {sum(p.numel() for p in gen.parameters()):,}")
    log.info(f"Discriminator params: {sum(p.numel() for p in disc.parameters()):,}")

    opt_g = torch.optim.Adam(gen.parameters(),  lr=s1["lr_gen"],  betas=(s1["beta1"], s1["beta2"]))
    opt_d = torch.optim.Adam(disc.parameters(), lr=s1["lr_disc"], betas=(s1["beta1"], s1["beta2"]))

    start_epoch   = 0
    best_val_loss = float("inf")

    # Resume
    last_ckpt = ckpt_dir / "last.pth"
    resume_path = args.resume
    if resume_path == "auto":
        resume_path = str(last_ckpt) if last_ckpt.exists() else None
    if resume_path and Path(resume_path).exists():
        start_epoch, best_val_loss = load_stage1(
            Path(resume_path), gen, disc, opt_g, opt_d, device
        )
        start_epoch += 1

    train_loader, val_loader = make_loaders(cfg)
    writer = SummaryWriter(str(log_dir))

    # Fixed z for sample grids
    fixed_z = torch.randn(s1["n_sample_images"], mc["z_dim"], device=device)

    n_critic  = s1["n_critic"]
    n_epochs  = s1["n_epochs"]
    t_start   = time.time()

    log.info(f"Training epochs {start_epoch}–{n_epochs}, n_critic={n_critic}")

    for epoch in range(start_epoch, n_epochs):
        gen.train()
        disc.train()

        d_losses, g_losses, gp_vals = [], [], []

        pbar = tqdm(train_loader, desc=f"S1 Epoch {epoch+1}/{n_epochs}", leave=False)
        for step, x_real_px in enumerate(pbar):
            x_real_px = x_real_px.to(device)

            with torch.no_grad():
                x_real = encode_batch(vqgan, x_real_px)   # (B, 3, 64, 64)

            B = x_real.size(0)

            # ── Discriminator update ──────────────────────────────────────────
            z = torch.randn(B, mc["z_dim"], device=device)
            with torch.no_grad():
                x_fake = gen(z)

            score_real, _ = disc(x_real)
            score_fake, _ = disc(x_fake.detach())
            gp = gradient_penalty(disc, x_real, x_fake.detach(), device, gp_w)

            loss_d = score_fake.mean() - score_real.mean() + gp

            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

            d_losses.append(loss_d.item())
            gp_vals.append(gp.item())

            # ── Generator update (every n_critic discriminator steps) ─────────
            if (step + 1) % n_critic == 0:
                z = torch.randn(B, mc["z_dim"], device=device)
                x_fake = gen(z)
                score_fake, _ = disc(x_fake)
                loss_g = -score_fake.mean()

                opt_g.zero_grad()
                loss_g.backward()
                opt_g.step()

                g_losses.append(loss_g.item())
                pbar.set_postfix(d=f"{loss_d.item():.3f}", g=f"{loss_g.item():.3f}")

        avg_d = sum(d_losses) / max(len(d_losses), 1)
        avg_g = sum(g_losses) / max(len(g_losses), 1)
        avg_gp = sum(gp_vals) / max(len(gp_vals), 1)

        writer.add_scalar("stage1/loss_d",  avg_d,  epoch)
        writer.add_scalar("stage1/loss_g",  avg_g,  epoch)
        writer.add_scalar("stage1/grad_pen", avg_gp, epoch)

        log.info(
            f"S1 Epoch {epoch+1}/{n_epochs} | "
            f"D={avg_d:.4f} | G={avg_g:.4f} | GP={avg_gp:.4f}"
        )

        # ── Validation (Wasserstein distance proxy on val bonafide) ───────────
        if (epoch + 1) % s1["val_interval"] == 0:
            gen.eval()
            disc.eval()
            w_vals = []
            with torch.no_grad():
                for x_val_px in val_loader:
                    x_val = encode_batch(vqgan, x_val_px.to(device))
                    z_v   = torch.randn(x_val.size(0), mc["z_dim"], device=device)
                    x_fk  = gen(z_v)
                    sr, _ = disc(x_val)
                    sf, _ = disc(x_fk)
                    w_vals.append((sf.mean() - sr.mean()).item())

            val_w = sum(w_vals) / max(len(w_vals), 1)
            writer.add_scalar("stage1/val_wasserstein", val_w, epoch)
            log.info(f"  val Wasserstein={val_w:.4f}")

            if val_w < best_val_loss:
                best_val_loss = val_w
                best_path     = ckpt_dir / "best.pth"
                save_stage1(best_path, gen, disc, opt_g, opt_d, epoch, best_val_loss)
                log.info(f"  NEW BEST → {best_path.name}")

        # ── Sample grid ───────────────────────────────────────────────────────
        if (epoch + 1) % s1["sample_interval"] == 0:
            gen.eval()
            with torch.no_grad():
                latents = gen(fixed_z)                    # (N, 3, 64, 64)
                samples = decode_batch(vqgan, latents)    # (N, 3, 256, 256)
                samples = (samples.clamp(-1, 1) + 1) / 2
            grid_path = sample_dir / f"epoch_{epoch+1:04d}.png"
            save_image(samples, grid_path, nrow=4)
            log.info(f"  Sample grid → {grid_path.name}")

        # ── Periodic checkpoint ───────────────────────────────────────────────
        if (epoch + 1) % s1["save_interval"] == 0:
            save_stage1(last_ckpt, gen, disc, opt_g, opt_d, epoch, best_val_loss)
            log.info(f"  Checkpoint saved (epoch {epoch+1})")

        elapsed = time.time() - t_start
        eta_s   = (elapsed / max(epoch - start_epoch + 1, 1)) * (n_epochs - epoch - 1)
        log.info(f"  ETA: {eta_s/3600:.1f}h")

    # Final save
    save_stage1(last_ckpt, gen, disc, opt_g, opt_d, epoch, best_val_loss)
    writer.close()
    log.info(f"Stage 1 complete. Best checkpoint: {ckpt_dir / 'best.pth'}")


# ---------------------------------------------------------------------------
# Stage 2 — Encoder training (ziz scheme)
# ---------------------------------------------------------------------------

def train_stage2(cfg: dict, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Stage 2 — Encoder (ziz) | device={device}")

    mc = cfg["model"]
    s2 = cfg["stage2"]

    result_dir = Path(cfg["result_path"])
    s1_ckpt    = result_dir / "stage1" / "checkpoint" / "best.pth"
    if not s1_ckpt.exists():
        # Try last.pth as fallback
        s1_ckpt = result_dir / "stage1" / "checkpoint" / "last.pth"
    if not s1_ckpt.exists():
        raise FileNotFoundError(
            f"Stage 1 checkpoint not found at {s1_ckpt}. "
            "Run --stage 1 first."
        )
    log.info(f"Loading Stage 1 from {s1_ckpt}")

    ckpt_dir   = result_dir / "stage2" / "checkpoint"
    log_dir    = result_dir / "stage2" / "log"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Models
    vqgan = load_vqgan(cfg, device)

    gen   = Generator(mc["z_dim"], mc["latent_channels"], mc["gen_base_channels"]).to(device)
    disc  = Discriminator(mc["latent_channels"], mc["disc_base_channels"]).to(device)
    enc   = Encoder(mc["z_dim"], mc["latent_channels"], mc["disc_base_channels"]).to(device)

    # Load Stage 1 weights (G + D frozen during Stage 2)
    s1 = torch.load(s1_ckpt, map_location=device, weights_only=False)
    gen.load_state_dict(s1["generator"])
    disc.load_state_dict(s1["discriminator"])
    gen.eval()
    disc.eval()
    for p in gen.parameters():  p.requires_grad_(False)
    for p in disc.parameters(): p.requires_grad_(False)

    log.info(f"Encoder params: {sum(p.numel() for p in enc.parameters()):,}")
    log.info(f"Generator and Discriminator frozen.")

    kappa  = mc["kappa"]
    opt_e  = torch.optim.Adam(enc.parameters(), lr=s2["lr_enc"], betas=(s2["beta1"], s2["beta2"]))

    start_epoch   = 0
    best_val_loss = float("inf")

    last_ckpt2 = ckpt_dir / "last.pth"
    resume_path = args.resume
    if resume_path == "auto":
        resume_path = str(last_ckpt2) if last_ckpt2.exists() else None
    if resume_path and Path(resume_path).exists():
        start_epoch, best_val_loss = load_stage2(Path(resume_path), enc, opt_e, device)
        start_epoch += 1

    train_loader, val_loader = make_loaders(cfg)
    writer = SummaryWriter(str(log_dir))

    n_epochs = s2["n_epochs"]
    t_start  = time.time()

    log.info(f"Encoder training: epochs {start_epoch}–{n_epochs}, kappa={kappa}")

    for epoch in range(start_epoch, n_epochs):
        enc.train()
        enc_losses = []

        pbar = tqdm(train_loader, desc=f"S2 Epoch {epoch+1}/{n_epochs}", leave=False)
        for x_real_px in pbar:
            B = x_real_px.size(0)

            # "ziz" scheme: sample z → G(z) → E(G(z)) ≈ z
            z = torch.randn(B, mc["z_dim"], device=device)

            with torch.no_grad():
                x_gen = gen(z)              # (B, 3, 64, 64) — generated latent

            z_hat = enc(x_gen)              # (B, z_dim)
            x_recon = gen(z_hat)            # (B, 3, 64, 64) — re-generated from E(G(z))

            loss_z = F.mse_loss(z_hat, z)   # ||E(G(z)) − z||²

            with torch.no_grad():
                _, feat_gen = disc(x_gen)
            _, feat_recon = disc(x_recon)
            loss_feat = F.mse_loss(feat_recon, feat_gen.detach())

            loss_enc = loss_z + kappa * loss_feat

            opt_e.zero_grad()
            loss_enc.backward()
            opt_e.step()

            enc_losses.append(loss_enc.item())
            pbar.set_postfix(enc=f"{loss_enc.item():.4f}")

        avg_enc = sum(enc_losses) / max(len(enc_losses), 1)
        writer.add_scalar("stage2/loss_enc", avg_enc, epoch)
        log.info(f"S2 Epoch {epoch+1}/{n_epochs} | enc_loss={avg_enc:.6f}")

        # ── Validation ────────────────────────────────────────────────────────
        if (epoch + 1) % s2["val_interval"] == 0:
            enc.eval()
            val_losses = []
            with torch.no_grad():
                for x_val_px in val_loader:
                    B_v = x_val_px.size(0)
                    z_v = torch.randn(B_v, mc["z_dim"], device=device)
                    x_g = gen(z_v)
                    z_h = enc(x_g)
                    x_r = gen(z_h)
                    lz  = F.mse_loss(z_h, z_v)
                    _, ff = disc(x_g)
                    _, fr = disc(x_r)
                    lf  = F.mse_loss(fr, ff)
                    val_losses.append((lz + kappa * lf).item())

            val_enc = sum(val_losses) / max(len(val_losses), 1)
            writer.add_scalar("stage2/val_loss_enc", val_enc, epoch)
            log.info(f"  val enc_loss={val_enc:.6f}")

            if val_enc < best_val_loss:
                best_val_loss = val_enc
                best_path2    = ckpt_dir / "best.pth"
                save_stage2(best_path2, enc, opt_e, epoch, best_val_loss)
                log.info(f"  NEW BEST → {best_path2.name}")

        # ── Periodic checkpoint ───────────────────────────────────────────────
        if (epoch + 1) % s2["save_interval"] == 0:
            save_stage2(last_ckpt2, enc, opt_e, epoch, best_val_loss)

        elapsed = time.time() - t_start
        eta_s   = (elapsed / max(epoch - start_epoch + 1, 1)) * (n_epochs - epoch - 1)
        log.info(f"  ETA: {eta_s/3600:.1f}h")

    save_stage2(last_ckpt2, enc, opt_e, epoch, best_val_loss)
    writer.close()
    log.info(f"Stage 2 complete. Best encoder: {ckpt_dir / 'best.pth'}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="f-AnoGAN training — iris PAD")
    parser.add_argument(
        "--config", default="iris_anogan/configs/fanogan_iris.yaml",
        help="Path to fanogan_iris.yaml",
    )
    parser.add_argument(
        "--stage", type=int, choices=[1, 2], required=True,
        help="1 = GAN training, 2 = Encoder training",
    )
    parser.add_argument(
        "--resume", default=None,
        help="Checkpoint to resume from. 'auto' = detect last.pth automatically.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.stage == 1:
        train_stage1(cfg, args)
    else:
        train_stage2(cfg, args)
