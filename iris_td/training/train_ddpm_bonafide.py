"""
Standalone DDPM training on bonafide iris images.

Trains LatentDDPM (iris_td/models/ddpm_model.py) on clean latents
using the standard noise-prediction MSE objective.

Identical hyperparameters to BBDM run:
  n_epochs=200, batch_size=16, accumulate_grad_batches=4, lr=1e-4

Checkpoints saved to: iris_td/results/ddpm_run1/DDPM/checkpoint/
  last.pth         — every epoch (for resume)
  top_model_epoch_{N}.pth — best val loss (for inference)

Usage:
  # New run
  cd ~/Documents/Geetanjali_PhD_IRIS_PAD
  conda run -n iris_pad python iris_td/training/train_ddpm_bonafide.py

  # Resume
  conda run -n iris_pad python iris_td/training/train_ddpm_bonafide.py \\
      --resume iris_td/results/ddpm_run1/DDPM/checkpoint/last.pth

  # Run inside tmux to survive disconnection
  tmux new -s ddpm_train
  conda run -n iris_pad python iris_td/training/train_ddpm_bonafide.py
  Ctrl+B D  (detach)
"""

import argparse
import json
import logging
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# ── Project paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
IRIS_TD_ROOT = Path(__file__).resolve().parents[1]
BBDM_ROOT    = PROJECT_ROOT / "BBDM"

for _p in [str(IRIS_TD_ROOT / "models"), str(BBDM_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ddpm_model import LatentDDPM  # noqa: E402

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# EMA helper (mirrors BBDM runners/base/EMA.py)
# ─────────────────────────────────────────────────────────────────────────────

class EMA:
    """Exponential Moving Average of model weights."""

    def __init__(self, decay: float = 0.995):
        self.decay = decay
        self.shadow: dict = {}
        self.backup: dict = {}

    def register(self, model: torch.nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: torch.nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name]
                    + (1.0 - self.decay) * param.data
                )

    def apply_shadow(self, model: torch.nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self, model: torch.nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading (reuse BBDM's custom_aligned, take B images only)
# ─────────────────────────────────────────────────────────────────────────────

def get_dataloader(cfg: dict, stage: str) -> DataLoader:
    """Load bonafide images using BBDM's custom_aligned dataset."""
    from datasets.custom import CustomAlignedDataset  # from BBDM/
    from Register import Registers  # noqa: F401 (side-effect: registers datasets)

    import argparse
    def _ns(d):
        ns = argparse.Namespace()
        for k, v in d.items():
            setattr(ns, k, _ns(v) if isinstance(v, dict) else v)
        return ns

    ds_cfg = _ns(cfg["data"]["dataset_config"])
    dataset = CustomAlignedDataset(ds_cfg, stage=stage)

    batch_size = cfg["data"][stage]["batch_size"]
    shuffle    = cfg["data"][stage].get("shuffle", stage == "train")
    num_workers = min(8, os.cpu_count() or 4)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=stage == "train",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Training step
# ─────────────────────────────────────────────────────────────────────────────

def compute_loss(model: LatentDDPM, x_clean: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Standard DDPM noise-prediction loss in latent space.

    1. Encode pixel image → latent z₀
    2. Sample random t, add noise: z_t = sqrt(ᾱ_t)·z₀ + sqrt(1-ᾱ_t)·ε
    3. UNet predicts ε̂ from z_t
    4. MSE(ε̂, ε)
    """
    x_clean = x_clean.to(device)
    B = x_clean.shape[0]

    with torch.no_grad():
        z0 = model.encode(x_clean)

    t = torch.randint(0, model.num_timesteps, (B,), device=device, dtype=torch.long)
    z_t, noise = model.q_sample(z0, t)
    noise_pred = model.unet(z_t, t)
    return F.mse_loss(noise_pred, noise)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    path: Path,
    model: LatentDDPM,
    optimizer: torch.optim.Optimizer,
    scheduler,
    ema: EMA,
    epoch: int,
    step: int,
    val_loss: float,
):
    torch.save({
        "epoch":           epoch,
        "step":            step,
        "val_loss":        val_loss,
        "model":           model.state_dict(),
        "model_state_dict": model.state_dict(),   # alias for anomaly_detector
        "ema_state_dict":  ema.shadow,
        "optimizer":       optimizer.state_dict(),
        "scheduler":       scheduler.state_dict() if scheduler is not None else None,
    }, path)


def load_checkpoint(
    path: Path,
    model: LatentDDPM,
    optimizer: torch.optim.Optimizer,
    scheduler,
    ema: EMA,
    device: torch.device,
):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    if "ema_state_dict" in ckpt:
        ema.shadow = {k: v.to(device) for k, v in ckpt["ema_state_dict"].items()}
    optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    epoch = ckpt.get("epoch", 0)
    step  = ckpt.get("step", 0)
    val_loss = ckpt.get("val_loss", float("inf"))
    log.info(f"Resumed from epoch {epoch}, step {step}, val_loss={val_loss:.6f}")
    return epoch, step, val_loss


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    # ── Config ────────────────────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    n_epochs   = cfg["training"]["n_epochs"]
    n_steps    = cfg["training"]["n_steps"]
    save_every = cfg["training"]["save_interval"]
    val_every  = cfg["training"]["validation_interval"]
    accum_grad = cfg["training"].get("accumulate_grad_batches", 4)

    result_path = Path(cfg.get("result_path", "iris_td/results/ddpm_run1")) / "DDPM"
    ckpt_dir    = result_path / "checkpoint"
    log_dir     = result_path / "log"
    sample_dir  = result_path / "samples"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    if device.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    log.info("Building LatentDDPM...")
    model = LatentDDPM(
        unet_config=cfg["model"]["unet_config"],
        vqgan_config=cfg["model"]["vqgan_config"],
        ddpm_config=cfg["model"]["ddpm_config"],
    ).to(device)

    n_params = sum(p.numel() for p in model.unet.parameters())
    log.info(f"UNet parameters: {n_params:,}")

    # ── Optimizer / Scheduler ─────────────────────────────────────────────────
    opt_cfg = cfg["model"]["optimizer"]
    optimizer = torch.optim.Adam(
        model.unet.parameters(),
        lr=opt_cfg["lr"],
        betas=(opt_cfg.get("beta1", 0.9), 0.999),
        weight_decay=opt_cfg.get("weight_decay", 0.0),
    )

    sch_cfg = cfg["model"]["lr_scheduler"]
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=sch_cfg["factor"],
        patience=sch_cfg["patience"],
        threshold=sch_cfg["threshold"],
        cooldown=sch_cfg["cooldown"],
        min_lr=sch_cfg["min_lr"],
    )

    # ── EMA ───────────────────────────────────────────────────────────────────
    ema_cfg   = cfg["model"]["EMA"]
    use_ema   = ema_cfg.get("use_ema", True)
    ema       = EMA(decay=ema_cfg.get("ema_decay", 0.995))
    ema_interval  = ema_cfg.get("update_ema_interval", 8)
    ema_start     = ema_cfg.get("start_ema_step", 30000)
    if use_ema:
        ema.register(model)
        log.info(f"EMA enabled (decay={ema.decay}, start_step={ema_start})")

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    best_ckpt_path = None

    resume_path = args.resume or (ckpt_dir / "last.pth" if (ckpt_dir / "last.pth").exists() else None)
    if resume_path and Path(resume_path).exists():
        start_epoch, global_step, best_val_loss = load_checkpoint(
            Path(resume_path), model, optimizer, scheduler, ema, device
        )
        start_epoch += 1  # continue from next epoch

    # ── Data ──────────────────────────────────────────────────────────────────
    log.info("Loading datasets...")
    train_loader = get_dataloader(cfg, "train")
    val_loader   = get_dataloader(cfg, "val")
    log.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ── TensorBoard ───────────────────────────────────────────────────────────
    writer = SummaryWriter(str(log_dir))

    # Save config snapshot
    with open(ckpt_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f)

    # ── Training ──────────────────────────────────────────────────────────────
    log.info(f"Starting training: epochs {start_epoch}–{n_epochs}, max_steps={n_steps}")
    log.info(f"Batch size: {cfg['data']['train']['batch_size']} × {accum_grad} accum = "
             f"{cfg['data']['train']['batch_size'] * accum_grad} effective")

    t_start = time.time()

    for epoch in range(start_epoch, n_epochs):
        if global_step >= n_steps:
            log.info(f"Reached n_steps={n_steps} — training complete.")
            break

        model.train()
        model.vqgan.eval()   # VQGAN always frozen

        epoch_loss = 0.0
        n_batches  = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}", leave=False)
        for batch_idx, ((x_clean, _), _) in enumerate(pbar):
            # dataset returns ((B_tensor, B_fname), (A_tensor, A_fname))
            # we only need x_clean = B (the clean target image)
            if global_step >= n_steps:
                break

            loss = compute_loss(model, x_clean, device)
            loss = loss / accum_grad
            loss.backward()

            if (batch_idx + 1) % accum_grad == 0:
                torch.nn.utils.clip_grad_norm_(model.unet.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

                # EMA update
                if use_ema and global_step >= ema_start and global_step % ema_interval == 0:
                    ema.update(model)

                global_step += 1

                train_loss_val = loss.item() * accum_grad
                epoch_loss += train_loss_val
                n_batches  += 1

                writer.add_scalar("train/loss", train_loss_val, global_step)
                pbar.set_postfix(loss=f"{train_loss_val:.4f}", step=global_step)

        epoch_loss = epoch_loss / max(n_batches, 1)
        log.info(f"Epoch {epoch+1} — train_loss={epoch_loss:.6f} | step={global_step}")

        # ── Validation ────────────────────────────────────────────────────────
        if (epoch + 1) % val_every == 0:
            model.eval()
            model.vqgan.eval()
            val_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for (x_clean, _), _ in tqdm(val_loader, desc="Val", leave=False):
                    val_loss += compute_loss(model, x_clean, device).item()
                    n_val += 1
            val_loss /= max(n_val, 1)
            log.info(f"  val_loss={val_loss:.6f} | lr={optimizer.param_groups[0]['lr']:.2e}")
            writer.add_scalar("val/loss", val_loss, global_step)
            scheduler.step(val_loss)

            # Top checkpoint
            if val_loss < best_val_loss:
                if best_ckpt_path and best_ckpt_path.exists():
                    best_ckpt_path.unlink()
                best_val_loss = val_loss
                best_ckpt_path = ckpt_dir / f"top_model_epoch_{epoch+1}.pth"
                # Use EMA weights for best checkpoint
                if use_ema:
                    ema.apply_shadow(model)
                save_checkpoint(best_ckpt_path, model, optimizer, scheduler, ema, epoch, global_step, val_loss)
                if use_ema:
                    ema.restore(model)
                log.info(f"  NEW BEST → {best_ckpt_path.name} (val_loss={val_loss:.6f})")
                # Stable alias for anomaly_detector
                best_alias = ckpt_dir / "top_model_epoch_best.pth"
                import shutil
                shutil.copy2(best_ckpt_path, best_alias)

        # ── Periodic checkpoint ───────────────────────────────────────────────
        if (epoch + 1) % save_every == 0:
            save_checkpoint(
                ckpt_dir / "last.pth",
                model, optimizer, scheduler, ema,
                epoch, global_step, best_val_loss,
            )
            log.info(f"  Checkpoint saved (epoch {epoch+1})")

        # ETA
        elapsed = time.time() - t_start
        epochs_done = epoch - start_epoch + 1
        epochs_left = n_epochs - epoch - 1
        eta_s = (elapsed / max(epochs_done, 1)) * epochs_left
        log.info(f"  ETA: {eta_s/3600:.1f}h")

    # Final checkpoint
    save_checkpoint(
        ckpt_dir / "last.pth",
        model, optimizer, scheduler, ema,
        epoch, global_step, best_val_loss,
    )
    writer.close()
    log.info(f"Training finished. Best val_loss={best_val_loss:.6f}")
    log.info(f"Best checkpoint: {best_ckpt_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="iris_td/configs/ddpm_iris.yaml",
        help="Path to ddpm_iris.yaml",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to checkpoint to resume from (default: auto-detect last.pth)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
