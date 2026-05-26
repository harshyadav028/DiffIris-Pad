"""
DDPM anomaly detector for iris PAD — AnoDDPM strategy (Wyatt et al. CVPR 2022).

This follows the AnoDDPM paper directly, NOT the BBDM scoring strategy:
  - 1 pass per image (no trajectory passes)
  - recon_score = MSE + LPIPS only
  - No trajectory_score, no combined_score

Inference configs:
  Config 1: gaussian t*=1000 steps=25  stochastic  (vanilla DDPM)
  Config 2: gaussian t*=500  steps=25  stochastic  (partial noising)
  Config 3: simplex  t*=500  steps=25  stochastic  (AnoDDPM)
  Config 4: simplex  t*=500  steps=50  DDIM        (AnoDDPM + DDIM)

All configs use:
  - Linear skip sampling (torch.linspace t_star→1)
  - LATENT space forward/reverse diffusion (z0 = model.encode(x0))
  - batch_size=32 (Configs 1-3) / batch_size=64 (Config 4), fp32, EMA weights

Usage:
    cd ~/Documents/Geetanjali_PhD_IRIS_PAD
    PYTHONPATH=BBDM:$PYTHONPATH conda run --no-capture-output -n iris_pad python \\
        iris_td/models/ddpm_anomaly_detector.py \\
        --config      iris_td/configs/ddpm_iris.yaml \\
        --checkpoint  iris_td/results/ddpm_run1/DDPM/checkpoint/top_model_epoch_best.pth \\
        --test_dir    iris_td/data/evaluation_sets/val/ \\
        --output_csv  iris_td/pad_scores/ddpm_val_simplex_tstar500_steps25.csv \\
        --labels      iris_td/labels/val_labels.csv \\
        --noise_type  simplex \\
        --t_star      500 \\
        --num_steps   25 \\
        --batch_size  32
"""

import argparse
import csv
import logging
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path("BBDM").resolve()))
sys.path.insert(0, str(Path(".").resolve()))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed_all(42)

# ── Optional simplex noise ────────────────────────────────────────────────────
try:
    from opensimplex import OpenSimplex
    SIMPLEX_AVAILABLE = True
except ImportError:
    SIMPLEX_AVAILABLE = False
    log.warning(
        "opensimplex not installed — simplex noise unavailable. "
        "Install: conda run -n iris_pad pip install opensimplex"
    )


# ─────────────────────────────────────────────────────────────────────────────
# SPEC 2 — EvalDataset for batched DataLoader inference
# ─────────────────────────────────────────────────────────────────────────────

class EvalDataset(Dataset):
    """Dataset that returns (image_tensor, img_path_str, label_str)."""

    def __init__(self, image_list: List[Tuple[Path, str]], transform):
        self.image_list = image_list
        self.transform  = transform

    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(self, idx: int):
        img_path, label_str = self.image_list[idx]
        x = self.transform(Image.open(img_path).convert("RGB"))
        return x, str(img_path), label_str


# ─────────────────────────────────────────────────────────────────────────────
# Simplex noise generator (vectorised — no nested H×W loop)
# ─────────────────────────────────────────────────────────────────────────────

def generate_simplex_noise(
    shape: tuple,
    device: str,
    scale: float = 0.1,
    octaves: int = 6,
    seed: int = 42,
) -> torch.Tensor:
    """
    Multi-scale simplex noise (AnoDDPM Fix 2).

    Spatially correlated noise — better than Gaussian for structured
    iris attack patterns (printed dots, contact lens rings, pixel grids).

    Args:
        shape:   (B, C, H, W)
        device:  'cuda' or 'cpu'
        scale:   0.05=large blobs, 0.1=medium (AnoDDPM default), 0.2=fine
        octaves: frequency layers (AnoDDPM uses 6)
        seed:    reproducibility

    Returns:
        Tensor (B, C, H, W) in [-1, 1]
    """
    if not SIMPLEX_AVAILABLE:
        log.warning("Falling back to Gaussian noise — install opensimplex")
        return torch.randn(shape, device=device)

    B, C, H, W = shape
    noise_np = np.zeros((B, C, H, W), dtype=np.float32)

    for c in range(C):
        gen = OpenSimplex(seed=seed + c)
        amplitude = 1.0
        frequency = scale
        total_amp = 0.0

        for _ in range(octaves):
            h_coords = np.arange(H) * frequency
            w_coords = np.arange(W) * frequency
            hh, ww   = np.meshgrid(h_coords, w_coords, indexing="ij")
            coords   = np.stack([hh.ravel(), ww.ravel()], axis=1)

            layer = np.array(
                [gen.noise2(float(x), float(y)) for x, y in coords],
                dtype=np.float32,
            ).reshape(H, W)

            noise_np[:, c] += amplitude * layer[np.newaxis]
            total_amp += amplitude
            amplitude *= 0.5
            frequency *= 2.0

        noise_np[:, c] /= total_amp

    mx = np.abs(noise_np).max()
    if mx > 1e-8:
        noise_np /= mx

    return torch.tensor(noise_np, dtype=torch.float32, device=device)


# ─────────────────────────────────────────────────────────────────────────────
# SPEC 5 — attack_type extraction from file path
# ─────────────────────────────────────────────────────────────────────────────

def extract_attack_type(img_path: Path, label_str: str) -> str:
    """Return 'Live' for bonafide, or attack type inferred from path."""
    if label_str == "bonafide":
        return "Live"
    for atype in [
        "Artifact", "CL", "E-display",
        "Fake_with_Add_On", "Generated",
        "Post-Mortem", "Printed", "Print_E-display",
    ]:
        if atype.lower() in str(img_path).lower():
            return atype
    return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Forward diffusion — operates on LATENT z0 (not pixel x0)
# ─────────────────────────────────────────────────────────────────────────────

def forward_diffusion(
    model,
    z0: torch.Tensor,
    t_star: int,
    device: str,
    noise_type: str = "simplex",
    simplex_scale: float = 0.1,
) -> tuple:
    """
    Add noise to latent z0 at timestep t_star.

    z0 must already be in latent space (model.encode(x0)).
    Formula: z_t = sqrt(ᾱ_t) * z0 + sqrt(1 - ᾱ_t) * ε

    Returns:
        z_t_star: noised latent
        noise:    the noise added
    """
    t_idx = min(t_star, len(model.alphas_cumprod) - 1)
    alpha_bar_t = model.alphas_cumprod[t_idx].to(device)

    if noise_type == "simplex":
        noise = generate_simplex_noise(
            z0.shape, device=device, scale=simplex_scale
        )
    else:
        noise = torch.randn_like(z0)

    z_t_star = (
        torch.sqrt(alpha_bar_t) * z0
        + torch.sqrt(1.0 - alpha_bar_t) * noise
    )
    return z_t_star, noise


# ─────────────────────────────────────────────────────────────────────────────
# SPEC 3 — reverse_diffusion with linear skip sampling
# ─────────────────────────────────────────────────────────────────────────────

def reverse_diffusion(
    model,
    z_t: torch.Tensor,
    t_star: int,
    num_steps: int,
    device: str,
    use_ddim: bool = False,
) -> torch.Tensor:
    """
    Denoise z_t back to ẑ₀ using linear skip sampling.

    Picks evenly spaced timesteps: torch.linspace(t_star, 1, num_steps).
    Matches BBDM skip_sample=True, sample_type=linear for fair comparison.

    Returns:
        z_recon: denoised latent (B, C_z, H_z, W_z)
    """
    t_star_idx = min(t_star, model.num_timesteps - 1)
    timesteps = torch.linspace(
        t_star_idx, 1, num_steps, dtype=torch.long, device=device
    )

    for i, t_val in enumerate(timesteps):
        t = t_val.expand(z_t.shape[0])

        with torch.no_grad():
            if use_ddim:
                try:
                    t_prev = (
                        timesteps[i + 1]
                        if i + 1 < len(timesteps)
                        else torch.zeros_like(t_val)
                    )
                    t_prev = t_prev.expand(z_t.shape[0])
                    z_t = model.ddim_sample(z_t, t, t_prev=t_prev, eta=0.0)
                except AttributeError:
                    log.warning(
                        "ddim_sample not available — falling back to p_sample."
                    )
                    z_t = model.p_sample(z_t, t)
            else:
                z_t = model.p_sample(z_t, t)
        # Break any reference chains to prevent memory accumulation over steps
        z_t = z_t.detach()

    return z_t


# ─────────────────────────────────────────────────────────────────────────────
# SPEC 7 — Model loading (EMA weights preferred)
# ─────────────────────────────────────────────────────────────────────────────

def load_ddpm_model(config_path: str, checkpoint_path: str, device: str):
    """
    Load LatentDDPM from iris_td checkpoint.

    Weight loading priority:
      1. ema_state_dict  → UNet EMA weights (best for inference)
      2. model_state_dict → full model weights
      3. model           → full model weights (fallback key)
    """
    cfg = yaml.safe_load(open(config_path))

    _iris_td_models = str(Path(__file__).resolve().parent)
    if _iris_td_models not in sys.path:
        sys.path.insert(0, _iris_td_models)
    from ddpm_model import LatentDDPM

    model = LatentDDPM(
        unet_config=cfg["model"]["unet_config"],
        vqgan_config=cfg["model"]["vqgan_config"],
        ddpm_config=cfg["model"]["ddpm_config"],
    )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "ema_state_dict" in ckpt and ckpt["ema_state_dict"]:
        ema_sd  = ckpt["ema_state_dict"]
        # EMA keys are "unet.xxx"; model.unet.state_dict() keys are "xxx"
        unet_ref = model.unet.state_dict()
        unet_sd  = {}
        for k, v in ema_sd.items():
            # Strip "unet." prefix if present
            unet_key = k[len("unet."):] if k.startswith("unet.") else k
            if unet_key in unet_ref:
                unet_sd[unet_key] = v
        if unet_sd:
            model.unet.load_state_dict(unet_sd, strict=False)
            log.info(f"Loaded EMA weights for inference ({len(unet_sd)} UNet keys)")
        else:
            log.warning("EMA state_dict present but no keys matched — falling back to model weights")
            if "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"], strict=False)
            elif "model" in ckpt:
                model.load_state_dict(ckpt["model"], strict=False)
    elif "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        log.info("Loaded model_state_dict weights")
    elif "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=False)
        log.info("Loaded model weights")
    else:
        model.load_state_dict(ckpt, strict=False)
        log.info("Loaded raw checkpoint weights")

    model.to(device)
    model.eval()
    log.info(f"LatentDDPM loaded: {checkpoint_path}")
    return model, cfg


# ─────────────────────────────────────────────────────────────────────────────
# SPEC 8 — Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="DDPM AnoDDPM inference for iris PAD scoring"
    )
    parser.add_argument("--config",
        default="iris_td/configs/ddpm_iris.yaml")
    parser.add_argument("--checkpoint",
        default="iris_td/results/ddpm_run1/DDPM/checkpoint/top_model_epoch_best.pth")
    parser.add_argument("--test_dir",
        default="iris_td/data/evaluation_sets/test/")
    parser.add_argument("--output_csv",
        default="iris_td/pad_scores/ddpm_test_simplex_tstar500_steps25.csv")
    parser.add_argument("--labels",
        default="iris_td/labels/test_labels.csv")
    parser.add_argument("--noise_type",
        choices=["gaussian", "simplex"], default="simplex",
        help=(
            "gaussian = TV static pixel noise (vanilla DDPM baseline). "
            "simplex  = spatially correlated blobs (AnoDDPM — recommended)."
        ))
    parser.add_argument("--simplex_scale",
        type=float, default=0.1,
        help="Simplex frequency: 0.05=large blobs, 0.1=medium, 0.2=fine")
    parser.add_argument("--t_star",
        type=int, default=500,
        help=(
            "Partial noising timestep. "
            "1000=full noise (vanilla). 500=AnoDDPM default."
        ))
    parser.add_argument("--num_steps",
        type=int, default=25,
        help=(
            "Reverse diffusion steps. "
            "25=AnoDDPM default (configs 1-3). "
            "100=DDIM minimum (config 4)."
        ))
    parser.add_argument("--use_ddim",
        action="store_true", default=False,
        help=(
            "Use deterministic DDIM sampling (config 4). "
            "Requires --num_steps 100 minimum."
        ))
    parser.add_argument("--image_size",
        type=int, default=256)
    parser.add_argument("--batch_size",
        type=int, default=32,
        help="DataLoader batch size (same as BBDM for fair comparison).")
    parser.add_argument("--split",
        choices=["test", "val"], default="test")
    parser.add_argument("--max_images",
        type=int, default=None,
        help="Limit number of images (for dry-run testing only).")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Ablation row label
    if args.t_star == 1000 and args.noise_type == "gaussian":
        row_label = "Config 1 — Vanilla DDPM"
    elif args.noise_type == "gaussian":
        row_label = f"Config 2 — Partial noise (t*={args.t_star})"
    elif not args.use_ddim:
        row_label = f"Config 3 — AnoDDPM simplex (t*={args.t_star})"
    else:
        row_label = f"Config 4 — AnoDDPM + DDIM (t*={args.t_star})"

    log.info("=" * 60)
    log.info("DDPM AnoDDPM Inference — Iris PAD")
    log.info(f"  {row_label}")
    log.info(f"  noise={args.noise_type} | t*={args.t_star} | "
             f"steps={args.num_steps} | ddim={args.use_ddim}")
    log.info(f"  batch={args.batch_size} | device={device}")
    log.info("=" * 60)

    # Transform — identical to BBDM for fair comparison
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    # LPIPS (alex net, fp32, frozen)
    import lpips as lpips_lib
    lpips_fn = lpips_lib.LPIPS(net="alex").to(device)
    lpips_fn.eval()
    for p in lpips_fn.parameters():
        p.requires_grad_(False)

    # Load model (EMA weights)
    model, _cfg = load_ddpm_model(args.config, args.checkpoint, device)

    # Load labels CSV
    labels = {}
    label_path = Path(args.labels)
    if label_path.exists():
        with open(label_path) as f:
            for row in csv.DictReader(f):
                labels[row.get("filename", "")] = row.get("label", "unknown")
        log.info(f"Labels loaded: {len(labels)} entries")

    # Collect images — handle images/ subfolder and flat layout
    test_dir   = Path(args.test_dir)
    all_images = []
    seen       = set()
    for subfolder in ["images", "Live", "Spoof", ""]:
        folder = test_dir / subfolder if subfolder else test_dir
        if not folder.exists():
            continue
        for img in sorted(folder.rglob("*.png")):
            if str(img) in seen:
                continue
            seen.add(str(img))
            lbl = labels.get(img.name, None)
            if lbl is None:
                lbl = (
                    "attack"
                    if ("Spoof" in str(img) or "spoof" in str(img))
                    else "bonafide"
                )
            all_images.append((img, lbl))

    if args.max_images:
        all_images = all_images[:args.max_images]

    n_bf  = sum(1 for _, l in all_images if l == "bonafide")
    n_atk = sum(1 for _, l in all_images if l == "attack")
    log.info(
        f"Dataset: {len(all_images)} images | "
        f"bonafide: {n_bf} | attack: {n_atk}"
    )
    if not all_images:
        log.error(f"No images found in {test_dir}. Check --test_dir path.")
        return

    # SPEC 2 — DataLoader
    dataset = EvalDataset(all_images, transform)
    loader  = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    rows = []

    for x0_batch, img_paths, label_strs in tqdm(loader, desc=row_label):
        x0_batch = x0_batch.to(device)     # (B, 3, H, W) fp32
        B = x0_batch.shape[0]
        t0 = time.time()

        with torch.no_grad():
            # Step 1 — Encode to latent (fp32)
            z0 = model.encode(x0_batch)    # (B, C_z, H_z, W_z)

            # Step 2 — Forward diffusion in LATENT space
            z_t, _ = forward_diffusion(
                model, z0, args.t_star, device,
                noise_type=args.noise_type,
                simplex_scale=args.simplex_scale,
            )

            # Step 3 — Reverse diffusion (linear skip)
            z_recon = reverse_diffusion(
                model, z_t, args.t_star,
                args.num_steps, device,
                use_ddim=args.use_ddim,
            )

            # Step 4 — Decode reconstruction to pixel space
            x_recon = model.decode(z_recon)    # (B, 3, H, W)

            # Step 5 — MSE per image
            mse_scores = F.mse_loss(
                x_recon, x0_batch, reduction="none"
            ).mean(dim=[1, 2, 3]).cpu()         # (B,)

            # Step 6 — LPIPS per image
            lpips_scores = lpips_fn(
                x_recon, x0_batch
            ).view(-1).cpu()                    # (B,)

            # Step 7 — recon_score = MSE + LPIPS
            recon_scores = mse_scores + lpips_scores   # (B,)

        elapsed  = time.time() - t0
        per_img  = elapsed / B

        for i in range(B):
            img_path  = Path(img_paths[i])
            label_str = label_strs[i]
            rows.append({
                "filename":      img_path.name,
                "label":         label_str,
                "attack_type":   extract_attack_type(img_path, label_str),
                "mse_score":     round(float(mse_scores[i]),   6),
                "lpips_score":   round(float(lpips_scores[i]), 6),
                "recon_score":   round(float(recon_scores[i]), 6),
                "inference_time": round(per_img, 4),
            })

    # SPEC 6 — Save CSV (6 scoring columns + inference_time)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "filename",
        "label",
        "attack_type",
        "mse_score",
        "lpips_score",
        "recon_score",
        "inference_time",
    ]

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    log.info(f"Saved {len(rows)} rows → {output_csv}")

    # Summary stats
    bf_recon  = [r["recon_score"] for r in rows if r["label"] == "bonafide"]
    atk_recon = [r["recon_score"] for r in rows if r["label"] == "attack"]
    if bf_recon and atk_recon:
        gap = np.mean(atk_recon) - np.mean(bf_recon)
        log.info(
            f"recon_score | bonafide={np.mean(bf_recon):.4f} "
            f"| attack={np.mean(atk_recon):.4f} | gap={gap:+.4f}"
        )
        log.info("GOOD — attacks score higher" if gap > 0
                 else "WARNING — gap negative, check model/config")

    if rows:
        avg_t = np.mean([r["inference_time"] for r in rows])
        log.info(f"Avg inference time: {avg_t:.3f}s/image")
        log.info(f"Val set (21380):  {21380 * avg_t / 3600:.1f}h")
        log.info(f"Test set (47434): {47434 * avg_t / 3600:.1f}h")


if __name__ == "__main__":
    main()
