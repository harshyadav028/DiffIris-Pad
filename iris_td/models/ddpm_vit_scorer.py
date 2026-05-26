"""
iris_td/models/ddpm_vit_scorer.py

DDPM Simplex+DDIM inference scored with ViT-B/16 cosine distance.

Scoring logic is identical to BBDM run_vit_scoring.py:
    vit_score = cosine_distance(ViT(original_x0), ViT(reconstructed_x0))

Bona-fide: x0 reconstructs faithfully  → small cosine distance → low score
Attack:    x0 pulled toward bona-fide  → large divergence     → high score

Config: simplex noise, t*=500, 100 DDIM steps (deterministic).

Usage (run from project root):
    cd /home/teaching/Documents/Geetanjali_PhD_IRIS_PAD
    PYTHONPATH=BBDM:$PYTHONPATH conda run --no-capture-output -n iris_pad python \\
        iris_td/models/ddpm_vit_scorer.py \\
        --split val \\
        --output_csv iris_td/pad_scores/ddpm_val_simplex_tstar500_ddim_steps100_vit.csv

    PYTHONPATH=BBDM:$PYTHONPATH conda run --no-capture-output -n iris_pad python \\
        iris_td/models/ddpm_vit_scorer.py \\
        --split test \\
        --output_csv iris_td/pad_scores/ddpm_test_simplex_tstar500_ddim_steps100_vit.csv
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
import timm
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

torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed_all(42)

# ImageNet normalisation constants (same as BBDM ViT scorer)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

try:
    from opensimplex import OpenSimplex
    SIMPLEX_AVAILABLE = True
except ImportError:
    SIMPLEX_AVAILABLE = False
    log.warning("opensimplex not installed — install: pip install opensimplex")


# ── Dataset ───────────────────────────────────────────────────────────────────

class EvalDataset(Dataset):
    def __init__(self, image_list: List[Tuple[Path, str]], transform):
        self.image_list = image_list
        self.transform  = transform

    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(self, idx: int):
        img_path, label_str = self.image_list[idx]
        x = self.transform(Image.open(img_path).convert("RGB"))
        return x, str(img_path), label_str


# ── Attack type extraction ────────────────────────────────────────────────────

_ATTACK_PREFIXES = [
    ("Print_and_E-display", "Print_E-display"),
    ("Print_E-display",     "Print_E-display"),
    ("E-display",           "E-display"),
    ("Fake",                "Fake_with_Add_On"),
    ("Artifact",            "Artifact"),
    ("CL",                  "CL"),
    ("Generated",           "Generated"),
    ("PostMortem",          "PostMortem"),
    ("Printed",             "Printed"),
]

def extract_attack_type(img_path: Path, label_str: str) -> str:
    if label_str == "bonafide":
        return "Live"
    name = img_path.name
    for prefix, atype in _ATTACK_PREFIXES:
        if name.startswith(prefix):
            return atype
    return "unknown"


# ── Simplex noise ─────────────────────────────────────────────────────────────

def generate_simplex_noise(
    shape: tuple,
    device: str,
    scale: float = 0.1,
    octaves: int = 6,
    seed: int = 42,
) -> torch.Tensor:
    if not SIMPLEX_AVAILABLE:
        log.warning("Falling back to Gaussian noise — install opensimplex")
        return torch.randn(shape, device=device)

    B, C, H, W = shape
    noise_np = np.zeros((B, C, H, W), dtype=np.float32)

    for c in range(C):
        gen = OpenSimplex(seed=seed + c)
        amplitude, frequency, total_amp = 1.0, scale, 0.0

        for _ in range(octaves):
            h_coords = np.arange(H) * frequency
            w_coords = np.arange(W) * frequency
            hh, ww   = np.meshgrid(h_coords, w_coords, indexing="ij")
            coords   = np.stack([hh.ravel(), ww.ravel()], axis=1)
            layer    = np.array(
                [gen.noise2(float(x), float(y)) for x, y in coords],
                dtype=np.float32,
            ).reshape(H, W)
            noise_np[:, c] += amplitude * layer[np.newaxis]
            total_amp  += amplitude
            amplitude  *= 0.5
            frequency  *= 2.0

        noise_np[:, c] /= total_amp

    mx = np.abs(noise_np).max()
    if mx > 1e-8:
        noise_np /= mx

    return torch.tensor(noise_np, dtype=torch.float32, device=device)


# ── DDPM model ────────────────────────────────────────────────────────────────

def load_ddpm_model(config_path: str, checkpoint_path: str, device: str):
    cfg = yaml.safe_load(open(config_path))

    _models_dir = str(Path(__file__).resolve().parent)
    if _models_dir not in sys.path:
        sys.path.insert(0, _models_dir)
    from ddpm_model import LatentDDPM

    model = LatentDDPM(
        unet_config=cfg["model"]["unet_config"],
        vqgan_config=cfg["model"]["vqgan_config"],
        ddpm_config=cfg["model"]["ddpm_config"],
    )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "ema_state_dict" in ckpt and ckpt["ema_state_dict"]:
        ema_sd   = ckpt["ema_state_dict"]
        unet_ref = model.unet.state_dict()
        unet_sd  = {
            (k[len("unet."):] if k.startswith("unet.") else k): v
            for k, v in ema_sd.items()
            if (k[len("unet."):] if k.startswith("unet.") else k) in unet_ref
        }
        if unet_sd:
            model.unet.load_state_dict(unet_sd, strict=False)
            log.info(f"Loaded EMA weights ({len(unet_sd)} UNet keys)")
        else:
            log.warning("EMA keys didn't match — falling back to model_state_dict")
            model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    elif "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        log.info("Loaded model_state_dict weights")
    else:
        model.load_state_dict(ckpt.get("model", ckpt), strict=False)
        log.info("Loaded model weights")

    model.to(device).eval()
    log.info(f"LatentDDPM loaded: {checkpoint_path}")
    return model


# ── Forward diffusion in latent space ─────────────────────────────────────────

def forward_diffusion(
    model,
    z0: torch.Tensor,
    t_star: int,
    device: str,
    simplex_scale: float = 0.1,
) -> torch.Tensor:
    t_idx       = min(t_star, len(model.alphas_cumprod) - 1)
    alpha_bar_t = model.alphas_cumprod[t_idx].to(device)
    noise       = generate_simplex_noise(z0.shape, device=device, scale=simplex_scale)
    return (
        torch.sqrt(alpha_bar_t) * z0
        + torch.sqrt(1.0 - alpha_bar_t) * noise
    )


# ── DDIM reverse diffusion ────────────────────────────────────────────────────

def reverse_diffusion_ddim(
    model,
    z_t: torch.Tensor,
    t_star: int,
    num_steps: int,
    device: str,
) -> torch.Tensor:
    t_star_idx = min(t_star, model.num_timesteps - 1)
    timesteps  = torch.linspace(t_star_idx, 1, num_steps, dtype=torch.long, device=device)

    for i, t_val in enumerate(timesteps):
        t      = t_val.expand(z_t.shape[0])
        t_prev = (
            timesteps[i + 1].expand(z_t.shape[0])
            if i + 1 < len(timesteps)
            else torch.zeros_like(t)
        )
        with torch.no_grad():
            try:
                z_t = model.ddim_sample(z_t, t, t_prev=t_prev, eta=0.0)
            except AttributeError:
                log.warning("ddim_sample not available — falling back to p_sample")
                z_t = model.p_sample(z_t, t)
        z_t = z_t.detach()

    return z_t


# ── ViT-B/16 (same as BBDM run_vit_scoring.py) ───────────────────────────────

def load_vit(device: str) -> torch.nn.Module:
    vit = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
    vit.eval().to(device)
    log.info("ViT-B/16 loaded (patch=16, embed_dim=768, ImageNet-21k pretrained)")
    return vit


def tensor_to_vit(x: torch.Tensor) -> torch.Tensor:
    """Convert [-1,1] DDPM tensor (B,3,H,W) → ViT input (B,3,224,224)."""
    x    = (x.clamp(-1.0, 1.0) + 1.0) / 2.0
    x    = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std  = torch.tensor(IMAGENET_STD,  device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


@torch.no_grad()
def extract_cls(vit: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """L2-normalised CLS token [B, 768]."""
    return F.normalize(vit(tensor_to_vit(x)), dim=-1)


def cosine_distance_batch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Element-wise cosine distance in [0, 2]. Higher = more different."""
    return (1.0 - (a * b).sum(dim=-1)).cpu()


# ── Argument parser ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="DDPM Simplex+DDIM iris PAD scoring via ViT-B/16 cosine distance"
    )
    p.add_argument("--config",
        default="iris_td/configs/ddpm_iris.yaml")
    p.add_argument("--checkpoint",
        default="iris_td/results/ddpm_run1/DDPM/checkpoint/top_model_epoch_best.pth")
    p.add_argument("--split",
        choices=["test", "val"], default="test")
    p.add_argument("--output_csv",
        default=None,
        help="Output CSV path. Auto-derived from split if omitted.")
    p.add_argument("--test_dir",
        default=None,
        help="Path to evaluation_sets/{split}/. Auto-derived from split if omitted.")
    p.add_argument("--labels",
        default=None,
        help="Path to labels CSV. Auto-derived from split if omitted.")
    p.add_argument("--t_star",      type=int,   default=500)
    p.add_argument("--num_steps",   type=int,   default=100)
    p.add_argument("--simplex_scale", type=float, default=0.1)
    p.add_argument("--image_size",  type=int,   default=256)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--device",
        default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_images",  type=int,   default=None,
        help="Limit images (dry-run only)")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Auto-derive paths from split
    if args.test_dir is None:
        args.test_dir = f"iris_td/data/evaluation_sets/{args.split}/"
    if args.labels is None:
        args.labels = f"iris_td/labels/{args.split}_labels.csv"
    if args.output_csv is None:
        args.output_csv = (
            f"iris_td/pad_scores/"
            f"ddpm_{args.split}_simplex_tstar{args.t_star}"
            f"_ddim_steps{args.num_steps}_vit.csv"
        )

    device = args.device
    log.info("=" * 60)
    log.info("DDPM Simplex+DDIM — ViT-B/16 Scoring")
    log.info(f"  split={args.split} | t*={args.t_star} | steps={args.num_steps} DDIM")
    log.info(f"  batch={args.batch_size} | device={device}")
    log.info(f"  output → {args.output_csv}")
    log.info("=" * 60)

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    # Load labels CSV
    labels = {}
    label_path = Path(args.labels)
    if label_path.exists():
        with open(label_path) as f:
            for row in csv.DictReader(f):
                labels[row.get("filename", "")] = row.get("label", "unknown")
        log.info(f"Labels loaded: {len(labels)} entries")

    # Collect images from images/ subfolder (flat layout)
    test_dir   = Path(args.test_dir)
    images_dir = test_dir / "images"
    search_dir = images_dir if images_dir.exists() else test_dir
    all_images = []
    for img in sorted(search_dir.rglob("*.png")):
        lbl = labels.get(img.name, None)
        # Fall back to filename prefix for None or "unknown" labels
        if lbl is None or lbl == "unknown":
            lbl = "bonafide" if img.name.startswith("Live") else "attack"
        all_images.append((img, lbl))

    if args.max_images:
        all_images = all_images[:args.max_images]

    n_bf  = sum(1 for _, l in all_images if l == "bonafide")
    n_atk = sum(1 for _, l in all_images if l == "attack")
    log.info(f"Dataset: {len(all_images)} images | bonafide={n_bf} | attack={n_atk}")

    if not all_images:
        log.error(f"No images found in {search_dir}. Check --test_dir.")
        return

    # Load DDPM model
    log.info("Loading LatentDDPM (EMA weights)...")
    model = load_ddpm_model(args.config, args.checkpoint, device)

    # Load ViT
    log.info("Loading ViT-B/16...")
    vit = load_vit(device)

    dataset = EvalDataset(all_images, transform)
    loader  = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device == "cuda"),
    )

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    start = time.time()

    for x0_batch, img_paths, label_strs in tqdm(loader, desc=f"Scoring [{args.split}]"):
        x0_batch = x0_batch.to(device)

        with torch.no_grad():
            # Step 1 — encode to latent
            z0 = model.encode(x0_batch)

            # Step 2 — forward diffusion: add simplex noise at t*
            z_t = forward_diffusion(
                model, z0, args.t_star, device,
                simplex_scale=args.simplex_scale,
            )

            # Step 3 — DDIM reverse: 100 steps
            z_recon = reverse_diffusion_ddim(
                model, z_t, args.t_star, args.num_steps, device,
            )

            # Step 4 — decode back to pixel space
            x_recon = model.decode(z_recon)

            # Step 5 — ViT embeddings (same as BBDM: compare original vs reconstructed)
            emb_orig  = extract_cls(vit, x0_batch)   # ViT(x0)
            emb_recon = extract_cls(vit, x_recon)     # ViT(x_recon)

            # Step 6 — cosine distance (identical formula to BBDM run_vit_scoring.py)
            vit_scores = cosine_distance_batch(emb_orig, emb_recon)

        for i in range(len(img_paths)):
            img_path = Path(img_paths[i])
            score    = float(vit_scores[i])
            rows.append({
                "filename":    img_path.name,
                "label":       label_strs[i],
                "attack_type": extract_attack_type(img_path, label_strs[i]),
                "vit_score":   round(score, 6) if np.isfinite(score) else float("nan"),
            })

    # Save CSV
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filename", "label", "attack_type", "vit_score"]
        )
        writer.writeheader()
        writer.writerows(rows)

    elapsed = time.time() - start
    log.info(f"Done: {len(rows)} images in {elapsed/60:.1f} min → {out_path}")

    # Score statistics
    bf_scores  = [r["vit_score"] for r in rows if r["label"] == "bonafide"
                  and np.isfinite(r["vit_score"])]
    atk_scores = [r["vit_score"] for r in rows if r["label"] != "bonafide"
                  and np.isfinite(r["vit_score"])]
    if bf_scores and atk_scores:
        gap = np.mean(atk_scores) - np.mean(bf_scores)
        log.info(
            f"vit_score | bonafide={np.mean(bf_scores):.4f} "
            f"| attack={np.mean(atk_scores):.4f} | gap={gap:+.4f}"
        )
        log.info("GOOD — attacks score higher" if gap > 0
                 else "WARNING — gap negative, check model/config")

    avg_t = elapsed / len(rows) if rows else 0
    log.info(f"Avg time/image: {avg_t:.3f}s")


if __name__ == "__main__":
    main()
