"""
f-AnoGAN anomaly detector for iris PAD.

Inference pipeline (fast — single forward pass per image):
    1. VQGAN encode:  x (3, 256, 256) → z_vq (3, 64, 64)
    2. Encoder:       z_vq → z_gan (z_dim,)
    3. Generator:     z_gan → z_vq_hat (3, 64, 64)
    4. VQGAN decode:  z_vq_hat → x_hat (3, 256, 256)
    5. Anomaly score: residual_loss + κ · feature_loss
        residual_loss = ||z_vq − z_vq_hat||²     (MSE in latent space)
        feature_loss  = ||f_D(z_vq) − f_D(z_vq_hat)||²
        mse_score     = ||x − x_hat||²            (pixel MSE — same as BBDM)
        lpips_score   = LPIPS(x, x_hat)           (perceptual — same as BBDM)
        recon_score   = mse_score + lpips_score    (primary comparison vs BBDM)

Resume/checkpoint support (mirrors iris_bbdm_pad/models/anomaly_detector.py):
    - Saves a JSON checkpoint every 50 batches listing processed filenames
    - On --resume: skips already-scored images, appends to existing CSV
    - Interruption-safe: no scored image is lost

Usage
-----
  cd ~/Documents/Geetanjali_PhD_IRIS_PAD

  # Fresh run
  python -W ignore iris_anogan/models/fanogan_anomaly_detector.py \\
      --config     iris_anogan/configs/fanogan_iris.yaml \\
      --s1_ckpt    iris_anogan/results/fanogan_run1/stage1/checkpoint/best.pth \\
      --s2_ckpt    iris_anogan/results/fanogan_run1/stage2/checkpoint/best.pth \\
      --test_dir   iris_bbdm_pad/data/evaluation_sets/test/ \\
      --labels     iris_bbdm_pad/data/evaluation_sets/test/labels.csv \\
      --output_csv iris_anogan/results/fanogan_run1/pad_scores/test_scores.csv

  # Resume after interruption
  python -W ignore iris_anogan/models/fanogan_anomaly_detector.py \\
      --config     iris_anogan/configs/fanogan_iris.yaml \\
      --s1_ckpt    iris_anogan/results/fanogan_run1/stage1/checkpoint/best.pth \\
      --s2_ckpt    iris_anogan/results/fanogan_run1/stage2/checkpoint/best.pth \\
      --test_dir   iris_bbdm_pad/data/evaluation_sets/test/ \\
      --labels     iris_bbdm_pad/data/evaluation_sets/test/labels.csv \\
      --output_csv iris_anogan/results/fanogan_run1/pad_scores/test_scores.csv \\
      --resume
"""

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional, Set

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
IRIS_ANOGAN  = Path(__file__).resolve().parents[1]
BBDM_ROOT    = PROJECT_ROOT / "BBDM"

for _p in [str(BBDM_ROOT), str(IRIS_ANOGAN)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model.VQGAN.vqgan import VQModel                                   # noqa: E402
from models.fanogan_model import Generator, Discriminator, Encoder      # noqa: E402
from data.iris_dataset import EvalDataset, extract_attack_type          # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

torch.manual_seed(42)
np.random.seed(42)

FIELDNAMES = [
    "filename", "label", "attack_type",
    "fanogan_score", "residual_score", "feature_score",
    "mse_score", "lpips_score", "recon_score",
    "vit_score",
    "inference_time",
]

# ImageNet normalisation (identical to iris_td/models/ddpm_vit_scorer.py)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Save scoring checkpoint every N batches (same cadence as BBDM)
CHECKPOINT_INTERVAL = 50


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _dict_to_ns(d: dict):
    import argparse as _ap
    ns = _ap.Namespace()
    for k, v in d.items():
        setattr(ns, k, _dict_to_ns(v) if isinstance(v, dict) else v)
    return ns


# ---------------------------------------------------------------------------
# ViT-B/16 cosine-distance scoring (identical to iris_td ddpm_vit_scorer.py)
#   vit_score = cosine_distance(ViT(x_input), ViT(x_reconstruction))
#   This is the scoring used for Diff-IrisPAD paper Table 2 — required for a
#   fair, like-for-like comparison.
# ---------------------------------------------------------------------------

def load_vit(device: torch.device):
    import timm
    vit = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
    vit.eval().to(device)
    for p in vit.parameters():
        p.requires_grad_(False)
    log.info("ViT-B/16 loaded (patch=16, embed_dim=768, ImageNet-21k pretrained)")
    return vit


def _tensor_to_vit(x: torch.Tensor) -> torch.Tensor:
    """[-1,1] tensor (B,3,H,W) → ViT input (B,3,224,224), ImageNet-normalised."""
    x = (x.clamp(-1.0, 1.0) + 1.0) / 2.0
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std  = torch.tensor(IMAGENET_STD,  device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


@torch.no_grad()
def _extract_cls(vit, x: torch.Tensor) -> torch.Tensor:
    """L2-normalised CLS token [B, 768]."""
    return F.normalize(vit(_tensor_to_vit(x)), dim=-1)


def _vit_cosine_distance(vit, x_input: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
    """Cosine distance in [0,2] between ViT(x_input) and ViT(x_recon)."""
    e_in  = _extract_cls(vit, x_input)
    e_rec = _extract_cls(vit, x_recon)
    return (1.0 - (e_in * e_rec).sum(dim=-1)).cpu()


def load_models(cfg: dict, s1_ckpt: str, s2_ckpt: str, device: torch.device):
    """Load VQGAN (frozen) + Generator + Discriminator + Encoder."""
    mc = cfg["model"]

    vq_ns = _dict_to_ns(cfg["vqgan"])
    vqgan = VQModel(**vars(vq_ns)).to(device).eval()
    for p in vqgan.parameters():
        p.requires_grad_(False)

    def _disabled_train(self, mode=True):
        return self
    vqgan.train = _disabled_train.__get__(vqgan, type(vqgan))
    log.info("VQGAN loaded (frozen)")

    gen  = Generator(mc["z_dim"], mc["latent_channels"], mc["gen_base_channels"]).to(device)
    disc = Discriminator(mc["latent_channels"], mc["disc_base_channels"]).to(device)
    s1   = torch.load(s1_ckpt, map_location=device, weights_only=False)
    gen.load_state_dict(s1["generator"])
    disc.load_state_dict(s1["discriminator"])
    gen.eval(); disc.eval()
    for p in gen.parameters():  p.requires_grad_(False)
    for p in disc.parameters(): p.requires_grad_(False)
    log.info(f"Stage 1 loaded: {s1_ckpt}")

    enc = Encoder(mc["z_dim"], mc["latent_channels"], mc["disc_base_channels"]).to(device)
    s2  = torch.load(s2_ckpt, map_location=device, weights_only=False)
    enc.load_state_dict(s2["encoder"])
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    log.info(f"Stage 2 loaded: {s2_ckpt}")

    return vqgan, gen, disc, enc


# ---------------------------------------------------------------------------
# Resume helpers  (mirrors iris_bbdm_pad/models/anomaly_detector.py)
# ---------------------------------------------------------------------------

def _load_scoring_checkpoint(ckpt_file: Path) -> Set[str]:
    """Return set of filenames already scored (from JSON checkpoint)."""
    if not ckpt_file.exists():
        return set()
    try:
        with open(ckpt_file) as f:
            data = json.load(f)
        processed = set(data.get("processed_files", []))
        log.info(f"Resume: {len(processed)} files already scored (from {ckpt_file.name})")
        return processed
    except Exception as e:
        log.warning(f"Could not read checkpoint {ckpt_file}: {e} — starting fresh")
        return set()


def _save_scoring_checkpoint(ckpt_file: Path, processed: Set[str], total: int):
    ckpt_file.parent.mkdir(parents=True, exist_ok=True)
    ckpt_file.write_text(json.dumps({
        "processed_files": sorted(processed),
        "n_processed": len(processed),
        "n_total": total,
    }, indent=2))


# ---------------------------------------------------------------------------
# Inference loop with resume support
# ---------------------------------------------------------------------------

def run_inference(
    cfg: dict,
    vqgan,
    gen,
    disc,
    enc,
    loader: DataLoader,
    device: torch.device,
    output_csv: Path,
    kappa: float = 1.0,
    compute_pixel: bool = True,
    lpips_fn=None,
    vit=None,
    resume: bool = False,
    ckpt_file: Optional[Path] = None,
    total_images: int = 0,
):
    """Score all images, writing to CSV with periodic checkpoint saves.

    Mirrors BBDM's process_dataset() resume logic exactly:
      - Loads processed filenames from JSON checkpoint on resume
      - Skips batches where all images are already done
      - Appends to existing CSV on resume
      - Saves checkpoint every CHECKPOINT_INTERVAL batches
    """
    ckpt_file = ckpt_file or (
        PROJECT_ROOT / "iris_anogan" / "checkpoints" / "scoring_checkpoint.json"
    )

    # Load existing checkpoint
    processed_files: Set[str] = _load_scoring_checkpoint(ckpt_file) if resume else set()

    # Open CSV in append mode if resuming, else write fresh
    csv_mode = "a" if resume and output_csv.exists() else "w"
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    n_skipped = 0
    n_scored  = 0

    with open(output_csv, csv_mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if csv_mode == "w":
            writer.writeheader()

        for batch_idx, (x_batch, img_paths, label_strs) in enumerate(
            tqdm(loader, desc="f-AnoGAN scoring")
        ):
            filenames = [Path(p).name for p in img_paths]

            # Skip batch if all images already scored
            if resume and all(fn in processed_files for fn in filenames):
                n_skipped += len(filenames)
                continue

            x_batch = x_batch.to(device)
            B = x_batch.size(0)
            t0 = time.time()

            with torch.no_grad():
                # 1. VQGAN encode
                z_vq = vqgan.encoder(x_batch)
                z_vq = vqgan.quant_conv(z_vq)          # (B, 3, 64, 64)

                # 2. f-AnoGAN Encoder: latent → z
                z_hat = enc(z_vq)                       # (B, z_dim)

                # 3. Generator: z → reconstructed latent
                z_vq_hat = gen(z_hat)                   # (B, 3, 64, 64)

                # 4. Latent-space scores
                residual = F.mse_loss(
                    z_vq_hat, z_vq, reduction="none"
                ).mean(dim=[1, 2, 3]).cpu()

                _, feat_real  = disc(z_vq)
                _, feat_recon = disc(z_vq_hat)
                feature = F.mse_loss(
                    feat_recon, feat_real, reduction="none"
                ).mean(dim=[1, 2, 3]).cpu()

                fanogan_score = residual + kappa * feature

                # 5. Pixel-space scores (same as BBDM: mse + lpips = recon_score)
                #    + ViT-B/16 cosine distance (same as Diff-IrisPAD Table 2)
                if compute_pixel or lpips_fn is not None or vit is not None:
                    z_q, _, _ = vqgan.quantize(z_vq_hat)
                    x_hat_px  = vqgan.decode(z_q)

                    pixel_mse = F.mse_loss(
                        x_hat_px, x_batch, reduction="none"
                    ).mean(dim=[1, 2, 3]).cpu()

                    lpips_score = (
                        lpips_fn(x_hat_px, x_batch).view(-1).cpu()
                        if lpips_fn is not None
                        else torch.zeros(B)
                    )

                    vit_score = (
                        _vit_cosine_distance(vit, x_batch, x_hat_px)
                        if vit is not None
                        else torch.zeros(B)
                    )
                else:
                    pixel_mse   = torch.zeros(B)
                    lpips_score = torch.zeros(B)
                    vit_score   = torch.zeros(B)

            elapsed = time.time() - t0
            per_img = elapsed / B

            rows = []
            for i in range(B):
                fn = filenames[i]
                if resume and fn in processed_files:
                    continue   # skip individual images within a partial batch

                pm  = float(pixel_mse[i])
                lps = float(lpips_score[i])
                rows.append({
                    "filename":       fn,
                    "label":          label_strs[i],
                    "attack_type":    extract_attack_type(img_paths[i], label_strs[i]),
                    "fanogan_score":  round(float(fanogan_score[i]), 6),
                    "residual_score": round(float(residual[i]),      6),
                    "feature_score":  round(float(feature[i]),       6),
                    "mse_score":      round(pm,       6),
                    "lpips_score":    round(lps,      6),
                    "recon_score":    round(pm + lps, 6),
                    "vit_score":      round(float(vit_score[i]), 6),
                    "inference_time": round(per_img,  4),
                })
                processed_files.add(fn)

            writer.writerows(rows)
            f.flush()
            n_scored += len(rows)

            # Save checkpoint every CHECKPOINT_INTERVAL batches
            if (batch_idx + 1) % CHECKPOINT_INTERVAL == 0:
                _save_scoring_checkpoint(ckpt_file, processed_files, total_images)

    # Final checkpoint
    _save_scoring_checkpoint(ckpt_file, processed_files, total_images)

    log.info(f"Scored: {n_scored} | Skipped (already done): {n_skipped}")
    return n_scored


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="f-AnoGAN inference — iris PAD")
    parser.add_argument("--config",
        default="iris_anogan/configs/fanogan_iris.yaml")
    parser.add_argument("--s1_ckpt",
        default="iris_anogan/results/fanogan_run1/stage1/checkpoint/best.pth",
        help="Stage 1 (GAN) checkpoint")
    parser.add_argument("--s2_ckpt",
        default="iris_anogan/results/fanogan_run1/stage2/checkpoint/best.pth",
        help="Stage 2 (Encoder) checkpoint")
    parser.add_argument("--test_dir",
        default="iris_bbdm_pad/data/evaluation_sets/test/",
        help="Directory containing test images (same 47434-image set as BBDM)")
    parser.add_argument("--labels",
        default="iris_bbdm_pad/data/evaluation_sets/test/labels.csv")
    parser.add_argument("--output_csv",
        default="iris_anogan/results/fanogan_run1/pad_scores/test_scores.csv")
    parser.add_argument("--scoring_checkpoint",
        default=None,
        help="JSON checkpoint file for resume (default: iris_anogan/checkpoints/scoring_checkpoint.json)")
    parser.add_argument("--resume", action="store_true",
        help="Resume from scoring checkpoint — skip already-scored images, append to CSV")
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--kappa",       type=float, default=1.0,
        help="Feature-matching weight in anomaly score")
    parser.add_argument("--no_lpips",    action="store_true",
        help="Skip LPIPS (faster; recon_score = mse_score only)")
    parser.add_argument("--no_pixel",    action="store_true",
        help="Skip all pixel-space scores (fastest — latent scores only)")
    parser.add_argument("--max_images",  type=int, default=None,
        help="Limit images for dry-run")
    parser.add_argument("--no_vit",      action="store_true",
        help="Skip ViT-B/16 cosine scoring (paper Diff-IrisPAD uses ViT scoring)")
    parser.add_argument("--split",
        choices=["test", "val"], default="test")
    return parser.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"f-AnoGAN inference | device={device} | kappa={args.kappa} | resume={args.resume}")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    vqgan, gen, disc, enc = load_models(cfg, args.s1_ckpt, args.s2_ckpt, device)

    # Optional LPIPS
    lpips_fn = None
    if not args.no_lpips:
        try:
            import lpips as lpips_lib
            lpips_fn = lpips_lib.LPIPS(net="alex").to(device).eval()
            for p in lpips_fn.parameters():
                p.requires_grad_(False)
            log.info("LPIPS (alex) enabled")
        except ImportError:
            log.warning("lpips not installed — recon_score = mse_score only. pip install lpips")

    # ViT-B/16 cosine scoring (same scoring as Diff-IrisPAD paper Table 2)
    vit = None
    if not args.no_vit:
        try:
            vit = load_vit(device)
        except Exception as e:
            log.warning(f"ViT not loaded ({e}) — vit_score will be 0. pip install timm")

    # Dataset
    dataset = EvalDataset(
        image_dir  = args.test_dir,
        labels_csv = args.labels,
        max_images = args.max_images,
    )
    if len(dataset) == 0:
        log.error(f"No images found in {args.test_dir}")
        return

    loader = DataLoader(
        dataset,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = 4,
        pin_memory  = True,
    )

    out_path = Path(args.output_csv)
    ckpt_file = Path(args.scoring_checkpoint) if args.scoring_checkpoint else (
        PROJECT_ROOT / "iris_anogan" / "checkpoints" / f"{args.split}_scoring_checkpoint.json"
    )

    n_scored = run_inference(
        cfg, vqgan, gen, disc, enc, loader, device,
        output_csv    = out_path,
        kappa         = args.kappa,
        compute_pixel = not args.no_pixel,
        lpips_fn      = lpips_fn,
        vit           = vit,
        resume        = args.resume,
        ckpt_file     = ckpt_file,
        total_images  = len(dataset),
    )

    log.info(f"Saved {n_scored} rows → {out_path}")

    # Summary stats from CSV
    import pandas as pd
    if out_path.exists():
        df = pd.read_csv(out_path)
        bf  = df[df["label"] == "bonafide"]["fanogan_score"]
        atk = df[df["label"] == "attack"]["fanogan_score"]
        if len(bf) and len(atk):
            gap = atk.mean() - bf.mean()
            log.info(
                f"fanogan_score | bonafide={bf.mean():.4f} "
                f"| attack={atk.mean():.4f} | gap={gap:+.4f}"
            )
            log.info("GOOD — attacks score higher" if gap > 0
                     else "WARNING — gap negative (check model / kappa)")

        avg_t = df["inference_time"].mean()
        log.info(f"Avg inference: {avg_t*1000:.1f}ms/image")
        log.info(f"Test set (47434): {47434 * avg_t / 3600:.1f}h estimate")


if __name__ == "__main__":
    main()
