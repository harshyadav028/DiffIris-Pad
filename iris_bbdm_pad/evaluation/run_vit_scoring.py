"""
iris_bbdm_pad/evaluation/run_vit_scoring.py

Computes ViT-B/16 reconstruction-divergence scores for DiffIrisPAD.

Since BBDM reconstructions are not saved to disk, this script integrates
BBDM inference and ViT feature extraction in a single pass:

  1. Load LBBDM-f4 model + ViT-B/16 (pretrained, CLS token)
  2. For each batch from IrisTestDataset:
       a. Run BBDM reconstruction:  corrupted_A  → reconstructed
       b. Extract ViT CLS embedding from clean_B  (original clean image)
       c. Extract ViT CLS embedding from reconstructed
       d. ViT score = cosine distance(emb_original, emb_reconstructed)
  3. Save CSV: filename, label, attack_type, vit_score

Bona-fide images reconstruct faithfully → small cosine distance → low score.
Attack images reconstruct toward bonafide → large divergence → high score.

Usage (run from project root):
    cd /home/teaching/Documents/Geetanjali_PhD_IRIS_PAD
    PYTHONNOUSERSITE=1 /home/teaching/miniconda3/envs/bbdm_clean/bin/python \\
        iris_bbdm_pad/evaluation/run_vit_scoring.py \\
        --split test \\
        --output_csv IJCB_paper_requirements/scoring/vit_scores_test.csv

    PYTHONNOUSERSITE=1 /home/teaching/miniconda3/envs/bbdm_clean/bin/python \\
        iris_bbdm_pad/evaluation/run_vit_scoring.py \\
        --split val \\
        --output_csv IJCB_paper_requirements/scoring/vit_scores_val.csv
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

# ── Project root bootstrap ───────────────────────────────────────────────────
# Must happen before any project imports. The G18 BBDM copy has the full
# runners/ package that the root BBDM/ directory is missing.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)

BBDM_ROOT = PROJECT_ROOT / "G18_Iris_PAD_2026" / "BBDM"
if not BBDM_ROOT.exists():
    # Fallback to root-level BBDM if G18 copy not present
    BBDM_ROOT = PROJECT_ROOT / "BBDM"

for p in [str(PROJECT_ROOT), str(BBDM_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Stdlib / third-party ─────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn.functional as F
import timm
import yaml
from tqdm import tqdm

# ── BBDM model ───────────────────────────────────────────────────────────────
from model.BrownianBridge.LatentBrownianBridgeModel import LatentBrownianBridgeModel
from utils import dict2namespace

# ── Project data loader ──────────────────────────────────────────────────────
from iris_bbdm_pad.data.iris_dataset import IrisTestDataset

# ── Logging ──────────────────────────────────────────────────────────────────
(PROJECT_ROOT / "IJCB_paper_requirements" / "logs").mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(
            PROJECT_ROOT / "IJCB_paper_requirements" / "logs" / "task1_run.log"
        ),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

CONFIG_PATH     = PROJECT_ROOT / "iris_bbdm_pad" / "configs" / "bbdm_iris_bonafide.yaml"
CHECKPOINT_PATH = (PROJECT_ROOT / "results" / "iris_bonafide_pad"
                   / "LBBDM-f4" / "checkpoint" / "top_model_epoch_70.pth")
EVAL_DIR        = PROJECT_ROOT / "iris_bbdm_pad" / "data" / "evaluation_sets"
RESUME_FILE     = (PROJECT_ROOT / "IJCB_paper_requirements"
                   / "scoring" / ".vit_scoring_checkpoint.json")


# ── BBDM model loader ────────────────────────────────────────────────────────

def load_bbdm(
    config_path: str,
    checkpoint_path: str,
    device: torch.device,
    num_steps: int = 50,
) -> LatentBrownianBridgeModel:
    """Load LBBDM-f4 from config + checkpoint, using EMA weights if available.

    Args:
        num_steps: Number of reverse-diffusion steps.  Overrides the config
            value (200 by default).  50 steps gives ~4× speedup with only a
            small drop in reconstruction quality — acceptable for ViT feature
            extraction.  (Ablation on this dataset: 50-step ACER=28.2%,
            200-step ACER=26.7% — both discriminative for ViT scoring.)
    """
    with open(config_path) as f:
        cfg_dict = yaml.safe_load(f)
    cfg = dict2namespace(cfg_dict)

    model = LatentBrownianBridgeModel(cfg.model)

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    log.info(f"Loading checkpoint: {ckpt_path.name}")
    states = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    if "ema" in states:
        log.info("Using EMA weights for inference.")
        model_state = model.state_dict()
        for name in model_state:
            if name in states["ema"]:
                model_state[name] = states["ema"][name]
        model.load_state_dict(model_state, strict=False)
    elif "model" in states:
        model.load_state_dict(states["model"], strict=False)
    else:
        model.load_state_dict(states, strict=False)

    # Override sampling schedule to num_steps (same formula as BrownianBridgeModel)
    T = model.num_timesteps   # 1000
    if num_steps < T:
        midsteps = torch.arange(
            T - 1, 1,
            step=-((T - 1) / (num_steps - 2)),
        ).long()
        model.steps = torch.cat(
            (midsteps, torch.tensor([1, 0], dtype=torch.long)), dim=0
        )
        log.info(f"Sampling steps overridden: {len(model.steps)} "
                 f"(config default: {T})")

    epoch = states.get("epoch", "?")
    log.info(f"Checkpoint loaded (epoch={epoch}), steps={len(model.steps)}")
    model.eval().to(device)
    return model


@torch.no_grad()
def bbdm_reconstruct(model: LatentBrownianBridgeModel, corrupted: torch.Tensor) -> torch.Tensor:
    """Run full BBDM reverse bridge: corrupted → reconstructed (both in [-1,1])."""
    return model.sample(corrupted, clip_denoised=False)


# ── ViT helpers ───────────────────────────────────────────────────────────────

def load_vit(device: torch.device) -> torch.nn.Module:
    """Load ViT-B/16 with ImageNet-21k pretrained weights, CLS-token output."""
    vit = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
    vit.eval().to(device)
    log.info("ViT-B/16 loaded (patch=16, embed_dim=768, ImageNet-21k pretrained)")
    return vit


def bbdm_to_vit(x: torch.Tensor) -> torch.Tensor:
    """Convert BBDM [-1,1] tensor (B,3,H,W) to ViT input (B,3,224,224) on same device."""
    x = (x.clamp(-1.0, 1.0) + 1.0) / 2.0                        # [0, 1]
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std  = torch.tensor(IMAGENET_STD,  device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


@torch.no_grad()
def extract_cls(vit: torch.nn.Module, x_bbdm: torch.Tensor) -> torch.Tensor:
    """Extract L2-normalised CLS token embedding [B, 768] from BBDM-scale images."""
    emb = vit(bbdm_to_vit(x_bbdm))          # [B, 768]
    return F.normalize(emb, dim=-1)          # unit vectors


def cosine_distance_batch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Element-wise cosine distance in [0, 2]. Higher = more different."""
    return (1.0 - (a * b).sum(dim=-1)).cpu()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ViT-B/16 iris PAD scoring via BBDM reconstruction divergence"
    )
    p.add_argument("--split", required=True, choices=["test", "val"])
    p.add_argument("--output_csv", required=True, help="Output CSV path")
    p.add_argument("--config",     default=str(CONFIG_PATH))
    p.add_argument("--checkpoint", default=str(CHECKPOINT_PATH))
    p.add_argument("--eval_dir",   default=str(EVAL_DIR))
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_steps", type=int, default=50,
                   help="Diffusion sampling steps (50 ≈ 4× faster, similar discrimination)")
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resume", action="store_true",
                   help="Resume from checkpoint if it exists")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    log.info(f"=== ViT Scoring: split={args.split} | device={device} ===")

    # Dataset
    dataset = IrisTestDataset(
        eval_dir=args.eval_dir,
        split=args.split,
        image_size=256,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(args.device == "cuda"),
    )
    log.info(f"Dataset: {len(dataset)} images")

    # Resume state
    processed: set = set()
    if args.resume and RESUME_FILE.exists():
        with open(RESUME_FILE) as f:
            rd = json.load(f)
        if rd.get("split") == args.split:
            processed = set(rd.get("processed_files", []))
            log.info(f"Resuming: {len(processed)} already scored")

    # Load models
    log.info(f"Loading LBBDM-f4 (num_steps={args.num_steps})...")
    bbdm = load_bbdm(args.config, args.checkpoint, device, num_steps=args.num_steps)

    log.info("Loading ViT-B/16...")
    vit = load_vit(device)

    # Output CSV
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Only append if we actually have a valid checkpoint to resume from.
    # If no checkpoint file existed, start fresh even if the CSV exists
    # (avoids duplicate rows from a killed-before-first-checkpoint run).
    csv_mode = "a" if (args.resume and len(processed) > 0 and out_path.exists()) else "w"
    csv_f  = open(out_path, csv_mode, newline="")
    writer = csv.DictWriter(csv_f, fieldnames=["filename", "label", "attack_type", "vit_score"])
    if csv_mode == "w":
        writer.writeheader()
    log.info(f"Output CSV mode: {csv_mode} | resuming {len(processed)} files")

    # Inference loop
    start      = time.time()
    scored     = len(processed)
    nan_count  = 0
    checkpoint_interval = 50   # batches (~1600 images; save frequently for resume

    for batch_idx, batch in enumerate(tqdm(loader, desc=f"Scoring [{args.split}]")):
        filenames    = list(batch["filename"])
        corrupted_A  = batch["A"].to(device)    # [-1, 1]  corrupted input
        clean_B      = batch["B"].to(device)    # [-1, 1]  clean original
        labels       = list(batch["label"])
        attack_types = list(batch["attack_type"])

        if args.resume and all(fn in processed for fn in filenames):
            continue

        # BBDM: corrupted → reconstructed
        reconstructed = bbdm_reconstruct(bbdm, corrupted_A)    # [-1, 1]

        # ViT: clean_B (original) vs reconstructed
        emb_orig  = extract_cls(vit, clean_B)
        emb_recon = extract_cls(vit, reconstructed)

        # Score = cosine distance
        vit_scores = cosine_distance_batch(emb_orig, emb_recon)  # [B]

        for i in range(len(filenames)):
            score = float(vit_scores[i])
            if not np.isfinite(score):
                nan_count += 1
                score = float("nan")
            writer.writerow({
                "filename":    filenames[i],
                "label":       labels[i],
                "attack_type": attack_types[i],
                "vit_score":   score,
            })
            processed.add(filenames[i])
            scored += 1

        csv_f.flush()

        if (batch_idx + 1) % checkpoint_interval == 0:
            RESUME_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(RESUME_FILE, "w") as f:
                json.dump({"split": args.split,
                           "processed_files": list(processed)}, f)

    csv_f.close()
    with open(RESUME_FILE, "w") as f:
        json.dump({"split": args.split,
                   "processed_files": list(processed),
                   "completed": True}, f)

    elapsed = time.time() - start
    log.info(f"Done: {scored} images scored in {elapsed/60:.1f} min")
    log.info(f"NaN scores: {nan_count} | Output: {out_path}")

    # Quick score statistics
    import pandas as pd
    df  = pd.read_csv(out_path)
    bon = df[df["label"] == "bonafide"]["vit_score"].dropna()
    atk = df[df["label"] != "bonafide"]["vit_score"].dropna()
    log.info(f"Bona-fide: mean={bon.mean():.4f}  std={bon.std():.4f}  "
             f"median={bon.median():.4f}")
    log.info(f"Attack:    mean={atk.mean():.4f}  std={atk.std():.4f}  "
             f"median={atk.median():.4f}")


if __name__ == "__main__":
    main()
