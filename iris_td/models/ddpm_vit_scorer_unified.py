"""
iris_td/models/ddpm_vit_scorer_unified.py

ViT-B/16 cosine-distance scorer for ALL 4 DDPM configs.
Identical scoring formula to ddpm_vit_scorer.py and BBDM run_vit_scoring.py.
Adds --noise_type gaussian|simplex and --use_ddim flag.

Configs:
  1. gaussian  t*=1000  steps=25  stochastic  (Vanilla)
  2. gaussian  t*=500   steps=25  stochastic  (Partial)
  3. simplex   t*=500   steps=25  stochastic  (AnoDDPM)
  4. simplex   t*=500   steps=50  ddim        (AnoDDPM+DDIM)

Usage:
  cd /home/teaching/Documents/Geetanjali_PhD_IRIS_PAD
  PYTHONNOUSERSITE=1 PYTHONPATH=BBDM:$PYTHONPATH \
    /home/teaching/miniconda3/envs/bbdm_clean/bin/python \
    iris_td/models/ddpm_vit_scorer_unified.py \
    --noise_type gaussian --t_star 1000 --num_steps 25 \
    --split val --output_csv iris_td/pad_scores/ddpm_val_gaussian_tstar1000_steps25_vit.csv
"""

import argparse, csv, logging, sys, time
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

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
torch.manual_seed(42); np.random.seed(42); torch.cuda.manual_seed_all(42)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

try:
    from opensimplex import OpenSimplex
    SIMPLEX_AVAILABLE = True
except ImportError:
    SIMPLEX_AVAILABLE = False

ATTACK_PREFIXES = [
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


class EvalDataset(Dataset):
    def __init__(self, image_list: List[Tuple[Path, str]], transform):
        self.image_list = image_list
        self.transform  = transform
    def __len__(self): return len(self.image_list)
    def __getitem__(self, idx):
        p, lbl = self.image_list[idx]
        return self.transform(Image.open(p).convert("RGB")), str(p), lbl


def extract_attack_type(img_path: Path, label_str: str) -> str:
    if label_str == "bonafide":
        return "Live"
    name = img_path.name
    for prefix, atype in ATTACK_PREFIXES:
        if name.startswith(prefix):
            return atype
    return "unknown"


def generate_simplex_noise(shape, device, scale=0.1, octaves=6, seed=42):
    if not SIMPLEX_AVAILABLE:
        return torch.randn(shape, device=device)
    B, C, H, W = shape
    noise_np = np.zeros((B, C, H, W), dtype=np.float32)
    for c in range(C):
        gen = OpenSimplex(seed=seed + c)
        amp, freq, total = 1.0, scale, 0.0
        for _ in range(octaves):
            hc = np.arange(H) * freq; wc = np.arange(W) * freq
            hh, ww = np.meshgrid(hc, wc, indexing="ij")
            coords = np.stack([hh.ravel(), ww.ravel()], axis=1)
            layer = np.array([gen.noise2(float(x), float(y)) for x, y in coords],
                             dtype=np.float32).reshape(H, W)
            noise_np[:, c] += amp * layer[np.newaxis]; total += amp
            amp *= 0.5; freq *= 2.0
        noise_np[:, c] /= total
    mx = np.abs(noise_np).max()
    if mx > 1e-8: noise_np /= mx
    return torch.tensor(noise_np, dtype=torch.float32, device=device)


def forward_diffusion(model, z0, t_star, device, noise_type="simplex", simplex_scale=0.1):
    t_idx = min(t_star, len(model.alphas_cumprod) - 1)
    alpha_bar_t = model.alphas_cumprod[t_idx].to(device)
    noise = (generate_simplex_noise(z0.shape, device=device, scale=simplex_scale)
             if noise_type == "simplex" else torch.randn_like(z0))
    return torch.sqrt(alpha_bar_t) * z0 + torch.sqrt(1.0 - alpha_bar_t) * noise


def reverse_diffusion(model, z_t, t_star, num_steps, device, use_ddim=False):
    t_star_idx = min(t_star, model.num_timesteps - 1)
    timesteps = torch.linspace(t_star_idx, 1, num_steps, dtype=torch.long, device=device)
    for i, t_val in enumerate(timesteps):
        t = t_val.expand(z_t.shape[0])
        with torch.no_grad():
            if use_ddim:
                t_prev = (timesteps[i + 1] if i + 1 < len(timesteps)
                          else torch.zeros_like(t_val)).expand(z_t.shape[0])
                try:
                    z_t = model.ddim_sample(z_t, t, t_prev=t_prev, eta=0.0)
                except AttributeError:
                    z_t = model.p_sample(z_t, t)
            else:
                z_t = model.p_sample(z_t, t)
        z_t = z_t.detach()
    return z_t


def load_ddpm_model(config_path, checkpoint_path, device):
    cfg = yaml.safe_load(open(config_path))
    _d = str(Path(__file__).resolve().parent)
    if _d not in sys.path: sys.path.insert(0, _d)
    from ddpm_model import LatentDDPM
    model = LatentDDPM(unet_config=cfg["model"]["unet_config"],
                       vqgan_config=cfg["model"]["vqgan_config"],
                       ddpm_config=cfg["model"]["ddpm_config"])
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "ema_state_dict" in ckpt and ckpt["ema_state_dict"]:
        ema_sd = ckpt["ema_state_dict"]; unet_ref = model.unet.state_dict()
        unet_sd = {(k[len("unet."):] if k.startswith("unet.") else k): v
                   for k, v in ema_sd.items()
                   if (k[len("unet."):] if k.startswith("unet.") else k) in unet_ref}
        if unet_sd: model.unet.load_state_dict(unet_sd, strict=False)
        else: model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    elif "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device).eval()
    log.info(f"LatentDDPM loaded: {checkpoint_path}")
    return model


def load_vit(device):
    vit = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
    vit.eval().to(device)
    log.info("ViT-B/16 loaded")
    return vit


def tensor_to_vit(x):
    x = (x.clamp(-1., 1.) + 1.) / 2.
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std  = torch.tensor(IMAGENET_STD,  device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


@torch.no_grad()
def extract_cls(vit, x):
    return F.normalize(vit(tensor_to_vit(x)), dim=-1)


def cosine_dist(a, b):
    return (1. - (a * b).sum(dim=-1)).cpu()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",      default="iris_td/configs/ddpm_iris.yaml")
    p.add_argument("--checkpoint",  default="iris_td/results/ddpm_run1/DDPM/checkpoint/top_model_epoch_best.pth")
    p.add_argument("--split",       choices=["val", "test"], default="test")
    p.add_argument("--noise_type",  choices=["gaussian", "simplex"], default="simplex")
    p.add_argument("--t_star",      type=int, default=500)
    p.add_argument("--num_steps",   type=int, default=25)
    p.add_argument("--use_ddim",    action="store_true", default=False)
    p.add_argument("--simplex_scale", type=float, default=0.1)
    p.add_argument("--image_size",  type=int, default=256)
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--test_dir",    default=None)
    p.add_argument("--labels",      default=None)
    p.add_argument("--output_csv",  default=None)
    p.add_argument("--max_images",  type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.test_dir is None:
        args.test_dir = f"iris_td/data/evaluation_sets/{args.split}/"
    if args.labels is None:
        args.labels = f"iris_td/labels/{args.split}_labels.csv"
    if args.output_csv is None:
        tag = f"ddpm_{args.split}_{args.noise_type}_tstar{args.t_star}"
        tag += "_ddim" if args.use_ddim else ""
        tag += f"_steps{args.num_steps}_vit.csv"
        args.output_csv = f"iris_td/pad_scores/{tag}"

    log.info("=" * 60)
    log.info(f"noise={args.noise_type} | t*={args.t_star} | steps={args.num_steps} | ddim={args.use_ddim}")
    log.info(f"split={args.split} | batch={args.batch_size} | device={device}")
    log.info(f"output → {args.output_csv}")
    log.info("=" * 60)

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    labels = {}
    lp = Path(args.labels)
    if lp.exists():
        with open(lp) as f:
            for row in csv.DictReader(f):
                labels[row.get("filename", "")] = row.get("label", "unknown")

    test_dir = Path(args.test_dir)
    images_dir = test_dir / "images"
    search_dir = images_dir if images_dir.exists() else test_dir
    all_images = []
    for img in sorted(search_dir.rglob("*.png")):
        lbl = labels.get(img.name, None)
        if lbl is None or lbl == "unknown":
            lbl = "bonafide" if img.name.startswith("Live") else "attack"
        all_images.append((img, lbl))
    if args.max_images:
        all_images = all_images[:args.max_images]

    n_bf  = sum(1 for _, l in all_images if l == "bonafide")
    n_atk = len(all_images) - n_bf
    log.info(f"Dataset: {len(all_images)} images | bonafide={n_bf} | attack={n_atk}")

    model = load_ddpm_model(args.config, args.checkpoint, device)
    vit   = load_vit(device)

    loader = DataLoader(EvalDataset(all_images, transform),
                        batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=(device == "cuda"))

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []; start = time.time()

    for x0_batch, img_paths, label_strs in tqdm(loader, desc=f"{args.noise_type} t*={args.t_star}"):
        x0_batch = x0_batch.to(device)
        with torch.no_grad():
            z0     = model.encode(x0_batch)
            z_t    = forward_diffusion(model, z0, args.t_star, device,
                                       args.noise_type, args.simplex_scale)
            z_recon = reverse_diffusion(model, z_t, args.t_star, args.num_steps,
                                        device, use_ddim=args.use_ddim)
            x_recon = model.decode(z_recon)
            emb_orig  = extract_cls(vit, x0_batch)
            emb_recon = extract_cls(vit, x_recon)
            scores    = cosine_dist(emb_orig, emb_recon)

        for i in range(len(img_paths)):
            p = Path(img_paths[i])
            rows.append({
                "filename":    p.name,
                "label":       label_strs[i],
                "attack_type": extract_attack_type(p, label_strs[i]),
                "vit_score":   round(float(scores[i]), 6),
            })

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "label", "attack_type", "vit_score"])
        w.writeheader(); w.writerows(rows)

    elapsed = time.time() - start
    bf_s  = [r["vit_score"] for r in rows if r["label"] == "bonafide"]
    atk_s = [r["vit_score"] for r in rows if r["label"] != "bonafide"]
    if bf_s and atk_s:
        gap = np.mean(atk_s) - np.mean(bf_s)
        log.info(f"bonafide={np.mean(bf_s):.4f} | attack={np.mean(atk_s):.4f} | gap={gap:+.4f}")
    log.info(f"Done: {len(rows)} images in {elapsed/60:.1f} min → {out_path}")
    log.info(f"Avg {elapsed/len(rows):.3f} s/img")


if __name__ == "__main__":
    main()
