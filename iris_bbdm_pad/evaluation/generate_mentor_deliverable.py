"""
Generate mentor deliverable: input/reconstruction/error-map pairs for each attack type.

Produces:
    input_images/   — corrupted input fed to BBDM (one per attack type + bonafide)
    recon_images/   — BBDM reconstruction output
    visualizations/ — 3-panel figure: Input | Reconstruction | L1 Error Map

Final deliverable zipped to: mentor_iris_recon_deliverable.zip

Usage:
    PYTHONNOUSERSITE=1 /home/teaching/miniconda3/envs/bbdm_clean/bin/python \
        iris_bbdm_pad/evaluation/generate_mentor_deliverable.py
"""

from __future__ import annotations

import json
import os
import sys
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

# ── Project / BBDM path setup ───────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BBDM_ROOT = PROJECT_ROOT / "G18_Iris_PAD_2026" / "BBDM"
if str(BBDM_ROOT) not in sys.path:
    sys.path.insert(0, str(BBDM_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Paths ────────────────────────────────────────────────────────────────────
CONFIG_PATH    = PROJECT_ROOT / "iris_bbdm_pad/configs/bbdm_iris_bonafide.yaml"
CKPT_PATH      = PROJECT_ROOT / "results/iris_bonafide_pad/LBBDM-f4/checkpoint/top_model_epoch_70.pth"
SCORES_CSV     = PROJECT_ROOT / "iris_bbdm_pad/results/test_pad_scores.csv"
TEST_IMAGES    = PROJECT_ROOT / "iris_bbdm_pad/data/evaluation_sets/test/images"
CORRUPTION_CFG = PROJECT_ROOT / "iris_bbdm_pad/data/bonafide_pairs/dataset_config.json"
OUT_DIR        = PROJECT_ROOT / "mentor_deliverable"
ZIP_PATH       = PROJECT_ROOT / "mentor_iris_recon_deliverable.zip"

IMAGE_SIZE = 256

ATTACK_TYPES = [
    "Live",           # bonafide
    "Artifact",
    "CL",
    "E-display",
    "Fake with Add On",
    "Generated",
    "PostMortem",
    "Print and E-display",
    "Printed",
]

# Safe filename stem for each attack type
SAFE_NAME = {
    "Live":                 "bonafide_Live",
    "Artifact":             "attack_Artifact",
    "CL":                   "attack_CL",
    "E-display":            "attack_E-display",
    "Fake with Add On":     "attack_FakeWithAddOn",
    "Generated":            "attack_Generated",
    "PostMortem":           "attack_PostMortem",
    "Print and E-display":  "attack_PrintAndEDisplay",
    "Printed":              "attack_Printed",
}


# ── Helper: tensor ↔ numpy ───────────────────────────────────────────────────

def tensor_to_uint8(t) -> np.ndarray:
    """[-1,1] CHW tensor → HWC uint8 numpy."""
    arr = t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    arr = (arr * 0.5 + 0.5) * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def uint8_to_tensor(arr: np.ndarray, device):
    """HWC uint8 numpy → [−1,1] 1CHW tensor."""
    import torch
    t = torch.from_numpy(arr.astype(np.float32) / 255.0)  # HWC [0,1]
    t = (t - 0.5) * 2.0                                   # HWC [-1,1]
    return t.permute(2, 0, 1).unsqueeze(0).to(device)     # 1CHW


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import torch
    from iris_bbdm_pad.models.anomaly_detector import BBDMAnomalyDetector
    from iris_bbdm_pad.data.corruption import apply_corruption_pipeline, filename_to_seed

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load corruption config
    with open(CORRUPTION_CFG) as f:
        cfg = json.load(f)["corruption_config"]

    # Load BBDM detector
    print("Loading BBDM model...")
    detector = BBDMAnomalyDetector(
        config_path=str(CONFIG_PATH),
        checkpoint_path=str(CKPT_PATH),
        device=device,
    )
    print("Model loaded.")

    # Load scores and pick samples that best contrast bonafide vs attacks visually.
    # Use recon_score (MSE+LPIPS, uncapped) — correlates directly with pixel L1 error.
    #   Bonafide → lowest recon_score (model reconstructs it best → low pixel error)
    #   Attacks  → highest recon_score (model struggles most → high pixel error)
    df = pd.read_csv(SCORES_CSV)
    selections = {}
    for at in ATTACK_TYPES:
        if at == "Live":
            subset = df[df["label"] == "bonafide"]
            idx = subset["recon_score"].idxmin()
            reason = f"lowest recon_score={subset.loc[idx,'recon_score']:.4f}"
        else:
            subset = df[df["attack_type"] == at]
            idx = subset["recon_score"].idxmax()
            reason = f"highest recon_score={subset.loc[idx,'recon_score']:.4f}"
        if len(subset) == 0:
            print(f"WARNING: no samples for {at}")
            continue
        selections[at] = subset.loc[idx, "filename"]
        print(f"  {at}: selected '{selections[at]}' ({reason})")

    # Create output folders
    input_dir = OUT_DIR / "input_images"
    recon_dir = OUT_DIR / "recon_images"
    viz_dir   = OUT_DIR / "visualizations"
    for d in (input_dir, recon_dir, viz_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Collect processed arrays for combined grid figure
    all_data = []   # list of (attack_type, corrupted_arr, recon_arr, error_map, mean, std)

    # Process each selection
    for at, fname in selections.items():
        safe = SAFE_NAME[at]
        img_path = TEST_IMAGES / fname
        if not img_path.exists():
            stem = Path(fname).stem
            for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"):
                candidate = TEST_IMAGES / (stem + ext)
                if candidate.exists():
                    img_path = candidate
                    break
            else:
                print(f"WARNING: image not found for {at}: {fname}")
                continue

        print(f"Processing {at} → {img_path.name}")

        # Load clean original, resize
        clean_pil = Image.open(img_path).convert("RGB").resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS
        )
        clean_arr = np.array(clean_pil)

        # Apply deterministic corruption (same as training/scoring)
        seed = filename_to_seed(Path(img_path.name).stem)
        corrupted_arr = apply_corruption_pipeline(
            clean_arr,
            seed=seed,
            target_size=IMAGE_SIZE,
            noise_sigma_range=tuple(cfg["noise_sigma_range"]),
            blur_kernel_choices=tuple(cfg["blur_kernel_choices"]),
            blur_sigma_range=tuple(cfg["blur_sigma_range"]),
            downscale_range=tuple(cfg["downscale_range"]),
        )

        # Run BBDM reconstruction
        corrupted_t = uint8_to_tensor(corrupted_arr, device)
        with torch.no_grad():
            recon_t = detector.reconstruct(corrupted_t)
        recon_arr = tensor_to_uint8(recon_t)

        # Save raw input and recon images
        Image.fromarray(corrupted_arr).save(input_dir / f"{safe}.png")
        Image.fromarray(recon_arr).save(recon_dir / f"{safe}.png")

        # Compute L1 error map (raw, unnormalized for stats)
        inp_f     = corrupted_arr.astype(np.float32) / 255.0
        recon_f   = recon_arr.astype(np.float32) / 255.0
        error_raw = np.abs(inp_f - recon_f).mean(axis=2)   # HW, [0,1]
        err_mean  = float(error_raw.mean())
        err_std   = float(error_raw.std())
        # Normalize per-image for display so jet colormap uses full range
        error_norm = (error_raw - error_raw.min()) / (error_raw.max() - error_raw.min() + 1e-8)

        all_data.append((at, corrupted_arr, recon_arr, error_norm, err_mean, err_std))

        # ── Individual per-attack figure (paper style) ───────────────────────
        label_str = "Bonafide (Live)" if at == "Live" else f"Attack: {at}"

        # gridspec: 3 image cols + 1 narrow colorbar col
        fig = plt.figure(figsize=(11, 3.8))
        fig.patch.set_facecolor("white")
        gs = fig.add_gridspec(
            2, 4,
            height_ratios=[10, 1],       # image row + text row
            width_ratios=[10, 10, 10, 1],
            hspace=0.08, wspace=0.06,
        )

        ax_inp  = fig.add_subplot(gs[0, 0])
        ax_rec  = fig.add_subplot(gs[0, 1])
        ax_err  = fig.add_subplot(gs[0, 2])
        ax_cb   = fig.add_subplot(gs[0, 3])
        ax_t0   = fig.add_subplot(gs[1, 0])
        ax_t1   = fig.add_subplot(gs[1, 1])
        ax_t2   = fig.add_subplot(gs[1, 2])

        for ax in (ax_t0, ax_t1, ax_t2):
            ax.axis("off")

        ax_inp.imshow(corrupted_arr)
        ax_inp.set_title("Input", fontsize=11, fontweight="bold", pad=4)
        ax_inp.axis("off")

        ax_rec.imshow(recon_arr)
        ax_rec.set_title("Reconstruction", fontsize=11, fontweight="bold", pad=4)
        ax_rec.axis("off")

        im = ax_err.imshow(error_norm, cmap="jet", vmin=0, vmax=1)
        ax_err.set_title("Error Map  |Input − Recon|", fontsize=11, fontweight="bold", pad=4)
        ax_err.axis("off")

        # Colorbar
        cb = fig.colorbar(im, cax=ax_cb)
        cb.ax.tick_params(labelsize=8)

        # mean ± std below error map only
        ax_t2.text(0.5, 0.6, f"{err_mean:.4f} ± {err_std:.4f}",
                   ha="center", va="center", fontsize=10, fontweight="bold",
                   transform=ax_t2.transAxes)

        # Row label on left
        fig.text(0.01, 0.55, label_str, va="center", ha="left",
                 fontsize=10, fontweight="bold", color="darkred" if at != "Live" else "darkgreen",
                 rotation=0)

        fig.savefig(viz_dir / f"{safe}.png", dpi=150, bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)
        print(f"  Saved visualization → {safe}.png  (error: {err_mean:.4f} ± {err_std:.4f})")

    # ── Combined grid figure (all 9 attack types, paper-ready) ───────────────
    print("\nGenerating combined grid figure...")
    n_rows = len(all_data)
    col_labels = ["Input", "Reconstruction", "Error Map  |Input − Recon|"]

    fig, axes = plt.subplots(
        n_rows, 4,
        figsize=(13, 3.2 * n_rows),
        gridspec_kw={"width_ratios": [10, 10, 10, 0.6], "wspace": 0.05, "hspace": 0.35},
    )
    fig.patch.set_facecolor("white")

    # Column headers on top row
    for c, title in enumerate(col_labels):
        axes[0, c].set_title(title, fontsize=12, fontweight="bold", pad=6)

    for r, (at, corrupted_arr, recon_arr, error_norm, err_mean, err_std) in enumerate(all_data):
        row_label = "Bonafide\n(Live)" if at == "Live" else at.replace(" ", "\n")

        axes[r, 0].imshow(corrupted_arr)
        axes[r, 0].axis("off")
        axes[r, 0].set_ylabel(row_label, fontsize=9, fontweight="bold", labelpad=4,
                               rotation=0, ha="right", va="center")

        axes[r, 1].imshow(recon_arr)
        axes[r, 1].axis("off")

        im = axes[r, 2].imshow(error_norm, cmap="jet", vmin=0, vmax=1)
        axes[r, 2].axis("off")
        axes[r, 2].text(
            0.5, -0.07,
            f"{err_mean:.4f} ± {err_std:.4f}",
            ha="center", va="top", fontsize=9, fontweight="bold",
            transform=axes[r, 2].transAxes,
        )

        # Colorbar per row
        cb = fig.colorbar(im, cax=axes[r, 3])
        cb.ax.tick_params(labelsize=7)
        # Only show tick labels on last row to reduce clutter
        if r < n_rows - 1:
            cb.ax.set_yticklabels([])

    fig.suptitle("BBDM Iris PAD — Reconstruction & Error Analysis", fontsize=13,
                 fontweight="bold", y=1.01)
    combined_path = viz_dir / "combined_all_attacks.png"
    fig.savefig(combined_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved combined grid → {combined_path.name}")

    # ── Zip everything ───────────────────────────────────────────────────────
    print(f"\nZipping to {ZIP_PATH} ...")
    ZIP_PATH.unlink(missing_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for folder in (input_dir, recon_dir, viz_dir):
            for fpath in sorted(folder.iterdir()):
                arcname = f"{folder.name}/{fpath.name}"
                zf.write(fpath, arcname)

    print(f"\nDone! Deliverable: {ZIP_PATH}")
    print(f"  input_images/  : {len(list(input_dir.iterdir()))} files")
    print(f"  recon_images/  : {len(list(recon_dir.iterdir()))} files")
    print(f"  visualizations/: {len(list(viz_dir.iterdir()))} files (9 individual + 1 combined grid)")


if __name__ == "__main__":
    main()
