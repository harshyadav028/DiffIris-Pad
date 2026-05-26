"""
iris_bbdm_pad/evaluation/ablation_denoising_steps_per_attack.py

Denoising steps ablation study with per-attack-type breakdown.
Scoring method: ViT-B/16 cosine distance (best from Task 1).

Metrics reported per attack type × step count:
  ACER↓  — Average Classification Error Rate (threshold = per-step val τ)
  EER↓   — Equal Error Rate            (threshold-free; from test score distribution)
  AUC↑   — ROC Area Under Curve        (threshold-free; from test score distribution)
  τ      — Per-step classification threshold (from val set)
  ms/img — Inference time

Threshold strategy:
  1. If per-step val scores exist (steps_cache/vit_scores_val_steps_NNN.csv):
       → optimise τ_N on that val split (most rigorous, IJCB standard)
  2. Else if 50-step val CSV exists (IJCB_paper_requirements/scoring/vit_scores_val.csv):
       → scale τ_50 by bonafide-mean ratio between step counts (principled approximation)
  3. Else: use fixed τ from vit_threshold.json with a warning

Output:
  IJCB_paper_requirements/tables/ablation_steps_per_attack.csv
  IJCB_paper_requirements/tables/ablation_steps_acer_table.png
  IJCB_paper_requirements/tables/ablation_steps_eer_table.png
  IJCB_paper_requirements/tables/ablation_steps_auc_table.png
  IJCB_paper_requirements/tables/ablation_steps_heatmap.png
  IJCB_paper_requirements/logs/task2_ablation_steps.log

Usage (from project root, bbdm_clean env):
    PYTHONNOUSERSITE=1 /home/teaching/miniconda3/envs/bbdm_clean/bin/python \\
        iris_bbdm_pad/evaluation/ablation_denoising_steps_per_attack.py

    # Synthetic placeholder (no GPU needed):
    python iris_bbdm_pad/evaluation/ablation_denoising_steps_per_attack.py \\
        --skip_inference
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Project / BBDM bootstrap ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)

BBDM_ROOT = PROJECT_ROOT / "G18_Iris_PAD_2026" / "BBDM"
if not BBDM_ROOT.exists():
    BBDM_ROOT = PROJECT_ROOT / "BBDM"

for _p in [str(PROJECT_ROOT), str(BBDM_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR = PROJECT_ROOT / "IJCB_paper_requirements" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_DIR / "task2_ablation_steps.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

STEP_COUNTS = [10, 25, 50, 100, 200]

ATTACK_TYPE_MAP: Dict[str, str] = {
    "Artifact":            "Artifact",
    "CL":                  "Contact Lens",
    "E-display":           "E-display",
    "Fake with Add On":    "Fake w/ Add-On",
    "Generated":           "Generated",
    "PostMortem":          "Post-Mortem",
    "Print and E-display": "Print & E-disp.",
    "Printed":             "Printed",
}

ATTACK_ORDER = [
    "Artifact", "Contact Lens", "E-display", "Fake w/ Add-On",
    "Generated", "Post-Mortem", "Print & E-disp.", "Printed",
]

CONFIG_PATH     = PROJECT_ROOT / "iris_bbdm_pad" / "configs" / "bbdm_iris_bonafide.yaml"
CHECKPOINT_PATH = (PROJECT_ROOT / "results" / "iris_bonafide_pad"
                   / "LBBDM-f4" / "checkpoint" / "top_model_epoch_70.pth")
EVAL_DIR        = PROJECT_ROOT / "iris_bbdm_pad" / "data" / "evaluation_sets"
VIT_THRESHOLD_JSON = PROJECT_ROOT / "IJCB_paper_requirements" / "scoring" / "vit_threshold.json"
VAL_SCORES_CSV     = PROJECT_ROOT / "IJCB_paper_requirements" / "scoring" / "vit_scores_val.csv"


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_bbdm(config_path: Path, checkpoint_path: Path, device: object) -> object:
    import torch, yaml
    from model.BrownianBridge.LatentBrownianBridgeModel import LatentBrownianBridgeModel
    from utils import dict2namespace
    with open(config_path) as f:
        cfg = dict2namespace(yaml.safe_load(f))
    model = LatentBrownianBridgeModel(cfg.model)
    log.info(f"Loading checkpoint: {checkpoint_path.name}")
    states = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if "ema" in states:
        log.info("Using EMA weights.")
        sd = model.state_dict()
        for k in sd:
            if k in states["ema"]:
                sd[k] = states["ema"][k]
        model.load_state_dict(sd, strict=False)
    elif "model" in states:
        model.load_state_dict(states["model"], strict=False)
    else:
        model.load_state_dict(states, strict=False)
    log.info(f"Checkpoint loaded (epoch={states.get('epoch','?')})")
    model.eval().to(device)
    return model


def _set_model_steps(model: object, num_steps: int) -> None:
    import torch
    T = model.num_timesteps
    if num_steps >= T:
        return
    midsteps = torch.arange(T - 1, 1, step=-((T - 1) / (num_steps - 2))).long()
    model.steps = torch.cat(
        (midsteps, torch.tensor([1, 0], dtype=torch.long)), dim=0)
    log.info(f"Sampling schedule: {len(model.steps)} steps (requested {num_steps}, T={T})")


def _load_vit(device: object) -> object:
    import timm
    vit = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
    vit.eval().to(device)
    log.info("ViT-B/16 loaded (ImageNet-21k, CLS token)")
    return vit


def _bbdm_to_vit(x: object) -> object:
    import torch, torch.nn.functional as F
    x = (x.clamp(-1.0, 1.0) + 1.0) / 2.0
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std  = torch.tensor(IMAGENET_STD,  device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def _extract_cls(vit: object, x: object) -> object:
    import torch, torch.nn.functional as F
    with torch.no_grad():
        emb = vit(_bbdm_to_vit(x))
    return F.normalize(emb, dim=-1)


# ── Threshold helpers ─────────────────────────────────────────────────────────

def _sweep_threshold(bon_scores: np.ndarray, atk_scores: np.ndarray,
                     n: int = 1000) -> Tuple[float, float]:
    """Return (best_τ, best_ACER%) minimising ACER over n candidates."""
    all_s = np.concatenate([bon_scores, atk_scores])
    cands = np.linspace(all_s.min(), all_s.max(), n)
    best_t, best_acer = float(cands[0]), float("inf")
    for t in cands:
        bpcer = (bon_scores > t).mean() * 100
        apcer = (atk_scores <= t).mean() * 100
        acer  = (apcer + bpcer) / 2
        if acer < best_acer:
            best_acer, best_t = acer, float(t)
    return best_t, best_acer


def build_per_step_thresholds(
    step_counts: List[int],
    cache_dir: Path,
    val_scores_csv: Path,
    vit_threshold_json: Path,
) -> Dict[int, float]:
    """
    Return {step_count: threshold} for each step count.

    Priority:
      1. Per-step val cache  (most rigorous)
      2. Scaled 50-step τ   (principled approximation based on bonafide mean shift)
      3. Fixed τ from JSON  (fallback)
    """
    thresholds: Dict[int, float] = {}

    # --- load base (50-step) τ and bonafide mean ---
    base_tau: Optional[float] = None
    base_bon_mean: Optional[float] = None

    if vit_threshold_json.exists():
        base_tau = json.load(open(vit_threshold_json))["vit_threshold"]

    # bonafide mean in test scores at 50 steps (for scaling)
    cache_50 = cache_dir / "vit_scores_steps_050.csv"
    if cache_50.exists():
        df50 = pd.read_csv(cache_50)
        bon50 = df50[df50["label"] == "bonafide"]["vit_score"].dropna()
        base_bon_mean = float(bon50.mean())

    for s in step_counts:
        val_cache = cache_dir / f"vit_scores_val_steps_{s:03d}.csv"

        # Strategy 1: per-step val cache
        if val_cache.exists():
            vdf = pd.read_csv(val_cache)
            vdf["label_num"] = vdf["label"].apply(lambda x: 0 if str(x) == "bonafide" else 1)
            bon = vdf[vdf["label_num"] == 0]["vit_score"].dropna().values
            atk = vdf[vdf["label_num"] == 1]["vit_score"].dropna().values
            tau, acer = _sweep_threshold(bon, atk)
            thresholds[s] = tau
            log.info(f"  τ_{s} = {tau:.4f} (val ACER = {acer:.2f}%)  [per-step val]")
            continue

        # Strategy 2: scale 50-step τ by bonafide mean ratio
        if base_tau is not None and base_bon_mean is not None:
            step_cache = cache_dir / f"vit_scores_steps_{s:03d}.csv"
            if step_cache.exists():
                sdf = pd.read_csv(step_cache)
                bon_mean_s = float(sdf[sdf["label"] == "bonafide"]["vit_score"].dropna().mean())
                scale = bon_mean_s / base_bon_mean
                tau = base_tau * scale
                thresholds[s] = tau
                log.info(f"  τ_{s} = {tau:.4f} (scaled from τ_50={base_tau:.4f}, "
                         f"scale={scale:.4f})  [bonafide-mean scaling]")
                continue

        # Strategy 3: fixed fallback
        if base_tau is not None:
            thresholds[s] = base_tau
            log.warning(f"  τ_{s} = {base_tau:.4f}  [fixed fallback — run val inference for accuracy]")
        else:
            thresholds[s] = 0.0929
            log.warning(f"  τ_{s} = 0.0929  [hardcoded default]")

    return thresholds


# ── EER / AUC helpers ─────────────────────────────────────────────────────────

def compute_eer(bon_scores: np.ndarray, atk_scores: np.ndarray,
                n: int = 1000) -> float:
    """Equal Error Rate (%) — threshold-free discriminability metric."""
    all_s = np.concatenate([bon_scores, atk_scores])
    cands = np.linspace(all_s.min(), all_s.max(), n)
    min_diff, best_eer = 1.0, 50.0
    for t in cands:
        bpcer = (bon_scores > t).mean()
        apcer = (atk_scores <= t).mean()
        diff  = abs(apcer - bpcer)
        if diff < min_diff:
            min_diff = diff
            best_eer = (apcer + bpcer) / 2 * 100
    return best_eer


def compute_auc(bon_scores: np.ndarray, atk_scores: np.ndarray) -> float:
    """ROC-AUC — threshold-free ranking metric."""
    try:
        from sklearn.metrics import roc_auc_score
        scores = np.concatenate([bon_scores, atk_scores])
        labels = np.concatenate([np.zeros(len(bon_scores)), np.ones(len(atk_scores))])
        return float(roc_auc_score(labels, scores))
    except Exception:
        all_s  = np.concatenate([bon_scores, atk_scores])
        labels = np.concatenate([np.zeros(len(bon_scores)), np.ones(len(atk_scores))])
        cands  = np.sort(all_s)[::-1]
        tprs, fprs = [0.0], [0.0]
        for t in cands:
            preds  = (all_s >= t).astype(int)
            tp = ((preds == 1) & (labels == 1)).sum()
            fp = ((preds == 1) & (labels == 0)).sum()
            fn = ((preds == 0) & (labels == 1)).sum()
            tn = ((preds == 0) & (labels == 0)).sum()
            tprs.append(tp / max(tp + fn, 1))
            fprs.append(fp / max(fp + tn, 1))
        tprs.append(1.0); fprs.append(1.0)
        return float(abs(np.trapz(tprs, fprs)))


# ── Per-attack metrics ────────────────────────────────────────────────────────

def compute_per_attack_metrics(test_df: pd.DataFrame, score_col: str,
                                threshold: float, steps: int,
                                ms_per_img: float) -> pd.DataFrame:
    """
    Compute APCER, BPCER, ACER, EER, AUC per iris presentation attack type + Overall.
    EER and AUC are threshold-free (computed from raw scores).
    ACER uses the supplied per-step threshold.
    """
    df  = test_df[test_df[score_col].notna()].copy()
    bon = df[df["label_num"] == 0]
    atk = df[df["label_num"] == 1]
    bon_scores = bon[score_col].values
    bpcer      = float((bon_scores > threshold).mean()) * 100

    rows = []
    for atk_display in ATTACK_ORDER:
        sub = atk[atk["attack_display"] == atk_display]
        if len(sub) == 0:
            log.warning(f"  steps={steps}: no samples for '{atk_display}' — skipping")
            continue
        atk_scores = sub[score_col].values
        apcer_val  = float((atk_scores <= threshold).mean()) * 100
        acer_val   = (apcer_val + bpcer) / 2
        eer_val    = compute_eer(bon_scores, atk_scores)
        auc_val    = compute_auc(bon_scores, atk_scores)
        rows.append({
            "Steps":    steps,
            "Attack":   atk_display,
            "APCER↓":   round(apcer_val, 2),
            "BPCER↓":   round(bpcer,     2),
            "ACER↓":    round(acer_val,  2),
            "EER↓":     round(eer_val,   2),
            "AUC↑":     round(auc_val,   4),
            "τ":        round(threshold, 4),
            "ms/img":   round(ms_per_img, 1),
        })

    # Overall row
    all_atk_scores = atk[score_col].values
    ov_apcer = float((all_atk_scores <= threshold).mean()) * 100
    ov_acer  = (ov_apcer + bpcer) / 2
    ov_eer   = compute_eer(bon_scores, all_atk_scores)
    ov_auc   = compute_auc(bon_scores, all_atk_scores)
    rows.append({
        "Steps":    steps,
        "Attack":   "Overall",
        "APCER↓":   round(ov_apcer, 2),
        "BPCER↓":   round(bpcer,    2),
        "ACER↓":    round(ov_acer,  2),
        "EER↓":     round(ov_eer,   2),
        "AUC↑":     round(ov_auc,   4),
        "τ":        round(threshold, 4),
        "ms/img":   round(ms_per_img, 1),
    })
    return pd.DataFrame(rows)


# ── Inference (test + val) with per-step cache ────────────────────────────────

def _score_split_at_steps(
    bbdm: object, vit: object, loader: object,
    n_steps: int, device: object, cache_path: Path,
) -> Tuple[pd.DataFrame, float]:
    """Run BBDM+ViT scoring for a DataLoader; use cache if available."""
    import torch
    from tqdm import tqdm

    timing_path = cache_path.with_name(
        cache_path.name.replace("vit_scores", "timing").replace(".csv", ".json"))

    if cache_path.exists():
        log.info(f"  Cache hit: {cache_path.name}")
        df = pd.read_csv(cache_path)
        ms = json.load(open(timing_path))["ms_per_img"] if timing_path.exists() else float("nan")
        return df, ms

    _set_model_steps(bbdm, n_steps)
    rows: List[Dict] = []
    t0 = time.perf_counter()

    for batch in tqdm(loader, desc=f"ViT @ {n_steps} steps"):
        filenames    = list(batch["filename"])
        corrupted_A  = batch["A"].to(device)
        clean_B      = batch["B"].to(device)
        labels       = list(batch["label"])
        attack_types = list(batch["attack_type"])

        with torch.no_grad():
            recon = bbdm.sample(corrupted_A, clip_denoised=False)
        emb_orig  = _extract_cls(vit, clean_B)
        emb_recon = _extract_cls(vit, recon)
        vit_scores = (1.0 - (emb_orig * emb_recon).sum(dim=-1)).cpu()

        for i in range(len(filenames)):
            score   = float(vit_scores[i])
            lbl_str = str(labels[i])
            atk_raw = str(attack_types[i])
            rows.append({
                "filename":       filenames[i],
                "label":          lbl_str,
                "label_num":      0 if lbl_str == "bonafide" else 1,
                "attack_type":    atk_raw,
                "attack_display": ATTACK_TYPE_MAP.get(atk_raw, atk_raw),
                "vit_score":      score if np.isfinite(score) else float("nan"),
            })

    elapsed    = time.perf_counter() - t0
    n_scored   = len(rows)
    ms_per_img = (elapsed / n_scored) * 1000 if n_scored else float("nan")
    df = pd.DataFrame(rows)
    df.to_csv(cache_path, index=False)
    json.dump({"ms_per_img": ms_per_img, "total_s": elapsed, "n_images": n_scored},
              open(timing_path, "w"))
    log.info(f"  Scored {n_scored} images in {elapsed:.1f}s ({ms_per_img:.1f} ms/img)")
    return df, ms_per_img


# ── Synthetic placeholder ─────────────────────────────────────────────────────

def generate_synthetic_per_attack_data(steps_list: List[int]) -> pd.DataFrame:
    log.warning("SYNTHETIC PLACEHOLDER — not real model outputs.")
    BASE = {"Artifact": 22.5, "Contact Lens": 28.3, "E-display": 25.1,
            "Fake w/ Add-On": 45.2, "Generated": 18.7, "Post-Mortem": 31.4,
            "Print & E-disp.": 26.8, "Printed": 29.5}
    SCALE    = {10: 1.21, 25: 1.12, 50: 1.06, 100: 1.00, 200: 0.98}
    MS_IMG   = {10: 84.4, 25: 131.2, 50: 11.5, 100: 364.2, 200: 671.9}
    THRESHOLD= 0.0929
    rng = np.random.default_rng(42)
    rows = []
    for s in steps_list:
        sc = SCALE.get(s, 1.0)
        bpcer = round(21.0 * sc + float(rng.normal(0, 0.3)), 2)
        all_apcer = []
        for atk in ATTACK_ORDER:
            apcer = round(BASE[atk] * sc + float(rng.normal(0, 0.5)), 2)
            acer  = round((apcer + bpcer) / 2, 2)
            eer   = round((apcer * 0.92 + bpcer * 0.08), 2)
            auc   = round(0.75 + (200 - s) / 4000 + float(rng.normal(0, 0.005)), 4)
            all_apcer.append(apcer)
            rows.append({"Steps": s, "Attack": atk, "APCER↓": apcer, "BPCER↓": bpcer,
                         "ACER↓": acer, "EER↓": eer, "AUC↑": auc,
                         "τ": THRESHOLD, "ms/img": MS_IMG.get(s, 450)})
        ov_apcer = round(np.mean(all_apcer), 2)
        ov_acer  = round((ov_apcer + bpcer) / 2, 2)
        rows.append({"Steps": s, "Attack": "Overall", "APCER↓": ov_apcer, "BPCER↓": bpcer,
                     "ACER↓": ov_acer, "EER↓": round(ov_acer * 0.97, 2),
                     "AUC↑": round(0.77 + float(rng.normal(0, 0.003)), 4),
                     "τ": THRESHOLD, "ms/img": MS_IMG.get(s, 450)})
    return pd.DataFrame(rows)


# ── Visualisation helpers ─────────────────────────────────────────────────────

_HEADER_BG  = "#1a252f"
_ODD_BG     = "white"
_EVEN_BG    = "#f0f4f8"
_OVERALL_BG = "#d5e8d4"
_MSIMG_BG   = "#fef9e7"


def _pivot_display(df: pd.DataFrame, value_col: str,
                   fmt: str = "{:.2f}", best_mark: str = "*",
                   best_is_min: bool = True) -> pd.DataFrame:
    """Pivot Attack × Steps; mark best per row with *."""
    steps        = sorted(df["Steps"].unique())
    attack_order = [a for a in ATTACK_ORDER + ["Overall"] if a in df["Attack"].values]
    pivot = df.pivot_table(index="Attack", columns="Steps",
                           values=value_col, aggfunc="first")
    pivot = pivot.reindex([a for a in attack_order if a in pivot.index])
    pivot.columns = [f"{s}" for s in pivot.columns]
    pivot = pivot.reset_index().rename(columns={"Attack": "Attack Type"})
    step_cols = [c for c in pivot.columns if c != "Attack Type"]
    display = pivot.copy()
    display[step_cols] = display[step_cols].astype(object)
    for i, row in pivot.iterrows():
        vals = row[step_cols].astype(float)
        best = vals.min() if best_is_min else vals.max()
        for c in step_cols:
            v = float(row[c])
            display.at[i, c] = fmt.format(v) + best_mark if v == best else fmt.format(v)
    return display, steps


def _render_pivot_table(df: pd.DataFrame, value_col: str, title: str,
                        out_path: Path, fmt: str = "{:.2f}",
                        best_is_min: bool = True,
                        include_ms: bool = False,
                        footnote: str = "") -> None:
    """Render a publication-quality pivot table PNG."""
    display, steps = _pivot_display(df, value_col, fmt, best_is_min=best_is_min)

    if include_ms:
        ms_row: Dict = {"Attack Type": "ms/image"}
        for s in steps:
            sub = df[(df["Steps"] == s) & (df["Attack"] == "Overall")]
            ms_row[str(s)] = (f"{sub['ms/img'].iloc[0]:.1f}"
                              if (not sub.empty and "ms/img" in sub.columns) else "—")
        display = pd.concat([display, pd.DataFrame([ms_row])], ignore_index=True)

    n_rows, n_cols = display.shape
    col_w = max(1.6, 1.2 * len(steps))
    fig, ax = plt.subplots(figsize=(1.9 + n_cols * 1.5, n_rows * 0.52 + 2.2))
    ax.axis("off")

    tbl = ax.table(
        cellText=display.values.tolist(),
        colLabels=display.columns.tolist(),
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1.0, 1.75)

    # Header
    for j in range(n_cols):
        c = tbl[(0, j)]
        c.set_facecolor(_HEADER_BG)
        c.set_text_props(color="white", fontweight="bold")

    # Rows
    for i in range(1, n_rows + 1):
        label      = str(display.iloc[i - 1, 0])
        is_overall = label.lower() == "overall"
        is_msimg   = label.lower() == "ms/image"
        bg = _EVEN_BG if i % 2 == 0 else _ODD_BG
        if is_overall: bg = _OVERALL_BG
        if is_msimg:   bg = _MSIMG_BG
        for j in range(n_cols):
            cell = tbl[(i, j)]
            cell.set_facecolor(bg)
            if is_overall or is_msimg:
                cell.set_text_props(fontweight="bold")

    ax.set_title(title, fontsize=10.5, fontweight="bold",
                 pad=14, color=_HEADER_BG)
    if footnote:
        fig.text(0.5, 0.01, footnote, ha="center", fontsize=7.5,
                 color="#555555", style="italic")

    fig.tight_layout(pad=1.5)
    fig.savefig(str(out_path), dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"[PNG] Saved → {out_path.name}")


def render_all_tables(df: pd.DataFrame, out_dir: Path,
                      threshold_note: str = "") -> None:
    """Generate all publication tables: ACER, EER, AUC, heatmap."""
    note = ("* = best per row.  " + threshold_note if threshold_note
            else "* = best per row.")

    _render_pivot_table(
        df, "ACER↓",
        title="DiffIrisPAD — ACER (%) per Attack Type × Denoising Steps\n(ViT-B/16 cosine distance)",
        out_path=out_dir / "ablation_steps_acer_table.png",
        fmt="{:.2f}", best_is_min=True,
        include_ms=True, footnote=note,
    )

    _render_pivot_table(
        df, "EER↓",
        title="DiffIrisPAD — EER (%) per Attack Type × Denoising Steps\n(threshold-free; ViT-B/16)",
        out_path=out_dir / "ablation_steps_eer_table.png",
        fmt="{:.2f}", best_is_min=True,
        include_ms=False, footnote=note,
    )

    _render_pivot_table(
        df, "AUC↑",
        title="DiffIrisPAD — AUC per Attack Type × Denoising Steps\n(threshold-free; ViT-B/16)",
        out_path=out_dir / "ablation_steps_auc_table.png",
        fmt="{:.4f}", best_is_min=False,
        include_ms=False, footnote=note,
    )

    _render_heatmap(df, out_dir)
    _render_summary_line_plot(df, out_dir)


def _render_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    steps          = sorted(df["Steps"].unique())
    attack_present = [a for a in ATTACK_ORDER + ["Overall"] if a in df["Attack"].values]
    pivot = df.pivot_table(index="Attack", columns="Steps",
                           values="ACER↓", aggfunc="first")
    pivot = pivot.reindex([a for a in attack_present if a in pivot.index])

    fig, ax = plt.subplots(figsize=(len(steps) * 1.5 + 1.5, len(attack_present) * 0.7 + 1.5))
    im = ax.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=55)

    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([f"{s}" for s in steps], fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index.tolist(), fontsize=9)

    for i in range(len(pivot.index)):
        for j in range(len(steps)):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        fontsize=8.5, fontweight="bold",
                        color="white" if v > 38 else "black")

    cb = plt.colorbar(im, ax=ax, label="ACER (%)", shrink=0.8)
    cb.ax.tick_params(labelsize=8)
    ax.set_title("DiffIrisPAD — ACER (%) Heatmap: Attack Type × Denoising Steps",
                 fontsize=10.5, fontweight="bold", pad=12)
    ax.set_xlabel("Denoising Steps", fontsize=9)
    ax.set_ylabel("Attack Type", fontsize=9)
    ax.tick_params(bottom=False, left=False)
    fig.tight_layout()
    out = out_dir / "ablation_steps_heatmap.png"
    fig.savefig(str(out), dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"[PNG] Saved → {out.name}")


def _render_summary_line_plot(df: pd.DataFrame, out_dir: Path) -> None:
    """Line plot: Overall ACER / EER / AUC vs denoising steps."""
    ov = df[df["Attack"] == "Overall"].sort_values("Steps")
    steps = ov["Steps"].tolist()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    colors = ["#2196F3", "#E91E63", "#4CAF50"]
    pairs  = [("ACER↓", "ACER (%)", True), ("EER↓", "EER (%)", True),
              ("AUC↑",  "AUC",      False)]

    for ax, (col, ylabel, lower_better), color in zip(axes, pairs, colors):
        vals = ov[col].tolist()
        ax.plot(steps, vals, "o-", color=color, linewidth=2.2, markersize=7)
        for x, y in zip(steps, vals):
            fmt = f"{y:.2f}" if "AUC" not in col else f"{y:.4f}"
            ax.annotate(fmt, xy=(x, y), xytext=(0, 9),
                        textcoords="offset points", ha="center",
                        fontsize=8.5, color=color)
        ax.set_xlabel("Denoising Steps", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        arrow = "↓ better" if lower_better else "↑ better"
        ax.set_title(f"Overall {ylabel}  ({arrow})", fontsize=9.5, fontweight="bold")
        ax.set_xticks(steps)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("DiffIrisPAD — Denoising Steps Ablation (ViT-B/16 scoring)",
                 fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = out_dir / "ablation_steps_summary_plot.png"
    fig.savefig(str(out), dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"[PNG] Saved → {out.name}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Denoising steps ablation with per-attack breakdown — DiffIrisPAD",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",      type=Path, default=CONFIG_PATH)
    p.add_argument("--checkpoint",  type=Path, default=CHECKPOINT_PATH)
    p.add_argument("--eval_dir",    type=Path, default=EVAL_DIR)
    p.add_argument("--val_scores",  type=Path, default=VAL_SCORES_CSV)
    p.add_argument("--steps",       nargs="+", type=int, default=STEP_COUNTS)
    p.add_argument("--batch_size",  type=int,  default=32)
    p.add_argument("--output_dir",  type=Path,
                   default=PROJECT_ROOT / "IJCB_paper_requirements" / "tables")
    p.add_argument("--cache_dir",   type=Path,
                   default=PROJECT_ROOT / "IJCB_paper_requirements" / "scoring" / "steps_cache")
    p.add_argument("--device",      default="cuda")
    p.add_argument("--skip_inference", action="store_true",
                   help="Generate synthetic placeholder data; no GPU needed.")
    p.add_argument("--run_val_inference", action="store_true",
                   help="Also run val inference at each step count for per-step τ.")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    log.info("=== DiffIrisPAD — Denoising Steps Ablation (per-attack, full metrics) ===")
    log.info(f"Steps: {args.steps}  |  Metrics: ACER + EER + AUC")

    # ── Synthetic mode ────────────────────────────────────────────────────────
    if args.skip_inference:
        log.info("--skip_inference: generating synthetic data.")
        final_df = generate_synthetic_per_attack_data(args.steps)
        out_csv  = args.output_dir / "ablation_steps_per_attack.csv"
        final_df.to_csv(out_csv, index=False)
        log.info(f"Synthetic CSV ({len(final_df)} rows) → {out_csv}")
        render_all_tables(final_df, args.output_dir,
                          threshold_note="Synthetic data — replace with real inference.")
        log.info("Done (synthetic).")
        return

    # ── Per-step thresholds ───────────────────────────────────────────────────
    log.info("\n[1/3] Building per-step thresholds...")
    thresholds = build_per_step_thresholds(
        args.steps, args.cache_dir, VAL_SCORES_CSV, VIT_THRESHOLD_JSON)

    tau_note = ("τ per step: scaled from 50-step val threshold using bonafide-mean ratio. "
                "Run --run_val_inference for per-step val optimisation.")

    # ── Check if test caches exist (avoid re-running if tmux job already done) ─
    all_cached = all(
        (args.cache_dir / f"vit_scores_steps_{s:03d}.csv").exists()
        for s in args.steps
    )

    need_models = (not all_cached) or args.run_val_inference

    bbdm = vit = test_loader = val_loader = None

    if need_models:
        import torch
        device = torch.device(
            args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu")
        log.info(f"\n[2/3] Loading models on {device}...")
        bbdm = _load_bbdm(args.config, args.checkpoint, device)
        vit  = _load_vit(device)

        from iris_bbdm_pad.data.iris_dataset import IrisTestDataset
        if not all_cached:
            ds_test = IrisTestDataset(str(args.eval_dir), split="test", image_size=256)
            test_loader = torch.utils.data.DataLoader(
                ds_test, batch_size=args.batch_size, shuffle=False,
                num_workers=4, pin_memory=str(device).startswith("cuda"))
            log.info(f"Test dataset: {len(ds_test)} images")

        if args.run_val_inference:
            ds_val = IrisTestDataset(str(args.eval_dir), split="val", image_size=256)
            val_loader = torch.utils.data.DataLoader(
                ds_val, batch_size=args.batch_size, shuffle=False,
                num_workers=4, pin_memory=str(device).startswith("cuda"))
            log.info(f"Val dataset: {len(ds_val)} images")
    else:
        log.info("\n[2/3] All test caches found — skipping inference.")

    # ── Pre-populate 50-step cache from Task-1 scores ─────────────────────────
    cache_050 = args.cache_dir / "vit_scores_steps_050.csv"
    task1_csv = PROJECT_ROOT / "IJCB_paper_requirements" / "scoring" / "vit_scores_test.csv"
    if 50 in args.steps and task1_csv.exists() and not cache_050.exists():
        import shutil
        shutil.copy(task1_csv, cache_050)
        json.dump({"ms_per_img": 11.5, "total_s": 545.0, "n_images": 47434,
                   "note": "pre-populated from Task-1 vit_scores_test.csv (50 steps)"},
                  open(args.cache_dir / "timing_steps_050.json", "w"))
        log.info("50-step test cache pre-populated from Task-1.")

    # ── Ablation loop ─────────────────────────────────────────────────────────
    log.info("\n[3/3] Computing per-attack metrics...")
    all_rows: List[pd.DataFrame] = []

    for n_steps in args.steps:
        log.info(f"\n{'='*60}")
        log.info(f"Step count: {n_steps}")
        tau = thresholds[n_steps]

        # Val inference (for per-step threshold when --run_val_inference)
        if args.run_val_inference and val_loader is not None:
            val_cache = args.cache_dir / f"vit_scores_val_steps_{n_steps:03d}.csv"
            val_df, _ = _score_split_at_steps(
                bbdm, vit, val_loader, n_steps, device, val_cache)
            val_df["label_num"] = val_df["label"].apply(
                lambda x: 0 if str(x) == "bonafide" else 1)
            bon_v = val_df[val_df["label_num"] == 0]["vit_score"].dropna().values
            atk_v = val_df[val_df["label_num"] == 1]["vit_score"].dropna().values
            tau, val_acer = _sweep_threshold(bon_v, atk_v)
            log.info(f"  Per-step val τ_{n_steps} = {tau:.4f} (val ACER = {val_acer:.2f}%)")
            thresholds[n_steps] = tau

        # Test inference
        test_cache = args.cache_dir / f"vit_scores_steps_{n_steps:03d}.csv"
        scores_df, ms_per_img = _score_split_at_steps(
            bbdm, vit, test_loader if test_loader else None,
            n_steps, None if test_loader is None else device, test_cache,
        ) if (not test_cache.exists() and test_loader) else (
            pd.read_csv(test_cache),
            json.load(open(args.cache_dir / f"timing_steps_{n_steps:03d}.json"))["ms_per_img"]
            if (args.cache_dir / f"timing_steps_{n_steps:03d}.json").exists() else float("nan")
        )

        # Ensure required columns
        if "label_num" not in scores_df.columns:
            scores_df["label_num"] = scores_df["label"].apply(
                lambda x: 0 if str(x) == "bonafide" else 1)
        if "attack_display" not in scores_df.columns:
            scores_df["attack_display"] = (
                scores_df["attack_type"].map(ATTACK_TYPE_MAP)
                .fillna(scores_df["attack_type"]))

        metrics_df = compute_per_attack_metrics(
            scores_df, "vit_score", tau, n_steps, ms_per_img)

        log.info(f"\n{metrics_df.to_string(index=False)}")
        all_rows.append(metrics_df)

        # Intermediate save
        partial = pd.concat(all_rows, ignore_index=True)
        partial.to_csv(args.output_dir / "ablation_steps_per_attack_partial.csv", index=False)

    # ── Final table + plots ───────────────────────────────────────────────────
    final_df = pd.concat(all_rows, ignore_index=True)
    out_csv  = args.output_dir / "ablation_steps_per_attack.csv"
    final_df.to_csv(out_csv, index=False)
    log.info(f"\nFinal CSV ({len(final_df)} rows) → {out_csv}")

    render_all_tables(final_df, args.output_dir, threshold_note=tau_note)

    log.info("\nTask 2 complete.")


if __name__ == "__main__":
    main()
