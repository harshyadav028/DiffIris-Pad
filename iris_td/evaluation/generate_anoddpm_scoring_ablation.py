"""
iris_td/evaluation/generate_anoddpm_scoring_ablation.py

Scoring method ablation for AnoDDPM+DDIM (Simplex, t*=500, 50 DDIM steps).
Mirrors Table 3 (DiffIrisPAD scoring ablation) exactly:
  three scorers with per-attack-type threshold protocol.

Scorers:
  1. ViT-only      : vit_score (cosine distance, no negation needed)
  2. Fixed Fusion  : 0.5 * vit_norm + 0.5 * lpips_norm
  3. Dynamic Fusion: alpha * vit_norm + (1-alpha) * lpips_norm,
                     alpha in {0.1,...,0.9} optimised on overall val ACER

LPIPS negated before fusion: attacks reconstruct more faithfully under
partial noising (t*=500), so raw lpips_attack < lpips_bonafide;
negation makes higher score = more anomalous (attack).

Normalisation: min-max computed on val set, applied to both val and test.

Per-attack threshold protocol: for each (scorer, attack_type), tau is
independently optimised on val(full bonafide + that attack type only)
by minimising ACER over 1,000 candidates.

Input CSVs (must be pre-computed before running this script):
  Val  ViT : iris_td/pad_scores/ddpm_val_simplex_tstar500_ddim_steps50_vit.csv
  Test ViT : iris_td/pad_scores/ddpm_test_simplex_tstar500_ddim_steps50_vit.csv
  Val  LPIPS: iris_td/pad_scores/ddpm_val_simplex_tstar500_ddim_steps50_fixed.csv
  Test LPIPS: iris_td/pad_scores/ddpm_test_simplex_tstar500_ddim_steps50_fixed.csv

Output:
  iris_td/final_results/anoddpm_scoring_ablation_per_attack.csv

Usage (CPU only, no GPU):
  cd /home/teaching/Documents/Geetanjali_PhD_IRIS_PAD
  PYTHONNOUSERSITE=1 /home/teaching/miniconda3/envs/bbdm_clean/bin/python \
      iris_td/evaluation/generate_anoddpm_scoring_ablation.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCORES_DIR   = PROJECT_ROOT / "iris_td" / "pad_scores"
OUT_DIR      = PROJECT_ROOT / "iris_td" / "final_results"
OUT_CSV      = OUT_DIR / "anoddpm_scoring_ablation_per_attack.csv"

VAL_VIT_CSV    = SCORES_DIR / "ddpm_val_simplex_tstar500_ddim_steps50_vit.csv"
TEST_VIT_CSV   = SCORES_DIR / "ddpm_test_simplex_tstar500_ddim_steps50_vit.csv"
VAL_LPIPS_CSV  = SCORES_DIR / "ddpm_val_simplex_tstar500_ddim_steps50_fixed.csv"
TEST_LPIPS_CSV = SCORES_DIR / "ddpm_test_simplex_tstar500_ddim_steps50_fixed.csv"

# Raw attack_type in CSV  →  display name (matches Table 3 convention)
ATTACK_TYPE_MAP = {
    "Artifact":             "Artifact",
    "CL":                   "Contact Lens",
    "E-display":            "E-display",
    "Fake with Add On":     "Fake w/ Add-On",
    "Fake_with_Add_On":     "Fake w/ Add-On",       # ddpm_vit_scorer.py variant
    "Generated":            "Generated",
    "PostMortem":           "Post-Mortem",
    "Print and E-display":  "Print & E-disp.",
    "Print_E-display":      "Print & E-disp.",       # ddpm_vit_scorer.py variant
    "Printed":              "Printed",
}

ATTACK_ORDER = [
    "Artifact", "Contact Lens", "E-display", "Fake w/ Add-On",
    "Generated", "Post-Mortem", "Print & E-disp.", "Printed",
]

N_CANDS   = 1000
ALPHA_GRID = np.round(np.arange(0.1, 1.0, 0.1), 1)   # {0.1, 0.2, ..., 0.9}


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def sweep_threshold(bon: np.ndarray, atk: np.ndarray, n: int = N_CANDS):
    all_s = np.concatenate([bon, atk])
    cands = np.linspace(all_s.min(), all_s.max(), n)
    best_t, best_a = float(cands[0]), float("inf")
    for t in cands:
        bpcer = float((bon > t).mean()) * 100.0
        apcer = float((atk <= t).mean()) * 100.0
        acer  = (apcer + bpcer) / 2.0
        if acer < best_a:
            best_a, best_t = acer, float(t)
    return best_t, best_a


def compute_eer(bon: np.ndarray, atk: np.ndarray, n: int = N_CANDS) -> float:
    all_s = np.concatenate([bon, atk])
    cands = np.linspace(all_s.min(), all_s.max(), n)
    min_diff, best_eer = 1.0, 50.0
    for t in cands:
        diff = abs(float((bon > t).mean()) - float((atk <= t).mean()))
        if diff < min_diff:
            min_diff = diff
            best_eer = (float((bon > t).mean()) + float((atk <= t).mean())) / 2.0 * 100.0
    return best_eer


def compute_auc(bon: np.ndarray, atk: np.ndarray) -> float:
    scores = np.concatenate([bon, atk])
    labels = np.concatenate([np.zeros(len(bon)), np.ones(len(atk))])
    return float(roc_auc_score(labels, scores))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def minmax_norm(s: pd.Series, mn: float, mx: float) -> pd.Series:
    return (s - mn) / (mx - mn + 1e-8)


def load_and_merge():
    """Load ViT + LPIPS CSVs, negate LPIPS, normalise, add fusion columns."""
    for p in [VAL_VIT_CSV, TEST_VIT_CSV, VAL_LPIPS_CSV, TEST_LPIPS_CSV]:
        if not p.exists():
            sys.exit(f"ERROR: missing required file: {p}")

    val_vit  = pd.read_csv(VAL_VIT_CSV)
    test_vit = pd.read_csv(TEST_VIT_CSV)
    val_lp   = pd.read_csv(VAL_LPIPS_CSV)[["filename", "lpips_score"]]
    test_lp  = pd.read_csv(TEST_LPIPS_CSV)[["filename", "lpips_score"]]

    # Negate LPIPS: raw attack < bonafide, so negate so attack > bonafide
    val_lp["lpips_score"]  = -val_lp["lpips_score"]
    test_lp["lpips_score"] = -test_lp["lpips_score"]

    val_df  = val_vit.merge(val_lp,  on="filename", how="inner")
    test_df = test_vit.merge(test_lp, on="filename", how="inner")

    # Normalise using val statistics → apply to test
    for col in ["vit_score", "lpips_score"]:
        mn = float(val_df[col].min())
        mx = float(val_df[col].max())
        val_df[col  + "_norm"] = minmax_norm(val_df[col],  mn, mx)
        test_df[col + "_norm"] = minmax_norm(test_df[col], mn, mx)

    # Remap attack_type to display names
    val_df["attack_type"]  = val_df["attack_type"].map(
        lambda x: ATTACK_TYPE_MAP.get(x, x))
    test_df["attack_type"] = test_df["attack_type"].map(
        lambda x: ATTACK_TYPE_MAP.get(x, x))

    return val_df, test_df


# ---------------------------------------------------------------------------
# Alpha search for dynamic fusion
# ---------------------------------------------------------------------------

def find_best_alpha(val_df: pd.DataFrame) -> float:
    """Find alpha in ALPHA_GRID minimising overall val ACER."""
    bon = val_df[val_df["label"] == "bonafide"]
    atk = val_df[val_df["label"] != "bonafide"]
    best_a, best_al = float("inf"), 0.5
    for al in ALPHA_GRID:
        s_bon = (al * bon["vit_score_norm"] + (1 - al) * bon["lpips_score_norm"]).values
        s_atk = (al * atk["vit_score_norm"] + (1 - al) * atk["lpips_score_norm"]).values
        _, acer = sweep_threshold(s_bon, s_atk)
        if acer < best_a:
            best_a, best_al = acer, float(al)
    return best_al


# ---------------------------------------------------------------------------
# Per-attack evaluation
# ---------------------------------------------------------------------------

def evaluate_scorer(val_df, test_df, score_col, model_name):
    val_bon  = val_df[val_df["label"] == "bonafide"][score_col].dropna().values
    test_bon = test_df[test_df["label"] == "bonafide"][score_col].dropna().values

    rows = []
    for atk_disp in ATTACK_ORDER:
        val_atk  = val_df[val_df["attack_type"] == atk_disp][score_col].dropna().values
        test_atk = test_df[test_df["attack_type"] == atk_disp][score_col].dropna().values
        if len(val_atk) == 0 or len(test_atk) == 0:
            print(f"  WARNING: no samples for {atk_disp} — skipping")
            continue

        tau, val_acer = sweep_threshold(val_bon, val_atk)
        bpcer = float((test_bon > tau).mean()) * 100.0
        apcer = float((test_atk <= tau).mean()) * 100.0
        acer  = (apcer + bpcer) / 2.0
        acc   = ((test_bon <= tau).sum() + (test_atk > tau).sum()) / (len(test_bon) + len(test_atk))
        eer   = compute_eer(test_bon, test_atk)
        auc   = compute_auc(test_bon, test_atk)

        print(f"  {atk_disp:<22} tau={tau:.4f} (val_ACER={val_acer:.2f}%)"
              f"  APCER={apcer:.2f}%  BPCER={bpcer:.2f}%  ACER={acer:.2f}%"
              f"  EER={eer:.2f}%  AUC={auc:.4f}")

        rows.append({
            "Model":       model_name,
            "Attack_Type": atk_disp,
            "Threshold":   round(tau,       4),
            "Accuracy":    round(float(acc), 4),
            "APCER":       round(apcer / 100, 4),
            "BPCER":       round(bpcer / 100, 4),
            "ACER":        round(acer  / 100, 4),
            "EER":         round(eer   / 100, 4),
            "AUC":         round(auc,        4),
            "val_ACER":    round(val_acer / 100, 4),
        })

    # Overall row (global tau)
    val_all  = val_df[val_df["label"] != "bonafide"][score_col].dropna().values
    test_all = test_df[test_df["label"] != "bonafide"][score_col].dropna().values
    tau_g, val_acer_g = sweep_threshold(val_bon, val_all)
    bpcer_g = float((test_bon > tau_g).mean()) * 100.0
    apcer_g = float((test_all <= tau_g).mean()) * 100.0
    acer_g  = (apcer_g + bpcer_g) / 2.0
    acc_g   = ((test_bon <= tau_g).sum() + (test_all > tau_g).sum()) / (len(test_bon) + len(test_all))
    rows.append({
        "Model":       model_name,
        "Attack_Type": "Overall",
        "Threshold":   round(tau_g,        4),
        "Accuracy":    round(float(acc_g),  4),
        "APCER":       round(apcer_g / 100, 4),
        "BPCER":       round(bpcer_g / 100, 4),
        "ACER":        round(acer_g  / 100, 4),
        "EER":         round(compute_eer(test_bon, test_all) / 100, 4),
        "AUC":         round(compute_auc(test_bon, test_all), 4),
        "val_ACER":    round(val_acer_g / 100, 4),
    })
    print(f"  {'Overall':<22} tau={tau_g:.4f} (val_ACER={val_acer_g:.2f}%)"
          f"  APCER={apcer_g:.2f}%  BPCER={bpcer_g:.2f}%  ACER={acer_g:.2f}%")

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading and merging CSVs...")
    val_df, test_df = load_and_merge()
    print(f"Val : {len(val_df):,} rows | "
          f"bonafide={sum(val_df['label']=='bonafide'):,} | "
          f"attack={sum(val_df['label']!='bonafide'):,}")
    print(f"Test: {len(test_df):,} rows | "
          f"bonafide={sum(test_df['label']=='bonafide'):,} | "
          f"attack={sum(test_df['label']!='bonafide'):,}")

    alpha = find_best_alpha(val_df)
    print(f"\nDynamic fusion: best alpha = {alpha}  "
          f"(fixed fusion always uses alpha=0.5)")

    val_df["fixed_score"]    = 0.5 * val_df["vit_score_norm"] + 0.5 * val_df["lpips_score_norm"]
    test_df["fixed_score"]   = 0.5 * test_df["vit_score_norm"] + 0.5 * test_df["lpips_score_norm"]
    val_df["dynamic_score"]  = alpha * val_df["vit_score_norm"] + (1 - alpha) * val_df["lpips_score_norm"]
    test_df["dynamic_score"] = alpha * test_df["vit_score_norm"] + (1 - alpha) * test_df["lpips_score_norm"]

    scorers = [
        ("vit_score",     "AnoDDPM_DDIM_ViT"),
        ("fixed_score",   "AnoDDPM_DDIM_Fixed"),
        ("dynamic_score", "AnoDDPM_DDIM_Dynamic"),
    ]

    all_rows = []
    for col, name in scorers:
        print(f"\n{'='*65}")
        print(f"Scorer: {name}  (column: {col})")
        print(f"{'='*65}")
        all_rows.extend(evaluate_scorer(val_df, test_df, col, name))

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n{'='*65}")
    print(f"Saved → {OUT_CSV}  ({len(df)} rows)")
    print(f"{'='*65}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
