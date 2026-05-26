"""
iris_bbdm_pad/evaluation/generate_ablation_steps_per_attack_threshold.py

Ablation Study 2 — Denoising Steps — Per-Attack-Type Threshold Protocol.

For every (step_count, attack_type) pair:
  1. Val set  : filter to bonafide (attack_type='Live') + that specific attack type only.
               Sweep 1,000 threshold candidates; pick τ that minimises ACER on val.
  2. Test set : apply the per-attack τ to the full bonafide test set + that attack type.
               Compute APCER, BPCER, ACER.
               Compute EER and AUC from raw scores (threshold-free).

For the 'Overall' row:
  Same protocol but threshold is optimised on val(bonafide + ALL attack types),
  and metrics are computed on test(bonafide + all attack types).
  This is the standard global threshold used in prior comparisons.

All val/test scores are read from pre-computed cache CSVs — no GPU required.
Val  CSVs: IJCB_paper_requirements/scoring/steps_cache/vit_scores_val_steps_NNN.csv
Test CSVs: IJCB_paper_requirements/scoring/steps_cache/vit_scores_steps_NNN.csv

Val CSV columns : filename, label, attack_type, vit_score
Test CSV columns: filename, label, label_num, attack_type, attack_display, vit_score

Val label values : 'bonafide', 'attack'
Val attack_type  : 'Live' (bonafide), 'Artifact', 'CL', 'E-display',
                   'Fake with Add On', 'Generated', 'PostMortem',
                   'Print and E-display', 'Printed'

Output:
  IJCB_paper_requirements/tables/ablation_steps_per_attack_perthreshold.csv
  (printed summary to stdout)

Usage (no GPU needed; any Python env with pandas + numpy + scikit-learn):
  python iris_bbdm_pad/evaluation/generate_ablation_steps_per_attack_threshold.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR    = PROJECT_ROOT / "IJCB_paper_requirements" / "scoring" / "steps_cache"
OUT_DIR      = PROJECT_ROOT / "IJCB_paper_requirements" / "tables"
OUT_CSV      = OUT_DIR / "ablation_steps_per_attack_perthreshold.csv"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STEP_COUNTS = [10, 25, 50, 100, 200]

# raw attack_type in CSV  →  display name used in paper
ATTACK_TYPE_MAP = {
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

N_THRESHOLD_CANDS = 1000   # sweep resolution for val optimisation


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def sweep_threshold(bon: np.ndarray, atk: np.ndarray,
                    n: int = N_THRESHOLD_CANDS):
    """
    Find τ* = argmin_{τ} ACER(τ) over n linearly spaced candidates.
    ACER(τ) = (APCER(τ) + BPCER(τ)) / 2
    APCER(τ) = fraction of attack samples with score ≤ τ  (mis-classified as bonafide)
    BPCER(τ) = fraction of bonafide samples with score > τ (mis-classified as attack)
    Returns (τ*, val_ACER%).
    """
    all_s  = np.concatenate([bon, atk])
    cands  = np.linspace(all_s.min(), all_s.max(), n)
    best_t = float(cands[0])
    best_a = float("inf")
    for t in cands:
        bpcer = float((bon > t).mean()) * 100.0
        apcer = float((atk <= t).mean()) * 100.0
        acer  = (apcer + bpcer) / 2.0
        if acer < best_a:
            best_a, best_t = acer, float(t)
    return best_t, best_a


def compute_eer(bon: np.ndarray, atk: np.ndarray,
                n: int = N_THRESHOLD_CANDS) -> float:
    """Equal Error Rate (%) — threshold-free."""
    all_s    = np.concatenate([bon, atk])
    cands    = np.linspace(all_s.min(), all_s.max(), n)
    min_diff = 1.0
    best_eer = 50.0
    for t in cands:
        bpcer = float((bon > t).mean())
        apcer = float((atk <= t).mean())
        diff  = abs(apcer - bpcer)
        if diff < min_diff:
            min_diff = diff
            best_eer = (apcer + bpcer) / 2.0 * 100.0
    return best_eer


def compute_auc(bon: np.ndarray, atk: np.ndarray) -> float:
    """ROC-AUC — threshold-free."""
    scores = np.concatenate([bon, atk])
    labels = np.concatenate([np.zeros(len(bon)), np.ones(len(atk))])
    return float(roc_auc_score(labels, scores))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    for s in STEP_COUNTS:
        val_csv    = CACHE_DIR / f"vit_scores_val_steps_{s:03d}.csv"
        test_csv   = CACHE_DIR / f"vit_scores_steps_{s:03d}.csv"
        timing_json= CACHE_DIR / f"timing_steps_{s:03d}.json"

        # --- sanity checks --------------------------------------------------
        for p in [val_csv, test_csv, timing_json]:
            if not p.exists():
                sys.exit(f"ERROR: required file missing: {p}")

        val_df  = pd.read_csv(val_csv)
        test_df = pd.read_csv(test_csv)
        ms_img  = json.load(open(timing_json))["ms_per_img"]

        # bonafide pools (full — no subsampling)
        val_bon  = val_df[val_df["label"] == "bonafide"]["vit_score"].dropna().values
        test_bon = test_df[test_df["label"] == "bonafide"]["vit_score"].dropna().values

        print(f"\n{'='*70}")
        print(f"Steps = {s}   |   val bonafide = {len(val_bon)}   "
              f"|   test bonafide = {len(test_bon)}   |   {ms_img:.1f} ms/img")
        print(f"{'='*70}")

        # --- per-attack threshold ------------------------------------------
        for raw_atk, display_atk in ATTACK_TYPE_MAP.items():

            # Val: full bonafide pool + all val samples of this attack type
            val_atk = val_df[val_df["attack_type"] == raw_atk]["vit_score"].dropna().values

            if len(val_atk) == 0:
                print(f"  WARNING: no val samples for attack='{raw_atk}' — skipping")
                continue

            # Optimise threshold on val(bonafide + this attack type)
            tau, val_acer = sweep_threshold(val_bon, val_atk)

            # Apply τ to test(bonafide + this attack type)
            test_atk = test_df[test_df["attack_type"] == raw_atk]["vit_score"].dropna().values

            if len(test_atk) == 0:
                print(f"  WARNING: no test samples for attack='{raw_atk}' — skipping")
                continue

            bpcer = float((test_bon > tau).mean()) * 100.0
            apcer = float((test_atk <= tau).mean()) * 100.0
            acer  = (apcer + bpcer) / 2.0
            eer   = compute_eer(test_bon, test_atk)
            auc   = compute_auc(test_bon, test_atk)

            print(f"  {display_atk:<22} | val_n={len(val_atk):5d}  test_n={len(test_atk):5d}"
                  f"  τ={tau:.4f}  (val_ACER={val_acer:.2f}%)"
                  f"  →  APCER={apcer:.2f}%  BPCER={bpcer:.2f}%  ACER={acer:.2f}%"
                  f"  EER={eer:.2f}%  AUC={auc:.4f}")

            rows.append({
                "Steps":    s,
                "Attack":   display_atk,
                "APCER↓":   round(apcer,    2),
                "BPCER↓":   round(bpcer,    2),
                "ACER↓":    round(acer,     2),
                "EER↓":     round(eer,      2),
                "AUC↑":     round(auc,      4),
                "τ":        round(tau,      4),
                "ms/img":   round(ms_img,   1),
                "val_ACER": round(val_acer, 2),
                "val_n_atk":len(val_atk),
                "test_n_atk":len(test_atk),
            })

        # --- Overall row: global threshold (val bonafide + ALL attacks) -----
        val_all_atk  = val_df[val_df["label"] == "attack"]["vit_score"].dropna().values
        test_all_atk = test_df[test_df["label"] == "attack"]["vit_score"].dropna().values

        tau_g, val_acer_g = sweep_threshold(val_bon, val_all_atk)

        bpcer_g = float((test_bon > tau_g).mean()) * 100.0
        apcer_g = float((test_all_atk <= tau_g).mean()) * 100.0
        acer_g  = (apcer_g + bpcer_g) / 2.0
        eer_g   = compute_eer(test_bon, test_all_atk)
        auc_g   = compute_auc(test_bon, test_all_atk)

        print(f"  {'Overall (global τ)':<22} | val_n={len(val_all_atk):5d}  "
              f"test_n={len(test_all_atk):5d}"
              f"  τ={tau_g:.4f}  (val_ACER={val_acer_g:.2f}%)"
              f"  →  APCER={apcer_g:.2f}%  BPCER={bpcer_g:.2f}%  ACER={acer_g:.2f}%"
              f"  EER={eer_g:.2f}%  AUC={auc_g:.4f}")

        rows.append({
            "Steps":    s,
            "Attack":   "Overall",
            "APCER↓":   round(apcer_g,    2),
            "BPCER↓":   round(bpcer_g,    2),
            "ACER↓":    round(acer_g,     2),
            "EER↓":     round(eer_g,      2),
            "AUC↑":     round(auc_g,      4),
            "τ":        round(tau_g,      4),
            "ms/img":   round(ms_img,     1),
            "val_ACER": round(val_acer_g, 2),
            "val_n_atk":len(val_all_atk),
            "test_n_atk":len(test_all_atk),
        })

    # --- Save ---------------------------------------------------------------
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n{'='*70}")
    print(f"Saved → {OUT_CSV}")
    print(f"Total rows: {len(df)}")
    print(f"{'='*70}\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
