"""
iris_td/evaluation/generate_ddpm_scoring_ablation.py

Generalised scoring-method ablation for any DDPM config.
Mirrors the AnoDDPM+DDIM ablation (Table 3 style):
  three scorers with per-attack-type threshold protocol.

Scorers:
  1. ViT-only      : vit_score (cosine distance)
  2. Fixed Fusion  : 0.5 * vit_norm + 0.5 * lpips_norm
  3. Dynamic Fusion: alpha * vit_norm + (1-alpha) * lpips_norm,
                     alpha in {0.1,...,0.9} optimised on overall val ACER

LPIPS negated before fusion: attacks reconstruct more faithfully under
partial noising, so raw lpips_attack < lpips_bonafide.

Normalisation: min-max computed on val set, applied to both val and test.

Per-attack threshold: tau independently optimised on
val(full bonafide + that attack type only), ACER-min sweep over 1,000 candidates.

Usage:
  cd /home/teaching/Documents/Geetanjali_PhD_IRIS_PAD
  PYTHONNOUSERSITE=1 /home/teaching/miniconda3/envs/bbdm_clean/bin/python \
      iris_td/evaluation/generate_ddpm_scoring_ablation.py \
      --config_name AnoDDPM_simplex_steps25 \
      --val_vit   iris_td/pad_scores/ddpm_val_simplex_tstar500_steps25_vit.csv \
      --test_vit  iris_td/pad_scores/ddpm_test_simplex_tstar500_steps25_vit.csv \
      --val_lpips iris_td/pad_scores/ddpm_val_simplex_tstar500_steps25_fixed.csv \
      --test_lpips iris_td/pad_scores/ddpm_test_simplex_tstar500_steps25_fixed.csv \
      --output    iris_td/final_results/anoddpm_simplex_steps25_scoring_ablation.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ATTACK_TYPE_MAP = {
    "Artifact":             "Artifact",
    "CL":                   "Contact Lens",
    "E-display":            "E-display",
    "Fake with Add On":     "Fake w/ Add-On",
    "Fake_with_Add_On":     "Fake w/ Add-On",
    "Generated":            "Generated",
    "PostMortem":           "Post-Mortem",
    "Print and E-display":  "Print & E-disp.",
    "Print_E-display":      "Print & E-disp.",
    "Printed":              "Printed",
}

ATTACK_ORDER = [
    "Artifact", "Contact Lens", "E-display", "Fake w/ Add-On",
    "Generated", "Post-Mortem", "Print & E-disp.", "Printed",
]

N_CANDS    = 1000
ALPHA_GRID = np.round(np.arange(0.1, 1.0, 0.1), 1)


def sweep_threshold(bon, atk, n=N_CANDS):
    cands = np.linspace(np.concatenate([bon, atk]).min(),
                        np.concatenate([bon, atk]).max(), n)
    best_t, best_a = float(cands[0]), float("inf")
    for t in cands:
        acer = (float((bon > t).mean()) + float((atk <= t).mean())) / 2.0 * 100.0
        if acer < best_a:
            best_a, best_t = acer, float(t)
    return best_t, best_a


def compute_eer(bon, atk, n=N_CANDS):
    cands = np.linspace(np.concatenate([bon, atk]).min(),
                        np.concatenate([bon, atk]).max(), n)
    min_diff, best_eer = 1.0, 50.0
    for t in cands:
        diff = abs(float((bon > t).mean()) - float((atk <= t).mean()))
        if diff < min_diff:
            min_diff = diff
            best_eer = (float((bon > t).mean()) + float((atk <= t).mean())) / 2.0 * 100.0
    return best_eer


def compute_auc(bon, atk):
    scores = np.concatenate([bon, atk])
    labels = np.concatenate([np.zeros(len(bon)), np.ones(len(atk))])
    return float(roc_auc_score(labels, scores))


def minmax_norm(s, mn, mx):
    return (s - mn) / (mx - mn + 1e-8)


def load_and_merge(val_vit, test_vit, val_lpips, test_lpips):
    for p in [val_vit, test_vit, val_lpips, test_lpips]:
        if not Path(p).exists():
            sys.exit(f"ERROR: missing required file: {p}")

    val_v  = pd.read_csv(val_vit)
    test_v = pd.read_csv(test_vit)
    val_l  = pd.read_csv(val_lpips)[["filename", "lpips_score"]]
    test_l = pd.read_csv(test_lpips)[["filename", "lpips_score"]]

    val_l["lpips_score"]  = -val_l["lpips_score"]
    test_l["lpips_score"] = -test_l["lpips_score"]

    val_df  = val_v.merge(val_l,  on="filename", how="inner")
    test_df = test_v.merge(test_l, on="filename", how="inner")

    for col in ["vit_score", "lpips_score"]:
        mn = float(val_df[col].min())
        mx = float(val_df[col].max())
        val_df[col  + "_norm"] = minmax_norm(val_df[col],  mn, mx)
        test_df[col + "_norm"] = minmax_norm(test_df[col], mn, mx)

    val_df["attack_type"]  = val_df["attack_type"].map(
        lambda x: ATTACK_TYPE_MAP.get(x, x))
    test_df["attack_type"] = test_df["attack_type"].map(
        lambda x: ATTACK_TYPE_MAP.get(x, x))

    return val_df, test_df


def find_best_alpha(val_df):
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

        print(f"  {atk_disp:<22} tau={tau:.4f} (val_ACER={val_acer:.2f}%)"
              f"  APCER={apcer:.2f}%  BPCER={bpcer:.2f}%  ACER={acer:.2f}%"
              f"  EER={compute_eer(test_bon, test_atk):.2f}%  AUC={compute_auc(test_bon, test_atk):.4f}")

        rows.append({
            "Model":       model_name,
            "Attack_Type": atk_disp,
            "APCER":       round(apcer / 100, 4),
            "BPCER":       round(bpcer / 100, 4),
            "ACER":        round(acer  / 100, 4),
            "Threshold":   round(tau,       4),
            "Accuracy":    round(float(acc), 4),
        })

    val_all  = val_df[val_df["label"] != "bonafide"][score_col].dropna().values
    test_all = test_df[test_df["label"] != "bonafide"][score_col].dropna().values
    tau_g, val_acer_g = sweep_threshold(val_bon, val_all)
    bpcer_g = float((test_bon > tau_g).mean()) * 100.0
    apcer_g = float((test_all <= tau_g).mean()) * 100.0
    acer_g  = (apcer_g + bpcer_g) / 2.0
    acc_g   = ((test_bon <= tau_g).sum() + (test_all > tau_g).sum()) / (len(test_bon) + len(test_all))

    rows.append({
        "Model":       model_name,
        "Attack_Type": "",
        "APCER":       round(apcer_g / 100, 4),
        "BPCER":       round(bpcer_g / 100, 4),
        "ACER":        round(acer_g  / 100, 4),
        "Threshold":   round(tau_g,        4),
        "Accuracy":    round(float(acc_g),  4),
    })
    print(f"  {'Overall':<22} tau={tau_g:.4f} (val_ACER={val_acer_g:.2f}%)"
          f"  APCER={apcer_g:.2f}%  BPCER={bpcer_g:.2f}%  ACER={acer_g:.2f}%")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config_name", required=True,
                   help="Short name for this config, used as model-name prefix in CSV")
    p.add_argument("--val_vit",   required=True)
    p.add_argument("--test_vit",  required=True)
    p.add_argument("--val_lpips", required=True)
    p.add_argument("--test_lpips",required=True)
    p.add_argument("--output",    required=True)
    args = p.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    print("Loading and merging CSVs...")
    val_df, test_df = load_and_merge(
        args.val_vit, args.test_vit, args.val_lpips, args.test_lpips)
    print(f"Val : {len(val_df):,} | bonafide={sum(val_df['label']=='bonafide'):,}"
          f" | attack={sum(val_df['label']!='bonafide'):,}")
    print(f"Test: {len(test_df):,} | bonafide={sum(test_df['label']=='bonafide'):,}"
          f" | attack={sum(test_df['label']!='bonafide'):,}")

    alpha = find_best_alpha(val_df)
    print(f"\nDynamic fusion: best alpha = {alpha}  (fixed always uses 0.5)")

    val_df["fixed_score"]    = 0.5 * val_df["vit_score_norm"]   + 0.5 * val_df["lpips_score_norm"]
    test_df["fixed_score"]   = 0.5 * test_df["vit_score_norm"]  + 0.5 * test_df["lpips_score_norm"]
    val_df["dynamic_score"]  = alpha * val_df["vit_score_norm"]  + (1 - alpha) * val_df["lpips_score_norm"]
    test_df["dynamic_score"] = alpha * test_df["vit_score_norm"] + (1 - alpha) * test_df["lpips_score_norm"]

    cn = args.config_name
    scorers = [
        ("vit_score",     f"{cn}_ViT"),
        ("fixed_score",   f"{cn}_Fixed"),
        ("dynamic_score", f"{cn}_Dynamic"),
    ]

    all_rows = []
    for col, name in scorers:
        print(f"\n{'='*65}")
        print(f"Scorer: {name}  (column: {col})")
        print(f"{'='*65}")
        all_rows.extend(evaluate_scorer(val_df, test_df, col, name))

    df = pd.DataFrame(all_rows)
    df.to_csv(out, index=False)
    print(f"\n{'='*65}")
    print(f"Saved → {out}  ({len(df)} rows)")
    print(f"{'='*65}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
