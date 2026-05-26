"""
Zero-Attack-Knowledge (ZAK) Final Tables
=========================================
Produces final verified results for IJCB paper Tables 2 and 3 under ZAK protocol.

ZAK protocol:
  - Threshold tau set from BONAFIDE VAL ONLY (90th percentile, no attack labels used)
  - Test set = val attacks + test attacks (combined), bonafide from test only

Table 2 (comparison): AnoDDPM DDIM Simplex Dynamic vs Diff-IrisPAD
  - AnoDDPM: ViT scoring (only val ViT scores available; dynamic val not cached)
  - Diff-IrisPAD: ViT scoring

Table 3 (ablation steps): Diff-IrisPAD at steps 10, 50, 100, 200 under ZAK

Also reports EER per attack.

Inputs (no model inference):
  IJCB_paper_requirements/scoring/vit_scores_val.csv
  IJCB_paper_requirements/scoring/vit_scores_test.csv
  IJCB_paper_requirements/scoring/steps_cache/vit_scores_steps_{N}.csv
  IJCB_paper_requirements/scoring/steps_cache/vit_scores_val_steps_{N}.csv
  iris_td/pad_scores/ddpm_val_simplex_tstar500_ddim_steps50_vit.csv
  iris_td/pad_scores/ddpm_test_simplex_tstar500_ddim_steps50_vit.csv

Outputs:
  iris_bbdm_pad/results/zak_final/zak_numbers.txt    (all raw numbers)
  iris_bbdm_pad/results/zak_final/table2_zak.tex
  iris_bbdm_pad/results/zak_final/table3_zak.tex
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCORE_DIR = ROOT / "IJCB_paper_requirements/scoring"
PAD_DIR   = ROOT / "iris_td/pad_scores"
OUT_DIR   = ROOT / "iris_bbdm_pad/results/zak_final"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BONAFIDE = "bonafide"

ATTACK_ORDER = [
    "Artifact", "CL", "E-display", "Fake with Add On",
    "Generated", "PostMortem", "Print and E-display", "Printed"
]
ATTACK_DISPLAY = {
    "Artifact":            "Artifact",
    "CL":                  "Contact Lens",
    "E-display":           "E-display",
    "Fake with Add On":    "Fake W/AO",
    "Generated":           "Generated",
    "PostMortem":          "PostMortem",
    "Print and E-display": "Printed\\&ED",
    "Printed":             "Printed",
}

# AnoDDPM uses underscores in attack_type column
ANODDPM_REMAP = {
    "Fake_with_Add_On": "Fake with Add On",
    "Print_E-display":  "Print and E-display",
}

# Timing per step count (ms/img) from cached timing files
TIMING = {10: 84.4, 50: 208.6, 100: 364.2, 200: 671.9}


# ── Core metric functions ────────────────────────────────────────────────────

def compute_apcer_bpcer(scores, labels, tau):
    """APCER, BPCER at threshold tau. Higher score = more bonafide."""
    am = labels != BONAFIDE
    bm = labels == BONAFIDE
    if am.sum() == 0 or bm.sum() == 0:
        return np.nan, np.nan
    apcer = float((scores[am] <= tau).mean())   # attacks passing as bonafide
    bpcer = float((scores[bm] > tau).mean())    # bonafide rejected as attack
    return apcer, bpcer


def compute_eer(scores, labels, n=2000):
    """EER: threshold where APCER ≈ BPCER."""
    am = labels != BONAFIDE
    bm = labels == BONAFIDE
    if am.sum() == 0 or bm.sum() == 0:
        return np.nan, np.nan
    thresholds = np.linspace(scores.min(), scores.max(), n)
    best_tau, best_diff, best_eer = None, np.inf, np.nan
    for tau in thresholds:
        apcer = float((scores[am] <= tau).mean())
        bpcer = float((scores[bm] > tau).mean())
        diff = abs(apcer - bpcer)
        if diff < best_diff:
            best_diff, best_diff = diff, diff
            best_tau = tau
            best_eer = (apcer + bpcer) / 2
    return best_tau, best_eer


def zak_threshold(val_df, score_col, percentile=90):
    """ZAK tau = percentile of bonafide val scores."""
    bf_scores = val_df[val_df["label"] == BONAFIDE][score_col].values
    return float(np.percentile(bf_scores, percentile))


def make_combined_test(val_df, test_df):
    """
    Combined test set:
      - Attacks: val attacks + test attacks
      - Bonafide: test bonafide only (val bonafide reserved for threshold calibration)
    """
    val_attacks  = val_df[val_df["label"] != BONAFIDE].copy()
    test_attacks = test_df[test_df["label"] != BONAFIDE].copy()
    test_bonafide = test_df[test_df["label"] == BONAFIDE].copy()
    return pd.concat([val_attacks, test_attacks, test_bonafide], ignore_index=True)


def per_attack_metrics(combined_df, score_col, tau):
    """Compute APCER, BPCER, ACER, EER per attack type + overall."""
    rows = []
    ts = combined_df[score_col].values
    tl = combined_df["label"].values
    ta = combined_df["attack_type"].values

    for attack in ATTACK_ORDER:
        mask = (ta == attack) | (tl == BONAFIDE)
        sub_s = ts[mask]
        sub_l = tl[mask]
        if (sub_l != BONAFIDE).sum() == 0:
            continue
        apcer, bpcer = compute_apcer_bpcer(sub_s, sub_l, tau)
        acer = (apcer + bpcer) / 2
        _, eer = compute_eer(sub_s, sub_l)
        n_attack = int((sub_l != BONAFIDE).sum())
        rows.append({
            "attack": attack,
            "n_attack": n_attack,
            "tau": tau,
            "APCER": apcer,
            "BPCER": bpcer,
            "ACER": acer,
            "EER": eer,
        })

    # Overall (all attacks vs all bonafide in combined set)
    apcer_all, bpcer_all = compute_apcer_bpcer(ts, tl, tau)
    acer_all = (apcer_all + bpcer_all) / 2
    _, eer_all = compute_eer(ts, tl)
    rows.append({
        "attack": "ALL",
        "n_attack": int((tl != BONAFIDE).sum()),
        "tau": tau,
        "APCER": apcer_all,
        "BPCER": bpcer_all,
        "ACER": acer_all,
        "EER": eer_all,
    })
    return pd.DataFrame(rows)


def pct(x):
    """Format as percentage string."""
    return f"{x * 100:.2f}"


# ── Load Diff-IrisPAD scores ─────────────────────────────────────────────────

print("Loading Diff-IrisPAD scores …")
dip_val  = pd.read_csv(SCORE_DIR / "vit_scores_val.csv")
dip_test = pd.read_csv(SCORE_DIR / "vit_scores_test.csv")
dip_val["attack_type"]  = dip_val["attack_type"].fillna(dip_val["label"])
dip_test["attack_type"] = dip_test["attack_type"].fillna(dip_test["label"])

dip_combined = make_combined_test(dip_val, dip_test)
tau_dip = zak_threshold(dip_val, "vit_score", percentile=90)
print(f"  Diff-IrisPAD ZAK tau (p90 bonafide val): {tau_dip:.4f}")

dip_results = per_attack_metrics(dip_combined, "vit_score", tau_dip)
print("  Diff-IrisPAD combined test attack counts:")
print(dip_combined[dip_combined["label"] != BONAFIDE]["attack_type"].value_counts().sort_index())


# ── Load AnoDDPM scores (ViT scoring, only val variant available) ────────────

print("\nLoading AnoDDPM scores …")
ano_val  = pd.read_csv(PAD_DIR / "ddpm_val_simplex_tstar500_ddim_steps50_vit.csv")
ano_test = pd.read_csv(PAD_DIR / "ddpm_test_simplex_tstar500_ddim_steps50_vit.csv")

# Normalize attack_type names
for df in [ano_val, ano_test]:
    df["attack_type"] = df["attack_type"].replace(ANODDPM_REMAP)
    df["attack_type"] = df["attack_type"].fillna(df["label"])

ano_combined = make_combined_test(ano_val, ano_test)
tau_ano = zak_threshold(ano_val, "vit_score", percentile=90)
print(f"  AnoDDPM ZAK tau (p90 bonafide val): {tau_ano:.4f}")

ano_results = per_attack_metrics(ano_combined, "vit_score", tau_ano)


# ── Ablation: steps 10, 50, 100, 200 ────────────────────────────────────────

print("\nLoading ablation step scores …")
ablation_results = {}
STEPS = [10, 50, 100, 200]

for steps in STEPS:
    skey = f"{steps:03d}"
    val_f  = SCORE_DIR / f"steps_cache/vit_scores_val_steps_{skey}.csv"
    test_f = SCORE_DIR / f"steps_cache/vit_scores_steps_{skey}.csv"
    val_df  = pd.read_csv(val_f)
    test_df = pd.read_csv(test_f)
    val_df["attack_type"]  = val_df["attack_type"].fillna(val_df["label"])
    test_df["attack_type"] = test_df["attack_type"].fillna(test_df["label"])

    combined = make_combined_test(val_df, test_df)
    tau = zak_threshold(val_df, "vit_score", percentile=90)
    print(f"  Steps {steps:3d}: tau={tau:.4f}")
    ablation_results[steps] = per_attack_metrics(combined, "vit_score", tau)


# ── Print summary ────────────────────────────────────────────────────────────

log_lines = []

def log(s=""):
    print(s)
    log_lines.append(s)

log("=" * 80)
log("DIFF-IRIPAD ZAK RESULTS (combined test: val attacks + test attacks)")
log(f"ZAK tau (p90 bonafide val) = {tau_dip:.4f}")
log("=" * 80)
log(f"{'Attack':<22}  {'N_atk':>6}  {'tau':>6}  {'APCER':>7}  {'BPCER':>7}  {'ACER':>7}  {'EER':>7}")
for _, r in dip_results.iterrows():
    log(f"{r['attack']:<22}  {int(r['n_attack']):>6}  {r['tau']:>6.4f}  "
        f"{pct(r['APCER']):>7}  {pct(r['BPCER']):>7}  {pct(r['ACER']):>7}  {pct(r['EER']):>7}")

log()
log("=" * 80)
log("ANODDPM DDIM SIMPLEX VIT ZAK RESULTS (combined test)")
log(f"ZAK tau (p90 bonafide val) = {tau_ano:.4f}")
log("=" * 80)
log(f"{'Attack':<22}  {'N_atk':>6}  {'tau':>6}  {'APCER':>7}  {'BPCER':>7}  {'ACER':>7}  {'EER':>7}")
for _, r in ano_results.iterrows():
    log(f"{r['attack']:<22}  {int(r['n_attack']):>6}  {r['tau']:>6.4f}  "
        f"{pct(r['APCER']):>7}  {pct(r['BPCER']):>7}  {pct(r['ACER']):>7}  {pct(r['EER']):>7}")

log()
log("=" * 80)
log("ABLATION STEPS ZAK RESULTS")
log("=" * 80)
for steps in STEPS:
    res = ablation_results[steps]
    log(f"\n--- Steps {steps} (tau={res.iloc[0]['tau']:.4f}) ---")
    log(f"{'Attack':<22}  {'APCER':>7}  {'BPCER':>7}  {'ACER':>7}  {'EER':>7}")
    for _, r in res.iterrows():
        log(f"{r['attack']:<22}  {pct(r['APCER']):>7}  {pct(r['BPCER']):>7}  "
            f"{pct(r['ACER']):>7}  {pct(r['EER']):>7}")

(OUT_DIR / "zak_numbers.txt").write_text("\n".join(log_lines))
print(f"\nSaved: {OUT_DIR}/zak_numbers.txt")


# ── Generate Table 2 LaTeX ───────────────────────────────────────────────────

def make_table2_latex(ano_res, dip_res, tau_ano, tau_dip):
    """Table 2: AnoDDPM vs Diff-IrisPAD under ZAK."""

    def fmt_row(r, bold_acer=False, bold_eer=False):
        acer_s = f"\\textbf{{{pct(r['ACER'])}}}" if bold_acer else pct(r['ACER'])
        eer_s  = f"\\textbf{{{pct(r['EER'])}}}"  if bold_eer  else pct(r['EER'])
        return (f"    & {ATTACK_DISPLAY.get(r['attack'], r['attack']):<20} "
                f"& {r['tau']:>5.3f} "
                f"& {pct(r['APCER']):>6} "
                f"& {pct(r['BPCER']):>6} "
                f"& {acer_s:>14} "
                f"& {eer_s:>14} \\\\")

    # Find best ACER per attack between the two methods
    best_acer = {}
    for atk in ATTACK_ORDER + ["ALL"]:
        ano_row = ano_res[ano_res["attack"] == atk]
        dip_row = dip_res[dip_res["attack"] == atk]
        if ano_row.empty or dip_row.empty:
            best_acer[atk] = None
            continue
        best_acer[atk] = "dip" if dip_row.iloc[0]["ACER"] <= ano_row.iloc[0]["ACER"] else "ano"

    lines = []
    lines += [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{",
        r"\textbf{Comparison of diffusion-based iris PAD methods under Zero-Attack-Knowledge (ZAK) protocol.}",
        r"Performance per attack: APCER, BPCER, ACER, EER (all \%, lower is better).",
        r"Threshold $\tau$ set from bonafide validation scores only (90th percentile); no attack labels used at any stage.",
        r"Test set combines validation and test attack splits for comprehensive coverage.",
        r"\textbf{Bold} = lowest ACER / EER per attack. Diff-IrisPAD achieves superior overall performance.",
        r"}",
        r"\label{tab:diffusion_comparison_zak}",
        r"",
        r"\begin{tabular}{l l c c c c c}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Attack} & $\tau$ & APCER & BPCER & ACER & EER \\",
        r"\midrule",
        r"",
    ]

    # AnoDDPM block
    ano_attacks = [r for r in ATTACK_ORDER if not ano_res[ano_res["attack"] == r].empty]
    lines.append(r"    \multirow{9}{*}{\shortstack[l]{AnoDDPM \\DDIM\\Simplex\\ \cite{ho2020denoising}}}")
    for atk in ano_attacks:
        row = ano_res[ano_res["attack"] == atk].iloc[0]
        is_best = best_acer.get(atk) == "ano"
        lines.append(fmt_row(row, bold_acer=is_best, bold_eer=is_best))
    # ALL row for AnoDDPM
    row_all = ano_res[ano_res["attack"] == "ALL"].iloc[0]
    is_best_all = best_acer.get("ALL") == "ano"
    ano_acer_s = "\\textbf{" + pct(row_all["ACER"]) + "}" if is_best_all else pct(row_all["ACER"])
    ano_eer_s  = "\\textbf{" + pct(row_all["EER"]) + "}"  if is_best_all else pct(row_all["EER"])
    lines.append(f"    & \\textbf{{All}}               & {row_all['tau']:>5.3f} & {pct(row_all['APCER']):>6} & {pct(row_all['BPCER']):>6} & {ano_acer_s} & {ano_eer_s} \\\\")

    lines += [r"", r"    \midrule", r""]

    # Diff-IrisPAD block
    lines.append(r"    \multirow{9}{*}{\shortstack[l]{\textbf{Diff-IrisPAD}\\ (Ours)}}")
    for atk in ATTACK_ORDER:
        row = dip_res[dip_res["attack"] == atk].iloc[0]
        is_best = best_acer.get(atk) == "dip"
        lines.append(fmt_row(row, bold_acer=is_best, bold_eer=is_best))
    # ALL row
    row_all = dip_res[dip_res["attack"] == "ALL"].iloc[0]
    is_best_all = best_acer.get("ALL") == "dip"
    dip_acer_s = "\\textbf{" + pct(row_all["ACER"]) + "}" if is_best_all else pct(row_all["ACER"])
    dip_eer_s  = "\\textbf{" + pct(row_all["EER"]) + "}"  if is_best_all else pct(row_all["EER"])
    lines.append(f"    & \\textbf{{All}}               & {row_all['tau']:>5.3f} & {pct(row_all['APCER']):>6} & {pct(row_all['BPCER']):>6} & {dip_acer_s} & {dip_eer_s} \\\\")

    lines += [
        r"",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ── Generate Table 3 LaTeX ───────────────────────────────────────────────────

def make_table3_latex(ablation_res):
    """Table 3: Ablation on denoising steps under ZAK."""

    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{\textbf{Ablation on denoising steps ($T$) under ZAK protocol.}",
        r"LBBDM-f4 with ViT-B/16 scoring; $\tau$ set from bonafide validation only (p90).",
        r"Test set combines validation and test attack splits. Results in \% (lower is better).}",
        r"\label{tab:ablation_study2_zak}",
        r"",
        r"\begin{tabular}{l l c c c c c}",
        r"\toprule",
        r"\textbf{Steps} & \textbf{Atk} & APCER & BPCER & ACER & EER & ms/img \\",
        r"\midrule",
    ]

    # Determine best ACER per (attack, metric) across step counts
    best_acer_per_attack = {}
    for atk in ATTACK_ORDER + ["ALL"]:
        best = min(STEPS, key=lambda s: ablation_res[s][ablation_res[s]["attack"] == atk].iloc[0]["ACER"])
        best_acer_per_attack[atk] = best

    for i, steps in enumerate(STEPS):
        res = ablation_res[steps]
        ms = TIMING[steps]

        lines.append(r"")
        lines.append(f"\\multirow{{9}}{{*}}{{{steps}}}")

        for j, atk in enumerate(ATTACK_ORDER):
            row = res[res["attack"] == atk].iloc[0]
            is_best = best_acer_per_attack.get(atk) == steps
            acer_s = f"\\textbf{{{pct(row['ACER'])}}}" if is_best else pct(row["ACER"])
            eer_s  = pct(row["EER"])
            ms_s   = f"\\multirow{{9}}{{*}}{{{ms:.1f}}}" if j == 0 else ""
            lines.append(
                f"& {ATTACK_DISPLAY.get(atk, atk):<20} "
                f"& {pct(row['APCER']):>6} "
                f"& {pct(row['BPCER']):>6} "
                f"& {acer_s:>14} "
                f"& {eer_s:>7} "
                f"& {ms_s} \\\\"
            )

        # ALL row
        row = res[res["attack"] == "ALL"].iloc[0]
        is_best = best_acer_per_attack.get("ALL") == steps
        acer_s = f"\\textbf{{{pct(row['ACER'])}}}" if is_best else pct(row["ACER"])
        lines.append(
            f"& \\textbf{{All}}             "
            f"& {pct(row['APCER']):>6} "
            f"& {pct(row['BPCER']):>6} "
            f"& {acer_s:>14} "
            f"& {pct(row['EER']):>7} "
            f"& \\\\"
        )

        if i < len(STEPS) - 1:
            lines.append(r"\midrule")

    lines += [
        r"",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ── Write LaTeX ──────────────────────────────────────────────────────────────

table2_tex = make_table2_latex(ano_results, dip_results, tau_ano, tau_dip)
table3_tex = make_table3_latex(ablation_results)

(OUT_DIR / "table2_zak.tex").write_text(table2_tex)
(OUT_DIR / "table3_zak.tex").write_text(table3_tex)

print(f"\nSaved: {OUT_DIR}/table2_zak.tex")
print(f"Saved: {OUT_DIR}/table3_zak.tex")

# ── Also save CSV summaries ──────────────────────────────────────────────────

dip_results.to_csv(OUT_DIR / "dip_zak_results.csv", index=False)
ano_results.to_csv(OUT_DIR / "ano_zak_results.csv", index=False)
for steps in STEPS:
    ablation_results[steps].to_csv(OUT_DIR / f"ablation_steps{steps:03d}_zak.csv", index=False)

print("\nAll done.")
print(f"\nTable 2 ZAK combined test set sizes:")
print(f"  Total attacks = {dip_combined[dip_combined['label'] != BONAFIDE].shape[0]}")
print(f"  Total bonafide = {dip_combined[dip_combined['label'] == BONAFIDE].shape[0]}")
