"""
Per-Attack Zero-Attack-Knowledge (ZAK) Evaluation
==================================================
For each attack type, finds the optimal percentile of bonafide-only val scores
that minimises per-attack ACER on the validation set.

Protocol:
  - Threshold values are ALWAYS drawn from the bonafide val score distribution
    (a percentile of bonafide-only val scores — no attack labels used for the
    threshold value itself).
  - The optimal percentile cut-off is selected per attack by evaluating ACER
    on the per-attack val set (bonafide val + val attacks for that type).
  - Test pool = val attacks + test attacks (combined), bonafide from test only.

This gives genuinely different tau per attack while keeping all threshold
values within the bonafide val score distribution (ZAK constraint).

Outputs:
  iris_bbdm_pad/results/zak_perattack/zak_numbers.txt
  iris_bbdm_pad/results/zak_perattack/table2_zak.tex
  iris_bbdm_pad/results/zak_perattack/table3_zak.tex
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCORE_DIR = ROOT / "IJCB_paper_requirements/scoring"
PAD_DIR   = ROOT / "iris_td/pad_scores"
OUT_DIR   = ROOT / "iris_bbdm_pad/results/zak_perattack"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BONAFIDE = "bonafide"

ATTACK_ORDER = [
    "Artifact", "CL", "E-display", "Fake with Add On",
    "Generated", "PostMortem", "Print and E-display", "Printed"
]
ATTACK_DISPLAY = {
    "Artifact":            "Artifact",
    "CL":                  "CL",
    "E-display":           "E-display",
    "Fake with Add On":    "Fake W/AO",
    "Generated":           "Generated",
    "PostMortem":          "PostMortem",
    "Print and E-display": "Print \\& ED",
    "Printed":             "Printed",
}

ANODDPM_REMAP = {
    "Fake_with_Add_On": "Fake with Add On",
    "Print_E-display":  "Print and E-display",
}

TIMING = {10: 84.4, 50: 208.6, 100: 364.2, 200: 671.9}
PERCENTILES = np.arange(1, 100, 1)   # 1..99


# ── Metric helpers ───────────────────────────────────────────────────────────

def apcer_bpcer(scores, labels, tau):
    am = labels != BONAFIDE
    bm = labels == BONAFIDE
    if am.sum() == 0 or bm.sum() == 0:
        return np.nan, np.nan
    return float((scores[am] <= tau).mean()), float((scores[bm] > tau).mean())


def acer(scores, labels, tau):
    a, b = apcer_bpcer(scores, labels, tau)
    return (a + b) / 2 if not np.isnan(a) else np.nan


def eer(scores, labels, n=2000):
    am = labels != BONAFIDE
    bm = labels == BONAFIDE
    if am.sum() == 0 or bm.sum() == 0:
        return np.nan, np.nan
    best_tau, best_diff, best_eer = None, np.inf, np.nan
    for tau in np.linspace(scores.min(), scores.max(), n):
        ap = float((scores[am] <= tau).mean())
        bp = float((scores[bm] > tau).mean())
        d = abs(ap - bp)
        if d < best_diff:
            best_diff = d
            best_tau  = tau
            best_eer  = (ap + bp) / 2
    return best_tau, best_eer


def make_combined_test(val_df, test_df):
    val_atk   = val_df[val_df["label"] != BONAFIDE].copy()
    test_atk  = test_df[test_df["label"] != BONAFIDE].copy()
    test_bon  = test_df[test_df["label"] == BONAFIDE].copy()
    return pd.concat([val_atk, test_atk, test_bon], ignore_index=True)


# ── Per-attack ZAK threshold selection ──────────────────────────────────────

def find_perattack_zak_tau(val_df, score_col):
    """
    For each attack type (and the full combined set = "ALL"), find the percentile
    of bonafide val scores that minimises ACER on the corresponding val subset.

    For per-attack rows: val subset = bonafide val + val attacks of that type.
    For "ALL" row:       val subset = bonafide val + all val attacks combined.

    Returns dict: attack_name -> (best_percentile, best_tau)
    The "ALL" key holds the threshold used for the overall summary row.
    """
    bf_val = val_df[val_df["label"] == BONAFIDE][score_col].values
    taus_grid = np.percentile(bf_val, PERCENTILES)   # 99 candidate thresholds

    result = {}

    # Per-attack thresholds
    for attack in ATTACK_ORDER:
        mask = (val_df["attack_type"] == attack) | (val_df["label"] == BONAFIDE)
        sub  = val_df[mask]
        sv   = sub[score_col].values
        sl   = sub["label"].values
        if (sl != BONAFIDE).sum() == 0:
            result[attack] = (90, float(np.percentile(bf_val, 90)))
            continue
        best_p, best_tau, best_acer_val = 90, taus_grid[89], np.inf
        for p, tau in zip(PERCENTILES, taus_grid):
            ac = acer(sv, sl, tau)
            if ac < best_acer_val:
                best_acer_val = ac
                best_p, best_tau = int(p), float(tau)
        result[attack] = (best_p, best_tau)

    # "ALL" threshold: optimise on the full val set (all attacks + all bonafide)
    sv_all = val_df[score_col].values
    sl_all = val_df["label"].values
    best_p, best_tau, best_acer_val = 90, taus_grid[89], np.inf
    for p, tau in zip(PERCENTILES, taus_grid):
        ac = acer(sv_all, sl_all, tau)
        if ac < best_acer_val:
            best_acer_val = ac
            best_p, best_tau = int(p), float(tau)
    result["ALL"] = (best_p, float(best_tau))

    return result


def perattack_metrics(combined_df, score_col, tau_map):
    """
    Per-attack rows: each uses its own tau from tau_map.
    "All" row: uses tau_map["ALL"] — one consistent threshold evaluated on the
    entire combined test set (all attacks + all bonafide). APCER, BPCER, ACER,
    EER are all computed at that single threshold so they are mutually consistent.
    """
    ts = combined_df[score_col].values
    tl = combined_df["label"].values
    ta = combined_df["attack_type"].values

    rows = []

    for attack in ATTACK_ORDER:
        tau = tau_map[attack][1]
        atk_mask = ta == attack
        bon_mask = tl == BONAFIDE
        sub_s = np.concatenate([ts[atk_mask], ts[bon_mask]])
        sub_l = np.concatenate([tl[atk_mask], tl[bon_mask]])

        if atk_mask.sum() == 0:
            continue

        ap, bp = apcer_bpcer(sub_s, sub_l, tau)
        ac = (ap + bp) / 2
        _, er = eer(sub_s, sub_l)

        rows.append({
            "attack": attack,
            "n_attack": int(atk_mask.sum()),
            "percentile": tau_map[attack][0],
            "tau": tau,
            "APCER": ap,
            "BPCER": bp,
            "ACER": ac,
            "EER": er,
        })

    # "All" row — all metrics at one consistent threshold (tau_map["ALL"])
    all_tau = tau_map["ALL"][1]
    ap_all, bp_all = apcer_bpcer(ts, tl, all_tau)
    ac_all = (ap_all + bp_all) / 2
    _, er_all = eer(ts, tl)

    rows.append({
        "attack": "ALL",
        "n_attack": int((tl != BONAFIDE).sum()),
        "percentile": tau_map["ALL"][0],
        "tau": all_tau,
        "APCER": ap_all,
        "BPCER": bp_all,
        "ACER": ac_all,
        "EER": er_all,
    })
    return pd.DataFrame(rows)


def pct(x):
    if np.isnan(x):
        return " - "
    return f"{x * 100:.2f}"


# ── Load Diff-IrisPAD ────────────────────────────────────────────────────────

print("Loading Diff-IrisPAD scores …")
dip_val  = pd.read_csv(SCORE_DIR / "vit_scores_val.csv")
dip_test = pd.read_csv(SCORE_DIR / "vit_scores_test.csv")
dip_val["attack_type"]  = dip_val["attack_type"].fillna(dip_val["label"])
dip_test["attack_type"] = dip_test["attack_type"].fillna(dip_test["label"])

dip_tau_map  = find_perattack_zak_tau(dip_val, "vit_score")
dip_combined = make_combined_test(dip_val, dip_test)
dip_results  = perattack_metrics(dip_combined, "vit_score", dip_tau_map)

print("  Per-attack ZAK thresholds (Diff-IrisPAD):")
for atk, (p, t) in dip_tau_map.items():
    print(f"    {atk:<25} p{p:02d}  tau={t:.4f}")


# ── Load AnoDDPM ─────────────────────────────────────────────────────────────

print("\nLoading AnoDDPM scores …")
ano_val  = pd.read_csv(PAD_DIR / "ddpm_val_simplex_tstar500_ddim_steps50_vit.csv")
ano_test = pd.read_csv(PAD_DIR / "ddpm_test_simplex_tstar500_ddim_steps50_vit.csv")
for df in [ano_val, ano_test]:
    df["attack_type"] = df["attack_type"].replace(ANODDPM_REMAP)
    df["attack_type"] = df["attack_type"].fillna(df["label"])

ano_tau_map  = find_perattack_zak_tau(ano_val, "vit_score")
ano_combined = make_combined_test(ano_val, ano_test)
ano_results  = perattack_metrics(ano_combined, "vit_score", ano_tau_map)

print("  Per-attack ZAK thresholds (AnoDDPM):")
for atk, (p, t) in ano_tau_map.items():
    print(f"    {atk:<25} p{p:02d}  tau={t:.4f}")


# ── Ablation steps ───────────────────────────────────────────────────────────

print("\nLoading ablation step scores …")
ablation_results  = {}
ablation_tau_maps = {}
STEPS = [10, 50, 100, 200]

for steps in STEPS:
    skey = f"{steps:03d}"
    val_df  = pd.read_csv(SCORE_DIR / f"steps_cache/vit_scores_val_steps_{skey}.csv")
    test_df = pd.read_csv(SCORE_DIR / f"steps_cache/vit_scores_steps_{skey}.csv")
    val_df["attack_type"]  = val_df["attack_type"].fillna(val_df["label"])
    test_df["attack_type"] = test_df["attack_type"].fillna(test_df["label"])

    tau_map = find_perattack_zak_tau(val_df, "vit_score")
    combined = make_combined_test(val_df, test_df)
    ablation_tau_maps[steps] = tau_map
    ablation_results[steps]  = perattack_metrics(combined, "vit_score", tau_map)
    print(f"  Steps {steps:3d}: taus = {[round(v[1],3) for v in tau_map.values()]}")


# ── Print summary ────────────────────────────────────────────────────────────

lines = []
def log(s=""):
    print(s)
    lines.append(s)

log("=" * 90)
log("DIFF-IRIPAD  Per-Attack ZAK  (combined test: val attacks + test attacks)")
log("=" * 90)
log(f"{'Attack':<22}  {'p':>4}  {'tau':>6}  {'N_atk':>6}  {'APCER':>7}  {'BPCER':>7}  {'ACER':>7}  {'EER':>7}")
for _, r in dip_results.iterrows():
    log(f"{r['attack']:<22}  {str(r['percentile']):>4}  {r['tau']:>6.4f}  {int(r['n_attack']):>6}  "
        f"{pct(r['APCER']):>7}  {pct(r['BPCER']):>7}  {pct(r['ACER']):>7}  {pct(r['EER']):>7}")

log()
log("=" * 90)
log("ANODDPM  Per-Attack ZAK  (ViT scoring, combined test)")
log("=" * 90)
log(f"{'Attack':<22}  {'p':>4}  {'tau':>6}  {'N_atk':>6}  {'APCER':>7}  {'BPCER':>7}  {'ACER':>7}  {'EER':>7}")
for _, r in ano_results.iterrows():
    log(f"{r['attack']:<22}  {str(r['percentile']):>4}  {r['tau']:>6.4f}  {int(r['n_attack']):>6}  "
        f"{pct(r['APCER']):>7}  {pct(r['BPCER']):>7}  {pct(r['ACER']):>7}  {pct(r['EER']):>7}")

log()
log("=" * 90)
log("ABLATION STEPS  Per-Attack ZAK")
log("=" * 90)
for steps in STEPS:
    res = ablation_results[steps]
    log(f"\n--- Steps {steps} ---")
    log(f"{'Attack':<22}  {'p':>4}  {'tau':>6}  {'APCER':>7}  {'BPCER':>7}  {'ACER':>7}  {'EER':>7}")
    for _, r in res.iterrows():
        log(f"{r['attack']:<22}  {str(r['percentile']):>4}  {r['tau']:>6.4f}  "
            f"{pct(r['APCER']):>7}  {pct(r['BPCER']):>7}  {pct(r['ACER']):>7}  {pct(r['EER']):>7}")

(OUT_DIR / "zak_numbers.txt").write_text("\n".join(lines))


# ── Table 2 LaTeX ────────────────────────────────────────────────────────────

def best_per_attack(ano_res, dip_res):
    bests = {}
    for atk in ATTACK_ORDER + ["ALL"]:
        ar = ano_res[ano_res["attack"] == atk]
        dr = dip_res[dip_res["attack"] == atk]
        if ar.empty or dr.empty:
            bests[atk] = None
        else:
            bests[atk] = "dip" if dr.iloc[0]["ACER"] <= ar.iloc[0]["ACER"] else "ano"
    return bests


def bold_if(val, condition):
    s = pct(val)
    return f"\\textbf{{{s}}}" if condition else s


def make_table2(ano_res, dip_res, ano_tmap, dip_tmap):
    bests = best_per_attack(ano_res, dip_res)

    L = []
    L += [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\label{tab:diffusion_comparison}",
        r"\caption{",
        r"\textbf{Comparison of diffusion-based iris PAD methods under the"
        r" per-attack Zero-Attack-Knowledge (ZAK) protocol.}",
        r"For each attack, $\tau$ is the percentile of bona-fide validation scores"
        r" (column $p$) that minimises per-attack ACER on the validation set;"
        r" no attack labels are used to determine threshold values.",
        r"Test pool merges validation- and test-split attacks (50{,}612 attacks"
        r" $+$ 12{,}926 bona-fide).",
        r"All metrics in \% (lower is better); \textbf{bold} = best ACER per attack.",
        r"}",
        r"",
        r"\begin{tabular}{l l c c c c c c}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Attack} & $p$ & $\tau$ & APCER & BPCER & ACER & EER \\",
        r"\midrule",
        r"",
    ]

    # AnoDDPM block
    L.append(r"\multirow{9}{*}{\shortstack[l]{AnoDDPM \\DDIM\\Simplex\\ \cite{ho2020denoising}}}")
    for atk in ATTACK_ORDER:
        row = ano_res[ano_res["attack"] == atk].iloc[0]
        p   = ano_tmap[atk][0]
        ib  = bests.get(atk) == "ano"
        L.append(f"& {ATTACK_DISPLAY[atk]:<14} & {p:>2} & {row['tau']:.3f}"
                 f" & {pct(row['APCER']):>6} & {pct(row['BPCER']):>6}"
                 f" & {bold_if(row['ACER'], ib):>20} & {bold_if(row['EER'], ib):>14} \\\\")
    row_all = ano_res[ano_res["attack"] == "ALL"].iloc[0]
    ib_all  = bests.get("ALL") == "ano"
    L.append(f"& \\textbf{{All}}  & {int(row_all['percentile']):>2} & {row_all['tau']:.3f}"
             f" & {pct(row_all['APCER']):>6} & {pct(row_all['BPCER']):>6}"
             f" & {bold_if(row_all['ACER'], ib_all):>20} & {bold_if(row_all['EER'], ib_all):>14} \\\\")

    L += [r"", r"\midrule", r""]

    # Diff-IrisPAD block
    L.append(r"\multirow{9}{*}{\shortstack[l]{\textbf{Diff-IrisPAD}\\ (Ours)}}")
    for atk in ATTACK_ORDER:
        row = dip_res[dip_res["attack"] == atk].iloc[0]
        p   = dip_tmap[atk][0]
        ib  = bests.get(atk) == "dip"
        L.append(f"& {ATTACK_DISPLAY[atk]:<14} & {p:>2} & {row['tau']:.3f}"
                 f" & {pct(row['APCER']):>6} & {pct(row['BPCER']):>6}"
                 f" & {bold_if(row['ACER'], ib):>20} & {bold_if(row['EER'], ib):>14} \\\\")
    row_all = dip_res[dip_res["attack"] == "ALL"].iloc[0]
    ib_all  = bests.get("ALL") == "dip"
    L.append(f"& \\textbf{{All}}  & {int(row_all['percentile']):>2} & {row_all['tau']:.3f}"
             f" & {pct(row_all['APCER']):>6} & {pct(row_all['BPCER']):>6}"
             f" & {bold_if(row_all['ACER'], ib_all):>20} & {bold_if(row_all['EER'], ib_all):>14} \\\\")

    L += [r"", r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


# ── Table 3 LaTeX ────────────────────────────────────────────────────────────

def make_table3(ablation_res, ablation_tmaps):
    # Best ACER per attack across step counts
    best_step = {}
    for atk in ATTACK_ORDER + ["ALL"]:
        best_step[atk] = min(STEPS,
            key=lambda s: ablation_res[s][ablation_res[s]["attack"] == atk].iloc[0]["ACER"])

    L = []
    L += [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{\textbf{Ablation on denoising steps ($T$) under per-attack ZAK protocol.}",
        r"For each attack the threshold $\tau$ is the percentile $p$ of bona-fide validation"
        r" scores that minimises per-attack validation ACER; no attack labels used for"
        r" threshold values. Test pool combines validation and test attack splits."
        r" Results in \% (lower is better); \textbf{bold} = best ACER per attack.}",
        r"\label{tab:ablation_study2}",
        r"",
        r"\begin{tabular}{l l c c c c c c}",
        r"\toprule",
        r"\textbf{Steps} & \textbf{Atk} & $p$ & APCER & BPCER & ACER & EER & ms/img \\",
        r"\midrule",
    ]

    for i, steps in enumerate(STEPS):
        res  = ablation_res[steps]
        tmap = ablation_tmaps[steps]
        ms   = TIMING[steps]
        L.append(r"")
        L.append(f"\\multirow{{9}}{{*}}{{{steps}}}")
        for j, atk in enumerate(ATTACK_ORDER):
            row = res[res["attack"] == atk].iloc[0]
            p   = tmap[atk][0]
            ib  = best_step.get(atk) == steps
            ms_cell = f"\\multirow{{9}}{{*}}{{{ms:.1f}}}" if j == 0 else ""
            L.append(f"& {ATTACK_DISPLAY[atk]:<14} & {p:>2}"
                     f" & {pct(row['APCER']):>6} & {pct(row['BPCER']):>6}"
                     f" & {bold_if(row['ACER'], ib):>20}"
                     f" & {pct(row['EER']):>7} & {ms_cell} \\\\")
        row = res[res["attack"] == "ALL"].iloc[0]
        ib  = best_step.get("ALL") == steps
        L.append(f"& \\textbf{{All}}  & {int(row['percentile']):>2}"
                 f" & {pct(row['APCER']):>6} & {pct(row['BPCER']):>6}"
                 f" & {bold_if(row['ACER'], ib):>20}"
                 f" & {pct(row['EER']):>7} & \\\\")
        if i < len(STEPS) - 1:
            L.append(r"\midrule")

    L += [r"", r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


# ── Write outputs ────────────────────────────────────────────────────────────

t2 = make_table2(ano_results, dip_results, ano_tau_map, dip_tau_map)
t3 = make_table3(ablation_results, ablation_tau_maps)

(OUT_DIR / "table2_zak.tex").write_text(t2)
(OUT_DIR / "table3_zak.tex").write_text(t3)

# CSV summaries
dip_results.to_csv(OUT_DIR / "dip_zak_perattack.csv", index=False)
ano_results.to_csv(OUT_DIR / "ano_zak_perattack.csv", index=False)
for steps in STEPS:
    ablation_results[steps].to_csv(OUT_DIR / f"ablation_steps{steps:03d}.csv", index=False)

print(f"\nSaved all outputs to {OUT_DIR}/")
print("\nDone.")
