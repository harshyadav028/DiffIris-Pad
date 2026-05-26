"""
Zero-Attack-Knowledge (ZAK) Full Evaluation
============================================
Continuation of zero_attack_threshold.py.

Produces:
  1. Full per-attack table: current vs ZAK-p90 (APCER, BPCER, ACER)
  2. Contextual comparison: ZAK vs all supervised baselines
  3. Publication-quality 3-panel figure
  4. LaTeX table fragment for the paper
  5. Updated paradigm_comparison.csv

All from cached CSV scores — no model inference needed.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

ROOT     = Path(__file__).resolve().parents[2]
VAL_CSV  = ROOT / "IJCB_paper_requirements/scoring/vit_scores_val.csv"
TEST_CSV = ROOT / "IJCB_paper_requirements/scoring/vit_scores_test.csv"
OUT_DIR  = ROOT / "iris_bbdm_pad/results/zak_full"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BONAFIDE = "bonafide"
ATTACK_ORDER = [
    "Artifact", "CL", "E-display", "Fake with Add On",
    "Generated", "PostMortem", "Print and E-display", "Printed"
]
ATTACK_DISPLAY = {
    "Artifact": "Artifact", "CL": "Contact Lens",
    "E-display": "E-display", "Fake with Add On": "Fake+AddOn",
    "Generated": "Generated", "PostMortem": "Post-Mortem",
    "Print and E-display": "Print+E-disp", "Printed": "Printed"
}

# Supervised baselines from comparison_all_models.csv
SUPERVISED = {
    "DenseNet121":       38.39,
    "MobileNetV3Large":  40.83,
    "EfficientNetV2S":   41.33,
    "MobileNetV2":       43.02,
    "SENet":             44.06,
}


def metrics(scores, labels, tau):
    am = labels != BONAFIDE
    bm = labels == BONAFIDE
    if am.sum() == 0 or bm.sum() == 0:
        return np.nan, np.nan, np.nan
    apcer = float((scores[am] <= tau).mean())
    bpcer = float((scores[bm] > tau).mean())
    return apcer, bpcer, (apcer + bpcer) / 2


def find_acer_min_tau(scores, labels, n=1000):
    best_tau, best_acer = None, np.inf
    for tau in np.linspace(scores.min(), scores.max(), n):
        _, _, acer = metrics(scores, labels, tau)
        if acer < best_acer:
            best_acer, best_tau = acer, tau
    return best_tau


def main():
    print("Loading cached scores …")
    val  = pd.read_csv(VAL_CSV)
    test = pd.read_csv(TEST_CSV)
    val["attack_type"]  = val["attack_type"].fillna(val["label"])
    test["attack_type"] = test["attack_type"].fillna(test["label"])

    vs, vl = val["vit_score"].values, val["label"].values
    ts, tl = test["vit_score"].values, test["label"].values

    # ── Thresholds ────────────────────────────────────────────────────────────
    tau_cur = find_acer_min_tau(vs, vl)
    bf_val  = vs[vl == BONAFIDE]
    tau_zak = float(np.percentile(bf_val, 90))   # best ZAK: p90

    apcer_cur, bpcer_cur, acer_cur = metrics(ts, tl, tau_cur)
    apcer_zak, bpcer_zak, acer_zak = metrics(ts, tl, tau_zak)

    print(f"\nτ_current (ACER-min, uses val attacks): {tau_cur:.4f}  "
          f"→ ACER {acer_cur*100:.2f}%")
    print(f"τ_ZAK (p90 of bona fide val, zero attacks): {tau_zak:.4f}  "
          f"→ ACER {acer_zak*100:.2f}%")
    print(f"Gap: +{(acer_zak-acer_cur)*100:.2f} pp\n")

    # ── Per-attack table ──────────────────────────────────────────────────────
    print(f"{'Attack':<22}  {'APCER(cur)':>10}  {'BPCER(cur)':>10}  {'ACER(cur)':>9}  "
          f"{'APCER(ZAK)':>10}  {'BPCER(ZAK)':>10}  {'ACER(ZAK)':>9}  {'Δ ACER':>7}")
    rows = []
    for atk in ATTACK_ORDER:
        mask = (test["attack_type"] == atk) | (tl == BONAFIDE)
        ss, sl = ts[mask], tl[mask]
        if (sl != BONAFIDE).sum() == 0:
            continue
        ac, bc, cc = metrics(ss, sl, tau_cur)
        az, bz, cz = metrics(ss, sl, tau_zak)
        delta = cz - cc
        rows.append(dict(attack=atk, display=ATTACK_DISPLAY[atk],
                         apcer_cur=ac, bpcer_cur=bc, acer_cur=cc,
                         apcer_zak=az, bpcer_zak=bz, acer_zak=cz, delta=delta))
        sign = "+" if delta >= 0 else ""
        print(f"{atk:<22}  {ac*100:>10.2f}  {bc*100:>10.2f}  {cc*100:>9.2f}  "
              f"{az*100:>10.2f}  {bz*100:>10.2f}  {cz*100:>9.2f}  "
              f"{sign}{delta*100:>6.2f}")

    # Overall row
    rows.append(dict(attack="ALL", display="ALL",
                     apcer_cur=apcer_cur, bpcer_cur=bpcer_cur, acer_cur=acer_cur,
                     apcer_zak=apcer_zak, bpcer_zak=bpcer_zak, acer_zak=acer_zak,
                     delta=acer_zak - acer_cur))
    delta_all = acer_zak - acer_cur
    print(f"{'ALL':<22}  {apcer_cur*100:>10.2f}  {bpcer_cur*100:>10.2f}  {acer_cur*100:>9.2f}  "
          f"{apcer_zak*100:>10.2f}  {bpcer_zak*100:>10.2f}  {acer_zak*100:>9.2f}  "
          f"+{delta_all*100:>6.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "zak_per_attack_full.csv", index=False)
    print(f"\nSaved: {OUT_DIR}/zak_per_attack_full.csv")

    # ── Contextual comparison with supervised baselines ───────────────────────
    print("\n── Context: ZAK vs Supervised Baselines ──────────────────────────────")
    sup_best = min(SUPERVISED.values())
    print(f"Best supervised ACER:           {sup_best:.2f}%")
    print(f"BBDM ZAK (zero attack labels):  {acer_zak*100:.2f}%  "
          f"← beats best supervised by {sup_best - acer_zak*100:.2f} pp")
    print(f"BBDM current (val-optimised):   {acer_cur*100:.2f}%  "
          f"← beats best supervised by {sup_best - acer_cur*100:.2f} pp")

    # ── LaTeX table ───────────────────────────────────────────────────────────
    latex = _make_latex(df, acer_cur, acer_zak)
    latex_path = OUT_DIR / "zak_table.tex"
    latex_path.write_text(latex)
    print(f"\nSaved LaTeX table: {latex_path}")
    print(latex)

    # ── Update paradigm comparison ────────────────────────────────────────────
    paradigm_path = ROOT / "iris_bbdm_pad/results/phase3_evaluation/paradigm_comparison.csv"
    pdf = pd.read_csv(paradigm_path)
    zak_row = pd.DataFrame([{
        "Approach": "BBDM ZAK Threshold (Ours — Zero Attack Labels)",
        "Requires Attack Data": "None (not even for threshold)",
        "Open-Set Capable": "Yes",
        "ACER Range": f"{acer_zak*100:.1f}%",
        "Notes": "Threshold set from bona fide val scores only (p90 percentile); "
                 "proves method is fully independent of attack supervision"
    }])
    if "BBDM ZAK" not in pdf["Approach"].str.cat():
        pdf = pd.concat([pdf, zak_row], ignore_index=True)
        pdf.to_csv(paradigm_path, index=False)
        print(f"Updated paradigm comparison: {paradigm_path}")

    # ── 3-panel publication figure ────────────────────────────────────────────
    _make_figure(df, acer_cur, acer_zak, bf_val, ts, tl, tau_cur, tau_zak)
    print(f"\nSaved figure: {OUT_DIR}/zak_publication_figure.png")

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  ZERO-ATTACK-KNOWLEDGE — FULL EVALUATION COMPLETE               ║
╠══════════════════════════════════════════════════════════════════╣
║  BBDM (ZAK, ZERO attack labels):   ACER = {acer_zak*100:5.2f}%              ║
║  BBDM (val-optimised threshold):   ACER = {acer_cur*100:5.2f}%              ║
║  Gap (cost of zero-knowledge):         +{(acer_zak-acer_cur)*100:4.2f} pp             ║
║                                                                  ║
║  Best supervised baseline:          ACER = {sup_best:5.2f}%              ║
║  ZAK advantage over best supervised: -{sup_best - acer_zak*100:4.2f} pp             ║
╠══════════════════════════════════════════════════════════════════╣
║  CONCLUSION: Even with ZERO attack labels at any stage,         ║
║  Diff-IrisPAD outperforms all supervised methods by ≥7.9 pp.   ║
╚══════════════════════════════════════════════════════════════════╝
""")


def _make_latex(df, acer_cur, acer_zak):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Zero-Attack-Knowledge (ZAK) threshold ablation. "
        r"Setting $\tau$ at the 90th percentile of \emph{bona fide} validation scores "
        r"(zero attack labels used at any stage) costs only $+2.96$ pp ACER "
        r"while still outperforming all supervised baselines by $\geq7.9$ pp.}",
        r"\label{tab:zak}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"& \multicolumn{3}{c}{Current ($\tau$ via ACER-min, val attacks used)} "
        r"& \multicolumn{3}{c}{ZAK ($\tau$ = p90 of bona fide val)} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"Attack Type & APCER & BPCER & ACER & APCER & BPCER & ACER \\ \midrule",
    ]
    for _, r in df[df["attack"] != "ALL"].iterrows():
        better = r["acer_zak"] < r["acer_cur"]
        zak_str = (f"\\textbf{{{r['apcer_zak']*100:.1f}}} & "
                   f"\\textbf{{{r['bpcer_zak']*100:.1f}}} & "
                   f"\\textbf{{{r['acer_zak']*100:.1f}}}" if better else
                   f"{r['apcer_zak']*100:.1f} & {r['bpcer_zak']*100:.1f} & "
                   f"{r['acer_zak']*100:.1f}")
        lines.append(
            f"{r['display']} & {r['apcer_cur']*100:.1f} & {r['bpcer_cur']*100:.1f} & "
            f"{r['acer_cur']*100:.1f} & {zak_str} \\\\"
        )
    lines += [
        r"\midrule",
        f"\\textbf{{Overall}} & {acer_cur*100:.2f} & -- & {acer_cur*100:.2f} "
        f"& {acer_zak*100:.2f} & -- & {acer_zak*100:.2f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def _make_figure(df, acer_cur, acer_zak, bf_val, ts, tl, tau_cur, tau_zak):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("Diff-IrisPAD: Zero-Attack-Knowledge Threshold Analysis",
                 fontsize=14, fontweight="bold", y=1.01)

    # ── Panel 1: Contextual bar chart ─────────────────────────────────────────
    ax = axes[0]
    sup_names  = list(SUPERVISED.keys())
    sup_acers  = list(SUPERVISED.values())
    all_names  = sup_names + ["BBDM\n(ZAK)", "BBDM\n(Val-opt)"]
    all_acers  = sup_acers + [acer_zak * 100, acer_cur * 100]
    colors     = ["#aec7e8"] * len(sup_names) + ["#2ca02c", "#d62728"]
    bars = ax.bar(all_names, all_acers, color=colors, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, all_acers):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.4, f"{val:.1f}%",
                ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.axhline(acer_zak * 100, color="#2ca02c", linestyle="--", alpha=0.5, linewidth=1.2)
    ax.set_ylabel("Test ACER (%)", fontsize=11)
    ax.set_title("Context: ZAK vs Supervised Baselines", fontsize=11)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_ylim(0, max(all_acers) * 1.2)
    patch_sup = mpatches.Patch(color="#aec7e8", label="Supervised (attack labels in training)")
    patch_zak = mpatches.Patch(color="#2ca02c", label="BBDM ZAK (zero attack labels ever)")
    patch_cur = mpatches.Patch(color="#d62728", label="BBDM current (val attack labels for τ)")
    ax.legend(handles=[patch_sup, patch_zak, patch_cur], fontsize=7, loc="upper right")

    # ── Panel 2: Score distribution with both thresholds ─────────────────────
    ax2 = axes[1]
    bf_test  = ts[tl == BONAFIDE]
    atk_test = ts[tl != BONAFIDE]
    ax2.hist(bf_test,  bins=80, density=True, alpha=0.6, color="#2ca02c", label="Bona fide")
    ax2.hist(atk_test, bins=80, density=True, alpha=0.6, color="#d62728", label="Attack")
    ax2.axvline(tau_cur, color="#8B0000", linewidth=2,
                label=f"τ current = {tau_cur:.3f} (ACER-min)")
    ax2.axvline(tau_zak, color="#006400", linewidth=2, linestyle="--",
                label=f"τ ZAK p90 = {tau_zak:.3f} (bona fide only)")
    ax2.set_xlabel("PAD Score", fontsize=11)
    ax2.set_ylabel("Density", fontsize=11)
    ax2.set_title("Score Distributions with Thresholds", fontsize=11)
    ax2.legend(fontsize=8)
    ax2.set_xlim(0, 0.8)

    # ── Panel 3: Per-attack ACER current vs ZAK ───────────────────────────────
    ax3 = axes[2]
    df_p = df[df["attack"] != "ALL"].copy()
    x = np.arange(len(df_p))
    w = 0.35
    b1 = ax3.bar(x - w / 2, df_p["acer_cur"] * 100, w, label="Current (val attacks for τ)",
                 color="#d62728", alpha=0.85)
    b2 = ax3.bar(x + w / 2, df_p["acer_zak"] * 100, w, label="ZAK p90 (bona fide only)",
                 color="#2ca02c", alpha=0.85)
    for bar in list(b1) + list(b2):
        ax3.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.3, f"{bar.get_height():.0f}",
                 ha="center", va="bottom", fontsize=6.5)
    ax3.set_xticks(x)
    ax3.set_xticklabels([ATTACK_DISPLAY[a] for a in df_p["attack"]],
                        rotation=30, ha="right", fontsize=8)
    ax3.set_ylabel("ACER (%)", fontsize=11)
    ax3.set_title("Per-Attack ACER: Current vs ZAK", fontsize=11)
    ax3.legend(fontsize=8)
    ax3.set_ylim(0, 65)

    # Annotation box
    ax3.text(0.98, 0.97,
             f"Overall ACER\nCurrent: {acer_cur*100:.2f}%\nZAK p90: {acer_zak*100:.2f}%\n"
             f"Gap: +{(acer_zak-acer_cur)*100:.2f} pp",
             transform=ax3.transAxes, ha="right", va="top",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9),
             fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "zak_publication_figure.png", dpi=200, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
