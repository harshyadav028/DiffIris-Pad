"""
Generate the f-AnoGAN paper comparison table — EXACT mirror of the Diff-IrisPAD
(BBDM) IJCB Table 2 protocol (iris_bbdm_pad/evaluation/zak_true_final.py).

Protocol (identical to BBDM paper Table 2):
  - Threshold τ = ZAK p=90: 90th percentile of BONAFIDE-ONLY validation scores.
    Attack labels are NEVER used to set τ (zero-attack-knowledge).
  - Combined test pool: attacks = val attacks + test attacks (50,612),
    bonafide = test bonafide only (12,926).
  - Metrics (STANDARD ISO 30107-3 convention, matches evaluate_pad.py & paper):
        APCER = fraction of ATTACK samples scoring <= τ   (attacks missed)
        BPCER = fraction of BONAFIDE samples scoring  > τ  (bonafide rejected)
        ACER  = (APCER + BPCER) / 2
        EER   = min over τ' of (APCER+BPCER)/2 at APCER≈BPCER
  - BPCER is constant across attack rows (same bonafide test pool); only APCER varies.
  - One row per attack type (paper order) + an "OverAll" row.

Usage
-----
  cd ~/Documents/Geetanjali_PhD_IRIS_PAD
  python iris_anogan/evaluation/generate_paper_table_fanogan.py \\
      --val_scores  iris_anogan/results/fanogan_run1/pad_scores/val_scores.csv \\
      --test_scores iris_anogan/results/fanogan_run1/pad_scores/test_scores.csv \\
      --score_col   fanogan_score \\
      --out_dir     iris_anogan/results/fanogan_run1/paper_table
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BONAFIDE = "bonafide"
ZAK_P    = 90

# Paper Table 2 attack order + display labels (mirrors zak_true_final.py)
ATTACK_ORDER = [
    "Artifact", "CL", "E-display", "Fake with Add On",
    "Generated", "PostMortem", "Print and E-display", "Printed",
]
ATTACK_DISPLAY = {
    "Artifact":            "Artifact",
    "CL":                  "Contact Lens",
    "E-display":           "E-display",
    "Fake with Add On":    "Fake W/AO",
    "Generated":           "Generated",
    "PostMortem":          "Post-Mortem",
    "Print and E-display": "Print \\& ED",
    "Printed":             "Printed",
}


# ── Metrics (STANDARD ISO — identical to BBDM zak_true_final.py) ─────────────────

def apcer_bpcer(scores, labels, tau):
    am = labels != BONAFIDE
    bm = labels == BONAFIDE
    if am.sum() == 0 or bm.sum() == 0:
        return np.nan, np.nan
    apcer = float((scores[am] <= tau).mean())   # attacks classified bonafide
    bpcer = float((scores[bm] >  tau).mean())   # bonafide classified attack
    return apcer, bpcer


def compute_eer(scores, labels, n=3000):
    am = labels != BONAFIDE
    bm = labels == BONAFIDE
    if am.sum() == 0 or bm.sum() == 0:
        return np.nan
    best_eer, best_diff = np.nan, np.inf
    for tau in np.linspace(scores.min(), scores.max(), n):
        ap = float((scores[am] <= tau).mean())
        bp = float((scores[bm] >  tau).mean())
        d  = abs(ap - bp)
        if d < best_diff:
            best_diff, best_eer = d, (ap + bp) / 2
    return best_eer


def zak_tau(val_df, score_col, p=ZAK_P):
    """τ = p-th percentile of BONAFIDE-ONLY validation scores."""
    bf = val_df.loc[val_df["label"] == BONAFIDE, score_col].values
    return float(np.percentile(bf, p)), len(bf)


def combined_test(val_df, test_df):
    """Attacks: val+test attacks. Bonafide: test only (val bonafide reserved for τ)."""
    return pd.concat([
        val_df[val_df["label"] != BONAFIDE],
        test_df[test_df["label"] != BONAFIDE],
        test_df[test_df["label"] == BONAFIDE],
    ], ignore_index=True)


def evaluate(combined_df, score_col, tau):
    ts = combined_df[score_col].values
    tl = combined_df["label"].values
    ta = combined_df["attack_type"].values
    rows = []
    for attack in ATTACK_ORDER:
        atk_mask = ta == attack
        bon_mask = tl == BONAFIDE
        if atk_mask.sum() == 0:
            continue
        sub_s = np.concatenate([ts[atk_mask], ts[bon_mask]])
        sub_l = np.concatenate([tl[atk_mask], tl[bon_mask]])
        ap, bp = apcer_bpcer(sub_s, sub_l, tau)
        er = compute_eer(sub_s, sub_l)
        rows.append(dict(attack=attack, n_attack=int(atk_mask.sum()),
                         APCER=ap, BPCER=bp, ACER=(ap+bp)/2, EER=er))
    ap_all, bp_all = apcer_bpcer(ts, tl, tau)
    er_all = compute_eer(ts, tl)
    rows.append(dict(attack="OverAll", n_attack=int((tl != BONAFIDE).sum()),
                     APCER=ap_all, BPCER=bp_all, ACER=(ap_all+bp_all)/2, EER=er_all))
    return pd.DataFrame(rows)


def pct(x):
    return f"{x*100:.2f}" if not np.isnan(x) else "--"


def to_latex(df, score_col, tau):
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{f-AnoGAN (Ours) per-attack PAD performance under the ZAK protocol "
        r"($p{=}90$, $\tau{=}%.3f$). Threshold from bona-fide validation scores only; "
        r"attacks = val+test pool. Same protocol as Diff-IrisPAD Table 2.}" % tau,
        r"\label{tab:fanogan_zak}",
        r"\begin{tabular}{lcccc}", r"\toprule",
        r"Attack & APCER & BPCER & ACER & EER \\", r"\midrule",
    ]
    for _, r in df.iterrows():
        name = ATTACK_DISPLAY.get(r["attack"], r["attack"])
        if r["attack"] == "OverAll":
            lines.append(r"\midrule")
            name = r"\textbf{OverAll}"
        lines.append(f"{name} & {pct(r.APCER)} & {pct(r.BPCER)} & {pct(r.ACER)} & {pct(r.EER)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def to_markdown(df):
    out = ["| Attack | APCER | BPCER | ACER | EER |", "|---|---|---|---|---|"]
    for _, r in df.iterrows():
        name = ATTACK_DISPLAY.get(r["attack"], r["attack"]).replace("\\&", "&")
        if r["attack"] == "OverAll":
            name = "**OverAll**"
        out.append(f"| {name} | {pct(r.APCER)} | {pct(r.BPCER)} | {pct(r.ACER)} | {pct(r.EER)} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_scores",  default="iris_anogan/results/fanogan_run1/pad_scores/val_scores.csv")
    ap.add_argument("--test_scores", default="iris_anogan/results/fanogan_run1/pad_scores/test_scores.csv")
    ap.add_argument("--score_col",   default="vit_score",
                    help="vit_score (matches Diff-IrisPAD paper Table 2), "
                         "fanogan_score (native), or recon_score (sMSE+LPIPS)")
    ap.add_argument("--out_dir",     default="iris_anogan/results/fanogan_run1/paper_table")
    ap.add_argument("--p", type=int, default=90, help="ZAK percentile")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    val  = pd.read_csv(args.val_scores)
    test = pd.read_csv(args.test_scores)

    tau, n_bf = zak_tau(val, args.score_col, p=args.p)
    pool = combined_test(val, test)
    n_atk = int((pool["label"] != BONAFIDE).sum())
    n_test_bf = int((pool["label"] == BONAFIDE).sum())

    print(f"Score column : {args.score_col}")
    print(f"ZAK τ (p={args.p}) : {tau:.4f}  (from {n_bf} val bonafide)")
    print(f"Attack pool  : {n_atk}   Test bonafide: {n_test_bf}")
    print()

    df = evaluate(pool, args.score_col, tau)
    print(f"{'Attack':<22}{'APCER':>8}{'BPCER':>8}{'ACER':>8}{'EER':>8}")
    print("-"*54)
    for _, r in df.iterrows():
        print(f"{ATTACK_DISPLAY.get(r['attack'], r['attack']).replace(chr(92)+'&','&'):<22}"
              f"{pct(r.APCER):>8}{pct(r.BPCER):>8}{pct(r.ACER):>8}{pct(r.EER):>8}")

    stem = f"fanogan_zak_p{args.p}_{args.score_col}"
    df.to_csv(out_dir / f"{stem}.csv", index=False)
    (out_dir / f"{stem}.tex").write_text(to_latex(df, args.score_col, tau))
    (out_dir / f"{stem}.md").write_text(
        f"## f-AnoGAN ZAK p={args.p} — `{args.score_col}` (τ={tau:.4f})\n\n"
        f"> Attacks: {n_atk} (val+test) · Bonafide: {n_test_bf} (test) · "
        f"Same protocol as Diff-IrisPAD Table 2.\n\n" + to_markdown(df) + "\n")
    print(f"\nSaved → {out_dir}/{stem}.{{csv,tex,md}}")


if __name__ == "__main__":
    main()
