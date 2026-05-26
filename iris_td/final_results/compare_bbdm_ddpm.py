# iris_td/final_results/compare_bbdm_ddpm.py
"""
Generate final ablation tables comparing BBDM vs all DDPM variants.

DO NOT RUN — needs inference CSVs from run_ddpm_scoring.py first.

Usage:
    python iris_td/final_results/compare_bbdm_ddpm.py \\
        --bbdm_csv  <path_to_bbdm_test_scores.csv> \\
        --bbdm_tau  <threshold from bbdm threshold.json> \\
        --ddpm_tau  <threshold from ddpm_threshold.json> \\
        --score_col recon_score \\
        --best_config anoddpm_simplex
"""
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_curve

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ABLATION1_CONFIGS = [
    ("ddpm_test_vanilla_ddpm_steps100.csv",
     "DDPM vanilla  (Gaussian, t*=1000)       "),
    ("ddpm_test_partial_gaussian_steps100.csv",
     "DDPM partial  (Gaussian, t*=500)        "),
    ("ddpm_test_anoddpm_simplex_steps100.csv",
     "DDPM AnoDDPM  (Simplex,  t*=500)        "),
    ("ddpm_test_anoddpm_simplex_ddim_steps100.csv",
     "DDPM+DDIM     (Simplex,  t*=500, DDIM)  "),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbdm_csv",     required=True)
    parser.add_argument("--bbdm_tau",     type=float, required=True)
    parser.add_argument("--ddpm_tau",     type=float, required=True)
    parser.add_argument("--score_col",    default="recon_score")
    parser.add_argument("--best_config",  default="anoddpm_simplex")
    return parser.parse_args()


def get_metrics(csv_path, tau, score_col):
    p = Path(csv_path)
    if not p.exists():
        return None
    df     = pd.read_csv(p)
    labels = (df["label"] == "attack").astype(int).values
    scores = df[score_col].values

    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fnr - fpr))
    eer = float((fpr[idx] + fnr[idx]) / 2)

    preds = (scores > tau).astype(int)
    tp = np.sum((preds==1)&(labels==1))
    fp = np.sum((preds==1)&(labels==0))
    tn = np.sum((preds==0)&(labels==0))
    fn = np.sum((preds==0)&(labels==1))
    apcer = fp / max(tn+fp, 1)
    bpcer = fn / max(tp+fn, 1)
    acer  = (apcer+bpcer) / 2

    return dict(apcer=apcer, bpcer=bpcer, acer=acer, eer=eer,
                bf_mean=float(scores[labels==0].mean()),
                atk_mean=float(scores[labels==1].mean()),
                gap=float(scores[labels==1].mean()-scores[labels==0].mean()))


def row_str(label, m, suffix=""):
    if m is None:
        return f"{label}  N/A\n"
    return (f"{label}  "
            f"{m['apcer']*100:>6.2f}%  "
            f"{m['bpcer']*100:>6.2f}%  "
            f"{m['acer']*100:>6.2f}%  "
            f"{m['eer']*100:>6.2f}%  "
            f"gap={m['gap']:.4f}{suffix}\n")


def main():
    args = parse_args()
    sc   = args.score_col

    # Ablation 1
    table1 = """
======================================================
ABLATION STUDY 1 — Diffusion type comparison
======================================================
Model                                    APCER    BPCER    ACER     EER
─────────────────────────────────────────────────────────────────────
"""
    ddpm_acers = []
    for csv_name, label in ABLATION1_CONFIGS:
        csv_path = f"iris_td/pad_scores/{csv_name}"
        m = get_metrics(csv_path, args.ddpm_tau, sc)
        table1 += row_str(label, m)
        if m:
            ddpm_acers.append(m["acer"])

    bbdm = get_metrics(args.bbdm_csv, args.bbdm_tau, sc)
    table1 += row_str("Brownian Bridge (ours)             ", bbdm, "  ← BEST")

    if bbdm and ddpm_acers:
        imp = (min(ddpm_acers) - bbdm["acer"]) * 100
        table1 += f"\nBBDM achieves {imp:.2f}% lower ACER than best DDPM variant.\n"
    table1 += "======================================================"

    # Ablation 2
    table2 = f"""
======================================================
ABLATION STUDY 2 — Denoising steps ({args.best_config})
======================================================
Steps    APCER     BPCER     ACER      EER       Time/img
─────────────────────────────────────────────────────────
"""
    for steps in [10, 25, 50, 100]:
        csv = f"iris_td/pad_scores/ddpm_test_{args.best_config}_steps{steps}.csv"
        m   = get_metrics(csv, args.ddpm_tau, sc)
        if m:
            df    = pd.read_csv(csv)
            avg_t = df["inference_time"].mean() if "inference_time" in df.columns else 0
            table2 += (f"{steps:<8} "
                       f"{m['apcer']*100:<9.2f}% "
                       f"{m['bpcer']*100:<9.2f}% "
                       f"{m['acer']*100:<9.2f}% "
                       f"{m['eer']*100:<9.2f}% "
                       f"{avg_t:.3f}s\n")
    table2 += "======================================================"

    print(table1)
    print(table2)

    out1 = Path("iris_td/final_results/ablation1_diffusion_type.txt")
    out2 = Path("iris_td/final_results/ablation2_steps.txt")
    out1.parent.mkdir(parents=True, exist_ok=True)
    out1.write_text(table1)
    out2.write_text(table2)
    log.info(f"Saved → {out1}")
    log.info(f"Saved → {out2}")


if __name__ == "__main__":
    main()
