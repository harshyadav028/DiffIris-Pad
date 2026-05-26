"""
iris_bbdm_pad/evaluation/generate_ijcb_tables.py

Assembles the 4-method scoring comparison table and per-attack ablation table,
saves them as CSVs and publication-quality PNG images.

Methods compared:
  1. LPIPS only              (existing, threshold=0.1923)
  2. ViT only                (new)
  3. ViT + LPIPS fixed 50/50 (from compute_dynamic_fusion.py)
  4. ViT + LPIPS dynamic     (from compute_dynamic_fusion.py)

Usage:
    cd /home/teaching/Documents/Geetanjali_PhD_IRIS_PAD
    python iris_bbdm_pad/evaluation/generate_ijcb_tables.py
"""

import json
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path

rcParams["font.family"] = "DejaVu Sans"

SCORING_DIR = Path("IJCB_paper_requirements/scoring")
OUT_DIR     = Path("IJCB_paper_requirements/tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ATTACK_ORDER = [
    "Artifact", "Contact Lens", "E-display", "Fake w/ Add-On",
    "Generated", "Post-Mortem", "Print & E-disp.", "Printed", "Overall",
]

# Map raw attack_type values from per_attack_metrics.csv to display names
LPIPS_ATTACK_MAP = {
    "Artifact":             "Artifact",
    "CL":                   "Contact Lens",
    "E-display":            "E-display",
    "Fake with Add On":     "Fake w/ Add-On",
    "Generated":            "Generated",
    "PostMortem":           "Post-Mortem",
    "Print and E-display":  "Print & E-disp.",
    "Printed":              "Printed",
}


def reorder_attacks(df: pd.DataFrame, attack_col: str = "Attack") -> pd.DataFrame:
    order_map = {a: i for i, a in enumerate(ATTACK_ORDER)}
    df = df.copy()
    df["_ord"] = df[attack_col].map(lambda x: order_map.get(x, 99))
    return df.sort_values("_ord").drop("_ord", axis=1).reset_index(drop=True)


def load_lpips_metrics() -> pd.DataFrame:
    """Load LPIPS per-attack metrics computed from per-sample scores.

    Accuracy is computed as (TP + TN) / (n_attack_subset + n_bonafide),
    NOT as 100 - ACER.  The per-sample scores come from all_scores_test.csv
    so that the formula is identical to the one used for the ViT/fusion methods.
    """
    with open("iris_bbdm_pad/results/phase3_evaluation/metrics_summary.json") as f:
        summary = json.load(f)
    threshold    = float(summary["threshold"])
    bpcer_global = float(summary["metrics"]["bpcer"])   # already a fraction in JSON

    # Load per-sample scores so we can compute real accuracy
    scores_df = pd.read_csv(str(SCORING_DIR / "all_scores_test.csv"))
    bonafide = scores_df[scores_df["label"] == "bonafide"]
    attacks  = scores_df[scores_df["label"] != "bonafide"]

    # Bonafide TN count is constant across all attack sub-tables
    tn_global = int((bonafide["lpips_score"] <= threshold).sum())
    n_bonafide = len(bonafide)

    rows = []
    for raw_type, display in LPIPS_ATTACK_MAP.items():
        atk_df = attacks[attacks["attack_type"] == raw_type]
        if atk_df.empty:
            continue
        n_atk   = len(atk_df)
        apcer   = float((atk_df["lpips_score"] <= threshold).mean())
        acer    = (apcer + bpcer_global) / 2.0
        tp      = int((atk_df["lpips_score"] > threshold).sum())
        acc     = (tp + tn_global) / (n_atk + n_bonafide)
        rows.append({
            "Method":  "LPIPS only",
            "Attack":  display,
            "APCER↓":  round(apcer,        6),
            "BPCER↓":  round(bpcer_global, 6),
            "ACER↓":   round(acer,         6),
            "Acc↑":    round(acc,          6),
            "τ":       round(threshold,    4),
        })

    # Overall row
    overall_apcer = float((attacks["lpips_score"] <= threshold).mean())
    overall_acer  = (overall_apcer + bpcer_global) / 2.0
    overall_tp    = int((attacks["lpips_score"] > threshold).sum())
    overall_acc   = (overall_tp + tn_global) / (len(attacks) + n_bonafide)
    rows.append({
        "Method":  "LPIPS only",
        "Attack":  "Overall",
        "APCER↓":  round(overall_apcer, 6),
        "BPCER↓":  round(bpcer_global,  6),
        "ACER↓":   round(overall_acer,  6),
        "Acc↑":    round(overall_acc,   6),
        "τ":       round(threshold,     4),
    })

    df = pd.DataFrame(rows)
    return reorder_attacks(df)


def load_method_metrics(csv_path: str, method_name: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.insert(0, "Method", method_name)
    return reorder_attacks(df)


def render_table_png(
    df: pd.DataFrame,
    title: str,
    out_path: Path,
    figwidth: float = None,
) -> None:
    """Render a DataFrame as a publication-quality table PNG."""
    n_rows, n_cols = df.shape
    fig_h = max(3.2, n_rows * 0.48 + 2.0)
    fig_w = figwidth or min(22, n_cols * 1.8)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    col_labels = df.columns.tolist()
    cell_text  = [[str(v) for v in row] for _, row in df.iterrows()]

    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.1, 1.75)

    header_color = "#1a252f"
    for j in range(n_cols):
        cell = tbl[(0, j)]
        cell.set_facecolor(header_color)
        cell.set_text_props(color="white", fontweight="bold")

    for i in range(1, n_rows + 1):
        row_label = str(df.iloc[i - 1, 0]) if "Attack" not in df.columns else str(df.iloc[i - 1].get("Attack", ""))
        is_overall = row_label.lower() == "overall"
        bg = "#f0f4f8" if i % 2 == 0 else "white"
        if is_overall:
            bg = "#d5e8d4"
        for j in range(n_cols):
            cell = tbl[(i, j)]
            cell.set_facecolor(bg)
            if is_overall:
                cell.set_text_props(fontweight="bold")

    ax.set_title(title, fontsize=10, fontweight="bold", pad=14, color="#1a252f")
    fig.tight_layout(pad=1.5)
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def main() -> None:
    print("=" * 60)
    print("Generating IJCB Task 1 Tables  (DiffIrisPAD)")
    print("=" * 60)

    # ── Load all four methods ────────────────────────────────────
    print("\nLoading per-attack results...")
    lpips_df  = load_lpips_metrics()
    vit_df    = load_method_metrics(str(SCORING_DIR / "vit_metrics_test.csv"),    "ViT only")
    fixed_df  = load_method_metrics(str(SCORING_DIR / "fixed_fusion_metrics_test.csv"),   "ViT+LPIPS (fixed)")
    dyn_df    = load_method_metrics(str(SCORING_DIR / "dynamic_fusion_metrics_test.csv"), "ViT+LPIPS (dynamic)")

    # ── Full ablation table (all methods stacked) ─────────────────
    ablation = pd.concat([lpips_df, vit_df, fixed_df, dyn_df], ignore_index=True)
    ablation.to_csv(OUT_DIR / "table_ablation_scoring.csv", index=False)
    print(f"\n[CSV] Ablation saved → {OUT_DIR}/table_ablation_scoring.csv")

    # ── Wide ACER comparison table ────────────────────────────────
    def extract_acer(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        return (
            df[["Attack", "ACER↓"]]
            .rename(columns={"ACER↓": col_name})
        )

    wide = pd.DataFrame({"Attack": ATTACK_ORDER})
    for df_, name in [
        (lpips_df, "LPIPS only"),
        (vit_df,   "ViT only"),
        (fixed_df, "ViT+LPIPS (fixed)"),
        (dyn_df,   "ViT+LPIPS (dynamic)"),
    ]:
        wide = wide.merge(extract_acer(df_, name), on="Attack", how="left")

    wide.to_csv(OUT_DIR / "table_scoring_comparison.csv", index=False)
    print(f"[CSV] Comparison saved → {OUT_DIR}/table_scoring_comparison.csv")

    # ── Bold-annotated comparison (best per row marked with *) ───
    score_cols = ["LPIPS only", "ViT only", "ViT+LPIPS (fixed)", "ViT+LPIPS (dynamic)"]
    wide_bold = wide.copy()
    for idx, row in wide_bold.iterrows():
        vals = pd.to_numeric(row[score_cols], errors="coerce")
        best = vals.min()
        for c in score_cols:
            if pd.notna(vals[c]) and vals[c] == best:
                wide_bold.at[idx, c] = f"*{row[c]}*"
    wide_bold.to_csv(OUT_DIR / "table_scoring_comparison_bold.csv", index=False)

    # ── Render PNGs ───────────────────────────────────────────────
    print("\nRendering table PNGs...")

    render_table_png(
        wide,
        title="Table: ACER (%) by Scoring Method — DiffIrisPAD (IJCB 2026)",
        out_path=OUT_DIR / "table_scoring_comparison.png",
        figwidth=14,
    )

    render_table_png(
        ablation,
        title="Ablation: All Scoring Methods × Per-Attack Metrics — DiffIrisPAD",
        out_path=OUT_DIR / "table_ablation_scoring.png",
        figwidth=20,
    )

    for method_name, df_ in [
        ("LPIPS only",           lpips_df),
        ("ViT only",             vit_df),
        ("ViT+LPIPS (fixed)",    fixed_df),
        ("ViT+LPIPS (dynamic)",  dyn_df),
    ]:
        sub = df_.drop("Method", axis=1)
        safe = (method_name.replace(" ", "_").replace("+", "plus")
                            .replace("(", "").replace(")", ""))
        render_table_png(
            sub,
            title=f"Per-Attack Metrics — {method_name} | DiffIrisPAD",
            out_path=OUT_DIR / f"table_ablation_{safe}.png",
            figwidth=12,
        )

    # ── Print summary ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Task 1 output summary:")
    print(f"  {SCORING_DIR}/  ← ViT score CSVs, threshold JSONs, all_scores_test.csv")
    print(f"  {OUT_DIR}/      ← comparison + ablation tables (CSV + PNG)")
    print("=" * 60)

    # Print the wide ACER table to stdout for quick review
    print("\nACER (%) comparison — test set:")
    print(wide.to_string(index=False))


if __name__ == "__main__":
    main()
