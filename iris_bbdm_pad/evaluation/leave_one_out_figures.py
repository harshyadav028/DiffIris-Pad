"""
Leave-One-Out Comparison Figures for BBDM Iris PAD.

Generates all figures for IJCB 2026 paper comparing BBDM (unsupervised,
trained on bona fide only) against supervised models from open_set_summary.csv.

All figures: 300 DPI, saved as both PNG and PDF.

Usage:
    python iris_bbdm_pad/evaluation/leave_one_out_figures.py \\
        --bbdm_detailed iris_bbdm_pad/results/leave_one_out/bbdm_open_set_detailed.csv \\
        --open_set_summary open_set_summary.csv \\
        --test_scores iris_bbdm_pad/results/test_pad_scores.csv \\
        --output_dir iris_bbdm_pad/results/leave_one_out/figures/ \\
        --scoring_method lpips_score
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
BBDM_COLOR = "#2a9d8f"       # teal
SUP_COLORS = ["#6a4c93", "#c77dff", "#a8dadc", "#457b9d", "#e76f51",
              "#f4a261", "#264653", "#8ecae6", "#219ebc", "#023047"]
BBDM_LABEL = "BBDM (ours)\n0 attacks in train"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, path_stem: Path, dpi: int = 300) -> None:
    """Save figure as PNG and PDF."""
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(path_stem.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight")
    logger.info(f"  Saved → {path_stem}.png / .pdf")


def _top_supervised_models(sup_df: pd.DataFrame, n: int = 3) -> list[str]:
    """Return top-N supervised model names ranked by lowest overall ACER.

    Uses rows with empty Attack_Type as 'overall'; falls back to mean across
    attack types if no such rows exist.
    """
    overall = sup_df[sup_df["Attack_Type"].isna() | (sup_df["Attack_Type"].astype(str) == "")]
    if overall.empty:
        overall = (
            sup_df.groupby("Model")["ACER"].mean()
            .reset_index()
            .sort_values("ACER")
        )
    else:
        overall = overall.sort_values("ACER")
    return overall["Model"].head(n).tolist()


def _get_per_attack(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the per-attack rows (non-empty Attack_Type)."""
    df = df.copy()
    df["Attack_Type"] = df["Attack_Type"].fillna("").astype(str).str.strip()
    return df[df["Attack_Type"] != ""].copy()


# ---------------------------------------------------------------------------
# Fig 1: Per-Attack ACER grouped bar chart
# ---------------------------------------------------------------------------

def fig_acer_comparison(
    bbdm_df: pd.DataFrame,
    sup_df: pd.DataFrame,
    output_dir: Path,
    n_sup_models: int = 3,
) -> None:
    """Grouped bar chart: BBDM vs top supervised models, per attack type."""
    bbdm_per = _get_per_attack(bbdm_df).set_index("Attack_Type")
    sup_per = _get_per_attack(sup_df)
    top_models = _top_supervised_models(sup_df, n=n_sup_models)

    attack_types = sorted(bbdm_per.index.tolist())
    n_attacks = len(attack_types)
    n_groups = 1 + len(top_models)

    fig, ax = plt.subplots(figsize=(max(12, n_attacks * 1.6), 6))
    x = np.arange(n_attacks)
    width = 0.8 / n_groups
    offsets = np.linspace(-(n_groups - 1) / 2, (n_groups - 1) / 2, n_groups) * width

    # BBDM bars
    bbdm_acers = [bbdm_per.loc[at, "ACER"] * 100 if at in bbdm_per.index else float("nan")
                  for at in attack_types]
    bars = ax.bar(x + offsets[0], bbdm_acers, width, color=BBDM_COLOR, label=BBDM_LABEL,
                  edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, bbdm_acers):
        if not np.isnan(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}",
                ha="center", va="bottom", fontsize=7.5, color=BBDM_COLOR, fontweight="bold",
            )

    # Supervised model bars
    for i, model in enumerate(top_models):
        model_df = sup_per[sup_per["Model"] == model].set_index("Attack_Type")
        acers = [model_df.loc[at, "ACER"] * 100 if at in model_df.index else float("nan")
                 for at in attack_types]
        ax.bar(x + offsets[i + 1], acers, width,
               color=SUP_COLORS[i % len(SUP_COLORS)],
               label=f"{model}\n(7/8 attacks in train)",
               edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(attack_types, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("ACER (%)", fontsize=11)
    ax.set_title("Per-Attack ACER: BBDM (Unsupervised) vs Top Supervised Models", fontsize=12)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_ylim(0, min(100, max(a for a in bbdm_acers if not np.isnan(a)) * 1.8 + 10))
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _save(fig, output_dir / "loo_acer_comparison")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 2: Per-Attack ACER heatmap
# ---------------------------------------------------------------------------

def fig_acer_heatmap(
    bbdm_df: pd.DataFrame,
    sup_df: pd.DataFrame,
    output_dir: Path,
    n_sup_models: int = 10,
) -> None:
    """Heatmap: attack types × models, coloured by ACER."""
    bbdm_per = _get_per_attack(bbdm_df).set_index("Attack_Type")[["ACER"]].copy()
    bbdm_per.columns = ["BBDM (Ours)"]

    sup_per = _get_per_attack(sup_df)
    top_models = _top_supervised_models(sup_df, n=n_sup_models)
    sup_pivot = sup_per[sup_per["Model"].isin(top_models)].pivot_table(
        index="Attack_Type", columns="Model", values="ACER"
    )

    # Combine: BBDM first, then supervised
    combined = bbdm_per.join(sup_pivot, how="outer")
    # Sort rows by BBDM ACER descending (hardest first)
    combined = combined.sort_values("BBDM (Ours)", ascending=False)

    data = combined.values * 100  # percent
    row_labels = combined.index.tolist()
    col_labels = combined.columns.tolist()

    fig, ax = plt.subplots(figsize=(max(12, len(col_labels) * 1.4), max(5, len(row_labels) * 0.9)))
    im = ax.imshow(data, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)

    # Cell text + bold where BBDM beats supervised
    for r in range(len(row_labels)):
        for c in range(len(col_labels)):
            val = data[r, c]
            if np.isnan(val):
                txt = "—"
                weight = "normal"
            else:
                # Bold if BBDM (col=0) value < supervised value in same row
                bbdm_val = data[r, 0]
                is_bbdm_col = (c == 0)
                beats = (not np.isnan(bbdm_val)) and (not np.isnan(val)) and (c > 0) and (bbdm_val < val)
                weight = "bold" if (is_bbdm_col or beats) else "normal"
                txt = f"{val:.1f}"
            color = "white" if val > 65 or np.isnan(val) else "black"
            if np.isnan(val):
                color = "gray"
            ax.text(c, r, txt, ha="center", va="center", fontsize=7,
                    color=color, fontweight=weight)

    plt.colorbar(im, ax=ax, label="ACER (%)", shrink=0.6)
    ax.set_title(
        "ACER (%) Heatmap — Green=Low (better). Bold=BBDM outperforms supervised",
        fontsize=11,
    )
    fig.tight_layout()
    _save(fig, output_dir / "loo_acer_heatmap")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 3: BBDM per-attack ROC curves
# ---------------------------------------------------------------------------

def fig_roc_curves(
    test_df: pd.DataFrame,
    bbdm_detailed_df: pd.DataFrame,
    output_dir: Path,
    scoring_method: str = "lpips_score",
) -> None:
    """One ROC curve per attack type, overlaid on a single axes."""
    bonafide_scores = test_df[test_df["label"] == "bonafide"][scoring_method].values
    attack_df = test_df[test_df["label"] == "attack"]
    attack_types = sorted(attack_df["attack_type"].unique())

    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="Random (AUC=0.50)")

    for i, at in enumerate(attack_types):
        attack_scores = attack_df.loc[attack_df["attack_type"] == at, scoring_method].values
        labels = np.concatenate([np.zeros(len(bonafide_scores)), np.ones(len(attack_scores))])
        scores = np.concatenate([bonafide_scores, attack_scores])
        fpr, tpr, _ = roc_curve(labels, scores)
        try:
            auc = roc_auc_score(labels, scores)
        except ValueError:
            auc = float("nan")
        ax.plot(fpr, tpr, color=cmap(i % 10), lw=1.8, label=f"{at} (AUC={auc:.4f})")

    ax.set_xlabel("False Positive Rate (BPCER)", fontsize=11)
    ax.set_ylabel("True Positive Rate (1 − APCER)", fontsize=11)
    ax.set_title("BBDM Leave-One-Out ROC Curves", fontsize=13)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, output_dir / "loo_roc_curves")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 4: Per-attack score distributions
# ---------------------------------------------------------------------------

def fig_score_distributions(
    test_df: pd.DataFrame,
    bbdm_detailed_df: pd.DataFrame,
    output_dir: Path,
    scoring_method: str = "lpips_score",
) -> None:
    """2×4 grid of KDE plots: bonafide vs each attack type."""
    from scipy.stats import gaussian_kde

    bonafide_scores = test_df[test_df["label"] == "bonafide"][scoring_method].values
    attack_df = test_df[test_df["label"] == "attack"]
    attack_types = sorted(attack_df["attack_type"].unique())
    det_df = bbdm_detailed_df.set_index("Attack_Type")

    n_cols = 4
    n_rows = int(np.ceil(len(attack_types) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.2, n_rows * 3.2))
    axes = axes.flatten()

    for i, at in enumerate(attack_types):
        ax = axes[i]
        attack_scores = attack_df.loc[attack_df["attack_type"] == at, scoring_method].values

        # Plot KDEs
        x_min = min(bonafide_scores.min(), attack_scores.min()) * 0.9
        x_max = max(bonafide_scores.max(), attack_scores.max()) * 1.1
        xs = np.linspace(x_min, x_max, 500)

        try:
            kde_bf = gaussian_kde(bonafide_scores, bw_method="silverman")
            ax.fill_between(xs, kde_bf(xs), alpha=0.35, color="#457b9d", label="Bona fide")
            ax.plot(xs, kde_bf(xs), color="#457b9d", lw=1.5)
        except Exception:
            pass

        try:
            kde_atk = gaussian_kde(attack_scores, bw_method="silverman")
            ax.fill_between(xs, kde_atk(xs), alpha=0.35, color="#e63946", label=at)
            ax.plot(xs, kde_atk(xs), color="#e63946", lw=1.5)
        except Exception:
            pass

        # Threshold line
        if at in det_df.index:
            thresh = det_df.loc[at, "Threshold"]
            acer = det_df.loc[at, "ACER"] * 100
            ax.axvline(thresh, color="black", lw=1.2, linestyle="--", alpha=0.8,
                       label=f"θ={thresh:.3f}")
            ax.set_title(f"{at}\n(ACER={acer:.2f}%)", fontsize=9)
        else:
            ax.set_title(at, fontsize=9)

        ax.set_xlabel(scoring_method, fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Hide unused axes
    for j in range(len(attack_types), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"BBDM PAD Score Distributions per Attack Type ({scoring_method})", fontsize=12, y=1.01)
    fig.tight_layout()
    _save(fig, output_dir / "loo_score_distributions")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 5: Overall comparison horizontal bar chart (all models + BBDM)
# ---------------------------------------------------------------------------

def fig_overall_comparison(
    bbdm_df: pd.DataFrame,
    sup_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Horizontal bar chart: all models sorted by overall ACER."""
    # Overall rows from supervised
    sup_overall = sup_df[sup_df["Attack_Type"].astype(str).str.strip() == ""]
    if sup_overall.empty:
        sup_overall = (
            sup_df.groupby("Model")["ACER"].mean()
            .reset_index()
            .rename(columns={"ACER": "ACER"})
        )
    else:
        sup_overall = sup_overall[["Model", "ACER"]].copy()

    # BBDM overall
    bbdm_overall = bbdm_df[bbdm_df["Attack_Type"].astype(str).str.strip() == ""][["Model", "ACER"]].copy()
    if bbdm_overall.empty:
        bbdm_overall = pd.DataFrame([{"Model": "BBDM (Ours)", "ACER": bbdm_df["ACER"].mean()}])

    all_df = pd.concat([sup_overall, bbdm_overall], ignore_index=True)
    all_df = all_df.drop_duplicates("Model").sort_values("ACER").reset_index(drop=True)
    all_df["ACER_pct"] = all_df["ACER"] * 100

    bbdm_name = bbdm_df["Model"].iloc[0]
    bbdm_rank = all_df[all_df["Model"] == bbdm_name].index[0]
    n = len(all_df)

    colors = [BBDM_COLOR if m == bbdm_name else "#b0b0b0" for m in all_df["Model"]]

    fig, ax = plt.subplots(figsize=(8, max(6, n * 0.35)))
    bars = ax.barh(range(n), all_df["ACER_pct"], color=colors, edgecolor="white", linewidth=0.3)
    ax.set_yticks(range(n))
    ax.set_yticklabels(all_df["Model"], fontsize=8)
    ax.set_xlabel("Overall ACER (%)", fontsize=11)
    ax.set_title("Overall ACER — All Models (sorted)", fontsize=12)

    # Label top-5 and bottom-5 bars + BBDM
    label_rows = set(range(min(5, n))) | set(range(max(0, n - 5), n)) | {bbdm_rank}
    for idx in label_rows:
        val = all_df["ACER_pct"].iloc[idx]
        ax.text(val + 0.3, idx, f"{val:.1f}%", va="center", fontsize=7.5,
                color=BBDM_COLOR if all_df["Model"].iloc[idx] == bbdm_name else "gray")

    # Vertical line at BBDM position
    bbdm_acer = all_df.loc[bbdm_rank, "ACER_pct"]
    ax.axvline(bbdm_acer, color=BBDM_COLOR, lw=1.2, linestyle="--", alpha=0.6)

    legend_patches = [
        mpatches.Patch(color=BBDM_COLOR, label=f"BBDM (ours) — rank {bbdm_rank + 1}/{n}"),
        mpatches.Patch(color="#b0b0b0", label="Supervised models"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, output_dir / "loo_overall_comparison")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 6: Radar / spider chart
# ---------------------------------------------------------------------------

def fig_radar_chart(
    bbdm_df: pd.DataFrame,
    sup_df: pd.DataFrame,
    output_dir: Path,
    n_sup_models: int = 2,
) -> None:
    """Spider chart: detection rate (1 − APCER) per attack type."""
    bbdm_per = _get_per_attack(bbdm_df).set_index("Attack_Type")
    sup_per = _get_per_attack(sup_df)
    top_models = _top_supervised_models(sup_df, n=n_sup_models)
    attack_types = sorted(bbdm_per.index.tolist())
    N = len(attack_types)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]   # close the polygon

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})

    # BBDM
    vals = [(1.0 - bbdm_per.loc[at, "APCER"]) if at in bbdm_per.index else 0.0
            for at in attack_types]
    vals += vals[:1]
    ax.plot(angles, vals, color=BBDM_COLOR, lw=2.2, label=f"BBDM (ours)")
    ax.fill(angles, vals, color=BBDM_COLOR, alpha=0.15)

    # Supervised models
    for i, model in enumerate(top_models):
        model_df = sup_per[sup_per["Model"] == model].set_index("Attack_Type")
        vals_sup = [(1.0 - model_df.loc[at, "APCER"]) if at in model_df.index else float("nan")
                    for at in attack_types]
        vals_sup += vals_sup[:1]
        c = SUP_COLORS[i % len(SUP_COLORS)]
        ax.plot(angles, vals_sup, color=c, lw=1.8, linestyle="--", label=model)
        ax.fill(angles, vals_sup, color=c, alpha=0.07)

    ax.set_thetagrids(np.degrees(angles[:-1]), attack_types, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], fontsize=7)
    ax.set_title("Detection Rate (1 − APCER) per Attack Type", fontsize=12, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, output_dir / "loo_radar_chart")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate leave-one-out comparison figures for BBDM Iris PAD",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--bbdm_detailed", type=Path,
                        default=Path("iris_bbdm_pad/results/leave_one_out/bbdm_open_set_detailed.csv"))
    parser.add_argument("--open_set_summary", type=Path,
                        default=Path("open_set_summary.csv"))
    parser.add_argument("--test_scores", type=Path,
                        default=Path("iris_bbdm_pad/results/test_pad_scores.csv"))
    parser.add_argument("--output_dir", type=Path,
                        default=Path("iris_bbdm_pad/results/leave_one_out/figures"))
    parser.add_argument("--scoring_method", type=str, default="lpips_score")
    args = parser.parse_args()

    # --- Load data ---
    logger.info(f"Loading BBDM detailed results: {args.bbdm_detailed}")
    bbdm_df = pd.read_csv(args.bbdm_detailed)
    bbdm_df["Attack_Type"] = bbdm_df["Attack_Type"].fillna("").astype(str).str.strip()

    if not args.open_set_summary.exists():
        logger.error(f"open_set_summary.csv not found: {args.open_set_summary}")
        sys.exit(1)
    logger.info(f"Loading supervised results: {args.open_set_summary}")
    sup_df = pd.read_csv(args.open_set_summary)
    # Normalise Attack_Type NaN → ""
    sup_df["Attack_Type"] = sup_df["Attack_Type"].fillna("").astype(str).str.strip()

    logger.info(f"Loading test scores: {args.test_scores}")
    test_df = pd.read_csv(args.test_scores)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Generate figures ---
    logger.info("[1/6] Fig 1: Per-attack ACER comparison bar chart")
    fig_acer_comparison(bbdm_df, sup_df, args.output_dir, n_sup_models=3)

    logger.info("[2/6] Fig 2: ACER heatmap")
    fig_acer_heatmap(bbdm_df, sup_df, args.output_dir, n_sup_models=10)

    logger.info("[3/6] Fig 3: ROC curves")
    fig_roc_curves(test_df, bbdm_df, args.output_dir, args.scoring_method)

    logger.info("[4/6] Fig 4: Score distributions")
    fig_score_distributions(test_df, bbdm_df, args.output_dir, args.scoring_method)

    logger.info("[5/6] Fig 5: Overall comparison bar chart")
    fig_overall_comparison(bbdm_df, sup_df, args.output_dir)

    logger.info("[6/6] Fig 6: Radar chart")
    fig_radar_chart(bbdm_df, sup_df, args.output_dir, n_sup_models=2)

    logger.info(f"[Done] All figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
