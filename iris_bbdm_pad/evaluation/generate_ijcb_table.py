"""
Generate IJCB 2026 comparison tables for the BBDM Iris PAD paper.

Produces LaTeX (booktabs) and Markdown tables combining:
- Shubham's supervised model results (open_set_summary.csv)
- Our BBDM unsupervised results (bbdm_open_set_summary.csv)

Tables generated:
  1. Per-attack ACER (%) — ijcb_comparison_acer.tex / .md
  2. Per-attack APCER (%) — ijcb_comparison_apcer.tex / .md

Usage:
    python iris_bbdm_pad/evaluation/generate_ijcb_table.py \\
        --bbdm_summary iris_bbdm_pad/results/leave_one_out/bbdm_open_set_summary.csv \\
        --open_set_summary open_set_summary.csv \\
        --output_dir iris_bbdm_pad/results/leave_one_out/tables/
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Short column headers for the paper (attack type → abbreviation)
ATTACK_ABBREVS = {
    "Artifact":           "Art.",
    "CL":                 "CL",
    "E-display":          "E-disp.",
    "Fake with Add On":   "Fake+",
    "Generated":          "Gen.",
    "PostMortem":         "P-M.",
    "Post-Mortem":        "P-M.",
    "Print and E-display": "P+E",
    "Printed":            "Print",
}


def _abbrev(at: str) -> str:
    return ATTACK_ABBREVS.get(at.strip(), at.strip()[:6])


def build_pivot(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Build a model × attack_type pivot table for the given metric.

    Args:
        df:     Combined DataFrame with Model, Attack_Type, and <metric>.
        metric: Column name ('ACER' or 'APCER').

    Returns:
        Pivot table; rows=models, columns=attack types + 'Avg'.
        Values are percentages (0–100).
    """
    per_attack = df[df["Attack_Type"].astype(str).str.strip() != ""].copy()
    per_attack[metric] = per_attack[metric] * 100   # convert to %

    pivot = per_attack.pivot_table(
        index="Model", columns="Attack_Type", values=metric, aggfunc="mean"
    )
    pivot.columns.name = None
    pivot.index.name = None

    # Compute average across available attack types per model
    pivot["Avg."] = pivot.mean(axis=1)

    # Sort: supervised models by Avg., BBDM last
    bbdm_mask = pivot.index.str.startswith("BBDM")
    sup_pivot = pivot[~bbdm_mask].sort_values("Avg.")
    bbdm_pivot = pivot[bbdm_mask]
    pivot = pd.concat([sup_pivot, bbdm_pivot])

    return pivot


def _format_val(val: float, bold: bool = False) -> str:
    """Format a metric value for LaTeX; NaN → '—'."""
    if np.isnan(val):
        return "—"
    s = f"{val:.1f}"
    return r"\textbf{" + s + "}" if bold else s


def _latex_table(
    pivot: pd.DataFrame,
    metric: str,
    caption: str,
    label: str,
) -> str:
    """Render pivot as a LaTeX booktabs table.

    Args:
        pivot:   Model × attack_type (% values).
        metric:  'ACER' or 'APCER' (for caption context).
        caption: Table caption string.
        label:   LaTeX \\label{} identifier.

    Returns:
        LaTeX table as a string.
    """
    attack_cols = [c for c in pivot.columns if c != "Avg."]
    all_cols = attack_cols + ["Avg."]
    abbrev_headers = [_abbrev(c) for c in attack_cols] + ["\\textbf{Avg.}"]

    n_cols = len(all_cols)
    col_spec = "l|" + "c" * n_cols

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        "Model & " + " & ".join(abbrev_headers) + r" \\",
        r"\midrule",
    ]

    bbdm_rows = [m for m in pivot.index if str(m).startswith("BBDM")]

    for model in pivot.index:
        if model == bbdm_rows[0] if bbdm_rows else None:
            lines.append(r"\midrule")
        is_bbdm = str(model).startswith("BBDM")

        row_vals = []
        for col in all_cols:
            val = pivot.loc[model, col] if col in pivot.columns else float("nan")
            # Bold if BBDM beats best supervised in that column
            if is_bbdm and not np.isnan(val) and col in pivot.columns:
                sup_vals = pivot.loc[~pivot.index.str.startswith("BBDM"), col].dropna()
                bold = bool(len(sup_vals) > 0 and val < sup_vals.min())
            else:
                bold = False
            row_vals.append(_format_val(val, bold=bold))

        model_cell = r"\textbf{" + str(model) + "}" if is_bbdm else str(model)
        lines.append(model_cell + " & " + " & ".join(row_vals) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def _markdown_table(pivot: pd.DataFrame, metric: str) -> str:
    """Render pivot as a GitHub-flavoured Markdown table.

    Args:
        pivot:  Model × attack_type (% values).
        metric: Metric name for header context.

    Returns:
        Markdown table string.
    """
    attack_cols = [c for c in pivot.columns if c != "Avg."]
    all_cols = attack_cols + ["Avg."]
    abbrevs = [_abbrev(c) for c in attack_cols] + ["**Avg.**"]

    header = "| Model | " + " | ".join(abbrevs) + " |"
    sep = "| --- | " + " | ".join(["---"] * len(all_cols)) + " |"

    rows = [header, sep]
    for model in pivot.index:
        is_bbdm = str(model).startswith("BBDM")
        model_cell = f"**{model}**" if is_bbdm else str(model)
        vals = []
        for col in all_cols:
            val = pivot.loc[model, col] if col in pivot.columns else float("nan")
            if np.isnan(val):
                vals.append("—")
            else:
                s = f"{val:.1f}"
                vals.append(f"**{s}**" if is_bbdm else s)
        rows.append("| " + model_cell + " | " + " | ".join(vals) + " |")

    return "\n".join(rows)


def generate_tables(
    combined_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Generate ACER and APCER tables in LaTeX and Markdown.

    Args:
        combined_df: DataFrame with Model, Attack_Type, ACER, APCER columns.
        output_dir:  Where to write .tex and .md files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric, short in [("ACER", "acer"), ("APCER", "apcer")]:
        if metric not in combined_df.columns:
            logger.warning(f"Column {metric} not found — skipping.")
            continue

        pivot = build_pivot(combined_df, metric)

        # LaTeX
        if metric == "ACER":
            caption = (
                r"Leave-One-Out " + metric + r" (\%) comparison. "
                r"Supervised models train on 7/8 attack types and test on the held-out type. "
                r"\textbf{BBDM (ours)} trains on 0 attack types (bona fide images only). "
                r"Lower is better. Bold = BBDM outperforms the best supervised model."
            )
        else:
            caption = (
                r"Per-attack " + metric + r" (\%). "
                r"APCER measures the fraction of attack samples incorrectly classified as bona fide. "
                r"Lower is better. Bold = BBDM outperforms the best supervised model."
            )

        label = f"tab:loo_{short}"
        tex = _latex_table(pivot, metric, caption, label)
        tex_path = output_dir / f"ijcb_comparison_{short}.tex"
        tex_path.write_text(tex, encoding="utf-8")
        logger.info(f"  Saved → {tex_path}")

        # Markdown
        md_header = f"## Per-Attack {metric} (%) — Leave-One-Out Evaluation\n\n"
        md_header += (
            f"> Supervised models: trained on 7/8 attack types, tested on held-out type.  \n"
            f"> **BBDM (ours)**: trained on **0 attack types** (bona fide only).  \n"
            f"> Lower {metric} = better detection.\n\n"
        )
        md = md_header + _markdown_table(pivot, metric)
        md_path = output_dir / f"ijcb_comparison_{short}.md"
        md_path.write_text(md, encoding="utf-8")
        logger.info(f"  Saved → {md_path}")

    logger.info(f"[Done] Tables written to {output_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate IJCB LaTeX + Markdown comparison tables",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--bbdm_summary",
        type=Path,
        default=Path("iris_bbdm_pad/results/leave_one_out/bbdm_open_set_summary.csv"),
        help="BBDM summary CSV (matches open_set_summary.csv format)",
    )
    parser.add_argument(
        "--open_set_summary",
        type=Path,
        default=Path("open_set_summary.csv"),
        help="Shubham's supervised model results CSV",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("iris_bbdm_pad/results/leave_one_out/tables"),
        help="Directory to write .tex and .md files",
    )
    args = parser.parse_args()

    logger.info(f"Loading BBDM summary: {args.bbdm_summary}")
    bbdm_df = pd.read_csv(args.bbdm_summary)
    bbdm_df["Attack_Type"] = bbdm_df["Attack_Type"].fillna("").astype(str)

    if not args.open_set_summary.exists():
        logger.warning(f"open_set_summary.csv not found — generating BBDM-only tables.")
        combined_df = bbdm_df
    else:
        logger.info(f"Loading supervised summary: {args.open_set_summary}")
        sup_df = pd.read_csv(args.open_set_summary)
        sup_df["Attack_Type"] = sup_df["Attack_Type"].fillna("").astype(str)
        combined_df = pd.concat([sup_df, bbdm_df], ignore_index=True)

    generate_tables(combined_df, args.output_dir)


if __name__ == "__main__":
    main()
