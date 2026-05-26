"""
Generate publication-ready LaTeX and markdown tables for the paper.

Usage:
    python iris_bbdm_pad/evaluation/generate_results_tables.py \
        --metrics_json iris_bbdm_pad/results/phase3_evaluation/metrics_summary.json \
        --per_attack_csv iris_bbdm_pad/results/phase3_evaluation/per_attack_metrics.csv \
        --ablation_csv iris_bbdm_pad/results/phase3_evaluation/ablation/ablation_denoising_steps.csv \
        --all_methods_csv iris_bbdm_pad/results/phase3_evaluation/all_methods_comparison.csv \
        --output_dir iris_bbdm_pad/results/phase3_evaluation/tables/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Hardcoded LivDet-Iris 2025 data (same as comparison_tables.py)
# ---------------------------------------------------------------------------

LIVDET_2025_TASK1: List[Dict[str, Any]] = [
    {
        "team": "Dermalog-Iris",
        "method": "Patch-based 3-network fusion",
        "type": "Supervised",
        "auroc": 0.9057,
        "apcer": 0.1069,
        "bpcer": 0.2826,
    },
    {
        "team": "MSU D-NetPAD",
        "method": "DenseNet121",
        "type": "Supervised",
        "auroc": 0.9014,
        "apcer": 0.0644,
        "bpcer": 0.4033,
    },
    {
        "team": "BUCEA",
        "method": "Attention multi-level fusion",
        "type": "Supervised",
        "auroc": 0.8984,
        "apcer": 0.1129,
        "bpcer": 0.3294,
    },
    {
        "team": "Baseline ViT+Synth",
        "method": "ViT + GAN synthetic data",
        "type": "Supervised",
        "auroc": 0.8648,
        "apcer": 0.2947,
        "bpcer": 0.0262,
    },
    {
        "team": "Baseline ViT",
        "method": "ViT vanilla",
        "type": "Supervised",
        "auroc": 0.8450,
        "apcer": 0.2894,
        "bpcer": 0.0245,
    },
    {
        "team": "Baseline ResNet",
        "method": "ResNet101",
        "type": "Supervised",
        "auroc": 0.8242,
        "apcer": 0.3003,
        "bpcer": 0.0224,
    },
    {
        "team": "MSU SPAD",
        "method": "DenseNet121",
        "type": "Supervised",
        "auroc": 0.8043,
        "apcer": 0.2538,
        "bpcer": 0.2762,
    },
    {
        "team": "EF 004",
        "method": "Dynamic ensemble",
        "type": "Supervised",
        "auroc": 0.6827,
        "apcer": 0.1662,
        "bpcer": 0.8706,
    },
    {
        "team": "HDA",
        "method": "CLIP + LoRA",
        "type": "Supervised",
        "auroc": 0.5692,
        "apcer": 0.8829,
        "bpcer": 0.0474,
    },
    {
        "team": "IIT Mandi (Ours)",
        "method": "BBDM Anomaly Detection",
        "type": "Unsupervised",
        "auroc": None,
        "apcer": 0.3304,
        "bpcer": 0.1993,
        "note": "Zero attack supervision",
    },
]

# Fallback BBDM values when metrics_summary.json is absent
_FALLBACK_METRICS: Dict[str, Any] = {
    "acer": 0.2649,
    "apcer": 0.3304,
    "bpcer": 0.1993,
    "eer": None,
    "auc": None,
    "accuracy": None,
    "precision": None,
    "recall": None,
    "f1": None,
    "tdr_at_fdr_0001": None,
    "tdr_at_fdr_001": None,
    "tdr_at_fdr_005": None,
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def pct(value: Optional[float], decimals: int = 2) -> str:
    """Format float (0-1 range) as percentage string."""
    if value is None:
        return "N/A"
    return f"{value * 100:.{decimals}f}"


def fmt_val(value: Optional[float], decimals: int = 4) -> str:
    """Format a float or return N/A."""
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}"


def tex_pct(value: Optional[float], decimals: int = 2) -> str:
    """LaTeX-safe percentage string."""
    if value is None:
        return "N/A"
    return f"{value * 100:.{decimals}f}\\%"


def tex_escape(text: str) -> str:
    """Escape special LaTeX characters in plain text."""
    replacements = [
        ("&", "\\&"),
        ("%", "\\%"),
        ("$", "\\$"),
        ("#", "\\#"),
        ("_", "\\_"),
        ("{", "\\{"),
        ("}", "\\}"),
        ("~", "\\textasciitilde{}"),
        ("^", "\\textasciicircum{}"),
    ]
    for orig, repl in replacements:
        text = text.replace(orig, repl)
    return text


def save_file(path: Path, content: str) -> None:
    """Write string content to file, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    logger.info("Saved: %s", path)


# ---------------------------------------------------------------------------
# Markdown table builder
# ---------------------------------------------------------------------------


def md_table(headers: List[str], rows: List[List[str]]) -> str:
    """Return a markdown table as a string."""
    sep = ["-" * max(len(h), 3) for h in headers]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LaTeX table builder
# ---------------------------------------------------------------------------


def latex_table(
    caption: str,
    label: str,
    headers: List[str],
    rows: List[List[str]],
    col_fmt: Optional[str] = None,
    note: Optional[str] = None,
) -> str:
    """Return a complete LaTeX booktabs table as a string."""
    n_cols = len(headers)
    if col_fmt is None:
        col_fmt = "l" + "r" * (n_cols - 1)

    tex_headers = [tex_escape(h) for h in headers]
    header_line = " & ".join(tex_headers) + " \\\\"

    body_lines: List[str] = []
    for row in rows:
        escaped = [tex_escape(str(c)) for c in row]
        body_lines.append("    " + " & ".join(escaped) + " \\\\")

    note_block = ""
    if note:
        note_block = (
            f"\n    \\multicolumn{{{n_cols}}}{{l}}"
            f"{{\\small\\textit{{{tex_escape(note)}}}}}\\\\"
        )

    lines = [
        "\\begin{table}[htbp]",
        "  \\centering",
        f"  \\caption{{{tex_escape(caption)}}}",
        f"  \\label{{{label}}}",
        f"  \\begin{{tabular}}{{{col_fmt}}}",
        "    \\toprule",
        f"    {header_line}",
        "    \\midrule",
    ]
    lines += body_lines
    lines += [
        "    \\bottomrule" + note_block,
        "  \\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table: main results
# ---------------------------------------------------------------------------


def table_main_results(
    metrics: Dict[str, Any], output_dir: Path
) -> None:
    """Generate table_main_results.md and .tex."""

    def get(key: str) -> Optional[float]:
        v = metrics.get(key)
        return float(v) if v is not None else None

    rows_data: List[Tuple[str, Optional[float], bool]] = [
        # (label, value_0to1, is_percentage)
        ("ACER", get("acer"), True),
        ("APCER", get("apcer"), True),
        ("BPCER", get("bpcer"), True),
        ("EER", get("eer"), True),
        ("AUC", get("auc"), False),
        ("Accuracy", get("accuracy"), True),
        ("Precision", get("precision"), True),
        ("Recall", get("recall"), True),
        ("F1 Score", get("f1"), False),
        ("TDR@FDR=0.1\\%", get("tdr_at_fdr_0001"), True),
        ("TDR@FDR=1.0\\%", get("tdr_at_fdr_001"), True),
        ("TDR@FDR=5.0\\%", get("tdr_at_fdr_005"), True),
    ]

    # Markdown (no LaTeX escaping)
    md_rows: List[List[str]] = []
    for label, val, is_pct in rows_data:
        label_clean = label.replace("\\%", "%").replace("\\", "")
        md_rows.append([label_clean, pct(val) + "%" if is_pct else fmt_val(val, 4)])

    md_content = "# Main Results\n\n" + md_table(["Metric", "Value"], md_rows)
    save_file(output_dir / "table_main_results.md", md_content)

    # LaTeX
    tex_rows: List[List[str]] = []
    for label, val, is_pct in rows_data:
        value_str = tex_pct(val) if is_pct else fmt_val(val, 4)
        tex_rows.append([label, value_str])

    tex_content = latex_table(
        caption="Main PAD Results — BBDM Anomaly Detection (lpips\\_score method)",
        label="tab:main_results",
        headers=["Metric", "Value"],
        rows=tex_rows,
        col_fmt="lr",
    )
    save_file(output_dir / "table_main_results.tex", tex_content)
    logger.info("[table_main_results] Written.")


# ---------------------------------------------------------------------------
# Table: per-attack results
# ---------------------------------------------------------------------------


def table_per_attack(
    per_attack_csv: Path, output_dir: Path
) -> None:
    """Generate table_per_attack.md and .tex."""
    headers = ["Attack Type", "N", "APCER%", "Detection Rate%", "Mean Score", "Std"]

    if not per_attack_csv.exists():
        logger.warning(
            "per_attack_csv not found: %s — generating placeholder.", per_attack_csv
        )
        placeholder_rows = [["(data not available)", "—", "—", "—", "—", "—"]]
        md_content = "# Per-Attack Results\n\n*(File not found — run evaluation first.)*\n\n"
        md_content += md_table(headers, placeholder_rows)
        save_file(output_dir / "table_per_attack.md", md_content)

        tex_content = latex_table(
            caption="Per-Attack Type Results",
            label="tab:per_attack",
            headers=headers,
            rows=placeholder_rows,
            col_fmt="lrrrrr",
            note="Data not available — run evaluation pipeline first.",
        )
        save_file(output_dir / "table_per_attack.tex", tex_content)
        return

    df = pd.read_csv(per_attack_csv)

    # Detect column names flexibly
    col_map: Dict[str, str] = {}
    for target, candidates in [
        ("attack", ["Attack_Type", "attack_type", "attack", "Attack"]),
        ("n", ["N", "n", "count", "Count", "num_samples"]),
        ("apcer", ["APCER", "apcer", "apcer_mean"]),
        ("det_rate", ["Detection_Rate", "detection_rate", "DR", "dr", "TPR", "tpr"]),
        ("mean_score", ["Mean_Score", "mean_score", "score_mean", "Score_Mean"]),
        ("std", ["Std", "std", "score_std", "Score_Std"]),
    ]:
        for c in candidates:
            if c in df.columns:
                col_map[target] = c
                break

    required = ["attack", "apcer"]
    missing = [k for k in required if k not in col_map]
    if missing:
        logger.warning(
            "Missing required columns %s in %s. Columns found: %s",
            missing,
            per_attack_csv,
            list(df.columns),
        )
        placeholder_rows = [["(column mismatch)", "—", "—", "—", "—", "—"]]
        md_content = "# Per-Attack Results\n\n*(Column mismatch — check CSV format.)*\n\n"
        md_content += md_table(headers, placeholder_rows)
        save_file(output_dir / "table_per_attack.md", md_content)
        save_file(
            output_dir / "table_per_attack.tex",
            latex_table(
                "Per-Attack Type Results",
                "tab:per_attack",
                headers,
                placeholder_rows,
                "lrrrrr",
            ),
        )
        return

    # Sort by APCER descending (hardest first)
    df = df.sort_values(col_map["apcer"], ascending=False)

    md_rows: List[List[str]] = []
    tex_rows: List[List[str]] = []

    for _, row in df.iterrows():
        attack = str(row[col_map["attack"]])
        n_val = str(int(row[col_map["n"]])) if "n" in col_map else "N/A"
        apcer_val = float(row[col_map["apcer"]])
        det_rate = (
            pct(1.0 - apcer_val) if "det_rate" not in col_map
            else pct(float(row[col_map["det_rate"]]))
        )
        mean_s = (
            fmt_val(float(row[col_map["mean_score"]]), 4)
            if "mean_score" in col_map
            else "N/A"
        )
        std_s = (
            fmt_val(float(row[col_map["std"]]), 4) if "std" in col_map else "N/A"
        )

        md_rows.append([attack, n_val, pct(apcer_val), det_rate, mean_s, std_s])
        tex_rows.append(
            [
                tex_escape(attack),
                n_val,
                tex_pct(apcer_val),
                tex_pct(1.0 - apcer_val) if "det_rate" not in col_map else tex_pct(float(row[col_map["det_rate"]])),
                mean_s,
                std_s,
            ]
        )

    md_content = (
        "# Per-Attack Results\n\nSorted by APCER descending (hardest attacks first).\n\n"
        + md_table(headers, md_rows)
    )
    save_file(output_dir / "table_per_attack.md", md_content)

    tex_content = latex_table(
        caption="Per-Attack Type PAD Results (sorted by APCER, hardest first)",
        label="tab:per_attack",
        headers=headers,
        rows=tex_rows,
        col_fmt="lrrrrr",
    )
    save_file(output_dir / "table_per_attack.tex", tex_content)
    logger.info("[table_per_attack] Written.")


# ---------------------------------------------------------------------------
# Table: ablation study
# ---------------------------------------------------------------------------


def table_ablation(
    ablation_csv: Path, output_dir: Path
) -> None:
    """Generate table_ablation.md and .tex."""
    headers = ["Steps", "ACER%", "APCER%", "BPCER%", "EER%", "AUC", "Time (ms)"]

    if not ablation_csv.exists():
        logger.warning(
            "ablation_csv not found: %s — generating placeholder.", ablation_csv
        )
        placeholder_rows = [
            ["10", "—", "—", "—", "—", "—", "—"],
            ["25", "—", "—", "—", "—", "—", "—"],
            ["50", "—", "—", "—", "—", "—", "—"],
            ["100", "—", "—", "—", "—", "—", "—"],
        ]
        md_content = (
            "# Ablation Study: Denoising Steps\n\n"
            "*(ablation_denoising_steps.csv not found — run ablation first.)*\n\n"
            + md_table(headers, placeholder_rows)
        )
        save_file(output_dir / "table_ablation.md", md_content)

        tex_content = latex_table(
            caption="Ablation Study: Effect of Number of Denoising Steps",
            label="tab:ablation",
            headers=headers,
            rows=placeholder_rows,
            col_fmt="lrrrrrr",
            note="Placeholder — run ablation evaluation to populate.",
        )
        save_file(output_dir / "table_ablation.tex", tex_content)
        return

    df = pd.read_csv(ablation_csv)

    # Detect columns
    col_map: Dict[str, str] = {}
    for target, candidates in [
        ("steps", ["Steps", "steps", "denoising_steps", "n_steps"]),
        ("acer", ["ACER", "acer"]),
        ("apcer", ["APCER", "apcer"]),
        ("bpcer", ["BPCER", "bpcer"]),
        ("eer", ["EER", "eer"]),
        ("auc", ["AUC", "auc", "auroc"]),
        ("time_ms", ["Time_ms", "time_ms", "inference_time_ms", "Time"]),
    ]:
        for c in candidates:
            if c in df.columns:
                col_map[target] = c
                break

    # Find best row by ACER
    best_idx: Optional[int] = None
    if "acer" in col_map:
        best_idx = int(df[col_map["acer"]].idxmin())

    def get_cell(row: Any, key: str, is_pct: bool = True) -> str:
        if key not in col_map:
            return "N/A"
        v = row[col_map[key]]
        if pd.isna(v):
            return "N/A"
        return pct(float(v)) if is_pct else fmt_val(float(v), 4)

    md_rows: List[List[str]] = []
    tex_rows: List[List[str]] = []

    for idx, row in df.iterrows():
        steps = str(int(row[col_map["steps"]])) if "steps" in col_map else str(idx)
        acer_s = get_cell(row, "acer")
        apcer_s = get_cell(row, "apcer")
        bpcer_s = get_cell(row, "bpcer")
        eer_s = get_cell(row, "eer")
        auc_s = get_cell(row, "auc", is_pct=False)
        time_s = get_cell(row, "time_ms", is_pct=False)

        is_best = best_idx is not None and idx == best_idx

        # Markdown: bold the best row
        if is_best:
            md_rows.append(
                [
                    f"**{steps}**",
                    f"**{acer_s}**",
                    f"**{apcer_s}**",
                    f"**{bpcer_s}**",
                    f"**{eer_s}**",
                    f"**{auc_s}**",
                    f"**{time_s}**",
                ]
            )
        else:
            md_rows.append([steps, acer_s, apcer_s, bpcer_s, eer_s, auc_s, time_s])

        # LaTeX: bold best with \textbf{}
        if is_best:
            tex_rows.append(
                [
                    f"\\textbf{{{steps}}}",
                    f"\\textbf{{{acer_s}\\%}}",
                    f"\\textbf{{{apcer_s}\\%}}",
                    f"\\textbf{{{bpcer_s}\\%}}",
                    f"\\textbf{{{eer_s}\\%}}",
                    f"\\textbf{{{auc_s}}}",
                    f"\\textbf{{{time_s}}}",
                ]
            )
        else:
            tex_rows.append(
                [
                    steps,
                    f"{acer_s}\\%",
                    f"{apcer_s}\\%",
                    f"{bpcer_s}\\%",
                    f"{eer_s}\\%",
                    auc_s,
                    time_s,
                ]
            )

    md_content = (
        "# Ablation Study: Denoising Steps\n\n"
        "Bold row indicates best ACER.\n\n"
        + md_table(headers, md_rows)
    )
    save_file(output_dir / "table_ablation.md", md_content)

    tex_content = latex_table(
        caption="Ablation Study: Effect of Number of Denoising Steps on PAD Performance",
        label="tab:ablation",
        headers=headers,
        rows=tex_rows,
        col_fmt="lrrrrrr",
        note="Bold indicates best ACER.",
    )
    save_file(output_dir / "table_ablation.tex", tex_content)
    logger.info("[table_ablation] Written.")


# ---------------------------------------------------------------------------
# Table: LivDet-Iris 2025 comparison
# ---------------------------------------------------------------------------


def table_livdet_comparison(output_dir: Path) -> None:
    """Generate table_livdet_comparison.md and .tex."""
    headers = ["Team", "Method", "Supervision", "AUROC", "APCER%", "BPCER%"]

    md_rows: List[List[str]] = []
    tex_rows: List[List[str]] = []

    for entry in LIVDET_2025_TASK1:
        is_ours = "Ours" in entry["team"]
        team_str = entry["team"]
        auroc_str = fmt_val(entry["auroc"], 4)
        apcer_str = pct(entry["apcer"])
        bpcer_str = pct(entry["bpcer"])

        if is_ours:
            md_rows.append(
                [
                    f"**{team_str}**",
                    f"**{entry['method']}**",
                    f"**{entry['type']}**",
                    f"**{auroc_str}**",
                    f"**{apcer_str}**",
                    f"**{bpcer_str}**",
                ]
            )
            tex_rows.append(
                [
                    f"\\textbf{{{tex_escape(team_str)}}}",
                    f"\\textbf{{{tex_escape(entry['method'])}}}",
                    f"\\textbf{{{tex_escape(entry['type'])}}}",
                    f"\\textbf{{{auroc_str}}}",
                    f"\\textbf{{{apcer_str}\\%}}",
                    f"\\textbf{{{bpcer_str}\\%}}",
                ]
            )
        else:
            md_rows.append(
                [team_str, entry["method"], entry["type"], auroc_str, apcer_str, bpcer_str]
            )
            tex_rows.append(
                [
                    tex_escape(team_str),
                    tex_escape(entry["method"]),
                    tex_escape(entry["type"]),
                    auroc_str,
                    f"{apcer_str}\\%",
                    f"{bpcer_str}\\%",
                ]
            )

    md_content = (
        "# LivDet-Iris 2025 Task 1 Comparison\n\n"
        "Our method is in **bold**. AUROC=N/A because BBDM threshold was set on val set only.\n\n"
        + md_table(headers, md_rows)
    )
    save_file(output_dir / "table_livdet_comparison.md", md_content)

    tex_content = latex_table(
        caption=(
            "Comparison with LivDet-Iris 2025 Task 1 participants. "
            "Our method (bold) uses no attack samples during training."
        ),
        label="tab:livdet2025",
        headers=headers,
        rows=tex_rows,
        col_fmt="llllrr",
    )
    save_file(output_dir / "table_livdet_comparison.tex", tex_content)
    logger.info("[table_livdet_comparison] Written.")


# ---------------------------------------------------------------------------
# Table: open-set supervised baselines comparison
# ---------------------------------------------------------------------------


def table_open_set_comparison(
    open_set_csv: Path, output_dir: Path
) -> None:
    """Generate table_open_set_comparison.md and .tex."""
    headers = ["Model", "Type", "ACER%", "APCER%", "BPCER%"]

    if not open_set_csv.exists():
        logger.warning(
            "open_set_csv not found: %s — generating placeholder.", open_set_csv
        )
        placeholder_rows = [["(file not found)", "—", "—", "—", "—"]]
        md_content = (
            "# Open-Set Supervised Baselines Comparison\n\n"
            "*(open_set_summary.csv not found.)*\n\n"
            + md_table(headers, placeholder_rows)
        )
        save_file(output_dir / "table_open_set_comparison.md", md_content)
        save_file(
            output_dir / "table_open_set_comparison.tex",
            latex_table(
                "Comparison with Open-Set Supervised Baselines",
                "tab:open_set",
                headers,
                placeholder_rows,
                "llrrr",
            ),
        )
        return

    df = pd.read_csv(open_set_csv)
    overall = df[
        df["Attack_Type"].isna() | (df["Attack_Type"].astype(str).str.strip() == "")
    ].copy()
    overall = overall.sort_values("ACER", ascending=True)

    md_rows: List[List[str]] = []
    tex_rows: List[List[str]] = []

    for _, row in overall.iterrows():
        model_name = str(row["Model"])
        acer_pct = f"{float(row['ACER']) * 100:.2f}"
        apcer_pct = f"{float(row['APCER']) * 100:.2f}"
        bpcer_pct = f"{float(row['BPCER']) * 100:.2f}"

        md_rows.append([model_name, "Supervised Open-Set", acer_pct, apcer_pct, bpcer_pct])
        tex_rows.append(
            [
                tex_escape(model_name),
                "Supervised Open-Set",
                f"{acer_pct}\\%",
                f"{apcer_pct}\\%",
                f"{bpcer_pct}\\%",
            ]
        )

    # Add our BBDM row (bold)
    our_acer = f"{0.2649 * 100:.2f}"
    our_apcer = f"{0.3304 * 100:.2f}"
    our_bpcer = f"{0.1993 * 100:.2f}"

    md_rows.append(
        [
            "**BBDM (Ours)**",
            "**Unsupervised (BBDM)**",
            f"**{our_acer}**",
            f"**{our_apcer}**",
            f"**{our_bpcer}**",
        ]
    )
    tex_rows.append(
        [
            "\\textbf{BBDM (Ours)}",
            "\\textbf{Unsupervised (BBDM)}",
            f"\\textbf{{{our_acer}\\%}}",
            f"\\textbf{{{our_apcer}\\%}}",
            f"\\textbf{{{our_bpcer}\\%}}",
        ]
    )

    md_content = (
        "# Open-Set Supervised Baselines Comparison\n\n"
        "Supervised models trained on 7/8 attack types. "
        "BBDM trained on bona fide only (0 attack types). "
        "Our method in **bold**.\n\n"
        + md_table(headers, md_rows)
    )
    save_file(output_dir / "table_open_set_comparison.md", md_content)

    tex_content = latex_table(
        caption=(
            "Comparison with open-set supervised baselines on the same dataset. "
            "Supervised models trained on 7 of 8 attack types. "
            "BBDM (bold) uses zero attack supervision."
        ),
        label="tab:open_set",
        headers=headers,
        rows=tex_rows,
        col_fmt="llrrr",
    )
    save_file(output_dir / "table_open_set_comparison.tex", tex_content)
    logger.info("[table_open_set_comparison] Written.")


# ---------------------------------------------------------------------------
# Table: scoring methods comparison
# ---------------------------------------------------------------------------


def table_scoring_methods(
    all_methods_csv: Path, output_dir: Path
) -> None:
    """Generate table_scoring_methods.md and .tex."""
    headers = ["Method", "ACER%", "APCER%", "BPCER%", "EER%", "AUC"]

    if not all_methods_csv.exists():
        logger.warning(
            "all_methods_csv not found: %s — generating placeholder with known values.",
            all_methods_csv,
        )
        # Use the one known result
        placeholder_rows = [
            ["lpips_score *", "26.49", "33.04", "19.93", "N/A", "N/A"],
            ["mse_score", "—", "—", "—", "—", "—"],
            ["combined_score", "—", "—", "—", "—", "—"],
        ]
        md_content = (
            "# Scoring Method Comparison\n\n"
            "\\* marks best method (by ACER). "
            "*(all_methods_comparison.csv not found — run scoring comparison.)*\n\n"
            + md_table(headers, placeholder_rows)
        )
        save_file(output_dir / "table_scoring_methods.md", md_content)

        tex_placeholder = [
            ["lpips\\_score", "26.49\\%", "33.04\\%", "19.93\\%", "N/A", "N/A"],
            ["mse\\_score", "---", "---", "---", "---", "---"],
            ["combined\\_score", "---", "---", "---", "---", "---"],
        ]
        tex_content = latex_table(
            caption="Comparison of PAD Scoring Methods",
            label="tab:scoring_methods",
            headers=headers,
            rows=tex_placeholder,
            col_fmt="lrrrrr",
            note="Placeholder — run scoring comparison pipeline to populate.",
        )
        save_file(output_dir / "table_scoring_methods.tex", tex_content)
        return

    df = pd.read_csv(all_methods_csv)

    col_map: Dict[str, str] = {}
    for target, candidates in [
        ("method", ["Method", "method", "scoring_method"]),
        ("acer", ["ACER", "acer"]),
        ("apcer", ["APCER", "apcer"]),
        ("bpcer", ["BPCER", "bpcer"]),
        ("eer", ["EER", "eer"]),
        ("auc", ["AUC", "auc", "auroc"]),
    ]:
        for c in candidates:
            if c in df.columns:
                col_map[target] = c
                break

    # Find best row by ACER
    best_idx: Optional[int] = None
    if "acer" in col_map:
        best_idx = int(df[col_map["acer"]].idxmin())

    def safe_pct(row: Any, key: str) -> str:
        if key not in col_map:
            return "N/A"
        v = row[col_map[key]]
        return "N/A" if pd.isna(v) else pct(float(v))

    def safe_val(row: Any, key: str) -> str:
        if key not in col_map:
            return "N/A"
        v = row[col_map[key]]
        return "N/A" if pd.isna(v) else fmt_val(float(v), 4)

    md_rows: List[List[str]] = []
    tex_rows: List[List[str]] = []

    for idx, row in df.iterrows():
        method_name = (
            str(row[col_map["method"]]) if "method" in col_map else f"method_{idx}"
        )
        is_best = best_idx is not None and idx == best_idx

        acer_s = safe_pct(row, "acer")
        apcer_s = safe_pct(row, "apcer")
        bpcer_s = safe_pct(row, "bpcer")
        eer_s = safe_pct(row, "eer")
        auc_s = safe_val(row, "auc")

        # Markdown: mark best with *
        label_md = f"{method_name} *" if is_best else method_name
        md_rows.append([label_md, acer_s, apcer_s, bpcer_s, eer_s, auc_s])

        # LaTeX: bold best
        method_tex = tex_escape(method_name)
        if is_best:
            tex_rows.append(
                [
                    f"\\textbf{{{method_tex}}}",
                    f"\\textbf{{{acer_s}\\%}}",
                    f"\\textbf{{{apcer_s}\\%}}",
                    f"\\textbf{{{bpcer_s}\\%}}",
                    f"\\textbf{{{eer_s}\\%}}",
                    f"\\textbf{{{auc_s}}}",
                ]
            )
        else:
            tex_rows.append(
                [
                    method_tex,
                    f"{acer_s}\\%",
                    f"{apcer_s}\\%",
                    f"{bpcer_s}\\%",
                    f"{eer_s}\\%",
                    auc_s,
                ]
            )

    md_content = (
        "# Scoring Method Comparison\n\n"
        "* marks best method (lowest ACER).\n\n"
        + md_table(headers, md_rows)
    )
    save_file(output_dir / "table_scoring_methods.md", md_content)

    tex_content = latex_table(
        caption="Comparison of PAD Scoring Methods (bold = best ACER)",
        label="tab:scoring_methods",
        headers=headers,
        rows=tex_rows,
        col_fmt="lrrrrr",
    )
    save_file(output_dir / "table_scoring_methods.tex", tex_content)
    logger.info("[table_scoring_methods] Written.")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate publication-ready LaTeX and markdown tables.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--metrics_json",
        type=Path,
        default=Path("iris_bbdm_pad/results/phase3_evaluation/metrics_summary.json"),
        help="Path to metrics_summary.json (overall BBDM metrics).",
    )
    parser.add_argument(
        "--per_attack_csv",
        type=Path,
        default=Path("iris_bbdm_pad/results/phase3_evaluation/per_attack_metrics.csv"),
        help="Path to per_attack_metrics.csv.",
    )
    parser.add_argument(
        "--ablation_csv",
        type=Path,
        default=Path(
            "iris_bbdm_pad/results/phase3_evaluation/ablation/ablation_denoising_steps.csv"
        ),
        help="Path to ablation_denoising_steps.csv.",
    )
    parser.add_argument(
        "--all_methods_csv",
        type=Path,
        default=Path(
            "iris_bbdm_pad/results/phase3_evaluation/all_methods_comparison.csv"
        ),
        help="Path to all_methods_comparison.csv (scoring method comparison).",
    )
    parser.add_argument(
        "--open_set_csv",
        type=Path,
        default=Path("open_set_summary.csv"),
        help="Path to open_set_summary.csv (project root).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("iris_bbdm_pad/results/phase3_evaluation/tables"),
        help="Directory where all table files are saved.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load metrics JSON
    metrics = dict(_FALLBACK_METRICS)
    if args.metrics_json.exists():
        with args.metrics_json.open("r", encoding="utf-8") as fh:
            loaded = json.load(fh)
        metrics.update(loaded)
        logger.info("Loaded metrics from %s", args.metrics_json)
    else:
        logger.warning(
            "metrics_json not found: %s — using hardcoded fallback values.",
            args.metrics_json,
        )

    logger.info("Generating table_main_results …")
    table_main_results(metrics, args.output_dir)

    logger.info("Generating table_per_attack …")
    table_per_attack(args.per_attack_csv, args.output_dir)

    logger.info("Generating table_ablation …")
    table_ablation(args.ablation_csv, args.output_dir)

    logger.info("Generating table_livdet_comparison …")
    table_livdet_comparison(args.output_dir)

    logger.info("Generating table_open_set_comparison …")
    table_open_set_comparison(args.open_set_csv, args.output_dir)

    logger.info("Generating table_scoring_methods …")
    table_scoring_methods(args.all_methods_csv, args.output_dir)

    logger.info(
        "All tables written to %s  (*.md and *.tex for each table)", args.output_dir
    )


if __name__ == "__main__":
    main()
