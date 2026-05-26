"""
Generate comparison tables: our BBDM method vs LivDet-Iris 2025 competitors,
LivDet-Iris 2023 results, AND our own Open_Set supervised baselines (10 models).

Usage:
    python iris_bbdm_pad/evaluation/comparison_tables.py \
        --our_results iris_bbdm_pad/results/phase3_evaluation/metrics_summary.json \
        --our_per_attack iris_bbdm_pad/results/phase3_evaluation/cross_attack_detailed.csv \
        --open_set_csv open_set_summary.csv \
        --output_dir iris_bbdm_pad/results/phase3_evaluation/
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Hardcoded competition results
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

LIVDET_2023: List[Dict[str, Any]] = [
    {"team": "BUCEA Algo 1", "acer": 0.2215, "setting": "Open-set (unweighted)"},
    {"team": "Fraunhofer IGD", "acer": 0.3731, "setting": "Open-set (weighted)"},
    {
        "team": "IIT Mandi (Ours)",
        "acer": 0.2649,
        "setting": "Unsupervised anomaly detection",
    },
]

PARADIGM_ROWS: List[Dict[str, Any]] = [
    {
        "approach": "Closed-set Supervised CNN",
        "requires_attack_data": "Yes (all types)",
        "open_set_capable": "No",
        "acer_range": "0.2% – 2.3%",
        "notes": "Requires attack samples for all test types",
    },
    {
        "approach": "Open-set Supervised CNN",
        "requires_attack_data": "Yes (partial)",
        "open_set_capable": "Partial",
        "acer_range": "1.6% – 50.3%",
        "notes": "Degrades on unseen attack types",
    },
    {
        "approach": "BBDM Anomaly Detection (Ours)",
        "requires_attack_data": "None",
        "open_set_capable": "Yes",
        "acer_range": "26.5%",
        "notes": "First BBDM-based iris PAD; truly zero-shot on attacks",
    },
]

# Our BBDM overall metrics (lpips_score method)
OUR_ACER = 0.2649
OUR_APCER = 0.3304
OUR_BPCER = 0.1993

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def pct(value: float) -> str:
    """Format a 0-1 float as XX.XX%."""
    return f"{value * 100:.2f}"


def fmt_auroc(value: Optional[float]) -> str:
    """Format AUROC or return N/A."""
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def save_markdown(path: Path, lines: List[str]) -> None:
    """Write list of strings as a markdown file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Saved markdown: %s", path)


def save_csv_from_rows(path: Path, fieldnames: List[str], rows: List[Dict]) -> None:
    """Write list of dicts to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Saved CSV: %s", path)


def md_table(headers: List[str], rows: List[List[str]]) -> List[str]:
    """Return markdown table lines."""
    sep = ["-" * max(len(h), 3) for h in headers]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return lines


# ---------------------------------------------------------------------------
# Table 1: LivDet-Iris 2025 Task 1 comparison
# ---------------------------------------------------------------------------


def table1_livdet2025(output_dir: Path) -> None:
    """Generate Table 1: LivDet-Iris 2025 Task 1 comparison."""
    headers = ["Team", "Method", "Supervision", "AUROC", "APCER%", "BPCER%", "Notes"]
    md_rows: List[List[str]] = []
    csv_rows: List[Dict] = []

    for entry in LIVDET_2025_TASK1:
        is_ours = "Ours" in entry["team"]
        team_md = f"**{entry['team']}**" if is_ours else entry["team"]
        note = entry.get("note", "")
        acer_val = (entry["apcer"] + entry["bpcer"]) / 2

        md_rows.append(
            [
                team_md,
                entry["method"],
                entry["type"],
                fmt_auroc(entry["auroc"]),
                pct(entry["apcer"]),
                pct(entry["bpcer"]),
                note,
            ]
        )
        csv_rows.append(
            {
                "Team": entry["team"],
                "Method": entry["method"],
                "Supervision": entry["type"],
                "AUROC": fmt_auroc(entry["auroc"]),
                "APCER%": pct(entry["apcer"]),
                "BPCER%": pct(entry["bpcer"]),
                "ACER%": pct(acer_val),
                "Notes": note,
            }
        )

    md_lines = [
        "# Table 1: LivDet-Iris 2025 Task 1 Comparison",
        "",
        "Our method marked with **bold**. ACER = (APCER + BPCER) / 2.",
        "",
    ] + md_table(headers, md_rows)

    save_markdown(output_dir / "comparison_livdet2025.md", md_lines)
    save_csv_from_rows(
        output_dir / "comparison_livdet2025.csv",
        ["Team", "Method", "Supervision", "AUROC", "APCER%", "BPCER%", "ACER%", "Notes"],
        csv_rows,
    )
    logger.info("[Table 1] LivDet-2025 comparison written.")


# ---------------------------------------------------------------------------
# Table 2: LivDet-Iris 2023 open-set comparison
# ---------------------------------------------------------------------------


def table2_livdet2023(output_dir: Path) -> None:
    """Generate Table 2: LivDet-Iris 2023 open-set comparison."""
    headers = ["Method", "ACER%", "Setting"]
    md_rows: List[List[str]] = []
    csv_rows: List[Dict] = []

    for entry in LIVDET_2023:
        is_ours = "Ours" in entry["team"]
        team_md = f"**{entry['team']}**" if is_ours else entry["team"]
        md_rows.append([team_md, pct(entry["acer"]), entry["setting"]])
        csv_rows.append(
            {
                "Method": entry["team"],
                "ACER%": pct(entry["acer"]),
                "Setting": entry["setting"],
            }
        )

    md_lines = [
        "# Table 2: LivDet-Iris 2023 Open-Set Comparison",
        "",
    ] + md_table(headers, md_rows)

    save_markdown(output_dir / "comparison_livdet2023.md", md_lines)
    save_csv_from_rows(
        output_dir / "comparison_livdet2023.csv",
        ["Method", "ACER%", "Setting"],
        csv_rows,
    )
    logger.info("[Table 2] LivDet-2023 comparison written.")


# ---------------------------------------------------------------------------
# Table 3: Paradigm comparison
# ---------------------------------------------------------------------------


def table3_paradigm(output_dir: Path) -> None:
    """Generate Table 3: Paradigm comparison."""
    headers = [
        "Approach",
        "Requires Attack Data",
        "Open-Set Capable",
        "ACER Range",
        "Notes",
    ]
    md_rows: List[List[str]] = []
    csv_rows: List[Dict] = []

    for entry in PARADIGM_ROWS:
        is_ours = "Ours" in entry["approach"]
        approach_md = f"**{entry['approach']}**" if is_ours else entry["approach"]
        md_rows.append(
            [
                approach_md,
                entry["requires_attack_data"],
                entry["open_set_capable"],
                entry["acer_range"],
                entry["notes"],
            ]
        )
        csv_rows.append(
            {
                "Approach": entry["approach"],
                "Requires Attack Data": entry["requires_attack_data"],
                "Open-Set Capable": entry["open_set_capable"],
                "ACER Range": entry["acer_range"],
                "Notes": entry["notes"],
            }
        )

    md_lines = [
        "# Table 3: Paradigm Comparison",
        "",
    ] + md_table(headers, md_rows)

    save_markdown(output_dir / "paradigm_comparison.md", md_lines)
    save_csv_from_rows(
        output_dir / "paradigm_comparison.csv",
        ["Approach", "Requires Attack Data", "Open-Set Capable", "ACER Range", "Notes"],
        csv_rows,
    )
    logger.info("[Table 3] Paradigm comparison written.")


# ---------------------------------------------------------------------------
# Table 4: vs Open_Set supervised baselines
# ---------------------------------------------------------------------------


def table4_open_set(open_set_csv: Path, output_dir: Path) -> None:
    """Generate Table 4: vs Open_Set supervised baselines."""
    if not open_set_csv.exists():
        logger.warning("open_set_csv not found: %s — skipping Table 4.", open_set_csv)
        return

    df = pd.read_csv(open_set_csv)
    # Overall rows: empty Attack_Type
    overall = df[df["Attack_Type"].isna() | (df["Attack_Type"].astype(str).str.strip() == "")]
    overall = overall.copy()
    overall = overall.sort_values("ACER", ascending=True)

    headers = [
        "Model",
        "Type",
        "Training_Attack_Types",
        "Overall_ACER%",
        "APCER%",
        "BPCER%",
        "Accuracy%",
    ]
    md_rows: List[List[str]] = []
    csv_rows: List[Dict] = []

    for _, row in overall.iterrows():
        model_name = str(row["Model"])
        acer_pct = f"{float(row['ACER']) * 100:.2f}"
        apcer_pct = f"{float(row['APCER']) * 100:.2f}"
        bpcer_pct = f"{float(row['BPCER']) * 100:.2f}"
        acc_pct = f"{float(row['Accuracy']) * 100:.2f}"

        md_rows.append(
            [
                model_name,
                "Supervised Open-Set",
                "7/8 attack types",
                acer_pct,
                apcer_pct,
                bpcer_pct,
                acc_pct,
            ]
        )
        csv_rows.append(
            {
                "Model": model_name,
                "Type": "Supervised Open-Set",
                "Training_Attack_Types": "7/8 attack types",
                "Overall_ACER%": acer_pct,
                "APCER%": apcer_pct,
                "BPCER%": bpcer_pct,
                "Accuracy%": acc_pct,
            }
        )

    # Add our BBDM row
    our_acer_pct = pct(OUR_ACER)
    our_apcer_pct = pct(OUR_APCER)
    our_bpcer_pct = pct(OUR_BPCER)

    md_rows.append(
        [
            "**BBDM (Ours)**",
            "Unsupervised (BBDM)",
            "0/8 attack types",
            our_acer_pct,
            our_apcer_pct,
            our_bpcer_pct,
            "N/A",
        ]
    )
    csv_rows.append(
        {
            "Model": "BBDM (Ours)",
            "Type": "Unsupervised (BBDM)",
            "Training_Attack_Types": "0/8 attack types",
            "Overall_ACER%": our_acer_pct,
            "APCER%": our_apcer_pct,
            "BPCER%": our_bpcer_pct,
            "Accuracy%": "N/A",
        }
    )

    md_lines = [
        "# Table 4: Comparison vs Open-Set Supervised Baselines (Same Dataset)",
        "",
        "Supervised models trained on 7 of 8 attack types; tested on all 8 (open-set).",
        "Our BBDM trained on bona fide only — zero attack supervision.",
        "",
    ] + md_table(headers, md_rows)

    save_markdown(output_dir / "comparison_open_set.md", md_lines)
    save_csv_from_rows(output_dir / "comparison_open_set.csv", headers, csv_rows)
    logger.info("[Table 4] Open-set comparison written (%d supervised models).", len(overall))


# ---------------------------------------------------------------------------
# Table 5: Per-attack comparison vs supervised
# ---------------------------------------------------------------------------


def table5_per_attack(
    open_set_csv: Path, our_per_attack_csv: Path, output_dir: Path
) -> None:
    """Generate Table 5: per-attack APCER comparison."""
    if not open_set_csv.exists():
        logger.warning("open_set_csv not found: %s — skipping Table 5.", open_set_csv)
        return

    if not our_per_attack_csv.exists():
        logger.warning(
            "cross_attack_detailed.csv not found: %s — skipping Table 5.",
            our_per_attack_csv,
        )
        return

    sup_df = pd.read_csv(open_set_csv)
    our_df = pd.read_csv(our_per_attack_csv)

    # Per-attack rows from supervised (non-empty Attack_Type)
    sup_per = sup_df[
        sup_df["Attack_Type"].notna()
        & (sup_df["Attack_Type"].astype(str).str.strip() != "")
    ].copy()

    # Models we compare against
    compare_models = [
        "DenseNet121",
        "DenseNet121_LastBlock",
        "MobileNetV3LargeModel_LastBlock",
    ]

    # Determine attack type column in our data
    our_attack_col = None
    for candidate in ["Attack_Type", "attack_type", "attack", "Attack"]:
        if candidate in our_df.columns:
            our_attack_col = candidate
            break
    if our_attack_col is None:
        logger.warning(
            "Cannot find attack type column in %s. Columns: %s",
            our_per_attack_csv,
            list(our_df.columns),
        )
        return

    our_apcer_col = None
    for candidate in ["APCER", "apcer", "apcer_mean"]:
        if candidate in our_df.columns:
            our_apcer_col = candidate
            break
    if our_apcer_col is None:
        logger.warning(
            "Cannot find APCER column in %s. Columns: %s",
            our_per_attack_csv,
            list(our_df.columns),
        )
        return

    # Collect all attack types from our data
    our_attacks = our_df[our_attack_col].dropna().unique().tolist()

    headers = (
        ["Attack_Type", "BBDM_APCER%"]
        + [f"{m}_APCER%" for m in compare_models]
        + ["Note"]
    )
    md_rows: List[List[str]] = []
    csv_rows: List[Dict] = []

    for attack in sorted(our_attacks):
        our_row = our_df[our_df[our_attack_col] == attack]
        our_apcer_val = (
            f"{float(our_row[our_apcer_col].values[0]) * 100:.2f}"
            if len(our_row) > 0
            else "N/A"
        )

        row_vals: List[str] = [attack, our_apcer_val]
        csv_dict: Dict[str, str] = {
            "Attack_Type": attack,
            "BBDM_APCER%": our_apcer_val,
        }

        note = ""
        for model in compare_models:
            sup_row = sup_per[
                (sup_per["Model"] == model) & (sup_per["Attack_Type"] == attack)
            ]
            if len(sup_row) > 0:
                val = f"{float(sup_row['APCER'].values[0]) * 100:.2f}"
            else:
                val = "N/A"
                if attack == "Artifact":
                    note = "Not in open-set test set"
            row_vals.append(val)
            csv_dict[f"{model}_APCER%"] = val

        row_vals.append(note)
        csv_dict["Note"] = note
        md_rows.append(row_vals)
        csv_rows.append(csv_dict)

    md_lines = [
        "# Table 5: Per-Attack APCER Comparison (BBDM vs Supervised)",
        "",
        "Note: Artifact attack type is present in our test data but was not",
        "included in the open-set supervised model evaluation.",
        "",
    ] + md_table(headers, md_rows)

    save_markdown(output_dir / "comparison_per_attack_vs_supervised.md", md_lines)
    save_csv_from_rows(
        output_dir / "comparison_per_attack_vs_supervised.csv", headers, csv_rows
    )
    logger.info("[Table 5] Per-attack comparison written.")


# ---------------------------------------------------------------------------
# Table 6: All models sorted by ACER
# ---------------------------------------------------------------------------


def table6_all_models(open_set_csv: Path, output_dir: Path) -> None:
    """Generate Table 6: All supervised models + BBDM sorted by ACER."""
    if not open_set_csv.exists():
        logger.warning("open_set_csv not found: %s — skipping Table 6.", open_set_csv)
        return

    df = pd.read_csv(open_set_csv)

    # Per-attack rows for best/worst
    per_attack = df[
        df["Attack_Type"].notna()
        & (df["Attack_Type"].astype(str).str.strip() != "")
    ].copy()

    # Overall rows
    overall = df[
        df["Attack_Type"].isna() | (df["Attack_Type"].astype(str).str.strip() == "")
    ].copy()

    headers = ["Model", "Overall_ACER%", "Best_Attack", "Worst_Attack"]
    md_rows: List[List[str]] = []
    csv_rows: List[Dict] = []

    model_data: List[Dict] = []
    for _, row in overall.iterrows():
        model_name = str(row["Model"])
        acer_val = float(row["ACER"])

        model_per = per_attack[per_attack["Model"] == model_name]
        if len(model_per) > 0:
            best_idx = model_per["ACER"].idxmin()
            worst_idx = model_per["ACER"].idxmax()
            best_attack = (
                f"{model_per.loc[best_idx, 'Attack_Type']} "
                f"({model_per.loc[best_idx, 'ACER'] * 100:.2f}%)"
            )
            worst_attack = (
                f"{model_per.loc[worst_idx, 'Attack_Type']} "
                f"({model_per.loc[worst_idx, 'ACER'] * 100:.2f}%)"
            )
        else:
            best_attack = "N/A"
            worst_attack = "N/A"

        model_data.append(
            {
                "model": model_name,
                "acer": acer_val,
                "best": best_attack,
                "worst": worst_attack,
            }
        )

    # Sort by ACER ascending
    model_data.sort(key=lambda x: x["acer"])

    for entry in model_data:
        acer_pct = f"{entry['acer'] * 100:.2f}"
        md_rows.append([entry["model"], acer_pct, entry["best"], entry["worst"]])
        csv_rows.append(
            {
                "Model": entry["model"],
                "Overall_ACER%": acer_pct,
                "Best_Attack": entry["best"],
                "Worst_Attack": entry["worst"],
            }
        )

    # Add BBDM row last
    md_rows.append(
        ["**BBDM (Ours)**", pct(OUR_ACER), "N/A (unsupervised)", "N/A (unsupervised)"]
    )
    csv_rows.append(
        {
            "Model": "BBDM (Ours)",
            "Overall_ACER%": pct(OUR_ACER),
            "Best_Attack": "N/A (unsupervised)",
            "Worst_Attack": "N/A (unsupervised)",
        }
    )

    md_lines = [
        "# Table 6: All Models Sorted by ACER (Supervised + BBDM Ours)",
        "",
        "Best_Attack and Worst_Attack refer to attack-type-specific ACER values.",
        "BBDM is appended at the end for reference — it uses ACER = (APCER + BPCER) / 2.",
        "",
    ] + md_table(headers, md_rows)

    save_markdown(output_dir / "comparison_all_models.md", md_lines)
    save_csv_from_rows(output_dir / "comparison_all_models.csv", headers, csv_rows)
    logger.info("[Table 6] All-models comparison written (%d models).", len(model_data))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate BBDM vs baseline comparison tables.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--our_results",
        type=Path,
        default=Path("iris_bbdm_pad/results/phase3_evaluation/metrics_summary.json"),
        help="Path to our BBDM metrics_summary.json (optional; overrides hardcoded values).",
    )
    parser.add_argument(
        "--our_per_attack",
        type=Path,
        default=Path(
            "iris_bbdm_pad/results/phase3_evaluation/cross_attack_detailed.csv"
        ),
        help="Path to cross_attack_detailed.csv for per-attack comparison.",
    )
    parser.add_argument(
        "--open_set_csv",
        type=Path,
        default=Path("open_set_summary.csv"),
        help="Path to open_set_summary.csv.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("iris_bbdm_pad/results/phase3_evaluation"),
        help="Directory where output files are saved.",
    )
    return parser.parse_args()


def load_our_metrics(json_path: Path) -> Dict[str, float]:
    """Load BBDM metrics from JSON if available, else return hardcoded defaults."""
    if json_path.exists():
        with json_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        logger.info("Loaded our metrics from %s", json_path)
        return data
    logger.warning(
        "metrics_summary.json not found at %s — using hardcoded BBDM values.", json_path
    )
    return {}


def main() -> None:
    """Entry point."""
    args = parse_args()

    # Optionally update globals from JSON
    metrics = load_our_metrics(args.our_results)
    global OUR_ACER, OUR_APCER, OUR_BPCER
    if "acer" in metrics:
        OUR_ACER = float(metrics["acer"])
    if "apcer" in metrics:
        OUR_APCER = float(metrics["apcer"])
    if "bpcer" in metrics:
        OUR_BPCER = float(metrics["bpcer"])

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating Table 1: LivDet-Iris 2025 Task 1 comparison …")
    table1_livdet2025(args.output_dir)

    logger.info("Generating Table 2: LivDet-Iris 2023 comparison …")
    table2_livdet2023(args.output_dir)

    logger.info("Generating Table 3: Paradigm comparison …")
    table3_paradigm(args.output_dir)

    logger.info("Generating Table 4: vs Open-Set supervised baselines …")
    table4_open_set(args.open_set_csv, args.output_dir)

    logger.info("Generating Table 5: Per-attack APCER comparison …")
    table5_per_attack(args.open_set_csv, args.our_per_attack, args.output_dir)

    logger.info("Generating Table 6: All models sorted by ACER …")
    table6_all_models(args.open_set_csv, args.output_dir)

    logger.info("All comparison tables written to %s", args.output_dir)


if __name__ == "__main__":
    main()
