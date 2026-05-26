"""
iris_bbdm_pad/evaluation/compute_vit_metrics.py

Finds the optimal ViT-only threshold (minimising ACER on val set) and
computes per-attack-type metrics on the test set.

Usage:
    cd /home/teaching/Documents/Geetanjali_PhD_IRIS_PAD
    python iris_bbdm_pad/evaluation/compute_vit_metrics.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Canonical attack display names (matches existing per_attack_metrics.csv)
# ---------------------------------------------------------------------------
ATTACK_RAW_TO_DISPLAY = {
    "Artifact":           "Artifact",
    "CL":                 "Contact Lens",
    "E-display":          "E-display",
    "Fake with Add On":   "Fake w/ Add-On",
    "Generated":          "Generated",
    "PostMortem":         "Post-Mortem",
    "Print and E-display": "Print & E-disp.",
    "Printed":            "Printed",
}

ATTACK_ORDER = [
    "Artifact", "Contact Lens", "E-display", "Fake w/ Add-On",
    "Generated", "Post-Mortem", "Print & E-disp.", "Printed", "Overall",
]


def normalise_attack_type(raw: str) -> str:
    return ATTACK_RAW_TO_DISPLAY.get(raw, raw)


def find_threshold(
    val_df: pd.DataFrame,
    score_col: str,
    n_candidates: int = 2000,
) -> tuple[float, float]:
    """Sweep threshold candidates to minimise ACER on validation set.

    Returns:
        (best_threshold, best_val_acer_pct)
    """
    scores = val_df[score_col].dropna()
    candidates = np.linspace(scores.quantile(0.01), scores.quantile(0.99), n_candidates)
    bonafide = val_df[(val_df["label"] == "bonafide") & val_df[score_col].notna()]
    attacks  = val_df[(val_df["label"] != "bonafide") & val_df[score_col].notna()]

    best_t, best_acer = None, float("inf")
    for t in candidates:
        bpcer = (bonafide[score_col] > t).mean()
        apcer = (attacks[score_col]  <= t).mean()
        acer  = (apcer + bpcer) / 2.0
        if acer < best_acer:
            best_acer, best_t = acer, float(t)
    return best_t, best_acer


def compute_per_attack(
    test_df: pd.DataFrame,
    score_col: str,
    threshold: float,
) -> pd.DataFrame:
    """Compute APCER, BPCER, ACER, Accuracy per attack type + Overall.

    Attack detection rule: score > threshold → classified as attack (positive).

    Args:
        test_df:   DataFrame with columns [label, attack_type, score_col].
                   label == 'bonafide' for live, anything else for attack.
        score_col: Column name of the PAD score.
        threshold: Decision threshold.

    Returns:
        DataFrame with columns [Attack, APCER↓, BPCER↓, ACER↓, Acc↑, τ].
    """
    bonafide = test_df[(test_df["label"] == "bonafide") & test_df[score_col].notna()]
    attacks  = test_df[(test_df["label"] != "bonafide") & test_df[score_col].notna()]

    # BPCER is global (all bonafide vs one threshold)
    bpcer_val = float((bonafide[score_col] > threshold).mean())

    rows = []
    for raw_type in attacks["attack_type"].unique():
        atk_df = attacks[attacks["attack_type"] == raw_type]
        display = normalise_attack_type(raw_type)
        apcer_val = float((atk_df[score_col] <= threshold).mean())
        acer_val  = (apcer_val + bpcer_val) / 2.0
        tp = int((atk_df[score_col] > threshold).sum())
        tn = int((bonafide[score_col] <= threshold).sum())
        acc = (tp + tn) / (len(atk_df) + len(bonafide))
        rows.append({
            "Attack":  display,
            "APCER↓":  round(apcer_val, 6),
            "BPCER↓":  round(bpcer_val, 6),
            "ACER↓":   round(acer_val,  6),
            "Acc↑":    round(acc,        6),
            "τ":       round(threshold,  4),
        })

    # Overall
    overall_apcer = float((attacks[score_col] <= threshold).mean())
    overall_acer  = (overall_apcer + bpcer_val) / 2.0
    overall_tn    = int((bonafide[score_col] <= threshold).sum())
    overall_tp    = int((attacks[score_col] > threshold).sum())
    overall_acc   = (overall_tp + overall_tn) / (len(attacks) + len(bonafide))
    rows.append({
        "Attack":  "Overall",
        "APCER↓":  round(overall_apcer, 6),
        "BPCER↓":  round(bpcer_val,     6),
        "ACER↓":   round(overall_acer,  6),
        "Acc↑":    round(overall_acc,   6),
        "τ":       round(threshold,     4),
    })

    df = pd.DataFrame(rows)
    # Reorder rows by canonical attack order
    order_map = {a: i for i, a in enumerate(ATTACK_ORDER)}
    df["_ord"] = df["Attack"].map(lambda x: order_map.get(x, 99))
    df = df.sort_values("_ord").drop("_ord", axis=1).reset_index(drop=True)
    return df


if __name__ == "__main__":
    SCORING_DIR = Path("IJCB_paper_requirements/scoring")

    val_csv  = SCORING_DIR / "vit_scores_val.csv"
    test_csv = SCORING_DIR / "vit_scores_test.csv"

    if not val_csv.exists() or not test_csv.exists():
        print(f"ERROR: Run run_vit_scoring.py first to generate score CSVs.")
        raise SystemExit(1)

    val_df  = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    print(f"Val  rows: {len(val_df)}  | Test rows: {len(test_df)}")
    print(f"Val  columns: {val_df.columns.tolist()}")
    print(f"Test attack types: {test_df['attack_type'].unique().tolist()}")

    # ── ViT-only threshold ────────────────────────────────────────
    vit_threshold, vit_val_acer = find_threshold(val_df, "vit_score")
    print(f"\nViT-only → threshold={vit_threshold:.4f}  val ACER={vit_val_acer:.2f}%")

    # ── Per-attack metrics on test ────────────────────────────────
    vit_metrics = compute_per_attack(test_df, "vit_score", vit_threshold)
    print("\nViT-only per-attack results (test set):")
    print(vit_metrics.to_string(index=False))

    # ── Save ─────────────────────────────────────────────────────
    out_metrics   = SCORING_DIR / "vit_metrics_test.csv"
    out_threshold = SCORING_DIR / "vit_threshold.json"

    vit_metrics.to_csv(out_metrics, index=False)
    print(f"\nSaved metrics → {out_metrics}")

    with open(out_threshold, "w") as f:
        json.dump(
            {"vit_threshold": vit_threshold, "vit_val_acer": vit_val_acer},
            f, indent=2,
        )
    print(f"Saved threshold → {out_threshold}")
