"""
Generate ddpm_comparison_fixed.csv matching combined.csv format.

Uses _fixed.csv score files (correct labels/attack_types extracted from filenames)
and negated reconstruction scores (score inversion fix for AnoDDPM at t*=500).

Per-attack threshold protocol matches BBDM (combined.csv):
  - For each attack type: optimize threshold on val (bonafide + attack type)
  - Evaluate test (bonafide + attack type) with that per-attack threshold
  - Overall row uses the global threshold from the JSON
"""
import json, numpy as np, pandas as pd
from pathlib import Path

ATTACK_TYPES = [
    'Artifact', 'CL', 'E-display', 'Fake with Add On',
    'Generated', 'PostMortem', 'Print and E-display', 'Printed',
]

NEGATE_COLS = ['mse_score', 'lpips_score', 'recon_score']

# ViT CSVs use underscores in attack_type — remap to match ATTACK_TYPES
ATTACK_TYPE_REMAP = {
    'Fake_with_Add_On':  'Fake with Add On',
    'Print_E-display':   'Print and E-display',
}


def load_and_fix(csv_path, negate=True):
    """Load CSV, optionally negate reconstruction scores, and remap attack_type names."""
    df = pd.read_csv(csv_path)
    if negate:
        for col in NEGATE_COLS:
            if col in df.columns:
                df[col] = -df[col]
    if 'attack_type' in df.columns:
        df['attack_type'] = df['attack_type'].replace(ATTACK_TYPE_REMAP)
    return df


def optimize_threshold(scores, labels):
    """Find threshold that minimises ACER on this subset."""
    best_acer = 1.0
    best_tau  = np.median(scores)
    for tau in np.unique(scores):
        pred  = (scores > tau).astype(int)
        tp = np.sum((pred==1)&(labels==1))
        fp = np.sum((pred==1)&(labels==0))
        tn = np.sum((pred==0)&(labels==0))
        fn = np.sum((pred==0)&(labels==1))
        apcer = fn / max(tp+fn, 1)
        bpcer = fp / max(tn+fp, 1)
        acer  = (apcer + bpcer) / 2
        if acer < best_acer:
            best_acer = acer
            best_tau  = tau
    return best_tau


def row_metrics(scores, labels, tau):
    pred  = (scores > tau).astype(int)
    tp = np.sum((pred==1)&(labels==1))
    fp = np.sum((pred==1)&(labels==0))
    tn = np.sum((pred==0)&(labels==0))
    fn = np.sum((pred==0)&(labels==1))
    apcer = fn / max(tp+fn, 1)   # ISO 30107-3: attacks missed / total attacks
    bpcer = fp / max(tn+fp, 1)   # ISO 30107-3: bonafide flagged / total bonafide
    acer  = (apcer + bpcer) / 2
    acc   = (tp+tn) / max(len(labels), 1)
    return apcer, bpcer, acer, acc


def compute_rows(model_name, val_csv, test_csv, thr_path, negate=True):
    """
    Returns list of dicts ready for DataFrame.
    Per-attack: threshold optimised on val (bonafide + attack type),
                evaluated on test (bonafide + attack type) — same protocol as BBDM.
    Overall: full test set at global threshold from JSON.
    """
    val_df  = load_and_fix(val_csv,  negate=negate)
    test_df = load_and_fix(test_csv, negate=negate)
    thr_data  = json.load(open(thr_path))
    score_col  = thr_data['best_method']
    global_tau = thr_data['best_threshold']

    print(f"  {model_name}: method={score_col}, global_tau={global_tau:.4f}")

    val_bf  = val_df[val_df['label']   == 'bonafide']
    test_bf = test_df[test_df['label'] == 'bonafide']
    rows = []

    for atype in ATTACK_TYPES:
        val_atk  = val_df[val_df['attack_type']   == atype]
        test_atk = test_df[test_df['attack_type'] == atype]
        if len(test_atk) == 0:
            print(f"    WARNING: no test samples for {atype}")
            continue

        # Optimise threshold on val subset
        val_sub    = pd.concat([val_bf, val_atk], ignore_index=True)
        val_scores = val_sub[score_col].values.astype(np.float64)
        val_labels = (val_sub['label'] == 'attack').astype(int).values
        tau = optimize_threshold(val_scores, val_labels)

        # Evaluate on test subset with per-attack threshold
        test_sub    = pd.concat([test_bf, test_atk], ignore_index=True)
        test_scores = test_sub[score_col].values.astype(np.float64)
        test_labels = (test_sub['label'] == 'attack').astype(int).values
        apcer, bpcer, acer, acc = row_metrics(test_scores, test_labels, tau)

        rows.append({
            'Model':       model_name,
            'Attack_Type': atype,
            'APCER':       round(apcer, 6),
            'BPCER':       round(bpcer, 6),
            'ACER':        round(acer,  6),
            'Threshold':   round(abs(tau), 6),
            'Accuracy':    round(acc,   6),
        })

    # Overall row uses global threshold
    all_scores = test_df[score_col].values.astype(np.float64)
    all_labels = (test_df['label'] == 'attack').astype(int).values
    apcer, bpcer, acer, acc = row_metrics(all_scores, all_labels, global_tau)
    rows.append({
        'Model':       model_name,
        'Attack_Type': '',
        'APCER':       round(apcer, 6),
        'BPCER':       round(bpcer, 6),
        'ACER':        round(acer,  6),
        'Threshold':   round(abs(global_tau), 6),
        'Accuracy':    round(acc,   6),
    })

    return rows


# ── Config definitions ───────────────────────────────────────────────────────
# (model_name, val_csv, test_csv, thr_path, negate)
configs = [
    ('DDPM_Vanilla_Gaussian_t1000',
     'iris_td/pad_scores/ddpm_val_gaussian_tstar1000_steps25_fixed.csv',
     'iris_td/pad_scores/ddpm_test_gaussian_tstar1000_steps25_fixed.csv',
     'iris_td/results/threshold_gaussian_tstar1000_fixed.json', True),

    ('DDPM_Partial_Gaussian_t500',
     'iris_td/pad_scores/ddpm_val_gaussian_tstar500_steps25_fixed.csv',
     'iris_td/pad_scores/ddpm_test_gaussian_tstar500_steps25_fixed.csv',
     'iris_td/results/threshold_gaussian_tstar500_fixed.json', True),

    ('DDPM_AnoDDPM_Simplex_t500',
     'iris_td/pad_scores/ddpm_val_simplex_tstar500_steps25_fixed.csv',
     'iris_td/pad_scores/ddpm_test_simplex_tstar500_steps25_fixed.csv',
     'iris_td/results/threshold_simplex_tstar500_fixed.json', True),

    ('DDPM_AnoDDPM_DDIM_Simplex_t500',
     'iris_td/pad_scores/ddpm_val_simplex_tstar500_ddim_steps50_fixed.csv',
     'iris_td/pad_scores/ddpm_test_simplex_tstar500_ddim_steps50_fixed.csv',
     'iris_td/results/threshold_ddim_tstar500_fixed.json', True),

    ('DDPM_AnoDDPM_DDIM_Simplex_t500_ViT',
     'iris_td/pad_scores/ddpm_val_simplex_tstar500_ddim_steps100_vit.csv',
     'iris_td/pad_scores/ddpm_test_simplex_tstar500_ddim_steps100_vit.csv',
     'iris_td/results/threshold_ddim_simplex_vit_fixed.json', False),
]

all_rows = []
for model_name, val_csv, test_csv, thr_path, negate in configs:
    missing = [p for p in [val_csv, test_csv, thr_path] if not Path(p).exists()]
    if missing:
        print(f'MISSING for {model_name}: {missing}')
        continue
    rows = compute_rows(model_name, val_csv, test_csv, thr_path, negate=negate)
    all_rows.extend(rows)
    print(f'  -> {len(rows)-1} attack types + overall row')

out_df = pd.DataFrame(all_rows, columns=[
    'Model', 'Attack_Type', 'APCER', 'BPCER', 'ACER', 'Threshold', 'Accuracy'
])
out_path = 'iris_td/final_results/ddpm_comparison_fixed.csv'
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
out_df.to_csv(out_path, index=False)
print(f'\nSaved {len(out_df)} rows -> {out_path}')
print()
print(out_df.to_string(index=False))
