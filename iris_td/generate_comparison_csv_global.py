"""
Generate ddpm_comparison_global.csv — global threshold version.

Same score files and ISO-correct APCER/BPCER as ddpm_comparison_fixed.csv,
but every per-attack row uses the single global threshold from the JSON
(no per-attack val optimisation).

APCER = FN / (FN + TP)  — attacks missed / total attacks      (ISO 30107-3)
BPCER = FP / (FP + TN)  — bonafide flagged / total bonafide   (ISO 30107-3)
"""
import json, numpy as np, pandas as pd
from pathlib import Path

ATTACK_TYPES = [
    'Artifact', 'CL', 'E-display', 'Fake with Add On',
    'Generated', 'PostMortem', 'Print and E-display', 'Printed',
]

NEGATE_COLS = ['mse_score', 'lpips_score', 'recon_score']


def load_and_fix(csv_path):
    df = pd.read_csv(csv_path)
    for col in NEGATE_COLS:
        if col in df.columns:
            df[col] = -df[col]
    return df


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


def compute_rows(model_name, test_csv, thr_path):
    """
    Per-attack rows: bonafide + attack type evaluated at global threshold.
    Overall row:     full test set at global threshold.
    """
    test_df    = load_and_fix(test_csv)
    thr_data   = json.load(open(thr_path))
    score_col  = thr_data['best_method']
    global_tau = thr_data['best_threshold']

    print(f"  {model_name}: method={score_col}, global_tau={global_tau:.4f}")

    test_bf = test_df[test_df['label'] == 'bonafide']
    rows = []

    for atype in ATTACK_TYPES:
        test_atk = test_df[test_df['attack_type'] == atype]
        if len(test_atk) == 0:
            print(f"    WARNING: no test samples for {atype}")
            continue

        test_sub    = pd.concat([test_bf, test_atk], ignore_index=True)
        test_scores = test_sub[score_col].values.astype(np.float64)
        test_labels = (test_sub['label'] == 'attack').astype(int).values
        apcer, bpcer, acer, acc = row_metrics(test_scores, test_labels, global_tau)

        rows.append({
            'Model':       model_name,
            'Attack_Type': atype,
            'APCER':       round(apcer, 6),
            'BPCER':       round(bpcer, 6),
            'ACER':        round(acer,  6),
            'Threshold':   round(abs(global_tau), 6),
            'Accuracy':    round(acc,   6),
        })

    # Overall row — full test set
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


configs = [
    ('DDPM_Vanilla_Gaussian_t1000',
     'iris_td/pad_scores/ddpm_test_gaussian_tstar1000_steps25_fixed.csv',
     'iris_td/results/threshold_gaussian_tstar1000_fixed.json'),

    ('DDPM_Partial_Gaussian_t500',
     'iris_td/pad_scores/ddpm_test_gaussian_tstar500_steps25_fixed.csv',
     'iris_td/results/threshold_gaussian_tstar500_fixed.json'),

    ('DDPM_AnoDDPM_Simplex_t500',
     'iris_td/pad_scores/ddpm_test_simplex_tstar500_steps25_fixed.csv',
     'iris_td/results/threshold_simplex_tstar500_fixed.json'),

    ('DDPM_AnoDDPM_DDIM_Simplex_t500',
     'iris_td/pad_scores/ddpm_test_simplex_tstar500_ddim_steps50_fixed.csv',
     'iris_td/results/threshold_ddim_tstar500_fixed.json'),
]

all_rows = []
for model_name, test_csv, thr_path in configs:
    missing = [p for p in [test_csv, thr_path] if not Path(p).exists()]
    if missing:
        print(f'MISSING for {model_name}: {missing}')
        continue
    rows = compute_rows(model_name, test_csv, thr_path)
    all_rows.extend(rows)
    print(f'  -> {len(rows)-1} attack types + overall row')

out_df = pd.DataFrame(all_rows, columns=[
    'Model', 'Attack_Type', 'APCER', 'BPCER', 'ACER', 'Threshold', 'Accuracy'
])
out_path = 'iris_td/final_results/ddpm_comparison_global.csv'
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
out_df.to_csv(out_path, index=False)
print(f'\nSaved {len(out_df)} rows -> {out_path}')
print()
print(out_df.to_string(index=False))
