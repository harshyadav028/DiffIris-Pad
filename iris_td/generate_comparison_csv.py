"""
Generate ddpm_comparison.csv matching combined_comparison.csv format exactly.

Per-attack rows: computed on (all bonafide + this attack type) with
threshold optimized on val subset (bonafide + attack type) — same
protocol as BBDM open-set rows in combined_comparison.csv.

Overall row: computed on full test set with globally optimised threshold.
"""
import json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score

ATTACK_DISPLAY = {
    'Artifact':         'Artifact',
    'CL':               'CL',
    'E-display':        'E-display',
    'Fake_with_Add_On': 'Fake with Add On',
    'Generated':        'Generated',
    'Post-Mortem':      'PostMortem',
    'Print_E-display':  'Print and E-display',
    'Printed':          'Printed',
}
ATTACK_TYPES = list(ATTACK_DISPLAY.keys())


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
        apcer = fp / max(tn+fp, 1)
        bpcer = fn / max(tp+fn, 1)
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
    apcer = fp / max(tn+fp, 1)
    bpcer = fn / max(tp+fn, 1)
    acer  = (apcer + bpcer) / 2
    acc   = (tp+tn) / max(len(labels), 1)
    return apcer, bpcer, acer, acc


def compute_rows(model_name, score_col,
                 val_csv, test_csv, global_thr_path):
    """
    Returns list of dicts ready for DataFrame.
    Per-attack: threshold optimised on val bonafide + attack type.
    Overall: global threshold from find_ddpm_threshold.py.
    """
    val_df  = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)
    global_thr = json.load(open(global_thr_path))
    global_tau = global_thr['best_threshold']

    # bonafide pools
    val_bf  = val_df[val_df['label']  == 'bonafide']
    test_bf = test_df[test_df['label'] == 'bonafide']

    rows = []

    # ── Per-attack rows (global threshold — same threshold for all rows) ──
    for atype in ATTACK_TYPES:
        test_atk = test_df[test_df['attack_type'] == atype]
        if len(test_atk) == 0:
            continue

        # Evaluate on bonafide + this attack type using the global threshold
        test_sub_df = pd.concat([test_bf, test_atk], ignore_index=True)
        test_scores = test_sub_df[score_col].values
        test_labels = (test_sub_df['label'] == 'attack').astype(int).values
        tau = global_tau
        apcer, bpcer, acer, acc = row_metrics(test_scores, test_labels, tau)

        rows.append({
            'Model':       model_name,
            'Attack_Type': ATTACK_DISPLAY[atype],
            'APCER':       round(apcer, 6),
            'BPCER':       round(bpcer, 6),
            'ACER':        round(acer,  6),
            'Threshold':   round(tau,   6),
            'Accuracy':    round(acc,   6),
        })

    # ── Overall row (blank Attack_Type = overall) ────────────────────────
    test_scores = test_df[score_col].values
    test_labels = (test_df['label'] == 'attack').astype(int).values
    apcer, bpcer, acer, acc = row_metrics(test_scores, test_labels, global_tau)
    rows.append({
        'Model':       model_name,
        'Attack_Type': '',
        'APCER':       round(apcer, 6),
        'BPCER':       round(bpcer, 6),
        'ACER':        round(acer,  6),
        'Threshold':   round(global_tau, 6),
        'Accuracy':    round(acc,   6),
    })

    return rows


# ── Config definitions ───────────────────────────────────────────────────────
configs = [
    ('DDPM_Vanilla_Gaussian_t1000',  'mse_score',
     'iris_td/pad_scores/ddpm_val_gaussian_tstar1000_steps25.csv',
     'iris_td/pad_scores/ddpm_test_gaussian_tstar1000_steps25.csv',
     'iris_td/results/threshold_gaussian_tstar1000.json'),

    ('DDPM_Partial_Gaussian_t500',   'mse_score',
     'iris_td/pad_scores/ddpm_val_gaussian_tstar500_steps25.csv',
     'iris_td/pad_scores/ddpm_test_gaussian_tstar500_steps25.csv',
     'iris_td/results/threshold_gaussian_tstar500.json'),

    ('DDPM_AnoDDPM_Simplex_t500',    'mse_score',
     'iris_td/pad_scores/ddpm_val_simplex_tstar500_steps25.csv',
     'iris_td/pad_scores/ddpm_test_simplex_tstar500_steps25.csv',
     'iris_td/results/threshold_simplex_tstar500.json'),

    ('DDPM_AnoDDPM_DDIM_Simplex_t500', 'mse_score',
     'iris_td/pad_scores/ddpm_val_simplex_tstar500_ddim_steps50.csv',
     'iris_td/pad_scores/ddpm_test_simplex_tstar500_ddim_steps50.csv',
     'iris_td/results/threshold_simplex_tstar500_ddim.json'),
]

all_rows = []
for model_name, score_col, val_csv, test_csv, thr_path in configs:
    missing = [p for p in [val_csv, test_csv, thr_path] if not Path(p).exists()]
    if missing:
        print(f'MISSING for {model_name}: {missing}'); continue
    rows = compute_rows(model_name, score_col, val_csv, test_csv, thr_path)
    all_rows.extend(rows)
    print(f'Done: {model_name}  ({len(rows)-1} attack types + overall)')

out_df   = pd.DataFrame(all_rows, columns=['Model','Attack_Type','APCER','BPCER','ACER','Threshold','Accuracy'])
out_path = 'iris_td/final_results/ddpm_comparison.csv'
out_df.to_csv(out_path, index=False)
print(f'\nSaved {len(out_df)} rows -> {out_path}')
print()
print(out_df.to_string(index=False))
