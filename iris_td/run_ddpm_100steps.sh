#!/bin/bash
set -e
cd /home/teaching/Documents/Geetanjali_PhD_IRIS_PAD

# Reduce fragmentation when GPU is partially occupied
export PYTORCH_ALLOC_CONF=expandable_segments:True

LOG=iris_td/final_results/ddpm_100steps_run.log
exec > >(tee -a "$LOG") 2>&1

echo "========================================================"
echo "AnoDDPM Config 3 — Simplex t*=500, 100 steps"
echo "Started: $(date)"
echo "========================================================"

# ── Step 1: Val scoring ──────────────────────────────────────
echo ""
echo "=== STEP 1: Val scoring (100 steps) ==="
PYTHONPATH=BBDM:$PYTHONPATH conda run -n iris_pad python \
  iris_td/models/ddpm_anomaly_detector.py \
  --config iris_td/configs/ddpm_iris.yaml \
  --checkpoint \
    iris_td/results/ddpm_run1/DDPM/checkpoint/top_model_epoch_best.pth \
  --test_dir iris_td/data/evaluation_sets/val/ \
  --output_csv \
    iris_td/pad_scores/ddpm_val_simplex_tstar500_steps100.csv \
  --labels iris_td/labels/val_labels.csv \
  --noise_type simplex \
  --t_star 500 \
  --num_steps 100 \
  --batch_size 48 \
  --split val
echo "=== VAL DONE: $(date) ==="

# ── Step 2: Test scoring ─────────────────────────────────────
echo ""
echo "=== STEP 2: Test scoring (100 steps) ==="
PYTHONPATH=BBDM:$PYTHONPATH conda run -n iris_pad python \
  iris_td/models/ddpm_anomaly_detector.py \
  --config iris_td/configs/ddpm_iris.yaml \
  --checkpoint \
    iris_td/results/ddpm_run1/DDPM/checkpoint/top_model_epoch_best.pth \
  --test_dir iris_td/data/evaluation_sets/test/ \
  --output_csv \
    iris_td/pad_scores/ddpm_test_simplex_tstar500_steps100.csv \
  --labels iris_td/labels/test_labels.csv \
  --noise_type simplex \
  --t_star 500 \
  --num_steps 100 \
  --batch_size 48 \
  --split test
echo "=== TEST DONE: $(date) ==="

# ── Step 3: Fix labels ───────────────────────────────────────
echo ""
echo "=== STEP 3: Fix labels ==="
conda run -n iris_pad python - <<'PYEOF'
import json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_curve, roc_auc_score

def fix_labels(csv_path, out_path):
    def extract(filename):
        fname = str(filename).lower().split('/')[-1]
        if fname.startswith('live_'):
            return 'bonafide', 'Live'
        patterns = [
            (['postmortem','post_mortem'],       'Post-Mortem'),
            (['print_and_e','print_e'],          'Print_E-display'),
            (['fake_with'],                      'Fake_with_Add_On'),
            (['generated_'],                     'Generated'),
            (['artifact_'],                      'Artifact'),
            (['printed_'],                       'Printed'),
            (['e-display_','e_display_'],        'E-display'),
            (['cl_','contactlens'],              'CL'),
        ]
        for prefixes, atype in patterns:
            for p in prefixes:
                if fname.startswith(p):
                    return 'attack', atype
        return 'attack', 'unknown_attack'

    df = pd.read_csv(csv_path)
    labels, atypes = [], []
    for _, row in df.iterrows():
        if row['label'] == 'bonafide' and row['attack_type'] == 'Live':
            labels.append('bonafide'); atypes.append('Live')
            continue
        if (row['label'] == 'attack' and
                row['attack_type'] not in ['unknown', 'unknown_attack', '']):
            labels.append('attack'); atypes.append(row['attack_type'])
            continue
        l, a = extract(row['filename'])
        labels.append(l); atypes.append(a)
    df['label']       = labels
    df['attack_type'] = atypes
    df.to_csv(out_path, index=False)
    print(f'Fixed: {Path(out_path).name}')
    print(f'  {df["label"].value_counts().to_dict()}')
    return df

fix_labels(
    'iris_td/pad_scores/ddpm_val_simplex_tstar500_steps100.csv',
    'iris_td/pad_scores/ddpm_val_simplex_tstar500_steps100_fixed.csv'
)
fix_labels(
    'iris_td/pad_scores/ddpm_test_simplex_tstar500_steps100.csv',
    'iris_td/pad_scores/ddpm_test_simplex_tstar500_steps100_fixed.csv'
)
print('Labels fixed.')
PYEOF
echo "=== LABELS DONE: $(date) ==="

# ── Step 4: Threshold tuning ─────────────────────────────────
echo ""
echo "=== STEP 4: Threshold tuning on fixed val CSV ==="
conda run -n iris_pad python \
  iris_td/training/find_ddpm_threshold.py \
  --scores_csv \
    iris_td/pad_scores/ddpm_val_simplex_tstar500_steps100_fixed.csv \
  --output \
    iris_td/results/threshold_simplex_tstar500_steps100_fixed.json
echo "=== THRESHOLD DONE: $(date) ==="

# ── Step 5: Compare 25 vs 100 vs BBDM ───────────────────────
echo ""
echo "=== STEP 5: Steps comparison ==="
conda run -n iris_pad python - <<'PYEOF'
import json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_curve, roc_auc_score

NEGATE = ['mse_score', 'lpips_score', 'recon_score']

def load(path, negate=True):
    df = pd.read_csv(path)
    if negate:
        for c in NEGATE:
            if c in df.columns:
                df[c] = -df[c]
    return df

def metrics(csv_p, thr_p, negate=True):
    df  = load(csv_p, negate)
    thr = json.load(open(thr_p))
    m   = thr['best_method']
    tau = thr['best_threshold']
    sc  = df[m].values.astype(float)
    lab = (df['label'] == 'attack').astype(int).values
    pred = (sc > tau).astype(int)
    tp = np.sum((pred == 1) & (lab == 1))
    fp = np.sum((pred == 1) & (lab == 0))
    tn = np.sum((pred == 0) & (lab == 0))
    fn = np.sum((pred == 0) & (lab == 1))
    ap = fp / max(tn + fp, 1)
    bp = fn / max(tp + fn, 1)
    ac = (ap + bp) / 2
    auc = roc_auc_score(lab, sc)
    fpr, tpr, _ = roc_curve(lab, sc)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fnr - fpr))
    er  = (fpr[idx] + fnr[idx]) / 2
    t   = df['inference_time'].mean() if 'inference_time' in df else 0.0
    return ap, bp, ac, er, auc, t, m

lines = []
lines.append('=' * 72)
lines.append('STEPS COMPARISON — AnoDDPM Simplex t*=500')
lines.append('=' * 72)
lines.append(f'  {"Config":<35} {"APCER":>7} {"BPCER":>7} {"ACER":>7} '
             f'{"AUC":>7} {"EER":>7} {"Time/img":>9}')
lines.append('  ' + '-' * 68)

# 25 steps
p25 = 'iris_td/pad_scores/ddpm_test_simplex_tstar500_steps25_fixed.csv'
t25 = 'iris_td/results/threshold_simplex_tstar500_fixed.json'
r25 = None
if Path(p25).exists() and Path(t25).exists():
    ap, bp, ac, er, auc, t, m = metrics(p25, t25, negate=True)
    r25 = (ap, bp, ac, er, auc, t)
    lines.append(f'  {"AnoDDPM Simplex  25 steps":<35} '
                 f'{ap*100:>6.2f}% {bp*100:>6.2f}% {ac*100:>6.2f}% '
                 f'{auc:>7.4f} {er*100:>6.2f}% {t:>8.3f}s')
else:
    lines.append(f'  {"AnoDDPM Simplex  25 steps":<35}  NOT FOUND')

# 100 steps
p100 = 'iris_td/pad_scores/ddpm_test_simplex_tstar500_steps100_fixed.csv'
t100 = 'iris_td/results/threshold_simplex_tstar500_steps100_fixed.json'
r100 = None
if Path(p100).exists() and Path(t100).exists():
    ap, bp, ac, er, auc, t, m = metrics(p100, t100, negate=True)
    r100 = (ap, bp, ac, er, auc, t)
    lines.append(f'  {"AnoDDPM Simplex 100 steps":<35} '
                 f'{ap*100:>6.2f}% {bp*100:>6.2f}% {ac*100:>6.2f}% '
                 f'{auc:>7.4f} {er*100:>6.2f}% {t:>8.3f}s')
else:
    lines.append(f'  {"AnoDDPM Simplex 100 steps":<35}  NOT FOUND')

# BBDM reference
lines.append('  ' + '-' * 68)
bbdf = pd.read_csv('iris_bbdm_pad/results/test_pad_scores.csv')
bbt  = json.load(open('iris_bbdm_pad/results/threshold.json'))
m    = bbt['best_method']
tau  = bbt['best_threshold']
sc   = bbdf[m].values.astype(float)
lab  = (bbdf['label'] == 'attack').astype(int).values
pred = (sc > tau).astype(int)
tp = np.sum((pred == 1) & (lab == 1)); fp = np.sum((pred == 1) & (lab == 0))
tn = np.sum((pred == 0) & (lab == 0)); fn = np.sum((pred == 0) & (lab == 1))
ap = fp / max(tn + fp, 1); bp = fn / max(tp + fn, 1); ac = (ap + bp) / 2
auc = roc_auc_score(lab, sc)
fpr, tpr, _ = roc_curve(lab, sc); fnr = 1 - tpr
idx = np.argmin(np.abs(fnr - fpr)); er = (fpr[idx] + fnr[idx]) / 2
lines.append(f'  {"BBDM 200 steps (reference)":<35} '
             f'{ap*100:>6.2f}% {bp*100:>6.2f}% {ac*100:>6.2f}% '
             f'{auc:>7.4f} {er*100:>6.2f}% {0.876:>8.3f}s')
lines.append('=' * 72)

# Verdict
if r25 and r100:
    acer_diff = (r25[2] - r100[2]) * 100
    auc_diff  = r100[4] - r25[4]
    lines.append('')
    lines.append('VERDICT:')
    lines.append(f'  25->100 steps ACER change : {acer_diff:+.2f}%  '
                 f'({"better" if acer_diff > 0 else "worse"} with 100 steps)')
    lines.append(f'  25->100 steps AUC  change : {auc_diff:+.4f}')
    if abs(acer_diff) < 1.0:
        lines.append('  CONCLUSION: Difference < 1% — 25 steps is sufficient.')
        lines.append('  Performance gap to BBDM is architectural, not computational.')
    elif acer_diff > 0:
        lines.append(f'  CONCLUSION: 100 steps improves ACER by {acer_diff:.2f}%.')
        lines.append('  Consider reporting 100-step result as primary DDPM result.')
    else:
        lines.append(f'  CONCLUSION: 100 steps is WORSE by {-acer_diff:.2f}% ACER.')
        lines.append('  Stick with 25-step result.')

output = '\n'.join(lines)
print(output)

out_path = 'iris_td/final_results/steps_comparison.txt'
with open(out_path, 'w') as f:
    f.write(output + '\n')
print(f'\nSaved to {out_path}')
PYEOF

echo ""
echo "========================================================"
echo "ALL DONE: $(date)"
echo "New files created:"
ls -lh iris_td/pad_scores/ddpm_*steps100*.csv 2>/dev/null
ls -lh iris_td/results/threshold_simplex_tstar500_steps100*.json 2>/dev/null
ls -lh iris_td/final_results/steps_comparison.txt 2>/dev/null
echo "========================================================"
