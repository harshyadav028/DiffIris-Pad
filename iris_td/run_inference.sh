#!/usr/bin/env bash
set -eo pipefail
cd ~/Documents/Geetanjali_PhD_IRIS_PAD
export PYTHONPATH="BBDM:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

CKPT="iris_td/results/ddpm_run1/DDPM/checkpoint/top_model_epoch_best.pth"
CFG="iris_td/configs/ddpm_iris.yaml"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║           DDPM INFERENCE PIPELINE — ALL 4 CONFIGS       ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo "  Started: $(ts)"
echo ""

# ─────────────────────────────────────────────────────────────
echo "▶ [$(ts)] VAL Config 1 — Gaussian t*=1000 steps=25 batch=48"
conda run --no-capture-output -n iris_pad python iris_td/models/ddpm_anomaly_detector.py \
  --config "$CFG" --checkpoint "$CKPT" \
  --test_dir iris_td/data/evaluation_sets/val/ \
  --output_csv iris_td/pad_scores/ddpm_val_gaussian_tstar1000_steps25.csv \
  --labels iris_td/labels/val_labels.csv \
  --noise_type gaussian --t_star 1000 --num_steps 25 --batch_size 48 --split val
echo "✔ [$(ts)] === CONFIG 1 VAL DONE ==="

# ─────────────────────────────────────────────────────────────
echo ""
echo "▶ [$(ts)] VAL Config 2 — Gaussian t*=500 steps=25 batch=48"
conda run --no-capture-output -n iris_pad python iris_td/models/ddpm_anomaly_detector.py \
  --config "$CFG" --checkpoint "$CKPT" \
  --test_dir iris_td/data/evaluation_sets/val/ \
  --output_csv iris_td/pad_scores/ddpm_val_gaussian_tstar500_steps25.csv \
  --labels iris_td/labels/val_labels.csv \
  --noise_type gaussian --t_star 500 --num_steps 25 --batch_size 48 --split val
echo "✔ [$(ts)] === CONFIG 2 VAL DONE ==="

# ─────────────────────────────────────────────────────────────
echo ""
echo "▶ [$(ts)] VAL Config 3 — Simplex t*=500 steps=25 batch=48"
conda run --no-capture-output -n iris_pad python iris_td/models/ddpm_anomaly_detector.py \
  --config "$CFG" --checkpoint "$CKPT" \
  --test_dir iris_td/data/evaluation_sets/val/ \
  --output_csv iris_td/pad_scores/ddpm_val_simplex_tstar500_steps25.csv \
  --labels iris_td/labels/val_labels.csv \
  --noise_type simplex --t_star 500 --num_steps 25 --batch_size 48 --split val
echo "✔ [$(ts)] === CONFIG 3 VAL DONE ==="

# ─────────────────────────────────────────────────────────────
echo ""
echo "▶ [$(ts)] VAL Config 4 — Simplex t*=500 steps=50 DDIM batch=64"
conda run --no-capture-output -n iris_pad python iris_td/models/ddpm_anomaly_detector.py \
  --config "$CFG" --checkpoint "$CKPT" \
  --test_dir iris_td/data/evaluation_sets/val/ \
  --output_csv iris_td/pad_scores/ddpm_val_simplex_tstar500_ddim_steps50.csv \
  --labels iris_td/labels/val_labels.csv \
  --noise_type simplex --t_star 500 --num_steps 50 --batch_size 64 --use_ddim --split val
echo "✔ [$(ts)] === CONFIG 4 VAL DONE ==="

# ─────────────────────────────────────────────────────────────
echo ""
echo "▶ [$(ts)] Verifying all VAL CSVs..."
python -c "
import pandas as pd
from pathlib import Path
configs = [
    ('Config1 Vanilla', 'iris_td/pad_scores/ddpm_val_gaussian_tstar1000_steps25.csv'),
    ('Config2 Partial', 'iris_td/pad_scores/ddpm_val_gaussian_tstar500_steps25.csv'),
    ('Config3 AnoDDPM', 'iris_td/pad_scores/ddpm_val_simplex_tstar500_steps25.csv'),
    ('Config4 +DDIM',   'iris_td/pad_scores/ddpm_val_simplex_tstar500_ddim_steps50.csv'),
]
all_ok = True
for name, path in configs:
    if not Path(path).exists():
        print(f'MISSING: {name}'); all_ok = False; continue
    df = pd.read_csv(path)
    bf = sum(df.label=='bonafide'); atk = sum(df.label=='attack')
    nan = df.isnull().any().any()
    ok = len(df) > 0 and not nan
    print(f'{'OK' if ok else 'FAIL'}: {name}  rows={len(df)} bonafide={bf} attack={atk}')
    if not ok: all_ok = False
print()
if not all_ok: raise SystemExit('SOME VAL CSVS FAILED')
print('ALL VAL OK — proceeding to threshold tuning')
"

# ─────────────────────────────────────────────────────────────
echo ""
echo "▶ [$(ts)] THRESHOLD 1 — Gaussian t*=1000"
conda run --no-capture-output -n iris_pad python iris_td/training/find_ddpm_threshold.py \
  --scores_csv iris_td/pad_scores/ddpm_val_gaussian_tstar1000_steps25.csv \
  --output iris_td/results/threshold_gaussian_tstar1000.json
echo "✔ [$(ts)] === THRESHOLD 1 DONE ==="

echo ""
echo "▶ [$(ts)] THRESHOLD 2 — Gaussian t*=500"
conda run --no-capture-output -n iris_pad python iris_td/training/find_ddpm_threshold.py \
  --scores_csv iris_td/pad_scores/ddpm_val_gaussian_tstar500_steps25.csv \
  --output iris_td/results/threshold_gaussian_tstar500.json
echo "✔ [$(ts)] === THRESHOLD 2 DONE ==="

echo ""
echo "▶ [$(ts)] THRESHOLD 3 — Simplex t*=500"
conda run --no-capture-output -n iris_pad python iris_td/training/find_ddpm_threshold.py \
  --scores_csv iris_td/pad_scores/ddpm_val_simplex_tstar500_steps25.csv \
  --output iris_td/results/threshold_simplex_tstar500.json
echo "✔ [$(ts)] === THRESHOLD 3 DONE ==="

echo ""
echo "▶ [$(ts)] THRESHOLD 4 — Simplex t*=500 DDIM"
conda run --no-capture-output -n iris_pad python iris_td/training/find_ddpm_threshold.py \
  --scores_csv iris_td/pad_scores/ddpm_val_simplex_tstar500_ddim_steps50.csv \
  --output iris_td/results/threshold_simplex_tstar500_ddim.json
echo "✔ [$(ts)] === THRESHOLD 4 DONE ==="

echo ""
echo "▶ [$(ts)] Threshold summary:"
python -c "
import json
from pathlib import Path
thresholds = [
    ('Config1 Vanilla', 'iris_td/results/threshold_gaussian_tstar1000.json'),
    ('Config2 Partial', 'iris_td/results/threshold_gaussian_tstar500.json'),
    ('Config3 AnoDDPM', 'iris_td/results/threshold_simplex_tstar500.json'),
    ('Config4 +DDIM',   'iris_td/results/threshold_simplex_tstar500_ddim.json'),
]
print('=== THRESHOLD RESULTS (VAL) ===')
all_ok = True
for name, path in thresholds:
    if not Path(path).exists():
        print(f'MISSING: {path}'); all_ok = False; continue
    t = json.load(open(path))
    print(f'{name}:  method={t[\"best_method\"]}  tau={t[\"best_threshold\"]:.6f}  ACER={t[\"best_acer\"]*100:.2f}%')
if not all_ok: raise SystemExit('SOME THRESHOLDS MISSING')
print('ALL THRESHOLDS OK — proceeding to test scoring')
"

# ─────────────────────────────────────────────────────────────
echo ""
echo "▶ [$(ts)] TEST Config 1 — Gaussian t*=1000 steps=25 batch=48"
conda run --no-capture-output -n iris_pad python iris_td/models/ddpm_anomaly_detector.py \
  --config "$CFG" --checkpoint "$CKPT" \
  --test_dir iris_td/data/evaluation_sets/test/ \
  --output_csv iris_td/pad_scores/ddpm_test_gaussian_tstar1000_steps25.csv \
  --labels iris_td/labels/test_labels.csv \
  --noise_type gaussian --t_star 1000 --num_steps 25 --batch_size 48 --split test
echo "✔ [$(ts)] === CONFIG 1 TEST DONE ==="

echo ""
echo "▶ [$(ts)] TEST Config 2 — Gaussian t*=500 steps=25 batch=48"
conda run --no-capture-output -n iris_pad python iris_td/models/ddpm_anomaly_detector.py \
  --config "$CFG" --checkpoint "$CKPT" \
  --test_dir iris_td/data/evaluation_sets/test/ \
  --output_csv iris_td/pad_scores/ddpm_test_gaussian_tstar500_steps25.csv \
  --labels iris_td/labels/test_labels.csv \
  --noise_type gaussian --t_star 500 --num_steps 25 --batch_size 48 --split test
echo "✔ [$(ts)] === CONFIG 2 TEST DONE ==="

echo ""
echo "▶ [$(ts)] TEST Config 3 — Simplex t*=500 steps=25 batch=48"
conda run --no-capture-output -n iris_pad python iris_td/models/ddpm_anomaly_detector.py \
  --config "$CFG" --checkpoint "$CKPT" \
  --test_dir iris_td/data/evaluation_sets/test/ \
  --output_csv iris_td/pad_scores/ddpm_test_simplex_tstar500_steps25.csv \
  --labels iris_td/labels/test_labels.csv \
  --noise_type simplex --t_star 500 --num_steps 25 --batch_size 48 --split test
echo "✔ [$(ts)] === CONFIG 3 TEST DONE ==="

echo ""
echo "▶ [$(ts)] TEST Config 4 — Simplex t*=500 steps=50 DDIM batch=64"
conda run --no-capture-output -n iris_pad python iris_td/models/ddpm_anomaly_detector.py \
  --config "$CFG" --checkpoint "$CKPT" \
  --test_dir iris_td/data/evaluation_sets/test/ \
  --output_csv iris_td/pad_scores/ddpm_test_simplex_tstar500_ddim_steps50.csv \
  --labels iris_td/labels/test_labels.csv \
  --noise_type simplex --t_star 500 --num_steps 50 --batch_size 64 --use_ddim --split test
echo "✔ [$(ts)] === CONFIG 4 TEST DONE ==="

# ─────────────────────────────────────────────────────────────
echo ""
echo "▶ [$(ts)] FINAL ABLATION TABLE"
python -c "
import json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_curve

def get_metrics(csv_path, thr_path):
    df  = pd.read_csv(csv_path)
    thr = json.load(open(thr_path))
    m   = thr['best_method']
    tau = thr['best_threshold']
    sc  = df[m].values
    lab = (df['label']=='attack').astype(int).values
    pred = (sc > tau).astype(int)
    tp=np.sum((pred==1)&(lab==1)); fp=np.sum((pred==1)&(lab==0))
    tn=np.sum((pred==0)&(lab==0)); fn=np.sum((pred==0)&(lab==1))
    apcer=fp/max(tn+fp,1); bpcer=fn/max(tp+fn,1); acer=(apcer+bpcer)/2
    fpr,tpr,_=roc_curve(lab,sc); fnr=1-tpr
    idx=np.argmin(np.abs(fnr-fpr)); eer=(fpr[idx]+fnr[idx])/2
    return apcer, bpcer, acer, eer

configs = [
    ('DDPM vanilla  (Gaussian, t*=1000, 25 steps)',
     'iris_td/pad_scores/ddpm_test_gaussian_tstar1000_steps25.csv',
     'iris_td/results/threshold_gaussian_tstar1000.json'),
    ('DDPM partial  (Gaussian, t*=500,  25 steps)',
     'iris_td/pad_scores/ddpm_test_gaussian_tstar500_steps25.csv',
     'iris_td/results/threshold_gaussian_tstar500.json'),
    ('DDPM AnoDDPM  (Simplex,  t*=500,  25 steps)',
     'iris_td/pad_scores/ddpm_test_simplex_tstar500_steps25.csv',
     'iris_td/results/threshold_simplex_tstar500.json'),
    ('DDPM+DDIM     (Simplex,  t*=500,  50 steps)',
     'iris_td/pad_scores/ddpm_test_simplex_tstar500_ddim_steps50.csv',
     'iris_td/results/threshold_simplex_tstar500_ddim.json'),
]

print('=' * 72)
print('FINAL ABLATION TABLE — TEST SET')
print('=' * 72)
print(f'{\"Model\":<44} {\"APCER\":>6} {\"BPCER\":>6} {\"ACER\":>6} {\"EER\":>6}')
print('-' * 72)
for name, csv_p, thr_p in configs:
    if not Path(csv_p).exists() or not Path(thr_p).exists():
        print(f'{name:<44}  NOT DONE YET'); continue
    ap,bp,ac,er = get_metrics(csv_p, thr_p)
    print(f'{name:<44} {ap*100:>5.2f}% {bp*100:>5.2f}% {ac*100:>5.2f}% {er*100:>5.2f}%')
print('-' * 72)
thr=json.load(open('iris_bbdm_pad/results/threshold.json'))
df=pd.read_csv('iris_bbdm_pad/results/test_pad_scores.csv')
m=thr['best_method']; tau=thr['best_threshold']
sc=df[m].values; lab=(df['label']=='attack').astype(int).values
pred=(sc>tau).astype(int)
tp=np.sum((pred==1)&(lab==1)); fp=np.sum((pred==1)&(lab==0))
tn=np.sum((pred==0)&(lab==0)); fn=np.sum((pred==0)&(lab==1))
ap=fp/max(tn+fp,1); bp=fn/max(tp+fn,1); ac=(ap+bp)/2
fpr,tpr,_=roc_curve(lab,sc); fnr=1-tpr; idx=np.argmin(np.abs(fnr-fpr)); er=(fpr[idx]+fnr[idx])/2
print(f'{\"Brownian Bridge (BBDM)\":<44} {ap*100:>5.2f}% {bp*100:>5.2f}% {ac*100:>5.2f}% {er*100:>5.2f}%  <- target')
print('=' * 72)
" | tee iris_td/final_results/ablation_table.txt

# ─────────────────────────────────────────────────────────────
echo ""
echo "▶ [$(ts)] PER-ATTACK BREAKDOWN"
python -c "
import json, numpy as np, pandas as pd
from pathlib import Path

configs_to_break = [
    ('DDPM AnoDDPM+DDIM (strongest)',
     'iris_td/pad_scores/ddpm_test_simplex_tstar500_ddim_steps50.csv',
     'iris_td/results/threshold_simplex_tstar500_ddim.json'),
    ('Brownian Bridge (BBDM)',
     'iris_bbdm_pad/results/test_pad_scores.csv',
     'iris_bbdm_pad/results/threshold.json'),
]
attack_types = ['Live','Artifact','CL','E-display','Fake_with_Add_On',
                'Generated','Post-Mortem','Printed','Print_E-display']

for model_name, csv_path, thr_path in configs_to_break:
    if not Path(csv_path).exists():
        print(f'MISSING: {csv_path}'); continue
    df=pd.read_csv(csv_path); thr=json.load(open(thr_path))
    m=thr['best_method']; tau=thr['best_threshold']
    print()
    print('=' * 65)
    print(f'PER-ATTACK BREAKDOWN — {model_name}')
    print('=' * 65)
    print(f'  threshold method: {m}  tau={tau:.4f}')
    print(f'{\"Attack Type\":<22} {\"Total\":>6} {\"Correct\":>8} {\"Rate\":>12}')
    print('-' * 65)
    for atype in attack_types:
        subset=df[df['attack_type']==atype]
        if len(subset)==0: continue
        scores=subset[m].values; preds=(scores>tau).astype(int)
        if atype=='Live':
            fn=np.sum(preds==1); total=len(subset); rate=fn/max(total,1); label='BPCER'
        else:
            fn=np.sum(preds==0); total=len(subset); rate=fn/max(total,1); label='miss rate'
        correct=total-fn
        print(f'  {atype:<20} {total:>6} {correct:>8} {rate*100:>10.2f}%  ({label})')
    print('-' * 65)
" | tee -a iris_td/final_results/ablation_table.txt

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║              ALL INFERENCE COMPLETE                      ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo "  Finished: $(ts)"
echo "  Results:  iris_td/final_results/ablation_table.txt"
