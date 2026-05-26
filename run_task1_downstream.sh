#!/bin/bash
# Run all Task 1 downstream steps after ViT scoring is complete.
# Run from project root:
#   bash run_task1_downstream.sh
set -e
cd /home/teaching/Documents/Geetanjali_PhD_IRIS_PAD

PYTHON="PYTHONNOUSERSITE=1 /home/teaching/miniconda3/envs/bbdm_clean/bin/python"
LOG="IJCB_paper_requirements/logs/task1_downstream.log"
mkdir -p IJCB_paper_requirements/logs

echo "========================================" | tee -a "$LOG"
echo "[$(date)] Task 1 downstream pipeline" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"

# ── Check scoring outputs exist ───────────────────────────────────
for f in IJCB_paper_requirements/scoring/vit_scores_test.csv \
          IJCB_paper_requirements/scoring/vit_scores_val.csv; do
    if [ ! -f "$f" ]; then
        echo "ERROR: $f not found. Run ViT scoring first." | tee -a "$LOG"
        exit 1
    fi
    rows=$(wc -l < "$f")
    echo "  $f: $rows rows" | tee -a "$LOG"
done

# ── Step 1C: Compute ViT-only threshold and per-attack metrics ────
echo "" | tee -a "$LOG"
echo "[$(date)] Step 1C: ViT-only metrics..." | tee -a "$LOG"
PYTHONNOUSERSITE=1 /home/teaching/miniconda3/envs/bbdm_clean/bin/python \
    iris_bbdm_pad/evaluation/compute_vit_metrics.py \
    2>&1 | tee -a "$LOG"

# ── Step 1D: Dynamic fusion metrics ──────────────────────────────
echo "" | tee -a "$LOG"
echo "[$(date)] Step 1D: Dynamic fusion metrics..." | tee -a "$LOG"
PYTHONNOUSERSITE=1 /home/teaching/miniconda3/envs/bbdm_clean/bin/python \
    iris_bbdm_pad/evaluation/compute_dynamic_fusion.py \
    2>&1 | tee -a "$LOG"

# ── Step 1E: Generate IJCB comparison tables ─────────────────────
echo "" | tee -a "$LOG"
echo "[$(date)] Step 1E: Generating IJCB tables..." | tee -a "$LOG"
PYTHONNOUSERSITE=1 /home/teaching/miniconda3/envs/bbdm_clean/bin/python \
    iris_bbdm_pad/evaluation/generate_ijcb_tables.py \
    2>&1 | tee -a "$LOG"

# ── Validation ────────────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "[$(date)] Validating outputs..." | tee -a "$LOG"
PYTHONNOUSERSITE=1 /home/teaching/miniconda3/envs/bbdm_clean/bin/python - <<'PYEOF' 2>&1 | tee -a "$LOG"
import pandas as pd
from pathlib import Path

files = {
    'IJCB_paper_requirements/scoring/vit_scores_test.csv':         (47434, ['filename','label','attack_type','vit_score']),
    'IJCB_paper_requirements/scoring/vit_scores_val.csv':          (21380, ['filename','label','attack_type','vit_score']),
    'IJCB_paper_requirements/scoring/vit_metrics_test.csv':        (9,     ['Attack','APCER↓','BPCER↓','ACER↓','Acc↑','τ']),
    'IJCB_paper_requirements/scoring/dynamic_fusion_metrics_test.csv': (9, ['Attack','APCER↓','BPCER↓','ACER↓','Acc↑','τ']),
    'IJCB_paper_requirements/scoring/fixed_fusion_metrics_test.csv':   (9, ['Attack','APCER↓','BPCER↓','ACER↓','Acc↑','τ']),
    'IJCB_paper_requirements/scoring/all_scores_test.csv':          (47434, ['filename','label','attack_type','lpips_score','vit_score','lpips_norm','vit_norm','fixed_score','dynamic_score']),
    'IJCB_paper_requirements/tables/table_scoring_comparison.csv':  (9,     ['Attack']),
    'IJCB_paper_requirements/tables/table_ablation_scoring.csv':    (36,    ['Method','Attack']),
}

all_ok = True
for fpath, (expected_rows, req_cols) in files.items():
    p = Path(fpath)
    if not p.exists():
        print(f'MISSING: {fpath}')
        all_ok = False
        continue
    df = pd.read_csv(fpath)
    row_ok  = len(df) == expected_rows
    col_ok  = all(c in df.columns for c in req_cols)
    status  = 'OK' if (row_ok and col_ok) else 'WARN'
    print(f'{status}: {fpath}  ({len(df)} rows, cols={df.columns.tolist()[:4]}...)')
    if not row_ok:
        print(f'  Expected {expected_rows} rows, got {len(df)}')
    if not col_ok:
        missing = [c for c in req_cols if c not in df.columns]
        print(f'  Missing columns: {missing}')

# List table PNGs
import os
pngs = list(Path('IJCB_paper_requirements/tables').glob('*.png'))
print(f'\nTable PNGs: {len(pngs)}')
for p in sorted(pngs):
    size_kb = p.stat().st_size // 1024
    print(f'  {p.name}  ({size_kb} KB)')

print('\nAll outputs validated.' if all_ok else '\nSome outputs need attention.')
PYEOF

echo "" | tee -a "$LOG"
echo "[$(date)] Task 1 downstream pipeline complete." | tee -a "$LOG"
echo "Output root: IJCB_paper_requirements/" | tee -a "$LOG"
