#!/bin/bash
# Full Task 1 pipeline in a persistent tmux session.
# Run once from any terminal:
#   bash run_task1_tmux.sh
#
# Detach from tmux at any time with Ctrl+B, D
# Reattach later with:  tmux attach -t task1

SESSION="task1"
PROJ="/home/teaching/Documents/Geetanjali_PhD_IRIS_PAD"
PYTHON="PYTHONNOUSERSITE=1 /home/teaching/miniconda3/envs/bbdm_clean/bin/python"

# Kill any stale session with the same name
tmux kill-session -t "$SESSION" 2>/dev/null || true

tmux new-session -d -s "$SESSION" -x 220 -y 50

tmux send-keys -t "$SESSION" "cd $PROJ" Enter
tmux send-keys -t "$SESSION" "echo '=== Task 1 pipeline started at '$(date) ===" Enter

# ── Step 1B-test: resume test-split ViT scoring ───────────────────
tmux send-keys -t "$SESSION" "
echo '' && echo '[STEP 1] ViT scoring — test split' &&
$PYTHON iris_bbdm_pad/evaluation/run_vit_scoring.py \
    --split test \
    --output_csv IJCB_paper_requirements/scoring/vit_scores_test.csv \
    --batch_size 32 --num_steps 50
" Enter

# ── Step 1B-val: val-split ViT scoring ───────────────────────────
tmux send-keys -t "$SESSION" "
echo '' && echo '[STEP 2] ViT scoring — val split' &&
$PYTHON iris_bbdm_pad/evaluation/run_vit_scoring.py \
    --split val \
    --output_csv IJCB_paper_requirements/scoring/vit_scores_val.csv \
    --batch_size 32 --num_steps 50
" Enter

# ── Step 1C: ViT-only threshold + per-attack metrics ─────────────
tmux send-keys -t "$SESSION" "
echo '' && echo '[STEP 3] ViT-only metrics (threshold + per-attack)' &&
$PYTHON iris_bbdm_pad/evaluation/compute_vit_metrics.py
" Enter

# ── Step 1D: Dynamic fusion ───────────────────────────────────────
tmux send-keys -t "$SESSION" "
echo '' && echo '[STEP 4] Dynamic fusion (LPIPS + ViT)' &&
$PYTHON iris_bbdm_pad/evaluation/compute_dynamic_fusion.py
" Enter

# ── Step 1E: IJCB comparison tables ──────────────────────────────
tmux send-keys -t "$SESSION" "
echo '' && echo '[STEP 5] Generating IJCB comparison tables' &&
$PYTHON iris_bbdm_pad/evaluation/generate_ijcb_tables.py
" Enter

# ── Final validation ──────────────────────────────────────────────
tmux send-keys -t "$SESSION" "
echo '' && echo '[STEP 6] Validating outputs' &&
$PYTHON - <<'PYEOF'
import pandas as pd
from pathlib import Path

checks = [
    ('IJCB_paper_requirements/scoring/vit_scores_test.csv',            47434),
    ('IJCB_paper_requirements/scoring/vit_scores_val.csv',             21380),
    ('IJCB_paper_requirements/scoring/vit_metrics_test.csv',               9),
    ('IJCB_paper_requirements/scoring/fixed_fusion_metrics_test.csv',      9),
    ('IJCB_paper_requirements/scoring/dynamic_fusion_metrics_test.csv',    9),
    ('IJCB_paper_requirements/scoring/all_scores_test.csv',            47434),
    ('IJCB_paper_requirements/tables/table_scoring_comparison.csv',        9),
    ('IJCB_paper_requirements/tables/table_ablation_scoring.csv',         36),
]
all_ok = True
for fpath, exp in checks:
    p = Path(fpath)
    if not p.exists():
        print(f'  MISSING : {fpath}'); all_ok = False; continue
    n = len(pd.read_csv(fpath))
    ok = abs(n - exp) <= 10
    print(f'  {\"OK\" if ok else \"WARN\"} : {fpath}  ({n} rows, expected ~{exp})')
    if not ok: all_ok = False

pngs = sorted(Path('IJCB_paper_requirements/tables').glob('*.png'))
print(f'\n  Table PNGs ({len(pngs)}):')
for p in pngs:
    print(f'    {p.name}  ({p.stat().st_size//1024} KB)')
print('\nAll done.' if all_ok else '\nSome outputs need attention.')
PYEOF
" Enter

tmux send-keys -t "$SESSION" "echo '' && echo '=== Task 1 complete at '$(date) '===' " Enter

# Attach so user can see it running
tmux attach -t "$SESSION"
