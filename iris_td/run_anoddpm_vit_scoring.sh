#!/usr/bin/env bash
# iris_td/run_anoddpm_vit_scoring.sh
#
# Computes ViT scores for AnoDDPM+DDIM (Simplex, t*=500, 50 DDIM steps)
# on val and test sets, then runs the scoring ablation (ViT / Fixed / Dynamic).
#
# Estimated time: ~1.5 hrs (val) + ~3.5 hrs (test) + minutes (fusion CPU)
#
# Run in tmux:
#   cd /home/teaching/Documents/Geetanjali_PhD_IRIS_PAD
#   tmux new -s anoddpm_vit
#   bash iris_td/run_anoddpm_vit_scoring.sh 2>&1 | tee iris_td/final_results/anoddpm_vit_run.log

set -eo pipefail
cd /home/teaching/Documents/Geetanjali_PhD_IRIS_PAD
export PYTHONNOUSERSITE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
PY=/home/teaching/miniconda3/envs/bbdm_clean/bin/python
PYTHONPATH=BBDM:$PYTHONPATH

ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "========================================================"
echo "AnoDDPM+DDIM ViT Scoring — Simplex t*=500, 50 DDIM steps"
echo "Started: $(ts)"
echo "========================================================"

# ── Step 1: Val ViT scoring ───────────────────────────────────
echo ""
echo "▶ [$(ts)] STEP 1: Val ViT scoring (50 DDIM steps)"
PYTHONPATH=BBDM:$PYTHONPATH $PY iris_td/models/ddpm_vit_scorer.py \
    --split     val \
    --t_star    500 \
    --num_steps 50 \
    --batch_size 32 \
    --output_csv iris_td/pad_scores/ddpm_val_simplex_tstar500_ddim_steps50_vit.csv
echo "✔ [$(ts)] Val ViT done"

# ── Step 2: Test ViT scoring ──────────────────────────────────
echo ""
echo "▶ [$(ts)] STEP 2: Test ViT scoring (50 DDIM steps)"
PYTHONPATH=BBDM:$PYTHONPATH $PY iris_td/models/ddpm_vit_scorer.py \
    --split     test \
    --t_star    500 \
    --num_steps 50 \
    --batch_size 32 \
    --output_csv iris_td/pad_scores/ddpm_test_simplex_tstar500_ddim_steps50_vit.csv
echo "✔ [$(ts)] Test ViT done"

# ── Step 3: Verify output CSVs ────────────────────────────────
echo ""
echo "▶ [$(ts)] STEP 3: Verifying output CSVs"
PYTHONPATH=BBDM:$PYTHONPATH $PY - <<'PYEOF'
import pandas as pd, sys
for split, path in [
    ("val",  "iris_td/pad_scores/ddpm_val_simplex_tstar500_ddim_steps50_vit.csv"),
    ("test", "iris_td/pad_scores/ddpm_test_simplex_tstar500_ddim_steps50_vit.csv"),
]:
    df = pd.read_csv(path)
    nans = df["vit_score"].isna().sum()
    bf   = (df["label"] == "bonafide").sum()
    atk  = (df["label"] != "bonafide").sum()
    gap  = df[df["label"]!="bonafide"]["vit_score"].mean() - df[df["label"]=="bonafide"]["vit_score"].mean()
    ok   = "OK" if nans == 0 and gap > 0 else "FAIL"
    print(f"[{ok}] {split}: {len(df)} rows | bonafide={bf} attack={atk} | NaNs={nans} | gap={gap:+.4f}")
    if ok == "FAIL":
        sys.exit("CSV verification failed — check log above")
print("All CSVs verified OK")
PYEOF
echo "✔ [$(ts)] Verification done"

# ── Step 4: Fusion ablation (CPU, fast) ───────────────────────
echo ""
echo "▶ [$(ts)] STEP 4: Fusion ablation (ViT / Fixed / Dynamic)"
PYTHONPATH=BBDM:$PYTHONPATH $PY iris_td/evaluation/generate_anoddpm_scoring_ablation.py
echo "✔ [$(ts)] Ablation done"

echo ""
echo "========================================================"
echo "ALL DONE: $(ts)"
echo "Results: iris_td/final_results/anoddpm_scoring_ablation_per_attack.csv"
echo "========================================================"
