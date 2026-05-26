"""
Script to programmatically create the 02_training_monitor.ipynb notebook.
Run this script once to generate the notebook file.
"""
import json
from pathlib import Path

def make_code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source if isinstance(source, list) else [source],
    }

def make_markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source if isinstance(source, list) else [source],
    }

cells = []

# ── Section 1: Setup ──────────────────────────────────────────────────────
cells.append(make_markdown_cell("# Phase 2: BBDM Training Monitor & PAD Score Analysis\n\nInteractive notebook for monitoring BBDM training progress and exploring PAD scores."))

cells.append(make_code_cell([
    "import sys, json, warnings\n",
    "from pathlib import Path\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib.cm as cm\n",
    "from PIL import Image\n",
    "warnings.filterwarnings('ignore')\n",
    "%matplotlib inline\n",
    "plt.rcParams['figure.dpi'] = 100\n",
    "\n",
    "PROJECT_ROOT = Path('..').resolve().parent  # Geetanjali_PhD_IRIS_PAD/\n",
    "RESULTS_DIR  = PROJECT_ROOT / 'iris_bbdm_pad' / 'results'\n",
    "BBDM_RESULTS = PROJECT_ROOT / 'results'\n",
    "DATA_DIR     = PROJECT_ROOT / 'iris_bbdm_pad' / 'data'\n",
    "print(f'Project root: {PROJECT_ROOT}')\n",
    "print(f'BBDM results: {BBDM_RESULTS}')\n",
]))

# ── Section 2: Training Progress ──────────────────────────────────────────
cells.append(make_markdown_cell("## Section 2: Training Progress\n\nParse BBDM training logs and visualize loss curve."))

cells.append(make_code_cell([
    "# Find the BBDM log directory\n",
    "log_dirs = list(BBDM_RESULTS.glob('*/LBBDM-f4/log'))\n",
    "if not log_dirs:\n",
    "    print('No log directory found. Training may not have started yet.')\n",
    "else:\n",
    "    log_dir = log_dirs[-1]\n",
    "    print(f'Log directory: {log_dir}')\n",
    "    json_logs = sorted(log_dir.glob('*.json'))\n",
    "    print(f'Found {len(json_logs)} JSON log files')\n",
]))

cells.append(make_code_cell([
    "# Parse training logs from TensorBoard event files or JSON\n",
    "try:\n",
    "    from torch.utils.tensorboard import SummaryWriter\n",
    "    from tensorboard.backend.event_processing import event_accumulator\n",
    "    ea = event_accumulator.EventAccumulator(str(log_dir))\n",
    "    ea.Reload()\n",
    "    tags = ea.Tags()['scalars']\n",
    "    print('Available scalar tags:', tags)\n",
    "    if 'loss/train' in tags:\n",
    "        events = ea.Scalars('loss/train')\n",
    "        steps = [e.step for e in events]\n",
    "        values = [e.value for e in events]\n",
    "        fig, ax = plt.subplots(figsize=(12, 4))\n",
    "        ax.plot(steps, values, linewidth=1, color='#2563eb', alpha=0.8)\n",
    "        ax.set_xlabel('Step'); ax.set_ylabel('Loss (L1)')\n",
    "        ax.set_title('BBDM Training Loss'); ax.grid(True, alpha=0.3)\n",
    "        plt.tight_layout(); plt.show()\n",
    "        print(f'Steps: {steps[0]} to {steps[-1]}, Final loss: {values[-1]:.6f}')\n",
    "except Exception as e:\n",
    "    print(f'Could not parse TensorBoard logs: {e}')\n",
    "    print('Training may still be in progress, or logs may be in a different format.')\n",
]))

cells.append(make_code_cell([
    "# Check training checkpoints\n",
    "ckpt_dirs = list(BBDM_RESULTS.glob('*/LBBDM-f4/checkpoint'))\n",
    "if ckpt_dirs:\n",
    "    ckpt_dir = ckpt_dirs[-1]\n",
    "    ckpts = sorted(ckpt_dir.glob('*.pth'))\n",
    "    print(f'Checkpoints in {ckpt_dir}:')\n",
    "    for c in ckpts:\n",
    "        size_mb = c.stat().st_size / 1e6\n",
    "        print(f'  {c.name}: {size_mb:.1f} MB')\n",
    "else:\n",
    "    print('No checkpoints found yet.')\n",
]))

# ── Section 3: Sample Reconstructions ────────────────────────────────────
cells.append(make_markdown_cell("## Section 3: Sample Reconstructions During Training\n\nVisualize how reconstruction quality improves over epochs."))

cells.append(make_code_cell([
    "# Find sample directories saved by BBDM during training\n",
    "sample_dirs = list(BBDM_RESULTS.glob('*/LBBDM-f4/sample/train_sample'))\n",
    "if not sample_dirs:\n",
    "    print('No sample images found. Training may not have started or sampling is disabled.')\n",
    "else:\n",
    "    sample_dir = sample_dirs[-1]\n",
    "    cond = sample_dir / 'condition.png'\n",
    "    recon = sample_dir / 'skip_sample.png'\n",
    "    gt = sample_dir / 'ground_truth.png'\n",
    "    if cond.exists() and recon.exists() and gt.exists():\n",
    "        fig, axes = plt.subplots(1, 3, figsize=(15, 5))\n",
    "        axes[0].imshow(Image.open(cond)); axes[0].set_title('Corrupted Input (A)'); axes[0].axis('off')\n",
    "        axes[1].imshow(Image.open(recon)); axes[1].set_title('BBDM Reconstruction'); axes[1].axis('off')\n",
    "        axes[2].imshow(Image.open(gt)); axes[2].set_title('Clean Target (B)'); axes[2].axis('off')\n",
    "        plt.suptitle('Latest Training Sample', fontsize=14, fontweight='bold')\n",
    "        plt.tight_layout(); plt.show()\n",
]))

# ── Section 4: PAD Score Analysis ────────────────────────────────────────
cells.append(make_markdown_cell("## Section 4: PAD Score Analysis\n\nLoad and analyze validation and test PAD scores."))

cells.append(make_code_cell([
    "val_csv  = RESULTS_DIR / 'val_pad_scores.csv'\n",
    "test_csv = RESULTS_DIR / 'test_pad_scores.csv'\n",
    "dfs = {}\n",
    "for name, p in [('val', val_csv), ('test', test_csv)]:\n",
    "    if p.exists():\n",
    "        dfs[name] = pd.read_csv(p)\n",
    "        print(f'{name}: {len(dfs[name])} rows, {dfs[name][\"label\"].value_counts().to_dict()}')\n",
    "    else:\n",
    "        print(f'{name} scores not found: {p}')\n",
    "df = dfs.get('val', dfs.get('test'))\n",
    "if df is not None: display(df.head())\n",
]))

cells.append(make_code_cell([
    "# Score statistics\n",
    "if df is not None:\n",
    "    score_cols = ['mse_score','lpips_score','recon_score','trajectory_score','combined_score']\n",
    "    available = [c for c in score_cols if c in df.columns]\n",
    "    for cls in ['bonafide','attack']:\n",
    "        subset = df[df['label']==cls]\n",
    "        print(f'\\n--- {cls.upper()} ({len(subset)} samples) ---')\n",
    "        for col in available:\n",
    "            print(f'  {col:25s}: mean={subset[col].mean():.4f}, std={subset[col].std():.4f}, '\n",
    "                  f'min={subset[col].min():.4f}, max={subset[col].max():.4f}')\n",
]))

cells.append(make_code_cell([
    "# Overlaid histograms\n",
    "if df is not None:\n",
    "    score_cols = ['mse_score','lpips_score','recon_score','trajectory_score','combined_score']\n",
    "    available = [c for c in score_cols if c in df.columns]\n",
    "    n = len(available)\n",
    "    fig, axes = plt.subplots(1, n, figsize=(5*n, 4))\n",
    "    if n == 1: axes = [axes]\n",
    "    for ax, col in zip(axes, available):\n",
    "        df[df['label']=='bonafide'][col].hist(bins=40, ax=ax, alpha=0.6, color='#3b82f6', label='Bonafide', density=True)\n",
    "        df[df['label']=='attack'][col].hist(bins=40, ax=ax, alpha=0.6, color='#ef4444', label='Attack', density=True)\n",
    "        ax.set_title(col); ax.legend(); ax.set_xlabel('Score'); ax.set_ylabel('Density')\n",
    "    plt.suptitle('PAD Score Distributions: Bonafide vs Attack', fontsize=13, fontweight='bold')\n",
    "    plt.tight_layout(); plt.show()\n",
]))

cells.append(make_code_cell([
    "# Box plots per attack type\n",
    "if df is not None and 'attack_type' in df.columns:\n",
    "    col = 'recon_score' if 'recon_score' in df.columns else df.columns[-1]\n",
    "    order = sorted(df['attack_type'].unique())\n",
    "    fig, ax = plt.subplots(figsize=(14, 5))\n",
    "    data_by_type = [df[df['attack_type']==t][col].values for t in order]\n",
    "    bp = ax.boxplot(data_by_type, labels=order, patch_artist=True)\n",
    "    for patch, t in zip(bp['boxes'], order):\n",
    "        patch.set_facecolor('#3b82f6' if t in ('Live','bonafide') else '#ef4444')\n",
    "        patch.set_alpha(0.6)\n",
    "    ax.set_xlabel('Attack Type'); ax.set_ylabel(col)\n",
    "    ax.set_title(f'{col} by Attack Type'); ax.tick_params(axis='x', rotation=30)\n",
    "    plt.tight_layout(); plt.show()\n",
]))

# ── Section 5: Threshold Analysis ────────────────────────────────────────
cells.append(make_markdown_cell("## Section 5: Threshold Analysis"))

cells.append(make_code_cell([
    "thr_path = RESULTS_DIR / 'threshold.json'\n",
    "if thr_path.exists():\n",
    "    thr_data = json.loads(thr_path.read_text())\n",
    "    print(f'Best method:    {thr_data[\"best_method\"]}')\n",
    "    print(f'Best threshold: {thr_data[\"best_threshold\"]:.4f}')\n",
    "    print(f'Best ACER:      {thr_data[\"best_acer\"]*100:.2f}%')\n",
    "    print('\\nPer-method summary:')\n",
    "    for m, v in thr_data['per_method'].items():\n",
    "        print(f'  {m:30s}: ACER={v[\"acer\"]*100:.2f}%  EER={v[\"eer\"]*100:.2f}%  thr={v[\"min_acer_threshold\"]:.4f}')\n",
    "else:\n",
    "    print('threshold.json not found. Run find_threshold.py first.')\n",
]))

cells.append(make_code_cell([
    "# Confusion matrix at selected threshold\n",
    "if df is not None and thr_path.exists():\n",
    "    import numpy as np\n",
    "    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay\n",
    "    best_method = thr_data['best_method']\n",
    "    col = best_method if best_method in df.columns else 'recon_score'\n",
    "    scores = df[col].values\n",
    "    labels_bin = (df['label'] == 'attack').astype(int).values\n",
    "    threshold = thr_data['best_threshold']\n",
    "    preds = (scores > threshold).astype(int)\n",
    "    cm = confusion_matrix(labels_bin, preds)\n",
    "    disp = ConfusionMatrixDisplay(cm, display_labels=['Bonafide','Attack'])\n",
    "    fig, ax = plt.subplots(figsize=(5, 4))\n",
    "    disp.plot(ax=ax, colorbar=False, cmap='Blues')\n",
    "    ax.set_title(f'Confusion Matrix @ threshold={threshold:.4f}'); plt.tight_layout(); plt.show()\n",
]))

# ── Section 6: Per-Attack-Type Breakdown ─────────────────────────────────
cells.append(make_markdown_cell("## Section 6: Per-Attack-Type Breakdown\n\nDetection rate per attack type at optimal threshold."))

cells.append(make_code_cell([
    "if df is not None and thr_path.exists():\n",
    "    import numpy as np\n",
    "    col = thr_data['best_method'] if thr_data['best_method'] in df.columns else 'recon_score'\n",
    "    thr = thr_data['best_threshold']\n",
    "    results = {}\n",
    "    for atype in df['attack_type'].unique():\n",
    "        sub = df[df['attack_type']==atype]\n",
    "        preds = (sub[col].values > thr).astype(int)\n",
    "        true_atk = (sub['label'] == 'attack').astype(int).values\n",
    "        det_rate = preds[true_atk==1].mean() if true_atk.sum() > 0 else np.nan\n",
    "        fa_rate  = preds[true_atk==0].mean() if (true_atk==0).sum() > 0 else np.nan\n",
    "        results[atype] = {'detection_rate': det_rate, 'false_alarm': fa_rate, 'n': len(sub)}\n",
    "    res_df = pd.DataFrame(results).T.sort_values('detection_rate', ascending=False)\n",
    "    display(res_df)\n",
    "    ax = res_df['detection_rate'].plot(kind='bar', figsize=(12,4), color='#3b82f6', alpha=0.7)\n",
    "    ax.set_title('Detection Rate per Attack Type'); ax.set_ylabel('Detection Rate')\n",
    "    ax.axhline(0.5, color='r', linestyle='--', label='50% baseline')\n",
    "    ax.tick_params(axis='x', rotation=30); ax.legend()\n",
    "    plt.tight_layout(); plt.show()\n",
]))

# ── Section 7: Failure Case Analysis ─────────────────────────────────────
cells.append(make_markdown_cell("## Section 7: Failure Case Analysis\n\nTop 10 false positives and false negatives at optimal threshold."))

cells.append(make_code_cell([
    "if df is not None and thr_path.exists():\n",
    "    col = thr_data['best_method'] if thr_data['best_method'] in df.columns else 'recon_score'\n",
    "    thr = thr_data['best_threshold']\n",
    "    df['pred'] = (df[col] > thr).astype(int)\n",
    "    df['is_attack'] = (df['label'] == 'attack').astype(int)\n",
    "    # False positives: bonafide scored as attack\n",
    "    fp = df[(df['pred']==1) & (df['is_attack']==0)].sort_values(col, ascending=False).head(10)\n",
    "    print(f'Top 10 False Positives (bonafide -> attack), score col: {col}')\n",
    "    display(fp[['filename','attack_type',col,'recon_score']].reset_index(drop=True))\n",
    "    # False negatives: attack scored as bonafide\n",
    "    fn = df[(df['pred']==0) & (df['is_attack']==1)].sort_values(col).head(10)\n",
    "    print(f'\\nTop 10 False Negatives (attack -> bonafide), score col: {col}')\n",
    "    display(fn[['filename','attack_type',col,'recon_score']].reset_index(drop=True))\n",
]))

# ── Section 8: Reconstruction Error Heatmaps ─────────────────────────────
cells.append(make_markdown_cell("## Section 8: Reconstruction Error Heatmaps (Paper Figure)\n\nPixel-wise absolute error between reconstruction and clean original."))

cells.append(make_code_cell([
    "heatmap_img = RESULTS_DIR / 'phase2_visualizations' / 'paper' / 'reconstruction_error_heatmaps.png'\n",
    "if heatmap_img.exists():\n",
    "    fig, ax = plt.subplots(figsize=(16, 12))\n",
    "    ax.imshow(Image.open(heatmap_img)); ax.axis('off')\n",
    "    ax.set_title('Reconstruction Error Heatmaps (Paper Figure)', fontsize=14)\n",
    "    plt.tight_layout(); plt.show()\n",
    "else:\n",
    "    print(f'Heatmap figure not found: {heatmap_img}')\n",
    "    print('Run anomaly_detector.py first to generate paper figures.')\n",
]))

# ── Section 9: Diffusion Trajectory ───────────────────────────────────────
cells.append(make_markdown_cell("## Section 9: Diffusion Trajectory Deep Dive (Paper Figure)"))

cells.append(make_code_cell([
    "traj_img = RESULTS_DIR / 'phase2_visualizations' / 'paper' / 'diffusion_denoising_frames.png'\n",
    "if traj_img.exists():\n",
    "    fig, ax = plt.subplots(figsize=(20, 6))\n",
    "    ax.imshow(Image.open(traj_img)); ax.axis('off')\n",
    "    ax.set_title('Diffusion Denoising Frames: Bonafide vs Attack', fontsize=14)\n",
    "    plt.tight_layout(); plt.show()\n",
    "else:\n",
    "    print(f'Denoising frames figure not found: {traj_img}')\n",
]))

cells.append(make_code_cell([
    "# Plot trajectory MSE curve from scores data\n",
    "traj_var_img = RESULTS_DIR / 'phase2_visualizations' / 'trajectory_variance_by_timestep.png'\n",
    "if traj_var_img.exists():\n",
    "    plt.imshow(Image.open(traj_var_img))\n",
    "    plt.axis('off'); plt.title('Trajectory Variance by Timestep'); plt.tight_layout(); plt.show()\n",
]))

# ── Section 10: Latent Space Analysis ────────────────────────────────────
cells.append(make_markdown_cell("## Section 10: Latent Space Analysis (Paper Figure)\n\nt-SNE and UMAP of VQGAN latent embeddings."))

cells.append(make_code_cell([
    "from sklearn.metrics import silhouette_score, davies_bouldin_score\n",
    "import numpy as np\n",
    "\n",
    "for fig_name in ['tsne_latent_embeddings.png','umap_latent_embeddings.png','tsne_by_attack_type.png']:\n",
    "    p = RESULTS_DIR / 'phase2_visualizations' / 'paper' / fig_name\n",
    "    if p.exists():\n",
    "        fig, ax = plt.subplots(figsize=(8, 7))\n",
    "        ax.imshow(Image.open(p)); ax.axis('off'); ax.set_title(fig_name[:-4])\n",
    "        plt.tight_layout(); plt.show()\n",
    "    else:\n",
    "        print(f'{fig_name} not found yet.')\n",
]))

# ── Section 11: Grad-CAM ──────────────────────────────────────────────────
cells.append(make_markdown_cell("## Section 11: Grad-CAM Interpretation (Paper Figure)\n\nUNet bottleneck attention patterns for bonafide vs attack."))

cells.append(make_code_cell([
    "gradcam_img = RESULTS_DIR / 'phase2_visualizations' / 'paper' / 'gradcam_unet_attention.png'\n",
    "if gradcam_img.exists():\n",
    "    fig, ax = plt.subplots(figsize=(12, 16))\n",
    "    ax.imshow(Image.open(gradcam_img)); ax.axis('off')\n",
    "    ax.set_title('Grad-CAM: UNet Middle Block Attention', fontsize=14)\n",
    "    plt.tight_layout(); plt.show()\n",
    "else:\n",
    "    print(f'Grad-CAM figure not found: {gradcam_img}')\n",
    "    print('Run anomaly_detector.py save_paper_figures() to generate.')\n",
]))

# ── Notebook structure ────────────────────────────────────────────────────
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "iris_pad",
            "language": "python",
            "name": "iris_pad",
        },
        "language_info": {
            "name": "python",
            "version": "3.8.0",
        },
    },
    "cells": cells,
}

out_path = Path(__file__).parent / "02_training_monitor.ipynb"
out_path.write_text(json.dumps(notebook, indent=1))
print(f"Created: {out_path}")
