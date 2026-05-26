"""Script to programmatically create 01_data_exploration.ipynb.

Run once to generate the notebook:
    python iris_bbdm_pad/notebooks/create_exploration_notebook.py
"""
import nbformat as nbf
from pathlib import Path


def make_md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def make_code(code: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(code)


nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}

nb.cells = [

    # -----------------------------------------------------------------------
    # Title
    # -----------------------------------------------------------------------
    make_md("""# Phase 1 Data Exploration — Iris PAD BBDM

Interactive notebook for validating Phase 1 pipeline outputs.
Run cells top-to-bottom after completing all three preprocessing steps.

> **Purpose**: Inspection only — does NOT run the pipeline.
"""),

    # -----------------------------------------------------------------------
    # Section 1: Setup & Imports
    # -----------------------------------------------------------------------
    make_md("## Section 1: Setup & Imports"),

    make_code("""\
import sys
sys.path.insert(0, '../..')   # project root
sys.path.insert(0, '../../iris_bbdm_pad/data')  # for corruption.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
import csv
import json
import random
from pathlib import Path
from PIL import Image

%matplotlib inline
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 100

# Paths (relative to project root)
PROJECT_ROOT = Path('../..')
PREPROCESSED_DIR = PROJECT_ROOT / 'iris_bbdm_pad/data/preprocessed'
PAIRS_DIR = PROJECT_ROOT / 'iris_bbdm_pad/data/bonafide_pairs'
EVAL_DIR = PROJECT_ROOT / 'iris_bbdm_pad/data/evaluation_sets'
VIS_DIR = PROJECT_ROOT / 'iris_bbdm_pad/results/phase1_visualizations'

print('Project root:', PROJECT_ROOT.resolve())
print('Preprocessed:', PREPROCESSED_DIR.exists())
print('Pairs:', PAIRS_DIR.exists())
print('Eval:', EVAL_DIR.exists())
"""),

    # -----------------------------------------------------------------------
    # Section 2: Preprocessing Results Overview
    # -----------------------------------------------------------------------
    make_md("""## Section 2: Preprocessing Results Overview

Load `preprocessed_metadata.csv` and summarize counts and segmentation success rates.
**Look for**: High Hough success rate (>80%), sensible split distributions.
"""),

    make_code("""\
meta_csv = PREPROCESSED_DIR / 'preprocessed_metadata.csv'
assert meta_csv.exists(), f"Missing: {meta_csv}"

df = pd.read_csv(meta_csv)
print(f"Total images processed: {len(df)}")
print(f"Successful:             {df['seg_success'].notna().sum()} (where success column is present)")
print()
print("Per-split counts:")
print(df.groupby('split').size().rename('count').to_string())
print()
print("Per-class counts:")
print(df.groupby('label').size().rename('count').to_string())
print()
print("Segmentation success rate:")
success_mask = df['seg_success'].astype(str).str.lower().isin(['true', '1'])
print(f"  Hough success : {success_mask.sum()} ({100*success_mask.mean():.1f}%)")
print(f"  Center-crop   : {(~success_mask).sum()} ({100*(~success_mask).mean():.1f}%)")
"""),

    make_code("""\
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Split × Class bar chart
split_class = df.groupby(['split', 'label']).size().unstack(fill_value=0)
split_class.plot(kind='bar', ax=axes[0], colormap='tab10')
axes[0].set_title('Image Counts by Split × Class', fontsize=12)
axes[0].set_xlabel('Split')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=0)
axes[0].legend(title='Label')

# Attack type bar chart (top 15)
attack_counts = df.groupby('attack_type').size().sort_values(ascending=False)
attack_counts.head(15).plot(kind='bar', ax=axes[1], color='steelblue')
axes[1].set_title('Image Counts by Attack Type', fontsize=12)
axes[1].set_xlabel('Attack Type')
axes[1].set_ylabel('Count')
axes[1].tick_params(axis='x', rotation=45, ha='right')

plt.tight_layout()
plt.show()
print("Bar charts displayed.")
"""),

    # -----------------------------------------------------------------------
    # Section 3: Visual Inspection
    # -----------------------------------------------------------------------
    make_md("""## Section 3: Visual Inspection of Preprocessed Images

Display 3×4 grid of random samples from different categories.
**Look for**: Clear iris structures, correct cropping, 256×256 RGB images.
"""),

    make_code("""\
# Build per-category samples
categories = df['attack_type'].dropna().unique()
print(f"Attack types in dataset: {list(categories)}")

# Sample a few from each category
samples = []
for cat in sorted(categories):
    cat_rows = df[(df['attack_type'] == cat) & (df['seg_success'].astype(str).str.lower().isin(['true', '1']))]
    if len(cat_rows) > 0:
        row = cat_rows.sample(1).iloc[0]
        samples.append(row)

# Show up to 12
n_show = min(12, len(samples))
n_rows, n_cols = 3, 4
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 12))
for idx, ax in enumerate(axes.flat):
    if idx < n_show:
        row = samples[idx]
        try:
            img = Image.open(row['preprocessed_path'])
            ax.imshow(img)
            ax.set_title(f"{row['attack_type']}\\n{row['label']}\\n{Path(row['preprocessed_path']).name[:18]}", fontsize=7)
        except Exception as e:
            ax.set_title(f"Error: {e}", fontsize=6)
    ax.axis('off')

plt.suptitle('Preprocessed Iris Samples by Category', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
"""),

    # -----------------------------------------------------------------------
    # Section 4: Corruption Quality Check
    # -----------------------------------------------------------------------
    make_md("""## Section 4: Corruption Quality Check

Load 5 random pairs from `bonafide_pairs/train/` and display side-by-side.
**Look for**: Visible corruption in A (noise, blur, resolution loss), clean B.
MSE should be in the range printed by pair_histogram.png.
"""),

    make_code("""\
from corruption import apply_corruption_pipeline, filename_to_seed

a_dir = PAIRS_DIR / 'train' / 'A'
b_dir = PAIRS_DIR / 'train' / 'B'

if a_dir.exists():
    a_files = sorted(a_dir.glob('*.png'))
    samples = random.sample(a_files, min(5, len(a_files)))

    fig, axes = plt.subplots(len(samples), 3, figsize=(12, 4 * len(samples)))
    if len(samples) == 1:
        axes = axes[np.newaxis, :]
    axes[0, 0].set_title('Corrupted (A)', fontweight='bold')
    axes[0, 1].set_title('Clean (B)', fontweight='bold')
    axes[0, 2].set_title('|A - B|', fontweight='bold')

    for row, a_path in enumerate(samples):
        b_path = b_dir / a_path.name
        try:
            a_img = np.array(Image.open(a_path))
            b_img = np.array(Image.open(b_path))
            diff = np.abs(a_img.astype(float) - b_img.astype(float))
            mse = float(np.mean(diff**2))

            axes[row, 0].imshow(a_img)
            axes[row, 1].imshow(b_img)
            axes[row, 2].imshow((diff / diff.max() * 255).astype(np.uint8))
            axes[row, 0].set_ylabel(f'MSE={mse:.1f}', fontsize=8, rotation=0, ha='right', va='center')
        except Exception as e:
            axes[row, 0].set_title(f'Error: {e}', fontsize=6)
        for col in range(3):
            axes[row, col].axis('off')

    plt.suptitle('Noisy-to-Clean Pair Quality Check', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()
else:
    print(f"Pairs directory not found: {a_dir}")
    print("Run prepare_bonafide_pairs.py first.")
"""),

    make_code("""\
# Load and display the pre-saved corruption_samples.png for quick reference
vis_path = PAIRS_DIR / 'corruption_samples.png'
if vis_path.exists():
    img = Image.open(vis_path)
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(img)
    ax.axis('off')
    ax.set_title('Pre-saved Corruption Samples', fontsize=12)
    plt.tight_layout()
    plt.show()
else:
    print(f"corruption_samples.png not found at {vis_path}")
"""),

    # -----------------------------------------------------------------------
    # Section 5: Corruption Parameter Distribution
    # -----------------------------------------------------------------------
    make_md("""## Section 5: Corruption Parameter Distribution

For 100 random filenames, sample the deterministic corruption parameters.
**Look for**: Uniform coverage of parameter ranges (histograms should be roughly flat).
"""),

    make_code("""\
from corruption import filename_to_seed, sample_corruption_params, get_default_corruption_config

cfg = get_default_corruption_config()
print("Corruption parameter ranges:")
for k, v in cfg.items():
    print(f"  {k}: {v}")

# Generate 100 synthetic filenames
np.random.seed(42)
random.seed(42)
filenames = [f"image_{i:05d}" for i in range(100)]

params_list = []
for fn in filenames:
    seed = filename_to_seed(fn)
    params = sample_corruption_params(seed, cfg)
    params_list.append(params)

params_df = pd.DataFrame(params_list)
print("\\nParameter statistics:")
print(params_df.describe().round(3))

fig, axes = plt.subplots(1, 4, figsize=(18, 4))
axes[0].hist(params_df['noise_sigma'], bins=20, edgecolor='black', color='steelblue')
axes[0].set_title('Noise Sigma Distribution')
axes[0].set_xlabel('sigma')
axes[1].hist(params_df['blur_sigma'], bins=20, edgecolor='black', color='coral')
axes[1].set_title('Blur Sigma Distribution')
axes[1].set_xlabel('sigma')
axes[2].hist(params_df['scale_factor'], bins=20, edgecolor='black', color='seagreen')
axes[2].set_title('Scale Factor Distribution')
axes[2].set_xlabel('scale factor')
axes[3].bar(*np.unique(params_df['blur_kernel'], return_counts=True), color='gold', edgecolor='black')
axes[3].set_title('Blur Kernel Distribution')
axes[3].set_xlabel('kernel size')
plt.tight_layout()
plt.show()
"""),

    # -----------------------------------------------------------------------
    # Section 6: Test/Val Set Distribution
    # -----------------------------------------------------------------------
    make_md("""## Section 6: Test/Val Set Distribution

Check class balance and attack type distribution in evaluation sets.
**Look for**: Both bonafide and attack classes present; all expected attack types present.
"""),

    make_code("""\
for split in ['test', 'val']:
    csv_path = EVAL_DIR / split / 'labels.csv'
    if not csv_path.exists():
        print(f"labels.csv not found for {split} — run prepare_test_dataset.py first.")
        continue

    df_split = pd.read_csv(csv_path)
    print(f"\\n{'='*50}")
    print(f"  {split.upper()} SET — {len(df_split)} images")
    print(f"{'='*50}")
    print(df_split['label'].value_counts().to_string())
    print("\\nPer attack type:")
    print(df_split['attack_type'].value_counts().to_string())

    # Verify no malformed labels
    valid_labels = {'bonafide', 'attack'}
    bad = df_split[~df_split['label'].isin(valid_labels)]
    if len(bad):
        print(f"WARNING: {len(bad)} rows with invalid labels: {bad['label'].unique()}")
    else:
        print("✓ All labels valid (bonafide/attack)")
"""),

    # -----------------------------------------------------------------------
    # Section 7: Dataset Class Smoke Test
    # -----------------------------------------------------------------------
    make_md("""## Section 7: Dataset Class Smoke Test

Verify tensor shapes, ranges, and dtypes from both dataset classes.
**Look for**: Shape (3, 256, 256), range approx [-1, 1], dtype float32.
"""),

    make_code("""\
from iris_bbdm_pad.data.iris_dataset import IrisBonafidePairDataset, IrisTestDataset

print("--- IrisBonafidePairDataset ---")
if (PAIRS_DIR / 'train' / 'A').exists():
    ds_train = IrisBonafidePairDataset(PAIRS_DIR, split='train')
    ds_val = IrisBonafidePairDataset(PAIRS_DIR, split='val')
    s = ds_train[0]
    print(f"  Train pairs : {len(ds_train)}")
    print(f"  Val pairs   : {len(ds_val)}")
    print(f"  A shape     : {s['A'].shape}  dtype={s['A'].dtype}")
    print(f"  B shape     : {s['B'].shape}  dtype={s['B'].dtype}")
    print(f"  A range     : [{s['A'].min():.3f}, {s['A'].max():.3f}]")
    print(f"  B range     : [{s['B'].min():.3f}, {s['B'].max():.3f}]")
    assert s['A'].shape == (3, 256, 256)
    assert -1.1 <= s['A'].min() <= 1.1
    print("  ✓ OK")
else:
    print("  Pairs not found — run prepare_bonafide_pairs.py first.")

print()
print("--- IrisTestDataset ---")
if (EVAL_DIR / 'test' / 'images').exists():
    ds_test = IrisTestDataset(EVAL_DIR, split='test')
    s2 = ds_test[0]
    print(f"  Test images : {len(ds_test)}")
    print(f"  A shape     : {s2['A'].shape}  dtype={s2['A'].dtype}")
    print(f"  B shape     : {s2['B'].shape}  dtype={s2['B'].dtype}")
    print(f"  label       : {s2['label']}  attack_type: {s2['attack_type']}")
    assert s2['A'].shape == (3, 256, 256)
    print("  ✓ OK")
else:
    print("  Eval dir not found — run prepare_test_dataset.py first.")
"""),

    make_code("""\
# Visualize 4 training samples (denormalize from [-1,1] to [0,1])
def denorm(t):
    return ((t * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)

if (PAIRS_DIR / 'train' / 'A').exists():
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes[0, 0].set_ylabel('Corrupted (A)', fontsize=10, rotation=0, ha='right', va='center')
    axes[1, 0].set_ylabel('Clean (B)', fontsize=10, rotation=0, ha='right', va='center')

    for col in range(4):
        s = ds_train[col * len(ds_train) // 4]
        axes[0, col].imshow(denorm(s['A']))
        axes[1, col].imshow(denorm(s['B']))
        for row in range(2):
            axes[row, col].axis('off')

    plt.suptitle('Training Sample Batch (4 pairs, denormalized)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()
"""),

    # -----------------------------------------------------------------------
    # Section 8: BBDM Config Validation
    # -----------------------------------------------------------------------
    make_md("""## Section 8: BBDM Config Validation

Validate the YAML config loads cleanly with `yaml.safe_load()` (no `!!python/tuple`).
**Look for**: Clean load, correct paths, n_steps ≥ 400000.
"""),

    make_code("""\
import yaml

config_path = PROJECT_ROOT / 'iris_bbdm_pad/configs/bbdm_iris_bonafide.yaml'
assert config_path.exists(), f"Config not found: {config_path}"

txt = config_path.read_text()

# Verify no !!python/tuple (only check non-comment lines)
bad_lines = [l for l in txt.splitlines() if '!!python/tuple' in l and not l.strip().startswith('#')]
assert not bad_lines, f"FAIL: !!python/tuple found: {bad_lines}"
print("✓ No !!python/tuple tags")

cfg = yaml.safe_load(txt)
print("✓ yaml.safe_load() successful")

print()
print("Key config values:")
print(f"  runner           : {cfg['runner']}")
print(f"  n_epochs         : {cfg['training']['n_epochs']}")
print(f"  n_steps          : {cfg['training']['n_steps']}  (must be >= 400000)")
print(f"  dataset_type     : {cfg['data']['dataset_type']}")
print(f"  dataset_path     : {cfg['data']['dataset_config']['dataset_path']}")
print(f"  image_size       : {cfg['data']['dataset_config']['image_size']}")
print(f"  batch_size(train): {cfg['data']['train']['batch_size']}")
print(f"  VQGAN ckpt       : {cfg['model']['VQGAN']['params']['ckpt_path']}")
print(f"  channel_mult     : {cfg['model']['BB']['params']['UNetParams']['channel_mult']}")
print(f"  channel_mult type: {type(cfg['model']['BB']['params']['UNetParams']['channel_mult']).__name__}")

# Verify paths exist
vqgan_path = PROJECT_ROOT / cfg['model']['VQGAN']['params']['ckpt_path']
dataset_path = PROJECT_ROOT / cfg['data']['dataset_config']['dataset_path']
print()
print(f"VQGAN checkpoint exists : {vqgan_path.exists()} ({vqgan_path})")
print(f"Dataset path exists     : {dataset_path.exists()} ({dataset_path})")

assert cfg['training']['n_steps'] >= 400000, f"n_steps too low: {cfg['training']['n_steps']}"
print()
print("✓ All config validations passed")
"""),
]

# Write notebook
out_path = Path(__file__).parent / '01_data_exploration.ipynb'
with open(out_path, 'w') as f:
    nbf.write(nb, f)

print(f"Notebook written to: {out_path}")
print(f"Cells: {len(nb.cells)}")
