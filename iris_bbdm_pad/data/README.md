# Phase 1: Data Preprocessing & Preparation

## What This Phase Does
Preprocesses raw iris images (segmentation via Hough transform, normalization to 256×256 RGB PNG),
creates noisy-to-clean training pairs from **bona fide images only** (Spoof images are never used
in training), and prepares flat evaluation datasets with both bona fide and attack images
for PAD scoring.

## Prerequisites
- Raw dataset in `Images/` with `train/`, `val/`, `test/` splits
  - Each split: `Live/` (flat) and `Spoof/{Artifact,CL,E-display,...}/` (subfolders)
  - Formats: `.bmp`, `.jpg`, `.png`, `.tiff` (mixed — all handled)
- BBDM repo cloned in `BBDM/` at project root
- VQGAN checkpoint at `BBDM/resources/vq-f4/model.ckpt` (verify: `ls -la BBDM/resources/vq-f4/model.ckpt`)
- Python dependencies installed:
  ```bash
  pip install opencv-python Pillow tqdm pandas matplotlib seaborn pyyaml scipy lpips pytorch-lightning
  ```

## Steps to Run (in order)

### Step 1: Preprocess iris images
```bash
python iris_bbdm_pad/data/iris_preprocessing.py \
    --input_dir Images/ \
    --output_dir iris_bbdm_pad/data/preprocessed/ \
    --image_size 256 \
    --workers 4
```
**What it does:** Segments each iris (Hough circle detection, center-crop fallback),
resizes to 256×256, converts to RGB PNG, mirrors the directory structure of `Images/`.
Saves progress to `iris_bbdm_pad/checkpoints/preprocessed_checkpoint.json`.

**Expected time:** ~15-30 min for ~14K images with 4 workers.

**Resume support:** Add `--resume` to skip already-processed files.

**Dry run (50 images):**
```bash
python iris_bbdm_pad/data/iris_preprocessing.py \
    --input_dir Images/ --output_dir iris_bbdm_pad/data/preprocessed/ \
    --image_size 256 --workers 4 --max_images 50
```

### Step 2: Create noisy-to-clean training pairs
```bash
python iris_bbdm_pad/data/prepare_bonafide_pairs.py \
    --source_dir iris_bbdm_pad/data/preprocessed/ \
    --output_dir iris_bbdm_pad/data/bonafide_pairs/ \
    --workers 4
```
**What it does:** Reads `preprocessed/{train,val}/Live/` only (Spoof images are
**never** included — an assertion will fail if any Spoof path appears). Creates:
- `bonafide_pairs/{train,val}/A/` — corrupted copies (Gaussian noise + blur + resolution degradation)
- `bonafide_pairs/{train,val}/B/` — clean originals (copies)

Filenames match exactly between A/ and B/ to ensure correct BBDM pairing.
Saves `bonafide_pairs/dataset_config.json` with corruption parameters and statistics.

**Note:** Run from project root so that `from corruption import ...` resolves correctly.
If running from a different directory, set `PYTHONPATH=iris_bbdm_pad/data:$PYTHONPATH`.

### Step 3: Prepare evaluation sets
```bash
python iris_bbdm_pad/data/prepare_test_dataset.py \
    --source_dir iris_bbdm_pad/data/preprocessed/ \
    --output_dir iris_bbdm_pad/data/evaluation_sets/
```
**What it does:** Copies all preprocessed images (Live + all Spoof attack types) from
`test/` and `val/` splits into flat directories. Adds attack-type prefix to filenames
to avoid collisions (e.g., `Printed_img001.png`, `Live_img002.png`). Writes `labels.csv`.

### Step 4: Validate with notebook
```bash
cd iris_bbdm_pad/notebooks
jupyter notebook 01_data_exploration.ipynb
```
Or run all cells from the project root:
```bash
jupyter nbconvert --to notebook --execute iris_bbdm_pad/notebooks/01_data_exploration.ipynb \
    --output iris_bbdm_pad/notebooks/01_data_exploration_executed.ipynb
```

## Expected Outputs

```
iris_bbdm_pad/
├── checkpoints/
│   └── preprocessed_checkpoint.json        # Resume state for preprocessing
├── configs/
│   └── bbdm_iris_bonafide.yaml             # BBDM training config (safe_load compatible)
├── data/
│   ├── preprocessed/
│   │   ├── preprocessed_metadata.csv       # original_path, preprocessed_path, label, attack_type, split, seg_success
│   │   ├── train/
│   │   │   ├── Live/                       # Bona fide PNGs (256×256 RGB)
│   │   │   └── Spoof/{Artifact,CL,...}/    # Attack PNGs (same structure as input)
│   │   ├── val/  (same structure)
│   │   └── test/ (same structure)
│   ├── bonafide_pairs/
│   │   ├── dataset_config.json             # Corruption parameters + pair counts + MSE stats
│   │   ├── corruption_samples.png          # 5 pairs: corrupted | clean | diff
│   │   ├── train/
│   │   │   ├── A/                          # Corrupted bona fide (BBDM condition input)
│   │   │   └── B/                          # Clean bona fide (BBDM target output)
│   │   └── val/  (same structure)
│   └── evaluation_sets/
│       ├── test/
│       │   ├── images/                     # Flat directory: all test images (Live + Spoof)
│       │   └── labels.csv                  # filename, label, attack_type
│       └── val/  (same structure)
└── results/
    └── phase1_visualizations/
        ├── preprocessing_pipeline.png
        ├── segmentation_results.png
        ├── format_comparison.png
        ├── corruption_intensity_spectrum.png
        ├── corruption_per_attack_comparison.png
        ├── pair_histogram.png
        ├── test_set_distribution.png
        └── attack_type_samples.png
```

## Visualizations Generated

| File | What it shows |
|------|---------------|
| `preprocessing_pipeline.png` | 5 rows × 7 cols: full pipeline for 5 raw images (Original → Gray → CLAHE → Hough → Cropped → Resized → RGB) |
| `segmentation_results.png` | 4×5 grid of 20 random preprocessed images, titled with segmentation method (Hough/center-crop) |
| `format_comparison.png` | One image per original format (.bmp, .jpg, .png, .tiff) before and after |
| `corruption_intensity_spectrum.png` | One image corrupted at 5 intensities (very mild → very strong) |
| `corruption_per_attack_comparison.png` | 2×4 grid: clean (top) vs corrupted (bottom) for 4 bona fide images |
| `pair_histogram.png` | Histogram of MSE values across all A/B pairs (shows corruption magnitude distribution) |
| `test_set_distribution.png` | Horizontal bars: sample count per attack type for test and val |
| `attack_type_samples.png` | 3×3 grid: one sample per attack type + bona fide |

All visualizations saved at 300 DPI as PNG.

## How to Verify Success

```bash
# 1. Check preprocessed images exist and are correct format
python3 -c "
from PIL import Image; from pathlib import Path; import numpy as np
pngs = list(Path('iris_bbdm_pad/data/preprocessed').rglob('*.png'))
print(f'Preprocessed PNGs: {len(pngs)}')
for p in pngs[:10]:
    img = Image.open(p)
    assert img.size == (256, 256) and img.mode == 'RGB'
print('Format OK (256x256 RGB)')
"

# 2. Check A/B filenames match
python3 -c "
from pathlib import Path
for split in ['train', 'val']:
    a = {p.name for p in Path(f'iris_bbdm_pad/data/bonafide_pairs/{split}/A').glob('*.png')}
    b = {p.name for p in Path(f'iris_bbdm_pad/data/bonafide_pairs/{split}/B').glob('*.png')}
    assert a == b, f'Mismatch in {split}: {a.symmetric_difference(b)}'
    print(f'{split}: {len(a)} matched pairs OK')
"

# 3. Check labels.csv columns
python3 -c "
import csv; from pathlib import Path
for split in ['test', 'val']:
    with open(f'iris_bbdm_pad/data/evaluation_sets/{split}/labels.csv') as f:
        rows = list(csv.DictReader(f))
    assert all(r['label'] in ('bonafide', 'attack') for r in rows)
    print(f'{split}: {len(rows)} labels OK')
"

# 4. Dataset smoke test
python3 -c "
import sys; sys.path.insert(0, '.')
from iris_bbdm_pad.data.iris_dataset import IrisBonafidePairDataset
ds = IrisBonafidePairDataset('iris_bbdm_pad/data/bonafide_pairs', split='train')
s = ds[0]
assert s['A'].shape == (3, 256, 256)
assert -1.1 <= s['A'].min() <= -0.8
print(f'Dataset OK: {len(ds)} pairs, range [{s[\"A\"].min():.2f}, {s[\"A\"].max():.2f}]')
"

# 5. YAML config validation
python3 -c "
import yaml; from pathlib import Path
txt = Path('iris_bbdm_pad/configs/bbdm_iris_bonafide.yaml').read_text()
lines_w_tag = [l for l in txt.splitlines() if '!!python/tuple' in l and not l.strip().startswith('#')]
assert not lines_w_tag
cfg = yaml.safe_load(txt)
print('YAML OK:', cfg['runner'], '|', cfg['data']['dataset_type'])
"
```

## Resume After Interruption

The preprocessing script saves a checkpoint after every image:
```bash
# Resume from where it left off
python iris_bbdm_pad/data/iris_preprocessing.py \
    --input_dir Images/ --output_dir iris_bbdm_pad/data/preprocessed/ \
    --image_size 256 --workers 4 --resume
```

Checkpoint file: `iris_bbdm_pad/checkpoints/preprocessed_checkpoint.json`

Structure:
```json
{
  "completed_files": ["Images/train/Live/img001.bmp", ...],
  "failed_files": [],
  "total_processed": 8420,
  "timestamp": "2026-03-19T15:43:29"
}
```

Pair creation and test set preparation are fast (copy operations) and do not need resume support.
If interrupted, simply re-run the script — it will overwrite previous outputs cleanly.

## Known Issues

1. **Import error when running prepare_bonafide_pairs.py** — The script uses `from corruption import ...`
   which requires `iris_bbdm_pad/data/` to be on `sys.path`. Run from the project root or set:
   ```bash
   PYTHONPATH=iris_bbdm_pad/data:$PYTHONPATH python iris_bbdm_pad/data/prepare_bonafide_pairs.py ...
   ```

2. **TIFF loading** — Some `.tiff` files may fail with cv2 on certain systems. The code falls back
   to PIL for TIFF loading automatically.

3. **Small images** — Images smaller than 50px in either dimension are skipped (logged in metadata CSV).

4. **Hough circle detection** — On blurry or low-contrast images, Hough may fail and fall back to
   center-crop. Check `segmentation_results.png` to verify crop quality. The `seg_success` column
   in `preprocessed_metadata.csv` tracks which method was used.

5. **Spoof images excluded from training** — An explicit `assert "Spoof" not in str(path)` in
   `prepare_bonafide_pairs.py` will raise `AssertionError` if a Spoof path is accidentally passed.
   This is intentional — bona fide only for BBDM training.
