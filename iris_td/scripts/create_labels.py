# iris_td/scripts/create_labels.py
import csv
from pathlib import Path

OUT_DIR = Path("iris_td/labels")
OUT_DIR.mkdir(parents=True, exist_ok=True)

for split in ["test", "val"]:
    folder = Path(f"iris_td/data/evaluation_sets/{split}")
    rows   = []
    for img in sorted(folder.rglob("*.png")):
        if "Spoof" in str(img) or "spoof" in str(img):
            label = "attack"
        elif "Live" in str(img) or "live" in str(img):
            label = "bonafide"
        else:
            label = "unknown"
        rows.append({"filename": img.name, "label": label})

    out = OUT_DIR / f"{split}_labels.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "label"])
        writer.writeheader()
        writer.writerows(rows)

    bonafide = sum(1 for r in rows if r["label"] == "bonafide")
    attack   = sum(1 for r in rows if r["label"] == "attack")
    print(f"{split}: {bonafide} bonafide, {attack} attack → {out}")
