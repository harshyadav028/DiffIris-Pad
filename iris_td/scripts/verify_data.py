# iris_td/scripts/verify_data.py
from pathlib import Path

print("Verifying data copy...")
all_ok = True

for split in ["train", "val", "test"]:
    for ab in ["A", "B"]:
        src = Path(f"iris_bbdm_pad/data/bonafide_pairs/{split}/{ab}")
        dst = Path(f"iris_td/data/bonafide_pairs/{split}/{ab}")
        src_n = len(list(src.rglob("*.png"))) if src.exists() else 0
        dst_n = len(list(dst.rglob("*.png"))) if dst.exists() else 0
        ok = src_n == dst_n and src_n > 0
        all_ok = all_ok and ok
        print(f"  {'OK' if ok else 'FAIL'} bonafide_pairs/{split}/{ab}: "
              f"src={src_n} dst={dst_n}")

for split in ["test", "val"]:
    src = Path(f"iris_bbdm_pad/data/evaluation_sets/{split}")
    dst = Path(f"iris_td/data/evaluation_sets/{split}")
    src_n = len(list(src.rglob("*.*"))) if src.exists() else 0
    dst_n = len(list(dst.rglob("*.*"))) if dst.exists() else 0
    ok = src_n == dst_n and src_n > 0
    all_ok = all_ok and ok
    print(f"  {'OK' if ok else 'FAIL'} evaluation_sets/{split}: "
          f"src={src_n} dst={dst_n}")

print(f"\n{'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}")
