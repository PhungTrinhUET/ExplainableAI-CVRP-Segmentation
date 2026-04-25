"""
Script chuẩn bị dữ liệu CVRP cho MMSegmentation.
Chạy: python prepare_data.py
"""
import os
import shutil
import random
from pathlib import Path

# ─── Cấu hình đường dẫn ───────────────────────────────────────────
CVRP_ROOT   = Path("CVRP_Rice/CVRP")
OUT_ROOT    = Path("data/CVRP")

MULTI_IMGS  = CVRP_ROOT / "FieldImages/multi_cultivars"
FIELD_IMGS  = CVRP_ROOT / "FieldImages/field_scenes"

SEED        = 42
VAL_SIZE    = 80     # đúng như paper: 727 train + 80 val

random.seed(SEED)

# ─── Tạo thư mục output ───────────────────────────────────────────
for split in ["train", "val"]:
    (OUT_ROOT / "img_dir" / split).mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "ann_dir" / split).mkdir(parents=True, exist_ok=True)

(OUT_ROOT / "img_dir" / "field_test").mkdir(parents=True, exist_ok=True)
(OUT_ROOT / "ann_dir" / "field_test").mkdir(parents=True, exist_ok=True)

# ─── Thu thập tất cả cặp (ảnh, annotation) từ multi_cultivars ────
pairs = []
for cultivar_dir in sorted(MULTI_IMGS.iterdir()):
    if not cultivar_dir.is_dir():
        continue
    img_dir = cultivar_dir / "Images"
    ann_dir = cultivar_dir / "Annotations"
    if not img_dir.exists():
        continue
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        ann_path = ann_dir / (img_path.stem + ".png")
        if ann_path.exists():
            pairs.append((img_path, ann_path))

print(f"Tổng cặp ảnh+annotation: {len(pairs)}")

# ─── Shuffle và chia train/val ────────────────────────────────────
random.shuffle(pairs)
val_pairs   = pairs[:VAL_SIZE]
train_pairs = pairs[VAL_SIZE:]

print(f"Train: {len(train_pairs)} | Val: {len(val_pairs)}")

def copy_pair(src_img, src_ann, split):
    dst_img = OUT_ROOT / "img_dir" / split / (src_img.stem + src_img.suffix)
    dst_ann = OUT_ROOT / "ann_dir" / split / (src_img.stem + ".png")
    shutil.copy2(src_img, dst_img)
    shutil.copy2(src_ann, dst_ann)

for img, ann in train_pairs:
    copy_pair(img, ann, "train")
for img, ann in val_pairs:
    copy_pair(img, ann, "val")

# ─── Copy field_scenes vào thư mục test riêng ────────────────────
fs_img_dir = FIELD_IMGS / "Images"
fs_ann_dir = FIELD_IMGS / "Annotations"
fs_count = 0
for img_path in sorted(fs_img_dir.iterdir()):
    if img_path.suffix.lower() not in {".jpg", ".jpeg"}:
        continue
    ann_path = fs_ann_dir / (img_path.stem + ".png")
    if ann_path.exists():
        shutil.copy2(img_path, OUT_ROOT / "img_dir" / "field_test" / img_path.name)
        shutil.copy2(ann_path, OUT_ROOT / "ann_dir" / "field_test" / ann_path.name)
        fs_count += 1

print(f"Field scenes (test): {fs_count}")
print()
print("Cấu trúc output:")
for p in sorted(OUT_ROOT.rglob("*")):
    if p.is_dir():
        n = len(list(p.glob("*")))
        print(f"  {p.relative_to(OUT_ROOT)}/  ({n} files)")

print()
print("Xong! Chạy tiếp: python verify_data.py")