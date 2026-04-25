"""
Kiểm tra nhanh dữ liệu đã chuẩn bị: kích thước ảnh, annotation values, cặp ảnh-mask.
Chạy: python verify_data.py
"""
from pathlib import Path
from PIL import Image
import random

OUT_ROOT = Path("data/CVRP")

for split in ["train", "val", "field_test"]:
    img_dir = OUT_ROOT / "img_dir" / split
    ann_dir = OUT_ROOT / "ann_dir" / split

    imgs = sorted(img_dir.glob("*"))
    anns = sorted(ann_dir.glob("*"))

    # Kiểm tra khớp tên
    img_stems = {p.stem for p in imgs}
    ann_stems = {p.stem for p in anns}
    missing = img_stems - ann_stems
    extra   = ann_stems - img_stems

    print(f"\n[{split}]")
    print(f"  Ảnh: {len(imgs)} | Mask: {len(anns)}")
    if missing:
        print(f"  CẢNH BÁO - Ảnh không có mask: {list(missing)[:5]}")
    if extra:
        print(f"  CẢNH BÁO - Mask không có ảnh: {list(extra)[:5]}")

    # Kiểm tra ngẫu nhiên 3 cặp
    samples = random.sample(list(imgs), min(3, len(imgs)))
    for img_path in samples:
        ann_path = ann_dir / (img_path.stem + ".png")
        img = Image.open(img_path)
        ann = Image.open(ann_path)

        # Kiểm tra annotation values (phải là 0 và 1)
        raw_vals = list(set(ann.getdata()))
        ok_vals  = all(v in (0, 1) for v in raw_vals)

        flag = "OK" if (img.size == ann.size and ok_vals) else "LỖI"
        print(f"  [{flag}] {img_path.name}: img={img.size} ann={ann.size} "
              f"ann_vals={sorted(raw_vals)}")

print("\nXác minh hoàn tất.")