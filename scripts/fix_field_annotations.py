"""
Convert annotation field_scenes từ RGB format (255,0,0)=panicle
sang Palette format index 0=background, 1=panicle.
"""
from pathlib import Path
from PIL import Image
import numpy as np

ANN_DIR = Path("data/CVRP/ann_dir/field_test")

converted = 0
errors = []
for ann_path in sorted(ANN_DIR.glob("*.png")):
    img = Image.open(ann_path)
    if img.mode == "P":
        # Kiểm tra xem đã đúng format chưa
        vals = set(img.getdata())
        if vals <= {0, 1}:
            continue  # đã OK

    arr = np.array(img.convert("RGB"))

    # (255, 0, 0) = panicle → index 1; còn lại → index 0
    mask = np.zeros((arr.shape[0], arr.shape[1]), dtype=np.uint8)
    panicle_pixels = (arr[:,:,0] > 128) & (arr[:,:,1] < 50) & (arr[:,:,2] < 50)
    mask[panicle_pixels] = 1

    unique_vals = np.unique(mask).tolist()
    if 1 not in unique_vals:
        errors.append(ann_path.name)

    # Lưu lại dạng Palette PNG
    out = Image.fromarray(mask, mode="L")  # 0/1 grayscale, MMSeg đọc được
    out.save(ann_path)
    converted += 1

print(f"Đã convert: {converted} files")
if errors:
    print(f"CẢNH BÁO - Không tìm thấy panicle trong: {errors[:10]}")
else:
    print("Tất cả annotations đều có panicle pixels.")

# Xác nhận lại
sample = list(ANN_DIR.glob("*.png"))[:3]
for p in sample:
    img = Image.open(p)
    import numpy as np
    arr = np.array(img)
    print(f"  {p.name}: mode={img.mode} vals={np.unique(arr).tolist()}")