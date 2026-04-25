"""
Giai đoạn 1 — Phân tích dataset CVRP (không cần GPU).
Tạo ra toàn bộ tables và figures cho phần Dataset Analysis trong báo cáo.

Chạy từ thư mục gốc: python scripts/analyze_dataset.py
"""
import os
import csv
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from PIL import Image

random.seed(42)

# ── Đường dẫn ────────────────────────────────────────────────────
CVRP_ROOT  = Path("CVRP_Rice/CVRP")
DATA_ROOT  = Path("data/CVRP")
FIG_DIR    = Path("results/figures/01_dataset_analysis")
TABLE_DIR  = Path("results/tables")

MC_ROOT    = CVRP_ROOT / "FieldImages/multi_cultivars"
FS_ROOT    = CVRP_ROOT / "FieldImages/field_scenes"
IN_ROOT    = CVRP_ROOT / "IndoorPanicleImages"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

# ── Style chung ──────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "font.size":      11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi":     150,
    "savefig.dpi":    200,
    "savefig.bbox":   "tight",
})

COLORS = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0"]   # blue, green, orange, purple


# ════════════════════════════════════════════════════════════════
# 1.1 — Bảng thống kê dataset
# ════════════════════════════════════════════════════════════════
def make_dataset_table():
    cultivars = [d for d in MC_ROOT.iterdir() if d.is_dir()]
    g_cultivars = [c for c in cultivars if c.name.startswith("G")]
    t_cultivars = [c for c in cultivars if c.name.startswith("T")]

    mc_imgs = sum(
        len(list((c / "Images").glob("*")))
        for c in cultivars if (c / "Images").exists()
    )
    mc_anns = sum(
        len(list((c / "Annotations").glob("*")))
        for c in cultivars if (c / "Annotations").exists()
    )
    fs_imgs = len(list((FS_ROOT / "Images").glob("*")))
    fs_anns = len(list((FS_ROOT / "Annotations").glob("*")))
    in_imgs = len(list(IN_ROOT.glob("*.png")))
    in_stems = len(set(p.stem.rsplit("_", 1)[0] for p in IN_ROOT.glob("*.png")))

    rows = [
        ["Loại dữ liệu", "Số giống", "Số ảnh", "Số annotation", "Kích thước ảnh"],
        ["Multi-cultivar (G - landrace)", len(g_cultivars), "-", "-", "1080×1920"],
        ["Multi-cultivar (T - modern)", len(t_cultivars), "-", "-", "1080×1920"],
        ["Multi-cultivar (tổng)", len(cultivars), mc_imgs, mc_anns, "1080×1920"],
        ["Field scenes", "-", fs_imgs, fs_anns, "biến đổi"],
        ["Indoor panicle", in_stems, in_imgs, 0, "1500×2007"],
        ["TỔNG", len(cultivars), mc_imgs + fs_imgs + in_imgs,
         mc_anns + fs_anns, "-"],
    ]
    path = TABLE_DIR / "dataset_statistics.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print(f"[1.1] Saved: {path}")
    for r in rows:
        print("     ", r)


# ════════════════════════════════════════════════════════════════
# 1.2 — Histogram số ảnh mỗi giống
# ════════════════════════════════════════════════════════════════
def plot_images_per_cultivar():
    cultivars = [d for d in MC_ROOT.iterdir() if d.is_dir()]
    counts = {
        c.name: len(list((c / "Images").glob("*")))
        for c in cultivars if (c / "Images").exists()
    }

    # Phân tách G và T
    g_counts = [v for k, v in counts.items() if k.startswith("G")]
    t_counts = [v for k, v in counts.items() if k.startswith("T")]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)

    for ax, data, label, color in zip(
        axes,
        [g_counts, t_counts],
        ["Giống G (Landrace)", "Giống T (Modern Cultivar)"],
        [COLORS[0], COLORS[1]],
    ):
        bins = range(min(data), max(data) + 2)
        ax.hist(data, bins=bins, color=color, edgecolor="white", linewidth=0.5)
        ax.set_xlabel("Số ảnh mỗi giống")
        ax.set_ylabel("Số giống")
        ax.set_title(f"{label} (n={len(data)}, avg={np.mean(data):.1f})")
        ax.axvline(np.mean(data), color="red", linestyle="--",
                   linewidth=1.5, label=f"Mean={np.mean(data):.1f}")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Phân bố số ảnh mỗi giống trong CVRP", fontweight="bold")
    plt.tight_layout()
    path = FIG_DIR / "fig_images_per_cultivar.png"
    fig.savefig(path)
    plt.close()
    print(f"[1.2] Saved: {path}")


# ════════════════════════════════════════════════════════════════
# 1.3 — Multiview sample (4 góc nhìn cùng 1 giống)
# ════════════════════════════════════════════════════════════════
def plot_multiview_sample():
    # Tìm giống có đủ >=4 ảnh với tên số thứ tự khác nhau
    cultivars = [d for d in MC_ROOT.iterdir() if d.is_dir()]
    candidate = None
    for c in cultivars:
        imgs = sorted((c / "Images").glob("*.jpg"))
        if len(imgs) >= 4:
            candidate = c
            break
    if candidate is None:
        print("[1.3] Không tìm thấy giống có đủ 4 ảnh, bỏ qua.")
        return

    imgs = sorted((candidate / "Images").glob("*.jpg"))[:4]
    view_labels = ["Góc nhìn 1", "Góc nhìn 2", "Góc nhìn 3", "Góc nhìn 4"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    for ax, img_path, label in zip(axes, imgs, view_labels):
        img = Image.open(img_path)
        # Thumbnail để tránh RAM lớn
        img.thumbnail((540, 960), Image.LANCZOS)
        ax.imshow(img)
        ax.set_title(label, fontsize=10)
        ax.axis("off")

    fig.suptitle(
        f"Ví dụ đa góc nhìn — Giống {candidate.name}",
        fontweight="bold", fontsize=13
    )
    plt.tight_layout()
    path = FIG_DIR / "fig_multiview_sample.png"
    fig.savefig(path)
    plt.close()
    print(f"[1.3] Saved: {path}")


# ════════════════════════════════════════════════════════════════
# 1.4 — Annotation overlay (6 cặp ảnh + mask)
# ════════════════════════════════════════════════════════════════
def plot_annotation_examples():
    # Lấy 6 giống ngẫu nhiên, mỗi giống lấy ảnh đầu tiên
    cultivars = [d for d in MC_ROOT.iterdir() if d.is_dir()]
    random.shuffle(cultivars)
    pairs = []
    for c in cultivars:
        imgs = list((c / "Images").glob("*.jpg"))
        if not imgs:
            continue
        img_path = imgs[0]
        ann_path = c / "Annotations" / (img_path.stem + ".png")
        if ann_path.exists():
            pairs.append((img_path, ann_path, c.name))
        if len(pairs) == 6:
            break

    fig, axes = plt.subplots(2, 6, figsize=(18, 7))
    # Hàng trên: ảnh gốc | Hàng dưới: ảnh + overlay đỏ

    for col, (img_path, ann_path, name) in enumerate(pairs):
        img = Image.open(img_path).convert("RGB")
        ann = Image.open(ann_path)
        img.thumbnail((400, 700), Image.LANCZOS)
        ann = ann.resize(img.size, Image.NEAREST)

        ann_arr = np.array(ann)
        img_arr = np.array(img)

        # Ảnh gốc
        axes[0, col].imshow(img_arr)
        axes[0, col].set_title(name, fontsize=8)
        axes[0, col].axis("off")

        # Overlay: bông lúa tô đỏ bán trong suốt
        overlay = img_arr.copy()
        mask = ann_arr == 1
        overlay[mask] = (
            overlay[mask] * 0.4 + np.array([220, 30, 30]) * 0.6
        ).astype(np.uint8)
        axes[1, col].imshow(overlay)
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("Ảnh gốc", fontsize=9, labelpad=4)
    axes[1, 0].set_ylabel("Ảnh + Annotation", fontsize=9, labelpad=4)

    red_patch = mpatches.Patch(color=(220/255, 30/255, 30/255), label="Bông lúa (panicle)")
    fig.legend(handles=[red_patch], loc="lower center", ncol=1, fontsize=10)
    fig.suptitle("Ví dụ annotation — 6 giống ngẫu nhiên", fontweight="bold", fontsize=13)
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    path = FIG_DIR / "fig_annotation_examples.png"
    fig.savefig(path)
    plt.close()
    print(f"[1.4] Saved: {path}")


# ════════════════════════════════════════════════════════════════
# 1.5 — Tỉ lệ pixel bông / nền
# ════════════════════════════════════════════════════════════════
def analyze_annotation_ratio():
    rows = [["split", "file", "panicle_pixels", "total_pixels", "ratio_pct"]]

    for split in ["train", "val", "field_test"]:
        ann_dir = DATA_ROOT / "ann_dir" / split
        ann_files = sorted(ann_dir.glob("*.png"))
        # Sample tối đa 200 ảnh mỗi split để nhanh
        sample = random.sample(ann_files, min(200, len(ann_files)))
        for ann_path in sample:
            ann = np.array(Image.open(ann_path))
            total   = ann.size
            panicle = int((ann == 1).sum())
            ratio   = panicle / total * 100
            rows.append([split, ann_path.name, panicle, total, f"{ratio:.2f}"])

    path = TABLE_DIR / "annotation_ratio.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

    # Tính tóm tắt
    data_by_split = {}
    for row in rows[1:]:
        s = row[0]
        r = float(row[4])
        data_by_split.setdefault(s, []).append(r)

    print(f"[1.5] Saved: {path}")
    print("      Tỉ lệ panicle pixel (%):")
    for split, vals in data_by_split.items():
        print(f"        {split}: mean={np.mean(vals):.1f}%  "
              f"min={np.min(vals):.1f}%  max={np.max(vals):.1f}%")

    # Boxplot
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = list(data_by_split.keys())
    data   = [data_by_split[k] for k in labels]
    bp = ax.boxplot(data, patch_artist=True, labels=labels)
    for patch, color in zip(bp["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("% pixel bông lúa / tổng pixel")
    ax.set_title("Phân bố tỉ lệ pixel bông trong annotation", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path2 = FIG_DIR / "fig_annotation_ratio_boxplot.png"
    fig.savefig(path2)
    plt.close()
    print(f"[1.5] Saved: {path2}")


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 55)
    print("Giai đoạn 1 — Phân tích Dataset CVRP")
    print("=" * 55)
    make_dataset_table()
    plot_images_per_cultivar()
    plot_multiview_sample()
    plot_annotation_examples()
    analyze_annotation_ratio()
    print("=" * 55)
    print("Hoàn tất Giai đoạn 1. Kết quả trong:")
    print(f"  Figures: {FIG_DIR}/")
    print(f"  Tables:  {TABLE_DIR}/")