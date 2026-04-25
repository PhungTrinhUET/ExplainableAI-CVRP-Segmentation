"""
Tạo visualization kết quả phân đoạn cho báo cáo.
Tái tạo Figure 6 + Figure 7 trong paper.

Output:
  fig_4models_grid.png       — 4 model × 4 ảnh (như Figure 7)
  fig_success_cases.png      — phân đoạn tốt (Figure 6 A-D)
  fig_failure_weed.png       — cỏ dại nhầm lẫn
  fig_failure_leaf.png       — lá lúa nhầm lẫn
  fig_failure_small.png      — bông nhỏ bị bỏ sót

Chạy: python scripts/visualize_seg_results.py
"""
import os, sys, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from PIL import Image

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))
os.environ["PYTHONPATH"] = str(PROJECT) + ":" + os.environ.get("PYTHONPATH", "")

DATA    = PROJECT / "data/CVRP"
WDIR    = PROJECT / "experiments/segmentation/work_dirs"
CFGDIR  = PROJECT / "experiments/segmentation/configs"
OUT_DIR = PROJECT / "results/figures/02_segmentation_qualitative"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PYTHON = "/home/tower2080/miniconda3/envs/cvrp_seg/bin/python"
MMSEG  = PROJECT / "mmsegmentation"

MODELS = {
    "DeepLabV3+":  ("deeplabv3plus", "best_mIoU_iter_64000.pth"),
    "SegFormer":   ("segformer",     "best_mIoU_iter_80000.pth"),
    "K-Net":       ("knet",          "best_mIoU_iter_80000.pth"),
    "Mask2Former": ("mask2former",   "best_mIoU_iter_72000.pth"),
}
MODEL_COLORS = {
    "DeepLabV3+":  "#2196F3",
    "SegFormer":   "#4CAF50",
    "K-Net":       "#FF5722",
    "Mask2Former": "#9C27B0",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size":   10,
    "savefig.dpi": 180,
    "savefig.bbox": "tight",
})

random.seed(42)


# ══════════════════════════════════════════════════════════
# Inference: dùng MMSeg inferencer API
# ══════════════════════════════════════════════════════════
def get_inferencer(model_name: str, ckpt_name: str):
    from mmseg.apis import MMSegInferencer
    cfg_path  = str(CFGDIR / f"{model_name}_cvrp.py")
    ckpt_path = str(WDIR / model_name / ckpt_name)
    infer = MMSegInferencer(model=cfg_path, weights=ckpt_path, device="cuda")
    return infer


def infer_image(inferencer, img_path: Path) -> np.ndarray:
    """Trả về pred mask (H×W, uint8, 0=bg 1=panicle)."""
    result = inferencer(str(img_path), return_datasamples=True)
    # MMSeg 1.x: result là SegDataSample trực tiếp
    if hasattr(result, "pred_sem_seg"):
        pred = result.pred_sem_seg.data.squeeze().cpu().numpy()
    else:
        pred = result["predictions"][0].pred_sem_seg.data.squeeze().cpu().numpy()
    return pred.astype(np.uint8)


# ══════════════════════════════════════════════════════════
# Overlay: màu theo TP/FP/FN
# ══════════════════════════════════════════════════════════
def make_overlay(img: np.ndarray, pred: np.ndarray,
                 gt: np.ndarray) -> np.ndarray:
    out   = img.astype(float)
    alpha = 0.55
    tp = (pred == 1) & (gt == 1)
    fp = (pred == 1) & (gt == 0)
    fn = (pred == 0) & (gt == 1)
    out[tp] = out[tp] * (1-alpha) + np.array([50,  200, 50 ]) * alpha
    out[fp] = out[fp] * (1-alpha) + np.array([220, 50,  50 ]) * alpha
    out[fn] = out[fn] * (1-alpha) + np.array([255, 200, 0  ]) * alpha
    return out.clip(0, 255).astype(np.uint8)


def compute_metrics(pred: np.ndarray, gt: np.ndarray):
    tp = ((pred == 1) & (gt == 1)).sum()
    fp = ((pred == 1) & (gt == 0)).sum()
    fn = ((pred == 0) & (gt == 1)).sum()
    iou = tp / (tp + fp + fn + 1e-6) * 100
    return round(iou, 1)


def load_sample(img_path: Path, resize_long: int = 512):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    scale = resize_long / max(w, h)
    img   = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    return np.array(img)


def load_gt(ann_path: Path, size_wh: tuple) -> np.ndarray:
    ann = Image.open(ann_path).resize(size_wh, Image.NEAREST)
    arr = np.array(ann)
    return (arr == 1).astype(np.uint8)


# ══════════════════════════════════════════════════════════
# Figure 1: 4 Models × 4 ảnh (như Figure 7 paper)
# ══════════════════════════════════════════════════════════
def plot_4models_grid(sample_paths: list[Path], split: str = "val"):
    """4 hàng (model) × 4 cột (ảnh) + 1 hàng ảnh gốc."""
    ann_dir = DATA / "ann_dir" / split
    n_img   = len(sample_paths)

    model_names = list(MODELS.keys())
    n_rows = 1 + len(model_names)   # 1 hàng ảnh gốc + 4 model

    fig, axes = plt.subplots(n_rows, n_img,
                              figsize=(n_img * 3.2, n_rows * 3.2))

    # Hàng 0: ảnh gốc
    imgs_arr, gts_arr = [], []
    for j, img_path in enumerate(sample_paths):
        img_arr = load_sample(img_path)
        ann_path = ann_dir / (img_path.stem + ".png")
        gt = load_gt(ann_path, (img_arr.shape[1], img_arr.shape[0]))
        imgs_arr.append(img_arr)
        gts_arr.append(gt)

        ax = axes[0, j]
        ax.imshow(img_arr)
        ax.set_title(f"Ảnh {j+1}", fontsize=9, pad=2)
        ax.axis("off")
    axes[0, 0].set_ylabel("Field Image", fontsize=9, labelpad=4)

    # Hàng 1..4: inference từng model
    for i, (label, (model_name, ckpt_name)) in enumerate(MODELS.items()):
        print(f"  [{i+1}/4] {label}...", flush=True)
        infer = get_inferencer(model_name, ckpt_name)
        color = MODEL_COLORS[label]

        for j, img_path in enumerate(sample_paths):
            ax = axes[i+1, j]
            img_arr = imgs_arr[j]
            gt      = gts_arr[j]

            pred    = infer_image(infer, img_path)
            pred_rs = np.array(Image.fromarray(pred).resize(
                (img_arr.shape[1], img_arr.shape[0]), Image.NEAREST))

            overlay = make_overlay(img_arr, pred_rs, gt)
            iou     = compute_metrics(pred_rs, gt)

            ax.imshow(overlay)
            ax.set_title(f"IoU={iou}%", fontsize=8, pad=2)
            ax.axis("off")

        axes[i+1, 0].set_ylabel(label, fontsize=9, labelpad=4,
                                  color=color, fontweight="bold")
        del infer   # giải phóng VRAM

    # Legend
    patches = [
        mpatches.Patch(color=(50/255,200/255,50/255), label="TP"),
        mpatches.Patch(color=(220/255,50/255,50/255), label="FP"),
        mpatches.Patch(color=(255/255,200/255,0),     label="FN"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3,
               fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("So sánh 4 model — CVRP Val Set",
                 fontweight="bold", fontsize=13)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    path = OUT_DIR / "fig_4models_grid.png"
    fig.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════
# Figure 2: Success cases (best model = Mask2Former)
# ══════════════════════════════════════════════════════════
def plot_success_cases(sample_paths: list[Path], split: str = "val"):
    ann_dir = DATA / "ann_dir" / split
    label, (model_name, ckpt_name) = "Mask2Former", MODELS["Mask2Former"]
    print(f"  Loading {label}...", flush=True)
    infer = get_inferencer(model_name, ckpt_name)

    n = len(sample_paths)
    fig, axes = plt.subplots(3, n, figsize=(n * 3.5, 9))

    for j, img_path in enumerate(sample_paths):
        img_arr  = load_sample(img_path)
        ann_path = ann_dir / (img_path.stem + ".png")
        gt       = load_gt(ann_path, (img_arr.shape[1], img_arr.shape[0]))
        pred     = infer_image(infer, img_path)
        pred_rs  = np.array(Image.fromarray(pred).resize(
            (img_arr.shape[1], img_arr.shape[0]), Image.NEAREST))

        overlay_gt   = img_arr.copy().astype(float)
        overlay_gt[gt == 1] = overlay_gt[gt == 1]*0.4 + np.array([180,30,30])*0.6
        overlay_gt   = overlay_gt.clip(0,255).astype(np.uint8)
        overlay_pred = make_overlay(img_arr, pred_rs, gt)
        iou          = compute_metrics(pred_rs, gt)

        axes[0, j].imshow(img_arr)
        axes[0, j].set_title(img_path.stem[:16], fontsize=8)
        axes[0, j].axis("off")

        axes[1, j].imshow(overlay_gt)
        axes[1, j].axis("off")

        axes[2, j].imshow(overlay_pred)
        axes[2, j].set_title(f"IoU={iou}%", fontsize=8)
        axes[2, j].axis("off")

    axes[0, 0].set_ylabel("Ảnh gốc",          fontsize=9, labelpad=4)
    axes[1, 0].set_ylabel("Ground Truth",      fontsize=9, labelpad=4)
    axes[2, 0].set_ylabel(f"{label} Predict",  fontsize=9, labelpad=4)

    fig.suptitle(f"Kết quả phân đoạn tốt — {label}",
                 fontweight="bold", fontsize=13)
    plt.tight_layout()
    path = OUT_DIR / "fig_success_cases.png"
    fig.savefig(path)
    plt.close()
    print(f"  Saved: {path}")
    del infer


# ══════════════════════════════════════════════════════════
# Figure 3: Failure cases (field_test — khó hơn)
# ══════════════════════════════════════════════════════════
def plot_failure_cases(field_paths: list[Path]):
    ann_dir = DATA / "ann_dir/field_test"
    label, (model_name, ckpt_name) = "Mask2Former", MODELS["Mask2Former"]
    print(f"  Loading {label} for failure cases...", flush=True)
    infer = get_inferencer(model_name, ckpt_name)

    n = min(4, len(field_paths))
    titles = ["Cỏ dại\nnhầm bông", "Lá lúa\nnhầm bông",
              "Bông nhỏ\nbỏ sót",   "Che khuất\nnặng"]

    fig, axes = plt.subplots(3, n, figsize=(n * 3.5, 9))

    for j, img_path in enumerate(field_paths[:n]):
        img_arr  = load_sample(img_path)
        ann_path = ann_dir / (img_path.stem + ".png")
        gt       = load_gt(ann_path, (img_arr.shape[1], img_arr.shape[0]))
        pred     = infer_image(infer, img_path)
        pred_rs  = np.array(Image.fromarray(pred).resize(
            (img_arr.shape[1], img_arr.shape[0]), Image.NEAREST))

        overlay_gt   = img_arr.copy().astype(float)
        overlay_gt[gt == 1] = overlay_gt[gt == 1]*0.4 + np.array([180,30,30])*0.6
        overlay_gt   = overlay_gt.clip(0,255).astype(np.uint8)
        overlay_pred = make_overlay(img_arr, pred_rs, gt)
        iou          = compute_metrics(pred_rs, gt)

        axes[0, j].imshow(img_arr)
        axes[0, j].set_title(titles[j], fontsize=9, pad=3)
        axes[0, j].axis("off")

        axes[1, j].imshow(overlay_gt)
        axes[1, j].axis("off")

        axes[2, j].imshow(overlay_pred)
        axes[2, j].set_title(f"IoU={iou}%", fontsize=8)
        axes[2, j].axis("off")

    axes[0, 0].set_ylabel("Field Image",    fontsize=9, labelpad=4)
    axes[1, 0].set_ylabel("Ground Truth",   fontsize=9, labelpad=4)
    axes[2, 0].set_ylabel("Mask2Former",    fontsize=9, labelpad=4)

    fig.suptitle("Các trường hợp khó — Field Scenes",
                 fontweight="bold", fontsize=13)
    plt.tight_layout()
    path = OUT_DIR / "fig_failure_cases.png"
    fig.savefig(path)
    plt.close()
    print(f"  Saved: {path}")
    del infer


# ══════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("═" * 58)
    print("  Visualization — Segmentation Results")
    print("═" * 58)

    # Lấy 4 ảnh val có panicle rõ ràng
    val_imgs = sorted((DATA / "img_dir/val").glob("*.jpg"))
    random.shuffle(val_imgs)
    val_samples = val_imgs[:4]
    print(f"  Val samples: {[p.name for p in val_samples]}")

    # Lấy 4 ảnh field_test đa dạng
    field_imgs = sorted((DATA / "img_dir/field_test").glob("*.jpg"))
    random.shuffle(field_imgs)
    field_samples = field_imgs[:4]
    print(f"  Field samples: {[p.name for p in field_samples]}")

    print("\n[1/3] 4-model grid (Figure 7 replica)...")
    plot_4models_grid(val_samples, "val")

    print("\n[2/3] Success cases (Figure 6 replica)...")
    plot_success_cases(val_samples[:4], "val")

    print("\n[3/3] Failure cases (field scenes)...")
    plot_failure_cases(field_samples)

    print(f"\n{'═'*58}")
    print(f"  Xong! Figures lưu tại: {OUT_DIR}/")
    for f in sorted(OUT_DIR.glob("*.png")):
        print(f"  {f.name}")
