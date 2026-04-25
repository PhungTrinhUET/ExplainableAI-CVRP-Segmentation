"""
xAI Pipeline — CVRP Rice Panicle Segmentation
==============================================
Methods:
  GradCAM++ [Chattopadhay et al., WACV 2018]  → DeepLabV3+, SegFormer, K-Net
  EigenCAM  [Muhammad & Yeasin, ICIP 2020]    → Mask2Former (Swin Transformer)

Research Questions:
  RQ1 — Vùng nào model focus khi predict bông?
  RQ2 — CNN vs Transformer: attention pattern khác nhau?
  RQ3 — Tại sao model fail trên cỏ dại/lá?
  RQ4 — Định lượng: Energy Pointing Game score

Outputs:
  fig_cam_4models_grid.png      (RQ1+RQ2)
  fig_cnn_vs_transformer.png    (RQ2)
  fig_failure_cam.png           (RQ3)
  fig_cam_quantitative.png      (RQ4)
  cam_quantitative.csv          (RQ4 table)

Chạy: python scripts/run_xai.py
"""
import os, sys, random, warnings, csv
import numpy as np
import cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from pathlib import Path
from PIL import Image

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ["PYTHONPATH"] = str(Path(__file__).parent.parent)

from pytorch_grad_cam import GradCAMPlusPlus, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from mmseg.apis import MMSegInferencer
from scripts.xai_wrapper import (
    MMSegCAMWrapper, preprocess_image,
    PanicleTarget, cam_iou, energy_pointing_game,
    swin_reshape_transform
)

PROJECT = Path(__file__).parent.parent
DATA    = PROJECT / "data/CVRP"
WDIR    = PROJECT / "experiments/segmentation/work_dirs"
CFGDIR  = PROJECT / "experiments/segmentation/configs"
OUT     = PROJECT / "results/figures/06_xai"

MODELS = {
    "DeepLabV3+":  ("deeplabv3plus", "best_mIoU_iter_64000.pth"),
    "SegFormer":   ("segformer",     "best_mIoU_iter_80000.pth"),
    "K-Net":       ("knet",          "best_mIoU_iter_80000.pth"),
    "Mask2Former": ("mask2former",   "best_mIoU_iter_72000.pth"),
}
COLORS = {
    "DeepLabV3+": "#2196F3", "SegFormer": "#4CAF50",
    "K-Net": "#FF5722",      "Mask2Former": "#9C27B0",
}
METHOD = {
    "DeepLabV3+": "GradCAM++", "SegFormer": "GradCAM++",
    "K-Net": "GradCAM++",      "Mask2Former": "EigenCAM",
}

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
                     "savefig.dpi":180,"savefig.bbox":"tight"})
random.seed(42)


# ─── Load helpers ────────────────────────────────────────────
def load_infer(label):
    folder, ckpt = MODELS[label]
    return MMSegInferencer(
        model=str(CFGDIR/f"{folder}_cvrp.py"),
        weights=str(WDIR/folder/ckpt), device="cuda")


def get_target_layer(model, label):
    """Spatial-output target layers phù hợp từng model."""
    if "DeepLabV3" in label:
        return [model.backbone.layer4[-1]]
    elif "SegFormer" in label:
        return [model.decode_head.fusion_conv]
    elif "K-Net" in label:
        return [model.decode_head.kernel_generate_head.fpn_bottleneck]
    else:  # Mask2Former — dùng backbone Swin
        return [model.backbone.layers[-1].blocks[-1]]


def make_cam(infer, label, tensor):
    wrapper = MMSegCAMWrapper(infer.model)
    tl = get_target_layer(infer.model, label)
    if label == "Mask2Former":
        cam_obj = EigenCAM(model=wrapper, target_layers=tl,
                           reshape_transform=swin_reshape_transform)
    else:
        cam_obj = GradCAMPlusPlus(model=wrapper, target_layers=tl)
    gcam = cam_obj(input_tensor=tensor.cuda(), targets=[PanicleTarget()])
    return gcam[0]   # (H, W) floats [0,1]


def load_sample(img_path, gt_dir, size=512):
    img_np, tensor = preprocess_image(img_path, target_size=size)
    ann = gt_dir / (Path(img_path).stem + ".png")
    gt  = (np.array(Image.open(ann).resize(
           (img_np.shape[1], img_np.shape[0]), Image.NEAREST)) == 1).astype(np.uint8)
    return img_np, tensor, gt


def overlay_cam(img_np, gcam):
    """CAM heatmap overlay lên ảnh gốc."""
    gcam_r = cv2.resize(gcam, (img_np.shape[1], img_np.shape[0]))
    return show_cam_on_image(img_np.astype(np.float32)/255., gcam_r, use_rgb=True)


# ─────────────────────────────────────────────────────────────
# Fig 1: 4 Models × 4 ảnh (RQ1 + RQ2)
# ─────────────────────────────────────────────────────────────
def fig_4models_grid(samples):
    print("\n[1/4] 4-model CAM grid...")
    n = len(samples)
    fig, axes = plt.subplots(5, n, figsize=(n*3.5, 5*3.5))

    for j, (img_np, tensor, gt, name) in enumerate(samples):
        axes[0,j].imshow(img_np)
        axes[0,j].set_title(f"Image {j+1}", fontsize=9)
        axes[0,j].axis("off")
    axes[0,0].set_ylabel("Input", fontsize=9, fontweight="bold")

    for i, label in enumerate(MODELS):
        print(f"   {label}...", flush=True)
        infer = load_infer(label)
        for j, (img_np, tensor, gt, _) in enumerate(samples):
            try:
                gcam = make_cam(infer, label, tensor)
                cam_img = overlay_cam(img_np, gcam)
                gcam_r  = cv2.resize(gcam, (gt.shape[1], gt.shape[0]))
                epg = energy_pointing_game(gcam_r, gt)
                axes[i+1,j].imshow(cam_img)
                axes[i+1,j].set_title(f"EPG={epg:.2f}", fontsize=8)
            except Exception as e:
                axes[i+1,j].imshow(img_np)
                axes[i+1,j].set_title("N/A", fontsize=8)
            axes[i+1,j].axis("off")
        color = COLORS[label]; method = METHOD[label]
        axes[i+1,0].set_ylabel(f"{label}\n({method})", fontsize=8,
                                color=color, fontweight="bold")
        del infer; torch.cuda.empty_cache()

    fig.suptitle("xAI — GradCAM++ / EigenCAM\nVùng màu nóng = model attention khi dự đoán bông lúa",
                 fontweight="bold", fontsize=12)
    plt.tight_layout()
    p = OUT/"gradcam/fig_cam_4models_grid.png"; p.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(p); plt.close()
    print(f"   Saved: {p.name}")


# ─────────────────────────────────────────────────────────────
# Fig 2: CNN vs Transformer (RQ2)
# ─────────────────────────────────────────────────────────────
def fig_cnn_vs_transformer(samples):
    print("\n[2/4] CNN vs Transformer...")
    pairs = [("DeepLabV3+", "CNN\n(GradCAM++)"),
             ("Mask2Former","Transformer\n(EigenCAM)")]
    n = min(3, len(samples))
    fig, axes = plt.subplots(3, n*2, figsize=(n*6, 9))

    for col_grp, (label, title) in enumerate(pairs):
        print(f"   {label}...", flush=True)
        infer = load_infer(label)
        color = COLORS[label]
        for j in range(n):
            img_np, tensor, gt, _ = samples[j]
            col = col_grp*n + j
            axes[0,col].imshow(img_np)
            if j == 0:
                axes[0,col].set_title(title, fontsize=10, color=color, fontweight="bold")
            axes[0,col].axis("off")
            try:
                gcam    = make_cam(infer, label, tensor)
                cam_img = overlay_cam(img_np, gcam)
                gcam_r  = cv2.resize(gcam, (gt.shape[1], gt.shape[0]))
                epg     = energy_pointing_game(gcam_r, gt)
                axes[1,col].imshow(cam_img)
                axes[1,col].set_title(f"EPG={epg:.2f}", fontsize=8)
                # GT overlay + CAM contour
                ov = img_np.copy().astype(float)
                ov[gt==1] = ov[gt==1]*0.5 + np.array([0,200,0])*0.5
                contours,_ = cv2.findContours((gcam_r>0.4).astype(np.uint8),
                                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(ov.astype(np.uint8), contours,-1,(255,0,0),2)
                axes[2,col].imshow(ov.clip(0,255).astype(np.uint8))
            except Exception:
                axes[1,col].imshow(img_np)
                axes[2,col].imshow(img_np)
            axes[1,col].axis("off"); axes[2,col].axis("off")
        del infer; torch.cuda.empty_cache()

    axes[0,0].set_ylabel("Input", fontsize=9)
    axes[1,0].set_ylabel("CAM Heatmap", fontsize=9)
    axes[2,0].set_ylabel("GT(xanh)+CAM contour(đỏ)", fontsize=9)

    fig.suptitle("CNN (GradCAM++) vs Transformer (EigenCAM)\n"
                 "EPG = Energy Pointing Game ↑ (model attend đúng vùng bông)",
                 fontweight="bold", fontsize=12)
    plt.tight_layout()
    p = OUT/"comparison/fig_cnn_vs_transformer.png"; p.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(p); plt.close()
    print(f"   Saved: {p.name}")


# ─────────────────────────────────────────────────────────────
# Fig 3: Failure analysis — CAM on field test (RQ3)
# ─────────────────────────────────────────────────────────────
def fig_failure_cam(field_samples):
    print("\n[3/4] Failure case CAM...")
    label  = "Mask2Former"
    infer  = load_infer(label)
    n      = len(field_samples)
    titles = ["Cỏ dại\nnhầm bông", "Lá lúa\nnhầm bông",
              "Bông nhỏ\nbỏ sót",   "Che khuất nặng"]

    fig, axes = plt.subplots(4, n, figsize=(n*3.5, 13))
    row_lbls  = ["Field image", "Ground Truth", "CAM Heatmap",
                 "GT(xanh) + FP(đỏ)"]

    for j, (img_np, tensor, gt, _) in enumerate(field_samples[:n]):
        # GT overlay
        gt_vis = img_np.copy().astype(float)
        gt_vis[gt==1] = gt_vis[gt==1]*0.4 + np.array([180,30,30])*0.6

        axes[0,j].imshow(img_np)
        axes[0,j].set_title(titles[j] if j<len(titles) else "", fontsize=9)
        axes[0,j].axis("off")
        axes[1,j].imshow(gt_vis.clip(0,255).astype(np.uint8)); axes[1,j].axis("off")

        try:
            gcam    = make_cam(infer, label, tensor)
            cam_img = overlay_cam(img_np, gcam)
            gcam_r  = cv2.resize(gcam, (gt.shape[1], gt.shape[0]))
            epg     = energy_pointing_game(gcam_r, gt)
            ciou    = cam_iou(gcam_r, gt)
            axes[2,j].imshow(cam_img)
            axes[2,j].set_title(f"EPG={epg:.2f}  CAM-IoU={ciou:.2f}", fontsize=7)
            axes[2,j].axis("off")

            ov      = img_np.copy().astype(float)
            ov[gt==1] = ov[gt==1]*0.5 + np.array([0,200,0])*0.5
            fp_mask = (gcam_r > 0.4) & (gt == 0)
            ov[fp_mask] = ov[fp_mask]*0.4 + np.array([220,50,50])*0.6
            axes[3,j].imshow(ov.clip(0,255).astype(np.uint8)); axes[3,j].axis("off")
        except Exception:
            for r in [2,3]: axes[r,j].imshow(img_np); axes[r,j].axis("off")

    for r, lbl in enumerate(row_lbls):
        axes[r,0].set_ylabel(lbl, fontsize=9)

    fig.suptitle(f"{label} — Failure Analysis (Field Scenes)\n"
                 "Xanh=GT bông | Đỏ=CAM nhầm (model attend vào vùng sai)",
                 fontweight="bold", fontsize=11)
    plt.tight_layout()
    p = OUT/"failure/fig_failure_cam.png"; p.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(p); plt.close()
    print(f"   Saved: {p.name}")
    del infer; torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────
# Fig 4 + Table: Quantitative EPG (RQ4)
# ─────────────────────────────────────────────────────────────
def fig_quantitative(val_samples, field_samples):
    print("\n[4/4] Quantitative EPG evaluation...")
    results = {m: {"val":[], "field":[]} for m in MODELS}
    N = min(8, len(val_samples), len(field_samples))

    for label in MODELS:
        print(f"   {label}...", flush=True)
        infer = load_infer(label)
        for split, samps in [("val", val_samples[:N]), ("field", field_samples[:N])]:
            for img_np, tensor, gt, _ in samps:
                try:
                    gcam   = make_cam(infer, label, tensor)
                    gcam_r = cv2.resize(gcam, (gt.shape[1], gt.shape[0]))
                    if gt.sum() > 0:
                        results[label][split].append(energy_pointing_game(gcam_r, gt))
                except Exception:
                    pass
        del infer; torch.cuda.empty_cache()

    # Lưu CSV
    csv_path = PROJECT/"results/tables/cam_quantitative.csv"
    rows = [["Model","Method","Val EPG (mean)","Val EPG (std)",
             "Field EPG (mean)","Field EPG (std)"]]
    for m, r in results.items():
        v = r["val"]; f = r["field"]
        rows.append([m, METHOD[m],
                     f"{np.mean(v):.3f}" if v else "-",
                     f"{np.std(v):.3f}"  if v else "-",
                     f"{np.mean(f):.3f}" if f else "-",
                     f"{np.std(f):.3f}"  if f else "-"])
    with open(csv_path,"w",newline="") as fh: csv.writer(fh).writerows(rows)

    # Bar chart
    labels = list(results.keys())
    x = np.arange(len(labels)); w = 0.35
    val_epg   = [np.mean(results[m]["val"])   if results[m]["val"]   else 0 for m in labels]
    field_epg = [np.mean(results[m]["field"]) if results[m]["field"] else 0 for m in labels]
    colors    = [COLORS[m] for m in labels]

    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x-w/2, val_epg,   w, label="Val Set",    color=colors, alpha=0.9)
    b2 = ax.bar(x+w/2, field_epg, w, label="Field Test", color=colors, alpha=0.45, hatch="//")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0,1); ax.set_ylabel("Energy Pointing Game Score ↑")
    ax.set_title("xAI Quantitative — Energy Pointing Game\n"
                 "Score cao = model attend đúng vùng bông lúa",
                 fontweight="bold")
    ax.legend(fontsize=10); ax.grid(axis="y",alpha=0.3)
    for bar,v in zip(b1,val_epg):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.01,
                f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")
    for bar,v in zip(b2,field_epg):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.01,
                f"{v:.2f}", ha="center", fontsize=9, color="gray")
    plt.tight_layout()
    p = OUT/"quantitative/fig_cam_quantitative.png"
    p.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(p); plt.close()
    print(f"   Saved: {p.name}")
    print(f"   CSV:   {csv_path.name}")

    # Print table
    print(f"\n   {'Model':<14} {'Method':<12} {'Val EPG':>9} {'Field EPG':>10}")
    print("   " + "-"*50)
    for row in rows[1:]:
        print(f"   {row[0]:<14} {row[1]:<12} {row[2]:>9} {row[4]:>10}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("═"*60)
    print("  xAI — CVRP Rice Panicle Segmentation")
    print("  GradCAM++ | EigenCAM | Energy Pointing Game")
    print("═"*60)

    # Load samples
    val_paths   = random.sample(sorted((DATA/"img_dir/val").glob("*.jpg")), 10)
    field_paths = random.sample(sorted((DATA/"img_dir/field_test").glob("*.jpg")), 10)

    def prep(paths, gt_dir, n=4):
        out = []
        for p in paths[:n]:
            img_np, tensor, gt = load_sample(p, gt_dir)
            out.append((img_np, tensor, gt, p.name))
        return out

    val_s   = prep(val_paths,   DATA/"ann_dir/val",        n=4)
    field_s = prep(field_paths, DATA/"ann_dir/field_test", n=4)

    print(f"\n  Val images:   {[s[3] for s in val_s]}")
    print(f"  Field images: {[s[3] for s in field_s]}")

    fig_4models_grid(val_s)
    fig_cnn_vs_transformer(val_s[:3])
    fig_failure_cam(field_s)
    fig_quantitative(val_s, field_s)

    print(f"\n{'═'*60}")
    print("  xAI hoàn tất!")
    for f in sorted(OUT.rglob("*.png")):
        print(f"  {f.relative_to(OUT)}")
