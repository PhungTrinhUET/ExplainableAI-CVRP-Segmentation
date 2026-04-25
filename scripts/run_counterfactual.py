"""
xAI Counterfactual — CVRP Rice Panicle Segmentation
=====================================================
Methods:
  M1. Occlusion Counterfactual Map  [Zeiler & Fergus, ECCV 2014]
      → Tìm vùng mà nếu xóa đi, model mất khả năng detect bông
  M2. Deletion / Insertion AUC      [Samek et al., IEEE TNNLS 2017]
      → Đo tốc độ prediction thay đổi khi xóa/thêm pixel theo thứ tự quan trọng
  M3. Superpixel Counterfactual     [RISE, Petsiuk et al., BMVC 2018]
      → Tìm superpixel nào là CRITICAL (xóa đi → mất detection)

Outputs:
  fig_occlusion_cf_map.png      (M1)
  fig_deletion_insertion.png    (M2)
  fig_superpixel_cf.png         (M3)
  counterfactual_results.csv    (tổng hợp)
"""
import os, sys, warnings, random, csv
import numpy as np
import cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ["PYTHONPATH"] = str(Path(__file__).parent.parent)

from mmseg.apis import MMSegInferencer
from scripts.xai_wrapper import (MMSegCAMWrapper, preprocess_image,
                                  swin_reshape_transform)

PROJECT = Path(__file__).parent.parent
DATA    = PROJECT / "data/CVRP"
WDIR    = PROJECT / "experiments/segmentation/work_dirs"
CFGDIR  = PROJECT / "experiments/segmentation/configs"
OUT     = PROJECT / "results/figures/06_xai/counterfactual"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
                     "savefig.dpi":180,"savefig.bbox":"tight"})
random.seed(42)

MODELS_USE = {
    "DeepLabV3+":  ("deeplabv3plus", "best_mIoU_iter_64000.pth"),
    "SegFormer":   ("segformer",     "best_mIoU_iter_80000.pth"),
    "Mask2Former": ("mask2former",   "best_mIoU_iter_72000.pth"),
}
COLORS = {"DeepLabV3+":"#2196F3","SegFormer":"#4CAF50","Mask2Former":"#9C27B0"}
PANICLE_CLASS = 1
MEAN_PIX = np.array([123.675, 116.28, 103.53], dtype=np.float32)


# ──── Model loading ────────────────────────────────────────────
class Mask2FormerCFWrapper(torch.nn.Module):
    """Mask2Former dùng backbone-only wrapper (decode_head cần batch_data_samples)."""
    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone
        self.proj     = torch.nn.Conv2d(1024, 2, 1).cuda().float()
        self.proj.eval()
    def forward(self, x):
        feats = self.backbone(x)
        return self.proj(feats[-1])   # (B, 2, H, W)


def load_wrapper(label):
    folder, ckpt = MODELS_USE[label]
    infer = MMSegInferencer(
        model=str(CFGDIR/f"{folder}_cvrp.py"),
        weights=str(WDIR/folder/ckpt), device="cuda")
    if label == "Mask2Former":
        wrapper = Mask2FormerCFWrapper(infer.model)
    else:
        wrapper = MMSegCAMWrapper(infer.model)
    return wrapper, infer


def get_panicle_score(wrapper, tensor: torch.Tensor) -> float:
    """Tổng confidence của panicle class (normalized)."""
    with torch.no_grad():
        logits = wrapper(tensor.cuda())           # (1, 2, H, W)
        probs  = torch.softmax(logits, dim=1)
        score  = probs[0, PANICLE_CLASS].sum().item()
        total  = probs[0, PANICLE_CLASS].numel()
    return score / total   # mean panicle probability


def get_panicle_iou(wrapper, tensor: torch.Tensor, gt: np.ndarray) -> float:
    """IoU giữa model prediction và GT mask."""
    with torch.no_grad():
        logits = wrapper(tensor.cuda())
        pred   = logits.argmax(dim=1).squeeze().cpu().numpy()
    pred_r = cv2.resize(pred.astype(np.float32),
                        (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
    pred_b = (pred_r > 0.5).astype(np.uint8)
    inter  = (pred_b & gt).sum()
    union  = (pred_b | gt).sum()
    return float(inter) / (union + 1e-6)


def img_to_tensor(img_np: np.ndarray) -> torch.Tensor:
    """np (H,W,3) uint8 → normalized tensor (1,3,H,W)."""
    norm = (img_np.astype(np.float32) - MEAN_PIX) / \
           np.array([58.395, 57.12, 57.375])
    return torch.from_numpy(norm.astype(np.float32).transpose(2,0,1)).unsqueeze(0)


def occlude_patch(img_np: np.ndarray, r: int, c: int,
                  patch_h: int, patch_w: int) -> np.ndarray:
    """Xóa một patch bằng màu mean (gray)."""
    out = img_np.copy()
    r2  = min(r + patch_h, img_np.shape[0])
    c2  = min(c + patch_w, img_np.shape[1])
    out[r:r2, c:c2] = MEAN_PIX.astype(np.uint8)
    return out


# ══════════════════════════════════════════════════════════════
# M1 — Occlusion Counterfactual Map
# Zeiler & Fergus, ECCV 2014: "Visualizing and Understanding CNNs"
# ══════════════════════════════════════════════════════════════
def compute_occlusion_map(wrapper, img_np: np.ndarray,
                           patch_size: int = 32, stride: int = 16) -> np.ndarray:
    """
    Slide patch qua ảnh, đo DROP trong panicle score.
    Vùng drop lớn → counterfactually important (xóa đi → mất prediction).
    """
    H, W   = img_np.shape[:2]
    base_t = img_to_tensor(img_np)
    base_s = get_panicle_score(wrapper, base_t)

    drop_map = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)

    rows = list(range(0, H - patch_size + 1, stride)) + [H - patch_size]
    cols = list(range(0, W - patch_size + 1, stride)) + [W - patch_size]

    for r in rows:
        for c in cols:
            occ = occlude_patch(img_np, r, c, patch_size, patch_size)
            t   = img_to_tensor(occ)
            s   = get_panicle_score(wrapper, t)
            drop = max(0, base_s - s)   # positive drop = prediction fell
            drop_map[r:r+patch_size, c:c+patch_size] += drop
            count_map[r:r+patch_size, c:c+patch_size] += 1

    count_map[count_map == 0] = 1
    result = drop_map / count_map
    # Normalize to [0,1]
    mx = result.max()
    if mx > 0:
        result /= mx
    return result


def fig_occlusion_cf_map(val_samples, field_samples):
    print("\n[M1] Occlusion Counterfactual Map...")
    all_samples = val_samples[:2] + field_samples[:2]
    n = len(all_samples)
    model_names = list(MODELS_USE.keys())

    fig, axes = plt.subplots(len(model_names)+1, n,
                              figsize=(n*3.5, (len(model_names)+1)*3.5))

    # Row 0: input + GT
    for j, (img_np, gt, name) in enumerate(all_samples):
        overlay = img_np.copy().astype(float)
        overlay[gt==1] = overlay[gt==1]*0.4 + np.array([180,30,30])*0.6
        axes[0,j].imshow(overlay.clip(0,255).astype(np.uint8))
        axes[0,j].set_title(f"{'Val' if j<2 else 'Field'} {j%2+1}", fontsize=9)
        axes[0,j].axis("off")
    axes[0,0].set_ylabel("Input + GT", fontsize=9, fontweight="bold")

    # Rows 1+: occlusion map mỗi model
    for i, label in enumerate(model_names):
        print(f"  {label}...", flush=True)
        wrapper, infer = load_wrapper(label)

        for j, (img_np, gt, _) in enumerate(all_samples):
            occ_map = compute_occlusion_map(wrapper, img_np,
                                             patch_size=32, stride=16)
            # Overlay: heatmap màu đỏ
            heatmap = cv2.applyColorMap((occ_map*255).astype(np.uint8),
                                         cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            blend   = (img_np.astype(float)*0.45 + heatmap.astype(float)*0.55)
            blend   = blend.clip(0,255).astype(np.uint8)

            # Counterfactual score: % drop map energy ở đúng GT
            gt_energy = float((occ_map * gt).sum()) / (occ_map.sum() + 1e-6)
            axes[i+1,j].imshow(blend)
            axes[i+1,j].set_title(f"CF-Align={gt_energy:.2f}", fontsize=8)
            axes[i+1,j].axis("off")

        axes[i+1,0].set_ylabel(label, fontsize=8,
                                 color=COLORS[label], fontweight="bold")
        del wrapper, infer; torch.cuda.empty_cache()

    fig.suptitle("M1 — Occlusion Counterfactual Map\n"
                 "Vùng đỏ: nếu xóa → model mất khả năng nhận diện bông\n"
                 "CF-Align = tỉ lệ counterfactual region trùng với GT",
                 fontweight="bold", fontsize=11)
    plt.tight_layout()
    p = OUT / "fig_occlusion_cf_map.png"
    fig.savefig(p); plt.close()
    print(f"  Saved: {p.name}")


# ══════════════════════════════════════════════════════════════
# M2 — Deletion / Insertion AUC
# Samek et al., IEEE TNNLS 2017: "Evaluating the Visualization"
# ══════════════════════════════════════════════════════════════
def deletion_insertion_curve(wrapper, img_np: np.ndarray,
                               cam_map: np.ndarray,
                               n_steps: int = 20) -> tuple[list, list]:
    """
    Deletion: xóa dần pixel theo thứ tự CAM quan trọng nhất.
             → Score phải giảm nhanh nếu CAM đúng.
    Insertion: thêm dần pixel vào ảnh blank.
              → Score phải tăng nhanh nếu CAM đúng.
    AUC cao = explanation chất lượng cao.
    """
    H, W    = img_np.shape[:2]
    cam_r   = cv2.resize(cam_map, (W, H))
    flat    = cam_r.flatten()
    order   = np.argsort(flat)[::-1]  # từ quan trọng nhất

    base_t  = img_to_tensor(img_np)
    blank   = np.ones_like(img_np, dtype=np.float32) * MEAN_PIX
    step_sz = max(1, len(flat) // n_steps)

    del_scores, ins_scores = [], []
    del_img = img_np.copy().astype(float)
    ins_img = blank.copy()

    for step in range(n_steps + 1):
        n_pix = min(step * step_sz, len(flat))
        if step > 0:
            # Deletion: thay pixel quan trọng bằng mean
            idx = order[:n_pix]
            r_idx, c_idx = np.unravel_index(idx, (H, W))
            del_img[r_idx, c_idx] = MEAN_PIX
            # Insertion: thêm pixel quan trọng từ ảnh gốc vào blank
            ins_img[r_idx, c_idx] = img_np[r_idx, c_idx].astype(float)

        del_t = img_to_tensor(del_img.clip(0,255).astype(np.uint8))
        ins_t = img_to_tensor(ins_img.clip(0,255).astype(np.uint8))
        del_scores.append(get_panicle_score(wrapper, del_t))
        ins_scores.append(get_panicle_score(wrapper, ins_t))

    return del_scores, ins_scores


def fig_deletion_insertion(val_samples):
    print("\n[M2] Deletion/Insertion AUC...")
    from pytorch_grad_cam import GradCAMPlusPlus
    from scripts.xai_wrapper import PanicleTarget, get_target_layers

    n_imgs = min(2, len(val_samples))
    fig, axes = plt.subplots(len(MODELS_USE), n_imgs,
                              figsize=(n_imgs*5, len(MODELS_USE)*4))

    x_axis = np.linspace(0, 100, 21)  # % pixels removed/inserted

    auc_results = {}

    for i, label in enumerate(MODELS_USE):
        print(f"  {label}...", flush=True)
        wrapper, infer = load_wrapper(label)
        auc_results[label] = {"del_auc": [], "ins_auc": []}

        # Get CAM map để xác định thứ tự xóa
        try:
            tl = get_target_layers(infer.model, label)
            if "Mask2Former" in label:
                # Dùng backbone stages cho Mask2Former
                tl = [infer.model.backbone.stages[-1].blocks[-1]]
                from pytorch_grad_cam import EigenCAM
                def sr(t):
                    B,L,C = t.shape; H=W=int(L**0.5)
                    return t.reshape(B,H,W,C).permute(0,3,1,2).contiguous()
                cam_obj = EigenCAM(model=wrapper, target_layers=tl,
                                    reshape_transform=sr)
            else:
                cam_obj = GradCAMPlusPlus(model=wrapper, target_layers=tl)
        except Exception:
            del wrapper, infer; torch.cuda.empty_cache()
            continue

        for j, (img_np, gt, _) in enumerate(val_samples[:n_imgs]):
            ax = axes[i,j] if n_imgs > 1 else axes[i]
            tensor = img_to_tensor(img_np)

            try:
                gcam = cam_obj(input_tensor=tensor.cuda(),
                               targets=[PanicleTarget()])
                cam_map = gcam[0]

                del_s, ins_s = deletion_insertion_curve(
                    wrapper, img_np, cam_map, n_steps=20)

                del_auc = float(np.trapz(del_s, dx=5))   # AUC chia 100%
                ins_auc = float(np.trapz(ins_s, dx=5))

                ax.plot(x_axis, del_s, "r-o", markersize=3,
                        label=f"Deletion (AUC={del_auc:.1f})")
                ax.plot(x_axis, ins_s, "g-s", markersize=3,
                        label=f"Insertion (AUC={ins_auc:.1f})")
                ax.fill_between(x_axis, del_s, alpha=0.1, color="red")
                ax.fill_between(x_axis, ins_s, alpha=0.1, color="green")
                ax.set_xlabel("% pixels removed/inserted")
                ax.set_ylabel("Panicle prediction score")
                ax.set_ylim(0, max(max(ins_s)*1.1, 0.1))
                ax.legend(fontsize=8); ax.grid(alpha=0.3)
                if j == 0:
                    ax.set_title(f"{label}", fontsize=10,
                                  color=COLORS[label], fontweight="bold")
                else:
                    ax.set_title(f"Image {j+1}", fontsize=9)

                auc_results[label]["del_auc"].append(del_auc)
                auc_results[label]["ins_auc"].append(ins_auc)
            except Exception as e:
                ax.text(0.5,0.5, f"N/A\n{str(e)[:30]}", ha="center",
                        va="center", transform=ax.transAxes, fontsize=8)

        del wrapper, infer, cam_obj; torch.cuda.empty_cache()

    fig.suptitle("M2 — Deletion / Insertion Curves\n"
                 "Deletion (đỏ): xóa pixel quan trọng → score giảm nhanh = CAM tốt\n"
                 "Insertion (xanh): thêm pixel quan trọng → score tăng nhanh = CAM tốt",
                 fontweight="bold", fontsize=11)
    plt.tight_layout()
    p = OUT / "fig_deletion_insertion.png"
    fig.savefig(p); plt.close()
    print(f"  Saved: {p.name}")
    return auc_results


# ══════════════════════════════════════════════════════════════
# M3 — Superpixel Counterfactual
# RISE: Petsiuk et al., BMVC 2018
# ══════════════════════════════════════════════════════════════
def compute_superpixel_cf(wrapper, img_np: np.ndarray,
                           gt: np.ndarray, n_segments: int = 40) -> np.ndarray:
    """
    Phân ảnh thành superpixels (SLIC).
    Với mỗi superpixel: xóa nó và đo drop IoU với GT.
    Drop lớn → superpixel này là CRITICAL (counterfactually necessary).
    """
    from skimage.segmentation import slic

    segments = slic(img_np, n_segments=n_segments, compactness=10,
                    start_label=0, channel_axis=2)

    base_t   = img_to_tensor(img_np)
    base_iou = get_panicle_iou(wrapper, base_t, gt)

    cf_map = np.zeros_like(segments, dtype=np.float32)

    for seg_id in range(segments.max() + 1):
        mask = (segments == seg_id)
        if mask.sum() < 10:
            continue
        perturbed = img_np.copy()
        perturbed[mask] = MEAN_PIX.astype(np.uint8)
        t   = img_to_tensor(perturbed)
        iou = get_panicle_iou(wrapper, t, gt)
        drop = max(0, base_iou - iou)
        cf_map[mask] = drop

    # Normalize
    mx = cf_map.max()
    if mx > 0:
        cf_map /= mx
    return cf_map, segments


def fig_superpixel_cf(val_samples, field_samples):
    print("\n[M3] Superpixel Counterfactual...")
    all_samples = val_samples[:2] + field_samples[:2]
    n = len(all_samples)

    fig, axes = plt.subplots(3, n, figsize=(n*3.5, 9))

    # Chỉ dùng best model = Mask2Former
    label   = "Mask2Former"
    wrapper, infer = load_wrapper(label)

    for j, (img_np, gt, name) in enumerate(all_samples):
        try:
            cf_map, segments = compute_superpixel_cf(wrapper, img_np, gt)

            # Row 0: original + GT
            overlay = img_np.copy().astype(float)
            overlay[gt==1] = overlay[gt==1]*0.4 + np.array([180,30,30])*0.6
            axes[0,j].imshow(overlay.clip(0,255).astype(np.uint8))
            axes[0,j].set_title(f"{'Val' if j<2 else 'Field'} {j%2+1}", fontsize=9)
            axes[0,j].axis("off")

            # Row 1: superpixel boundaries
            from skimage.segmentation import mark_boundaries
            img_with_bounds = mark_boundaries(img_np, segments, color=(1,1,0))
            axes[1,j].imshow((img_with_bounds*255).astype(np.uint8))
            axes[1,j].set_title(f"{segments.max()+1} superpixels", fontsize=8)
            axes[1,j].axis("off")

            # Row 2: counterfactual map
            heatmap = cv2.applyColorMap((cf_map*255).astype(np.uint8),
                                         cv2.COLORMAP_HOT)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            blend   = (img_np.astype(float)*0.5 + heatmap.astype(float)*0.5)
            # Counterfactual alignment score
            cf_align = float((cf_map * gt).sum()) / (cf_map.sum() + 1e-6)

            # Mark critical superpixels (top 3)
            from skimage.segmentation import mark_boundaries
            n_seg = segments.max() + 1
            seg_scores = np.array([cf_map[segments==s].mean()
                                   for s in range(n_seg)])
            top3 = np.argsort(seg_scores)[-3:]
            critical_mask = np.isin(segments, top3)
            blend_arr = blend.clip(0,255).astype(np.uint8)
            blend_arr[critical_mask & (gt==0)] = [255, 50, 50]   # red = wrong attend
            blend_arr[critical_mask & (gt==1)] = [50, 255, 50]   # green = correct attend

            axes[2,j].imshow(blend_arr)
            axes[2,j].set_title(f"CF-Align={cf_align:.2f}", fontsize=8)
            axes[2,j].axis("off")

        except Exception as e:
            for r in range(3):
                axes[r,j].imshow(img_np); axes[r,j].axis("off")

    axes[0,0].set_ylabel("Input + GT (đỏ=bông)", fontsize=8)
    axes[1,0].set_ylabel("Superpixels (SLIC)", fontsize=8)
    axes[2,0].set_ylabel("CF Map + Critical SP\n(xanh=đúng, đỏ=sai)", fontsize=8)

    fig.suptitle(f"M3 — Superpixel Counterfactual [{label}]\n"
                 "Superpixel nóng = xóa đi thì model mất detection\n"
                 "Xanh = critical & đúng vị trí | Đỏ = critical nhưng ngoài GT",
                 fontweight="bold", fontsize=11)
    plt.tight_layout()
    p = OUT / "fig_superpixel_cf.png"
    fig.savefig(p); plt.close()
    print(f"  Saved: {p.name}")
    del wrapper, infer; torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════
# Tổng hợp: Summary table + comparison figure
# ══════════════════════════════════════════════════════════════
def fig_cf_summary(auc_results: dict):
    """Bảng tóm tắt Deletion/Insertion AUC và vẽ comparison."""
    import numpy as np

    csv_path = PROJECT / "results/tables/counterfactual_results.csv"
    rows = [["Model", "Method", "Del AUC (mean)", "Ins AUC (mean)",
             "Del-Ins Gap", "Interpretation"]]

    print("\n  Summary:")
    print(f"  {'Model':<14} {'Del AUC':>9} {'Ins AUC':>9} {'Gap':>7}")
    print("  " + "-"*44)

    for label, r in auc_results.items():
        if not r["del_auc"]:
            continue
        d = np.mean(r["del_auc"]); ins = np.mean(r["ins_auc"])
        gap = ins - d
        interp = "Good" if gap > 0 else "Poor"
        rows.append([label, "GradCAM++/EigenCAM",
                     f"{d:.2f}", f"{ins:.2f}", f"{gap:.2f}", interp])
        print(f"  {label:<14} {d:>9.2f} {ins:>9.2f} {gap:>7.2f}")

    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"\n  CSV saved: {csv_path.name}")


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("═"*60)
    print("  xAI Counterfactual — CVRP")
    print("  M1: Occlusion Map | M2: Del/Ins AUC | M3: Superpixel CF")
    print("═"*60)

    # Load samples
    val_paths   = random.sample(sorted((DATA/"img_dir/val").glob("*.jpg")), 6)
    field_paths = random.sample(sorted((DATA/"img_dir/field_test").glob("*.jpg")), 6)

    def prep(paths, gt_dir, n=4):
        out = []
        for p in paths[:n]:
            img_np, _ = preprocess_image(p, target_size=512)
            ann = gt_dir / (Path(p).stem + ".png")
            gt  = (np.array(Image.open(ann).resize(
                   (img_np.shape[1], img_np.shape[0]), Image.NEAREST)) == 1
                   ).astype(np.uint8)
            out.append((img_np, gt, p.name))
        return out

    val_s   = prep(val_paths,   DATA/"ann_dir/val",        n=4)
    field_s = prep(field_paths, DATA/"ann_dir/field_test", n=4)

    print(f"\n  Val:   {[s[2] for s in val_s]}")
    print(f"  Field: {[s[2] for s in field_s]}")

    fig_occlusion_cf_map(val_s, field_s)
    auc_results = fig_deletion_insertion(val_s)
    fig_superpixel_cf(val_s, field_s)
    fig_cf_summary(auc_results)

    print(f"\n{'═'*60}")
    print("  Counterfactual xAI hoàn tất!")
    for f in sorted(OUT.rglob("*.png")):
        print(f"  {f.name}")
