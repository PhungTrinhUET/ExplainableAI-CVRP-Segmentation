"""
Vẽ bảng kết quả dạng hình ảnh + bar chart so sánh với paper.
Chạy: python scripts/plot_results_table.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUT_DIR = Path("results/figures/03_segmentation_curves")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["DeepLabV3+", "SegFormer", "K-Net", "Mask2Former"]
COLORS = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0"]

# ── Kết quả của chúng ta ──────────────────────────────────────
OUR_VAL   = {"DeepLabV3+": 65.66, "SegFormer": 66.61,
             "K-Net": 75.56, "Mask2Former": 77.58}
OUR_FIELD = {"DeepLabV3+": 63.46, "SegFormer": 63.12,
             "K-Net": 69.23, "Mask2Former": 72.00}

# ── Kết quả paper ────────────────────────────────────────────
PAP_VAL   = {"DeepLabV3+": 71.02, "SegFormer": 71.19,
             "K-Net": 72.53, "Mask2Former": 73.38}
PAP_FIELD = {"DeepLabV3+": 62.82, "SegFormer": 68.24,
             "K-Net": 67.34, "Mask2Former": 76.56}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "savefig.dpi": 200, "savefig.bbox": "tight"})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

x = np.arange(len(MODELS))
w = 0.35

for ax, our, paper, title in [
    (ax1, OUR_VAL,   PAP_VAL,   "Val Set — Multi-Cultivar (Table 2)"),
    (ax2, OUR_FIELD, PAP_FIELD, "Field Scenes (Table 3)"),
]:
    our_vals   = [our[m]   for m in MODELS]
    paper_vals = [paper[m] for m in MODELS]

    bars1 = ax.bar(x - w/2, our_vals,   w, label="Kết quả của chúng ta",
                   color=COLORS, alpha=0.85, edgecolor="white")
    bars2 = ax.bar(x + w/2, paper_vals, w, label="Paper (Tang et al. 2025)",
                   color=COLORS, alpha=0.40, edgecolor=COLORS, linewidth=1.5,
                   hatch="//")

    for bar, val in zip(bars1, our_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar, val in zip(bars2, paper_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, fontsize=10)
    ax.set_ylabel("Panicle IoU (%)")
    ax.set_ylim(55, 90)
    ax.set_title(title, fontweight="bold", fontsize=12)
    ax.grid(axis="y", alpha=0.3)

# Legend chung
solid = mpatches.Patch(color="gray", alpha=0.85, label="Kết quả của chúng ta")
hatch = mpatches.Patch(color="gray", alpha=0.40, hatch="//", label="Paper (Tang et al. 2025)")
fig.legend(handles=[solid, hatch], loc="lower center", ncol=2,
           fontsize=10, bbox_to_anchor=(0.5, -0.05))

fig.suptitle("So sánh Panicle IoU — Kết quả thực nghiệm vs Paper",
             fontweight="bold", fontsize=14)
plt.tight_layout(rect=[0, 0.06, 1, 0.97])

path = OUT_DIR / "fig_results_comparison.png"
fig.savefig(path)
plt.close()
print(f"Saved: {path}")

# ── Radar chart ───────────────────────────────────────────────
fig2, axes = plt.subplots(1, 2, figsize=(12, 5),
                           subplot_kw=dict(polar=True))

for ax, our, paper, title in [
    (axes[0], OUR_VAL,   PAP_VAL,   "Val Set"),
    (axes[1], OUR_FIELD, PAP_FIELD, "Field Test"),
]:
    N = len(MODELS)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    our_v   = [our[m]   for m in MODELS] + [our[MODELS[0]]]
    paper_v = [paper[m] for m in MODELS] + [paper[MODELS[0]]]

    ax.plot(angles, our_v,   'o-', linewidth=2, color="#9C27B0", label="Chúng ta")
    ax.fill(angles, our_v,          alpha=0.25, color="#9C27B0")
    ax.plot(angles, paper_v, 's--', linewidth=2, color="#FF9800", label="Paper")
    ax.fill(angles, paper_v,        alpha=0.15, color="#FF9800")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(MODELS, fontsize=9)
    ax.set_ylim(55, 85)
    ax.set_title(title, fontweight="bold", pad=15)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

fig2.suptitle("Radar Chart — Panicle IoU theo Model",
              fontweight="bold", fontsize=13)
plt.tight_layout()
path2 = OUT_DIR / "fig_radar_comparison.png"
fig2.savefig(path2)
plt.close()
print(f"Saved: {path2}")
