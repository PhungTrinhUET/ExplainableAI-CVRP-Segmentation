"""
Parse log files từ MMSeg và vẽ training curves.
Có thể chạy TRONG LÚC training đang chạy để xem tiến trình.

Chạy: python scripts/plot_training_curves.py
      python scripts/plot_training_curves.py --model deeplabv3plus
"""
import re, argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

WDIR    = Path("experiments/segmentation/work_dirs")
OUT_DIR = Path("results/figures/03_segmentation_curves")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_COLORS = {
    "deeplabv3plus": "#2196F3",
    "segformer":     "#4CAF50",
    "knet":          "#FF5722",
    "mask2former":   "#9C27B0",
}
MODEL_LABELS = {
    "deeplabv3plus": "DeepLabV3+",
    "segformer":     "SegFormer",
    "knet":          "K-Net",
    "mask2former":   "Mask2Former",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "figure.dpi": 130,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def parse_log(log_path: Path):
    """Trích loss và val mIoU từ MMSeg log file."""
    train_iters, train_loss = [], []
    val_iters,   val_miou   = [], []

    iter_re  = re.compile(r"Iter\(train\)\s+\[\s*(\d+)/\d+\].*?loss:\s+([\d.]+)")
    val_re   = re.compile(r"Iter\(val\)\s+\[\s*(\d+)/\d+\].*?mIoU:\s+([\d.]+)")

    text = log_path.read_text(errors="ignore")
    for m in iter_re.finditer(text):
        train_iters.append(int(m.group(1)))
        train_loss.append(float(m.group(2)))
    for m in val_re.finditer(text):
        val_iters.append(int(m.group(1)))
        val_miou.append(float(m.group(2)))

    return train_iters, train_loss, val_iters, val_miou


def find_latest_log(model_dir: Path):
    logs = sorted(model_dir.glob("train_*.log"))
    return logs[-1] if logs else None


def plot_curves(models_data: dict):
    """Vẽ 2 subplots: Loss (train) + mIoU (val) cho tất cả model."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    has_loss = has_miou = False

    for model_name, data in models_data.items():
        train_iters, train_loss, val_iters, val_miou = data
        color = MODEL_COLORS.get(model_name, "gray")
        label = MODEL_LABELS.get(model_name, model_name)

        if train_iters:
            # Smooth loss với rolling mean window=10
            smoothed = []
            window = min(10, len(train_loss))
            for i in range(len(train_loss)):
                start = max(0, i - window // 2)
                end   = min(len(train_loss), i + window // 2 + 1)
                smoothed.append(sum(train_loss[start:end]) / (end - start))

            ax1.plot(train_iters, train_loss, alpha=0.2, color=color, linewidth=0.8)
            ax1.plot(train_iters, smoothed, color=color, linewidth=2,
                     label=f"{label} (last: {train_loss[-1]:.3f})")
            has_loss = True

        if val_iters:
            ax2.plot(val_iters, val_miou, color=color, linewidth=2,
                     marker="o", markersize=4,
                     label=f"{label} (best: {max(val_miou):.2f}%)")
            has_miou = True

    # Paper reference lines
    paper_iou = {
        "DeepLabV3+": 71.02, "Mask2Former": 73.38,
        "SegFormer": 71.19, "K-Net": 72.53
    }
    for name, iou in paper_iou.items():
        color = MODEL_COLORS.get(
            [k for k, v in MODEL_LABELS.items() if v == name][0] if
            [k for k, v in MODEL_LABELS.items() if v == name] else "", "gray"
        )
        ax2.axhline(iou, linestyle="--", color=color, alpha=0.4, linewidth=1)

    # ax1 — Loss
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Training Loss")
    ax1.set_title("Training Loss theo Iteration")
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x))
    ))
    ax1.grid(alpha=0.3)
    if has_loss:
        ax1.legend(fontsize=9)

    # ax2 — mIoU
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Val mIoU (%)")
    ax2.set_title("Validation mIoU theo Iteration\n(nét đứt = kết quả paper)")
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x))
    ))
    ax2.set_ylim(bottom=0)
    ax2.grid(alpha=0.3)
    if has_miou:
        ax2.legend(fontsize=9)

    fig.suptitle("CVRP — So sánh 4 model Semantic Segmentation",
                 fontweight="bold", fontsize=14)
    plt.tight_layout()

    path = OUT_DIR / "fig_training_curves.png"
    fig.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def print_summary(models_data: dict):
    """In bảng tóm tắt tiến trình."""
    print(f"\n{'Model':<16} {'Iter':>8} {'Loss(last)':>11} {'mIoU(best)':>11} {'ETA'}")
    print("-" * 60)
    for model_name, (ti, tl, vi, vm) in models_data.items():
        label    = MODEL_LABELS.get(model_name, model_name)
        cur_iter = ti[-1] if ti else 0
        cur_loss = f"{tl[-1]:.4f}" if tl else "—"
        best_iou = f"{max(vm):.2f}%" if vm else "—"
        print(f"{label:<16} {cur_iter:>8} {cur_loss:>11} {best_iou:>11}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="", help="Tên model cụ thể, bỏ trống = tất cả")
    args = parser.parse_args()

    models_data = {}
    for model_dir in sorted(WDIR.iterdir()):
        if not model_dir.is_dir():
            continue
        if args.model and model_dir.name != args.model:
            continue
        log = find_latest_log(model_dir)
        if log is None:
            continue
        ti, tl, vi, vm = parse_log(log)
        if ti:
            models_data[model_dir.name] = (ti, tl, vi, vm)
            print(f"  Parsed {model_dir.name}: {len(ti)} train pts, {len(vi)} val pts")

    if not models_data:
        print("Chưa có log nào. Training đang chạy? Thử lại sau.")
    else:
        print_summary(models_data)
        plot_curves(models_data)
        print(f"\nFigures lưu tại: {OUT_DIR}/")