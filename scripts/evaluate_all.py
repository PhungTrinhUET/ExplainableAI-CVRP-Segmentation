"""
Evaluate tất cả model đã train, tạo bảng kết quả đầy đủ.
Tái tạo Table 2 (multi-cultivar) và Table 3 (field scenes) trong paper.

Chạy: python scripts/evaluate_all.py
      python scripts/evaluate_all.py --model deeplabv3plus
"""
import os, sys, re, csv, subprocess, argparse
from pathlib import Path

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))
os.environ["PYTHONPATH"] = str(PROJECT) + ":" + os.environ.get("PYTHONPATH", "")

PYTHON   = "/home/tower2080/miniconda3/envs/cvrp_seg/bin/python"
MMSEG    = PROJECT / "mmsegmentation"
CFG_DIR  = PROJECT / "experiments/segmentation/configs"
WDIR     = PROJECT / "experiments/segmentation/work_dirs"
TABLE_DIR = PROJECT / "results/tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["deeplabv3plus", "segformer", "knet", "mask2former"]
LABELS = {
    "deeplabv3plus": "DeepLabV3+",
    "segformer":     "SegFormer",
    "knet":          "K-Net",
    "mask2former":   "Mask2Former",
}

# Kết quả paper để so sánh
PAPER = {
    "val": {
        "deeplabv3plus": dict(P=79.83, R=86.55, F=83.05, IoU=71.02),
        "mask2former":   dict(P=81.12, R=88.49, F=84.64, IoU=73.38),
        "segformer":     dict(P=78.76, R=88.11, F=83.17, IoU=71.19),
        "knet":          dict(P=82.37, R=85.85, F=84.08, IoU=72.53),
    },
    "field_test": {
        "deeplabv3plus": dict(P=82.73, R=74.18, F=76.89, IoU=62.82),
        "mask2former":   dict(P=89.39, R=84.60, F=86.49, IoU=76.56),
        "segformer":     dict(P=79.28, R=84.08, F=80.91, IoU=68.24),
        "knet":          dict(P=85.52, R=77.50, F=80.13, IoU=67.34),
    }
}


def find_checkpoint(model_name: str) -> Path | None:
    d = WDIR / model_name
    bests = sorted(d.glob("best_mIoU_iter_*.pth"))
    if bests:
        return bests[-1]
    iters = sorted(d.glob("iter_*.pth"))
    return iters[-1] if iters else None


def run_test(model_name: str, split: str) -> dict:
    """
    Chạy mmseg test.py và parse confusion matrix để tính
    Precision, Recall, F-score, IoU cho class panicle (index 1).
    """
    ckpt = find_checkpoint(model_name)
    if ckpt is None:
        return {}

    cfg     = CFG_DIR / f"{model_name}_cvrp.py"
    out_dir = WDIR / model_name / f"eval_{split}"
    out_dir.mkdir(exist_ok=True)

    # Override test dataloader để trỏ đúng split
    cfg_opts = (
        f"test_dataloader.dataset.data_prefix.img_path=img_dir/{split} "
        f"test_dataloader.dataset.data_prefix.seg_map_path=ann_dir/{split}"
    )

    cmd = [
        PYTHON, str(MMSEG / "tools/test.py"),
        str(cfg), str(ckpt),
        "--work-dir", str(out_dir),
        "--cfg-options", cfg_opts,
    ]

    env = {**os.environ, "PYTHONPATH": str(PROJECT)}
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    output = result.stdout + result.stderr

    # Parse từ output
    metrics = parse_metrics(output)
    return metrics


def parse_metrics(output: str) -> dict:
    """
    Parse output của mmseg test.py.
    MMSeg 1.x in ra dạng:
      per class results:
      +------------+-------+-------+
      | background | 99.xx | 99.xx |
      | panicle    | xx.xx | xx.xx |
    và summary:
      aAcc: xx.xx  mIoU: xx.xx  mAcc: xx.xx
    """
    metrics = {}

    # aAcc, mIoU, mAcc
    for key in ["aAcc", "mIoU", "mAcc"]:
        m = re.search(rf"{key}:\s+([\d.]+)", output)
        if m:
            metrics[key] = float(m.group(1))

    # Per-class IoU từ bảng
    lines = output.splitlines()
    for i, line in enumerate(lines):
        if "panicle" in line.lower():
            nums = re.findall(r"[\d.]+", line)
            if len(nums) >= 2:
                try:
                    metrics["panicle_IoU"] = float(nums[-2])
                    metrics["panicle_Acc"] = float(nums[-1])
                except Exception:
                    pass

    # Tính Precision/Recall/F từ IoU và Acc (approximate)
    # IoU = TP/(TP+FP+FN), Recall = TP/(TP+FN) = Acc_panicle
    iou = metrics.get("panicle_IoU", metrics.get("mIoU", 0))
    rec = metrics.get("panicle_Acc", 0)      # Recall ≈ per-class accuracy
    if iou > 0 and rec > 0:
        # IoU = TP/(TP+FP+FN), Recall = TP/(TP+FN)
        # → TP+FP+FN = TP/IoU, TP+FN = TP/Recall
        # → Precision = TP/(TP+FP) = IoU*Recall/(Recall - IoU + IoU*Recall/100)
        # Simplified: P = IoU / (Recall/100 + IoU/100 - IoU*Recall/10000) * (1/100)
        R = rec / 100
        I = iou / 100
        if R + I - R * I > 0:
            P = I / (R + I - R * I)
            metrics["Precision"] = round(P * 100, 2)
            metrics["Recall"]    = round(rec, 2)
            metrics["IoU"]       = round(iou, 2)
            metrics["Fscore"]    = round(
                2 * P * R / (P + R) * 100 if (P + R) > 0 else 0, 2
            )

    return metrics


def save_table(results: dict, split: str, filename: str):
    """Lưu CSV và in bảng so sánh với paper."""
    header = [
        "Model",
        "Precision", "Recall", "F-score", "IoU",
        "Paper_P",   "Paper_R", "Paper_F",  "Paper_IoU",
        "IoU_diff"
    ]
    rows = [header]

    print(f"\n{'─'*72}")
    print(f"  {'Split: ' + split.upper():<20}")
    print(f"  {'Model':<14} {'Precision':>10} {'Recall':>8} "
          f"{'F-score':>8} {'IoU':>6}   {'Paper IoU':>10} {'Δ':>6}")
    print(f"  {'─'*68}")

    for model_name in MODELS:
        m = results.get(model_name, {})
        p = PAPER[split].get(model_name, {})
        if not m:
            print(f"  {LABELS[model_name]:<14} {'—':>10} {'—':>8} {'—':>8} {'—':>6}")
            continue

        iou_diff = ""
        if m.get("IoU") and p.get("IoU"):
            iou_diff = f"{m['IoU'] - p['IoU']:+.2f}"

        rows.append([
            LABELS[model_name],
            m.get("Precision", "—"), m.get("Recall", "—"),
            m.get("Fscore", "—"),    m.get("IoU", "—"),
            p.get("P", "—"),         p.get("R", "—"),
            p.get("F", "—"),         p.get("IoU", "—"),
            iou_diff
        ])

        print(f"  {LABELS[model_name]:<14} "
              f"{str(m.get('Precision','—')):>10} "
              f"{str(m.get('Recall','—')):>8} "
              f"{str(m.get('Fscore','—')):>8} "
              f"{str(m.get('IoU','—')):>6}   "
              f"{str(p.get('IoU','—')):>10} "
              f"{iou_diff:>6}")

    path = TABLE_DIR / filename
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print(f"\n  CSV lưu tại: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="", help="Tên model cụ thể")
    args = parser.parse_args()

    target = [args.model] if args.model else MODELS

    print("═" * 72)
    print("  CVRP — Evaluation Pipeline")
    print("═" * 72)

    val_results  = {}
    test_results = {}

    for model_name in target:
        ckpt = find_checkpoint(model_name)
        if ckpt is None:
            print(f"\n[SKIP] {LABELS.get(model_name, model_name)}: Chưa có checkpoint.")
            continue

        print(f"\n[{LABELS.get(model_name, model_name)}] checkpoint: {ckpt.name}")

        print("  → Đánh giá trên val set (multi-cultivar)...")
        val_results[model_name]  = run_test(model_name, "val")

        print("  → Đánh giá trên field_test (field scenes)...")
        test_results[model_name] = run_test(model_name, "field_test")

    if val_results:
        save_table(val_results,  "val",        "seg_multicult_results.csv")
        save_table(test_results, "field_test", "seg_fieldscene_results.csv")
    else:
        print("\nChưa có kết quả. Train xong ít nhất 1 model rồi chạy lại.")

    print("\n" + "═" * 72)