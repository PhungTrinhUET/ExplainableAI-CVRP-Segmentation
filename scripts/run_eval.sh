#!/bin/bash
# Evaluate tất cả 4 model trên 2 split: val + field_test
# Chạy: bash scripts/run_eval.sh
set -e
cd /home/tower2080/Documents/CVRP_xAI
export PYTHONPATH=/home/tower2080/Documents/CVRP_xAI:$PYTHONPATH
PYTHON=/home/tower2080/miniconda3/envs/cvrp_seg/bin/python
TEST=$PWD/mmsegmentation/tools/test.py

declare -A CKPTS=(
    [deeplabv3plus]="best_mIoU_iter_64000.pth"
    [segformer]="best_mIoU_iter_80000.pth"
    [knet]="best_mIoU_iter_80000.pth"
    [mask2former]="best_mIoU_iter_72000.pth"
)

parse_result() {
    # In: panicle IoU, Acc, mIoU, aAcc
    local output="$1"
    local p_iou=$(echo "$output" | grep "panicle" | grep -oP '\|\s*\K[\d.]+' | head -1)
    local p_acc=$(echo "$output" | grep "panicle" | grep -oP '\|\s*\K[\d.]+' | tail -1)
    local miou=$(echo "$output"  | grep -oP 'mIoU:\s*\K[\d.]+' | tail -1)
    local aacc=$(echo "$output"  | grep -oP 'aAcc:\s*\K[\d.]+' | tail -1)
    echo "panicle_IoU=$p_iou | panicle_Acc=$p_acc | mIoU=$miou | aAcc=$aacc"
}

echo "════════════════════════════════════════════════"
echo "  CVRP Evaluation — $(date '+%H:%M %d/%m/%Y')"
echo "════════════════════════════════════════════════"

for model in deeplabv3plus segformer knet mask2former; do
    ckpt="${CKPTS[$model]}"
    cfg="experiments/segmentation/configs/${model}_cvrp.py"
    ckpt_path="experiments/segmentation/work_dirs/$model/$ckpt"

    echo ""
    echo "──────────── $model ────────────"

    # --- VAL SET (Table 2) ---
    echo -n "  [val]        "
    out_val=$($PYTHON "$TEST" "$cfg" "$ckpt_path" \
        --work-dir "experiments/segmentation/work_dirs/$model/eval_val" \
        --cfg-options \
            "test_dataloader.dataset.data_prefix.img_path=img_dir/val" \
            "test_dataloader.dataset.data_prefix.seg_map_path=ann_dir/val" \
        2>&1)
    parse_result "$out_val"

    # --- FIELD TEST (Table 3, default test_dataloader) ---
    echo -n "  [field_test]  "
    out_field=$($PYTHON "$TEST" "$cfg" "$ckpt_path" \
        --work-dir "experiments/segmentation/work_dirs/$model/eval_field" \
        2>&1)
    parse_result "$out_field"
done

echo ""
echo "════════════════════════════════════════════════"
echo "  Xong! Kết quả lưu trong work_dirs/*/eval_*/"
