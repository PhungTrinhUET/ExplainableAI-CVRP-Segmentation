#!/bin/bash
# ================================================================
# Chạy training 4 model CVRP lần lượt trên 1 GPU (RTX 2080 Ti)
# Thời gian ước tính: DeepLab~8h, SegFormer~8h, KNet~10h, M2F~12h
#
# Cách chạy:
#   conda activate cvrp_seg
#   cd /home/tower2080/Documents/CVRP_xAI
#   nohup bash scripts/train_all.sh > train_all.log 2>&1 &
#   tail -f train_all.log
#
# Chạy 1 model:
#   bash scripts/train_all.sh deeplabv3plus
# ================================================================
set -e

PROJECT=/home/tower2080/Documents/CVRP_xAI
PYTHON=/home/tower2080/miniconda3/envs/cvrp_seg/bin/python
TRAIN=$PROJECT/mmsegmentation/tools/train.py
CFGS=$PROJECT/experiments/segmentation/configs
WDIR=$PROJECT/experiments/segmentation/work_dirs

export PYTHONPATH=$PROJECT:$PYTHONPATH

declare -A CONFIGS=(
    [deeplabv3plus]="deeplabv3plus_cvrp.py"
    [segformer]="segformer_cvrp.py"
    [knet]="knet_cvrp.py"
    [mask2former]="mask2former_cvrp.py"
)
# Thứ tự: nhẹ → nặng
ORDER=(deeplabv3plus segformer knet mask2former)
[ $# -gt 0 ] && ORDER=("$@")

echo "========================================================"
echo "CVRP Training — $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "========================================================"

for MODEL in "${ORDER[@]}"; do
    CFG="${CONFIGS[$MODEL]}"
    OUT="$WDIR/$MODEL"
    mkdir -p "$OUT"

    echo ""
    echo "──────────────────────────────────────────────────────"
    echo " Model: $MODEL | Start: $(date '+%H:%M:%S %d/%m')"
    echo "──────────────────────────────────────────────────────"

    RESUME=""
    if ls "$OUT"/iter_*.pth 1>/dev/null 2>&1; then
        LATEST=$(ls -t "$OUT"/iter_*.pth 2>/dev/null | head -1)
        echo " Tìm thấy checkpoint → resume từ: $(basename $LATEST)"
        RESUME="--resume"
    fi

    $PYTHON "$TRAIN" "$CFGS/$CFG" \
        --work-dir "$OUT" \
        $RESUME \
        2>&1 | tee "$OUT/train_$(date +%Y%m%d_%H%M).log"

    echo " Xong: $MODEL — $(date '+%H:%M:%S %d/%m')"
    echo " VRAM sau khi xong: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
done

echo ""
echo "========================================================"
echo " Tất cả model xong! $(date)"
echo " Chạy tiếp: python scripts/evaluate_all.py"
echo "========================================================"