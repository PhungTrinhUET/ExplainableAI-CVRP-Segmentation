#!/bin/bash
# Xem tóm tắt tiến trình training — chạy ở terminal mới
# Dùng: bash scripts/monitor_training.sh
# Dùng: bash scripts/monitor_training.sh deeplabv3plus

MODEL=${1:-""}
WDIR=/home/tower2080/Documents/CVRP_xAI/experiments/segmentation/work_dirs

clear
echo "════════════════════════════════════════════════════"
echo "  CVRP Training Monitor — $(date '+%H:%M %d/%m/%Y')"
echo "════════════════════════════════════════════════════"

echo ""
echo "── GPU Status ──────────────────────────────────────"
nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader | awk -F',' '{
    printf "  GPU: %s\n  Temp: %s°C | Usage: %s | VRAM: %s / %s\n", $1,$2,$3,$4,$5}'

echo ""
echo "── Training Processes ──────────────────────────────"
ps aux | grep "train.py" | grep -v grep | awk '{printf "  PID %s | CPU %.0f%% | RAM %.0fMB\n", $2, $3, $6/1024}'

echo ""
echo "── Checkpoints đã lưu ──────────────────────────────"
for d in "$WDIR"/*/; do
    name=$(basename "$d")
    [ -n "$MODEL" ] && [ "$name" != "$MODEL" ] && continue
    ckpts=$(ls "$d"iter_*.pth 2>/dev/null | wc -l)
    best=$(ls "$d"best_*.pth 2>/dev/null | head -1 | xargs -I{} basename {} 2>/dev/null)
    last_iter=$(ls -t "$d"iter_*.pth 2>/dev/null | head -1 | grep -o 'iter_[0-9]*' | grep -o '[0-9]*')
    printf "  %-14s: %d ckpts | last iter: %-6s | best: %s\n" \
        "$name" "$ckpts" "${last_iter:-N/A}" "${best:-N/A}"
done

echo ""
echo "── Log mới nhất (50 dòng cuối) ─────────────────────"
if [ -n "$MODEL" ]; then
    LOG=$(ls -t "$WDIR/$MODEL"/train_*.log 2>/dev/null | head -1)
else
    LOG=$(ls -t "$WDIR"/*/train_*.log 2>/dev/null | head -1)
fi

if [ -f "$LOG" ]; then
    echo "  File: $LOG"
    grep -E "Iter\(train\)|Iter\(val\)|mIoU|ERROR" "$LOG" 2>/dev/null | tail -20 | \
    awk '{
        # Highlight loss và mIoU
        gsub(/loss: [0-9.]+/, "\033[33m&\033[0m")
        gsub(/mIoU: [0-9.]+/, "\033[32m&\033[0m")
        gsub(/ERROR/, "\033[31mERROR\033[0m")
        print "  "$0
    }'
else
    echo "  Chưa có log file."
fi

echo ""
echo "════════════════════════════════════════════════════"
echo "  Refresh: bash scripts/monitor_training.sh"
echo "  Watch mode: watch -n 30 bash scripts/monitor_training.sh"