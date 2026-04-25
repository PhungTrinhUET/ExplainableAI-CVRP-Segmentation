#!/bin/bash
# ================================================================
# Tải pretrained weights cho 4 model
# Chạy: bash scripts/download_pretrained.sh
# ================================================================
set -e
mkdir -p checkpoints
cd checkpoints

echo "========================================================"
echo "Tải pretrained weights..."
echo "========================================================"

# ── 1. Swin-B (dùng cho Mask2Former + K-Net) ─────────────────
FILE="swin_base_patch4_window12_384_22k.pth"
if [ ! -f "$FILE" ]; then
    echo "[1/3] Tải Swin-B pretrained (ImageNet-22K)..."
    wget -q --show-progress \
        "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_base_patch4_window12_384_22k_20220317-e5c09f74.pth" \
        -O "$FILE"
    echo "    OK: $FILE"
else
    echo "[1/3] $FILE đã có sẵn, bỏ qua."
fi

# ── 2. MiT-B2 (dùng cho SegFormer) ───────────────────────────
FILE="mit_b2.pth"
if [ ! -f "$FILE" ]; then
    echo "[2/3] Tải MiT-B2 pretrained (ImageNet-1K)..."
    wget -q --show-progress \
        "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b2_20220624-66e8bf70.pth" \
        -O "$FILE"
    echo "    OK: $FILE"
else
    echo "[2/3] $FILE đã có sẵn, bỏ qua."
fi

# ── 3. ResNet-101 (dùng cho DeepLabV3+) ──────────────────────
FILE="resnet101_v1c.pth"
if [ ! -f "$FILE" ]; then
    echo "[3/3] Tải ResNet-101 pretrained (ImageNet-1K)..."
    wget -q --show-progress \
        "https://download.openmmlab.com/pretrain/third_party/resnet101_v1c-e67eebb6.pth" \
        -O "$FILE"
    echo "    OK: $FILE"
else
    echo "[3/3] $FILE đã có sẵn, bỏ qua."
fi

cd ..
echo ""
echo "========================================================"
echo "Tất cả pretrained weights đã sẵn sàng trong checkpoints/"
ls -lh checkpoints/
echo "========================================================"