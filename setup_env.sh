#!/bin/bash
# ================================================================
# Tạo môi trường conda cvrp_seg cho dự án CVRP
# GPU: RTX 2080 Ti (11GB), CUDA 12.2
# ================================================================
set -e

ENV_NAME="cvrp_seg"
WORK_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "========================================================"
echo "  Tạo môi trường: $ENV_NAME"
echo "  Thư mục dự án:  $WORK_DIR"
echo "========================================================"

# ── 1. Tạo conda env ─────────────────────────────────────────
conda create -n "$ENV_NAME" python=3.9 -y
echo "[1/6] Đã tạo conda env: $ENV_NAME"

# ── 2. Cài PyTorch 2.1.0 + CUDA 12.1 (tương thích CUDA 12.2) ─
conda run -n "$ENV_NAME" pip install \
    torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121
echo "[2/6] Đã cài PyTorch 2.1.0 + cu121"

# ── 3. Cài OpenMMLab tools ───────────────────────────────────
conda run -n "$ENV_NAME" pip install -U openmim
conda run -n "$ENV_NAME" mim install "mmengine==0.10.3"
conda run -n "$ENV_NAME" mim install "mmcv==2.1.0"
echo "[3/6] Đã cài mmengine + mmcv"

# ── 4. Clone + cài MMSegmentation v1.2.2 ─────────────────────
if [ ! -d "$WORK_DIR/mmsegmentation" ]; then
    git clone -b v1.2.2 \
        https://github.com/open-mmlab/mmsegmentation.git \
        "$WORK_DIR/mmsegmentation"
fi
conda run -n "$ENV_NAME" \
    bash -c "cd '$WORK_DIR/mmsegmentation' && pip install -v -e ."
echo "[4/6] Đã cài MMSegmentation v1.2.2"

# ── 5. Cài thư viện bổ sung ──────────────────────────────────
conda run -n "$ENV_NAME" pip install \
    matplotlib numpy pillow scipy scikit-learn \
    pandas seaborn tqdm open3d
echo "[5/6] Đã cài thư viện bổ sung"

# ── 6. Kiểm tra môi trường ───────────────────────────────────
echo ""
echo "[6/6] Kiểm tra môi trường..."
conda run -n "$ENV_NAME" python -c "
import torch, mmcv, mmseg
print('  Python: OK')
print(f'  PyTorch:    {torch.__version__}')
print(f'  CUDA avail: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:        {torch.cuda.get_device_name(0)}')
    print(f'  VRAM:       {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB')
print(f'  MMCV:       {mmcv.__version__}')
print(f'  MMSeg:      {mmseg.__version__}')
"

echo ""
echo "========================================================"
echo "  Môi trường '$ENV_NAME' sẵn sàng!"
echo "  Kích hoạt bằng: conda activate $ENV_NAME"
echo "========================================================"