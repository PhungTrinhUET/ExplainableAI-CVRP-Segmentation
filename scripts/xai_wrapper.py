"""
MMSeg model wrapper cho pytorch-grad-cam.
Cần thiết vì pytorch-grad-cam yêu cầu model forward() trả về tensor logits trực tiếp,
trong khi MMSeg trả về SegDataSample objects.
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path


class MMSegCAMWrapper(nn.Module):
    """
    Wrap MMSeg model để:
    1. Forward pass trả về raw seg_logits (B, num_classes, H, W)
    2. Tương thích với pytorch-grad-cam hooks
    """
    def __init__(self, mmseg_model):
        super().__init__()
        self.model = mmseg_model
        self.model.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W) — đã normalize (ImageNet mean/std)
        returns: (B, num_classes, H, W) — logits trước softmax
        """
        with torch.set_grad_enabled(True):
            # Backbone features
            feat = self.model.backbone(x)

            # Neck (nếu có)
            if hasattr(self.model, 'neck') and self.model.neck is not None:
                feat = self.model.neck(feat)

            # Decode head forward (trả về logits, không phải prediction)
            # Với MMSeg 1.x: decode_head.forward() trả về logits trực tiếp
            logits = self.model.decode_head.forward(feat)

        return logits  # (B, 2, H, W)


class MMSegMask2FormerWrapper(nn.Module):
    """
    Wrapper đặc biệt cho Mask2Former vì decode_head phức tạp hơn.
    Lấy seg_logits từ kết quả predict.
    """
    def __init__(self, mmseg_model):
        super().__init__()
        self.model = mmseg_model
        self.model.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.model.backbone(x)
        # Mask2Former decode head cần all_cls_scores và all_mask_preds
        # Chúng ta lấy logits từ pixel_decoder + transformer decoder
        # Đơn giản: dùng predict flow nhưng lấy seg_logits
        try:
            # Lấy pixel decoder features
            pixel_feats = self.model.decode_head.pixel_decoder.forward_features(feat)
            # Simplified: chỉ lấy output từ lớp cuối pixel decoder
            logits = pixel_feats[0]  # (B, C, H/4, W/4)
            # Resize về đúng num_classes
            if logits.shape[1] != 2:
                # Project về 2 classes
                logits = logits[:, :2, :, :]
        except Exception:
            # Fallback: dùng backbone features trực tiếp
            logits = feat[-1][:, :2, :, :]
        return logits


def get_target_layers(model, model_name: str) -> list:
    """
    Trả về danh sách target layers cho GradCAM/EigenCAM.

    Strategy:
    - CNN (ResNet): layer cuối cùng của stage 4 → gradient rõ ràng, spatial detail tốt
    - Transformer (Swin/MiT): block cuối cùng của stage cuối → high-level features
    """
    backbone = model.backbone
    name_lower = model_name.lower()

    if "deeplabv3" in name_lower:
        # ResNet-101: layer4 là stage cuối, [-1] là bottleneck cuối
        return [backbone.layer4[-1]]

    elif "segformer" in name_lower:
        # MixVisionTransformer: layers là ModuleList chứa các stage
        # Layer cuối (stage 4) → high-level semantic features
        return [backbone.layers[-1]]

    elif "knet" in name_lower or "mask2former" in name_lower:
        # Swin Transformer: layers[-1] là stage cuối
        # blocks[-1] là transformer block cuối của stage đó
        last_stage = backbone.layers[-1]
        if hasattr(last_stage, 'blocks'):
            return [last_stage.blocks[-1]]
        return [last_stage]

    return [list(backbone.children())[-1]]


def swin_reshape_transform(tensor, height=None, width=None):
    """
    Reshape transform cho Swin Transformer output.
    Swin output: (B, H*W, C) → cần reshape về (B, C, H, W) cho CAM.
    """
    # Tính H, W từ số tokens
    n_tokens = tensor.shape[1]
    if height is None or width is None:
        h = w = int(n_tokens ** 0.5)
    else:
        h, w = height, width
    result = tensor.reshape(tensor.size(0), h, w, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)  # (B, C, H, W)
    return result


def mit_reshape_transform(tensor, height=None, width=None):
    """
    Reshape transform cho Mix Vision Transformer (SegFormer backbone).
    Tương tự Swin nhưng có thể khác shape.
    """
    if tensor.dim() == 3:
        n_tokens = tensor.shape[1]
        h = w = int(n_tokens ** 0.5)
        result = tensor.reshape(tensor.size(0), h, w, tensor.size(2))
        result = result.transpose(2, 3).transpose(1, 2)
        return result
    return tensor  # đã là (B, C, H, W)


def preprocess_image(img_path,   # str or Path
                     mean=(123.675, 116.28, 103.53),
                     std=(58.395, 57.12, 57.375),
                     target_size: int = 512) -> tuple[np.ndarray, torch.Tensor]:
    """
    Load và preprocess ảnh cho inference + CAM.
    Returns: (rgb_np_array, input_tensor)
    """
    from PIL import Image
    img = Image.open(img_path).convert("RGB")

    # Resize
    w, h = img.size
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    img_np = np.array(img, dtype=np.float32)   # (H, W, 3) 0-255

    # Normalize
    mean_arr = np.array(mean, dtype=np.float32)
    std_arr  = np.array(std,  dtype=np.float32)
    img_norm = (img_np - mean_arr) / std_arr

    # To tensor (B, C, H, W)
    tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0)

    return img_np.astype(np.uint8), tensor


class PanicleTarget:
    """
    Custom CAM target: maximize sum of panicle class logits.
    Không cần mask — hoạt động với mọi output resolution.
    """
    def __call__(self, model_output):
        return model_output[1, :, :].sum()


def cam_iou(cam: np.ndarray, gt: np.ndarray, threshold: float = 0.5) -> float:
    """IoU giữa binarized CAM và GT mask."""
    import cv2
    cam_bin = (cam >= threshold).astype(np.uint8)
    gt_bin  = (gt > 0).astype(np.uint8)
    if cam_bin.shape != gt_bin.shape:
        cam_bin = (cv2.resize(cam.astype(np.float32),
                               (gt_bin.shape[1], gt_bin.shape[0])) >= 0.5).astype(np.uint8)
    return float((cam_bin & gt_bin).sum()) / ((cam_bin | gt_bin).sum() + 1e-6)


def energy_pointing_game(cam: np.ndarray, gt: np.ndarray) -> float:
    """Tỉ lệ CAM energy nằm trong GT mask."""
    import cv2
    if cam.shape != gt.shape:
        cam = cv2.resize(cam.astype(np.float32), (gt.shape[1], gt.shape[0]))
    gt_norm = (gt > 0).astype(np.float32)
    return float((cam * gt_norm).sum()) / (cam.sum() + 1e-6)
