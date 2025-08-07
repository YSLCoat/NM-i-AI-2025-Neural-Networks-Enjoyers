import torch
import numpy as np
import cv2
from unet_model import get_unet_model
import matplotlib.pyplot as plt

# Device
device = torch.device("cpu")  # Match original
print(f"Ensemble inference device: {device}")

# Global state for loaded models
_models = [
    "tumor-segmentation/models/unet_model_7_0.pth",
    "tumor-segmentation/models/unet_model_6_8.pth",
    "tumor-segmentation/models/unet_model_6_7.pth"
]
_resize_shape = (512, 512)
_threshold = 0.5


def load_ensemble(model_paths: list):
    """
    Load multiple U-Net models into global _models list.
    Call this once before using predict().
    """
    global _models
    _models = []
    for path in model_paths:
        model = get_unet_model(in_channels=1, out_classes=1)
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        _models.append(model)
    print(f"Loaded {len(_models)} models for ensemble inference.")


def preprocess(img: np.ndarray, resize_shape=(512, 512)) -> torch.Tensor:
    # Convert to grayscale if input is color
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)

    # Resize
    if resize_shape:
        target_h, target_w = resize_shape
        img = cv2.resize(img, (target_h, target_w), interpolation=cv2.INTER_AREA)

    # Add channel and batch dims: (1, 1, H, W)
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=0)

    tensor = torch.tensor(img, dtype=torch.float32).to(device)
    return tensor

def debug_thresholds(pred_resized: np.ndarray):
    """
    Show different thresholded masks using OpenCV windows.
    """
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
        mask = (pred_resized > thresh).astype(np.uint8) * 255  # scale to 0–255
        cv2.imshow(f"Threshold: {thresh}", mask)

    print("Press any key to close all threshold windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def postprocess(pred: torch.Tensor, target_size: tuple) -> np.ndarray:
    # pred shape: (1, 1, H, W)
    pred_np = pred.squeeze().cpu().numpy()  # (H, W)

    # Resize back to original image size
    pred_resized = cv2.resize(pred_np, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)

    # print("<<<<<<<<<<<<  DEBUG  >>>>>>>>>>>>>>>")
    # debug_thresholds(pred_resized)

    # Threshold to binary mask
    binary_mask = (pred_resized > _threshold).astype(np.uint8)  # 0 or 1

    # --- Postprocessing ---
    # Remove small objects using connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    min_size = 5
    cleaned_mask = np.zeros_like(binary_mask)

    for i in range(1, num_labels):  # Skip background
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            cleaned_mask[labels == i] = 1

    # Morphological closing
    kernel = np.ones((3, 3), np.uint8)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)

    # Convert to [0, 255] RGB
    cleaned_mask = (cleaned_mask * 255).astype(np.uint8)
    return cv2.cvtColor(cleaned_mask, cv2.COLOR_GRAY2RGB)


@torch.no_grad()
def predict(img: np.ndarray) -> np.ndarray:
    """
    Run ensemble prediction on a single input image.
    Args:
        img (np.ndarray): Grayscale or RGB image, shape (H, W) or (H, W, 3)
    Returns:
        np.ndarray: RGB binary mask, shape (H, W, 3), values in {0, 255}
    """
    if not _models:
        raise RuntimeError("No models loaded. Call load_ensemble(model_paths) first.")

    original_size = img.shape[:2]
    input_tensor = preprocess(img, resize_shape=_resize_shape)

    preds = []
    for model in _models:
        out = model(input_tensor)  # (1, 1, H, W)
        prob = torch.sigmoid(out)
        preds.append(prob)

    avg_pred = torch.stack(preds, dim=0).mean(dim=0)  # shape: (1, 1, H, W)
    ret = postprocess(avg_pred, original_size)
    return ret
