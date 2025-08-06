import torch
import numpy as np
import cv2
from unet_model import get_unet_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Inference device: {device}")
model = get_unet_model(in_channels=1, out_classes=1)
model.load_state_dict(torch.load("tumor-segmentation/models/unet_model_6_2.pth", map_location=device))
model.to(device)
model.eval()

def preprocess(img: np.ndarray, resize_shape=(512, 512)) -> torch.Tensor:
    # Convert to grayscale (if input is color)
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)

    # # Log transform to compress dynamic range
    # img = np.log1p(img)

    # # Z-score normalization
    # img = (img - img.mean()) / (img.std() + 1e-5)

    # Resize if specified
    if resize_shape:
        target_h, target_w = resize_shape
        img = cv2.resize(img, (target_h, target_w), interpolation=cv2.INTER_AREA)

    # Add channel and batch dims: (1, 1, H, W)
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=0)

    tensor = torch.tensor(img, dtype=torch.float32).to(device)
    return tensor

def postprocess(pred: torch.Tensor, target_size: tuple) -> np.ndarray:
    # pred shape: (1, 1, H_pred, W_pred)
    pred_np = pred.squeeze().cpu().numpy()  # (H_pred, W_pred)

    # Resize back to original image size
    pred_resized = cv2.resize(pred_np, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)

    # Threshold to binary mask
    binary_mask = (pred_resized > 0.5).astype(np.uint8)  # values: 0 or 1

    # --- NEW CLEANING STEPS BELOW ---

    # Remove small objects using connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    min_size = 5  # minimum area of object to keep (tune this value)
    cleaned_mask = np.zeros_like(binary_mask)

    for i in range(1, num_labels):  # skip background (label 0)
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            cleaned_mask[labels == i] = 1

    # Optional: apply morphological closing to fill small holes
    kernel = np.ones((3, 3), np.uint8)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
    # cleaned_mask = binary_mask
    # Convert to [0, 255] and 3-channel RGB
    cleaned_mask = (cleaned_mask * 255).astype(np.uint8)
    return cv2.cvtColor(cleaned_mask, cv2.COLOR_GRAY2RGB)


def predict(img: np.ndarray) -> np.ndarray:
    original_size = img.shape[:2]  # (height, width)
    input_tensor = preprocess(img)
    with torch.no_grad():
        output = model(input_tensor)
    return postprocess(output, original_size)
