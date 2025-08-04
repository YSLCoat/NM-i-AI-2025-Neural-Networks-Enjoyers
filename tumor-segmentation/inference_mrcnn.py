import os
import numpy as np
import cv2
import tensorflow as tf

from mrcnn import model as modellib
from mrcnn.config import Config
from mrcnn import visualize

# === CONFIG ===
class InferenceConfig(Config):
    NAME = "tumor"
    NUM_CLASSES = 1 + 1  # background + tumor
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    DETECTION_MIN_CONFIDENCE = 0.5

    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

config = InferenceConfig()
config.display()

# === SETUP MODEL ===
MODEL_PATH = os.path.join("tumor-segmentation","logs", "train", "mask_rcnn_tumor_20250804T2137.h5")

model = modellib.MaskRCNN(mode="inference", model_dir="logs/train", config=config)
model.load_weights(MODEL_PATH, by_name=True)

# === PREDICTION FUNCTION ===
def predict(img: np.ndarray) -> np.ndarray:
    # Convert grayscale to RGB if needed
    if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = img[:, :, :3]  # Remove alpha channel

    # Ensure uint8 type
    if img.dtype != np.uint8:
        img = (255 * (img - img.min()) / (img.max() - img.min())).astype(np.uint8)

    # Run detection
    results = model.detect([img], verbose=0)
    r = results[0]

    # Create an empty mask (same size as input)
    mask_combined = np.zeros(img.shape[:2], dtype=np.uint8)

    # Combine all instance masks (optional: you could keep them separate if needed)
    for i in range(r['masks'].shape[-1]):
        mask_combined = np.logical_or(mask_combined, r['masks'][:, :, i])

    # # Morphological cleanup
    # mask_combined = mask_combined.astype(np.uint8)
    # kernel = np.ones((3, 3), np.uint8)
    # mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, kernel)

    # Convert to 3-channel RGB
    mask_rgb = (mask_combined * 255).astype(np.uint8)
    mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_GRAY2RGB)

    return mask_rgb
