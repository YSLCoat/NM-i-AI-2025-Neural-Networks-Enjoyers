import os
import numpy as np
import matplotlib.pyplot as plt

from mrcnn import utils, model as modellib, visualize
from mrcnn.config import Config
from mrcnn_dataset import TumorDataset

import cv2
import numpy as np


# === CONFIG (Must match training setup) ===
class InferenceConfig(Config):
    NAME = "tumor"
    NUM_CLASSES = 1 + 1  # background + tumor
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.1
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512


# === Dice Score Calculation ===
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    intersection = np.logical_and(y_true, y_pred).sum()
    union = y_true.sum() + y_pred.sum()
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice


# === MAIN EVALUATION ===
def main():
    ROOT_DIR = os.path.abspath(".")
    MODEL_PATH = os.path.join(ROOT_DIR, "tumor-segmentation", "logs", "train", "mask_rcnn_tumor_20250804T2137.h5")
    VAL_DIR = os.path.join(ROOT_DIR, "tumor-segmentation","datasets", "val")
    SAVE_DIR = os.path.join(ROOT_DIR, "tumor-segmentation","eval_outputs")
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Load config
    config = InferenceConfig()
    config.display()

    # Load model
    model = modellib.MaskRCNN(mode="inference", model_dir=ROOT_DIR, config=config)
    model.load_weights(MODEL_PATH, by_name=True)
    print(f"Loaded weights from {MODEL_PATH}")

    # Load dataset
    dataset_val = TumorDataset()
    dataset_val.load_tumor(VAL_DIR)
    dataset_val.prepare()

    # Metrics
    from mrcnn.utils import compute_ap
    APs = []
    dice_scores = []

    for image_id in dataset_val.image_ids:
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(
            dataset_val, config, image_id, use_mini_mask=False
        )

        molded_image = np.expand_dims(modellib.mold_image(image, config), 0)
        results = model.detect([image], verbose=0)
        r = results[0]

        # Compute AP
        AP, precisions, recalls, overlaps = compute_ap(
            gt_bbox, gt_class_id, gt_mask,
            r["rois"], r["class_ids"], r["scores"], r["masks"]
        )
        APs.append(AP)

        # Compute Dice Score
        pred_mask = r['masks']
        gt_mask_combined = np.any(gt_mask, axis=-1)
        pred_mask_combined = np.any(pred_mask, axis=-1) if pred_mask.size > 0 else np.zeros_like(gt_mask_combined)
        dice = dice_coefficient(gt_mask_combined, pred_mask_combined)
        dice_scores.append(dice)

        # Save visualization
        filename = os.path.basename(dataset_val.image_info[image_id]['path'])
        save_path = os.path.join(SAVE_DIR, f"pred_{filename}")
        visualize.display_instances(
            image, r['rois'], r['masks'], r['class_ids'],
            dataset_val.class_names, r['scores'],
            show_bbox=True, show_mask=True,
            title=f"AP: {AP:.2f}, Dice: {dice:.2f}"
        )

    # Print Summary
    print(f"\n=== Evaluation Summary ===")
    print(f"Mean Average Precision (mAP): {np.mean(APs):.4f}")
    print(f"Mean Dice Score: {np.mean(dice_scores):.4f}")
    print(f"Saved results to: {SAVE_DIR}")


if __name__ == "__main__":
    main()
