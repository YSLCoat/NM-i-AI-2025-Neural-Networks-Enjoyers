import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import os
import argparse
from tqdm import tqdm
import cv2

from utils import dice_score
from model import UNet
from dataset import val_transform  # Import the transform from dataset.py for consistency
import config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "saved_models/unet_tumor_segmentation.pth"
print(f"Inference running on device: {DEVICE}")

# The val_transform is now imported from dataset.py to guarantee it's identical.

model = UNet(in_channels=1, out_channels=1).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(DEVICE), weights_only=False))
model.eval()

def postprocess(pred: torch.Tensor) -> np.ndarray:
    """
    Cleans the mask on the full-sized (padded) canvas without resizing.
    """
    # pred shape: (1, 1, H_pred, W_pred)
    pred_np = pred.squeeze().cpu().numpy()  # (H_pred, W_pred)

    # --- NO RESIZING ---
    # Threshold the mask directly
    binary_mask = (pred_np > 0.5).astype(np.uint8)

    # Remove small objects
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    min_size = 5
    cleaned_mask = np.zeros_like(binary_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            cleaned_mask[labels == i] = 1

    # Morphological closing
    kernel = np.ones((3, 3), np.uint8)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)

    # Convert to [0, 255] and 3-channel RGB for evaluation/visualization
    cleaned_mask = (cleaned_mask * 255).astype(np.uint8)
    return cv2.cvtColor(cleaned_mask, cv2.COLOR_GRAY2RGB)

def predict(img: np.ndarray) -> np.ndarray:
    """
    Takes a raw numpy image and returns a cleaned segmentation mask of the same size.
    """
    # --- 1. Store the original image's shape ---
    original_height, original_width = img.shape[:2]

    # Handle grayscale conversion if necessary
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)

    # Apply transforms (including resizing for the model)
    transformed = val_transform(image=img)
    image_tensor = transformed["image"]
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

    # Get model prediction
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.sigmoid(logits)
        predicted_mask = probabilities.float()

    # Post-process the mask (cleaning, etc.)
    # The output here is still at the model's size (e.g., 992, 400, 3)
    processed_mask = postprocess(predicted_mask)

    # --- 2. Resize the final mask back to the original image's dimensions ---
    final_mask = cv2.resize(processed_mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

    return final_mask

def main():
    parser = argparse.ArgumentParser(description="Evaluate segmentation model performance.")
    parser.add_argument("--folder_path", type=str, required=True, help="Path to the folder containing 'imgs' and 'labels' subfolders.")
    args = parser.parse_args()

    imgs_dir = os.path.join(args.folder_path, "imgs")
    labels_dir = os.path.join(args.folder_path, "labels")

    image_files = sorted([f for f in os.listdir(imgs_dir) if f.startswith('patient_') and f.endswith('.png')])
    dice_scores = []

    print(f"Evaluating {len(image_files)} images from {args.folder_path}...")

    for img_file in tqdm(image_files, desc="Calculating Dice Scores"):
        img_path = os.path.join(imgs_dir, img_file)
        label_file = img_file.replace("patient_", "segmentation_")
        label_path = os.path.join(labels_dir, label_file)

        if not os.path.exists(label_path):
            continue

        # 1. Load raw image and label
        image_np = np.array(Image.open(img_path).convert("L"), dtype=np.float32)
        label_np = np.array(Image.open(label_path).convert("L")) / 255.0 # Normalize to 0/1

        # 2. Get the padded prediction from the model
        prediction_np = predict(image_np) # This is already a 3-channel RGB numpy array

        # 3. Apply the *same* validation transform to the ground truth label
        # This pads it to match the prediction's canvas.
        padded_label_data = val_transform(image=image_np, mask=label_np)
        padded_label_np = padded_label_data['mask'].numpy() # (H, W)

        # Convert padded label to 3-channel RGB for Dice score comparison
        label_3d = np.stack([padded_label_np] * 3, axis=-1)
        label_3d = (label_3d * 255).astype(np.uint8)

        # 4. Calculate Dice score on the aligned, padded images
        score = dice_score(y_true=label_3d, y_pred=prediction_np)
        dice_scores.append(score)

    if dice_scores:
        avg_dice_score = np.mean(dice_scores)
        print(f"\n✅ Evaluation Complete.")
        print(f"Average Dice Score: {avg_dice_score:.4f}")
    else:
        print("No images were evaluated.")

if __name__ == '__main__':
    main()