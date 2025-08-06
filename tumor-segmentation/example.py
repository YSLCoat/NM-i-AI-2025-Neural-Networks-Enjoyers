import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import os
import argparse
from tqdm import tqdm

from utils import dice_score
from model import UNet
import config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "saved_models/unet_tumor_segmentation.pth"
print(f"Inference running on device: {DEVICE}")

val_transform = A.Compose([
    A.PadIfNeeded(min_height=config.IMG_HEIGHT, min_width=config.IMG_WIDTH, border_mode=0),
    A.Normalize(mean=[0.0], std=[1.0]),
    ToTensorV2(),
])


model = UNet(in_channels=1, out_channels=1).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(DEVICE)))
model.eval()


def predict(img: np.ndarray) -> np.ndarray:
    """
    Takes a numpy array (color or grayscale) and returns a segmentation mask.
    """
    # --- START OF FIX: Universal Grayscale Conversion ---
    # This block ensures the input is always single-channel grayscale,
    # whether it comes from the API (3-channel) or a local script.
    if img.ndim == 3 and img.shape[2] == 3:
        # Convert 3-channel (e.g., RGB) to 1-channel grayscale
        pil_image = Image.fromarray(img)
        img = np.array(pil_image.convert("L"), dtype=np.float32)
    # --- END OF FIX ---

    # The input 'img' is now guaranteed to be grayscale (H, W)
    transformed = val_transform(image=img)
    image_tensor = transformed["image"]
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.sigmoid(logits)
        predicted_mask = (probabilities > 0.5).float()

    predicted_mask_np = predicted_mask.squeeze(0).squeeze(0).cpu().numpy()
    final_segmentation_2d = (predicted_mask_np * 255).astype(np.uint8)
    final_segmentation_3d = np.stack([final_segmentation_2d] * 3, axis=-1)

    return final_segmentation_3d


def main():
    parser = argparse.ArgumentParser(description="Evaluate segmentation model performance.")
    parser.add_argument("--folder_path", type=str, required=True, help="Path to the folder containing 'imgs' and 'labels' subfolders.")
    args = parser.parse_args()

    imgs_dir = os.path.join(args.folder_path, "imgs")
    labels_dir = os.path.join(args.folder_path, "labels")

    image_files = sorted([f for f in os.listdir(imgs_dir) if f.startswith('patient_') and f.endswith('.png')])
    dice_scores = []

    label_pad_transform = A.PadIfNeeded(min_height=config.IMG_HEIGHT, min_width=config.IMG_WIDTH, border_mode=0)

    print(f"Evaluating {len(image_files)} images from {args.folder_path}...")

    for img_file in tqdm(image_files, desc="Calculating Dice Scores"):
        img_path = os.path.join(imgs_dir, img_file)
        label_file = img_file.replace("patient_", "segmentation_")
        label_path = os.path.join(labels_dir, label_file)

        if not os.path.exists(label_path):
            continue

        # Load image and ensure it's converted to 1-channel Grayscale
        image_np = np.array(Image.open(img_path).convert("L"))

        # Load label as Grayscale
        label_np = np.array(Image.open(label_path).convert("L"))

        # Pad the ground truth label to match the model's output size
        padded_label = label_pad_transform(image=label_np)["image"]
        label_3d = np.stack([padded_label] * 3, axis=-1)

        # Get prediction
        prediction_np = predict(image_np)

        # Calculate Dice score on full-sized, padded data
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