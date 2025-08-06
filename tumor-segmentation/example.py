import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image # <--- Add this import

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
    # --- START OF MODIFICATION ---
    # The input 'img' is a NumPy array. We will replicate the training preprocessing.

    # 1. Check if the input is a 3-channel color image
    if img.ndim == 3 and img.shape[2] == 3:
        # 2. Convert NumPy array to PIL Image
        pil_image = Image.fromarray(img.astype(np.uint8))
        # 3. Convert to grayscale ("L" mode), just like in the TumorDataset
        grayscale_pil_image = pil_image.convert("L")
        # 4. Convert back to a NumPy array for further processing
        img = np.array(grayscale_pil_image, dtype=np.float32)

    # --- END OF MODIFICATION ---

    original_height, original_width = img.shape
    
    transformed = val_transform(image=img)
    image_tensor = transformed["image"]
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.sigmoid(logits)
        predicted_mask = (probabilities > 0.5).float()

    predicted_mask = predicted_mask.squeeze(0).squeeze(0).cpu().numpy()
    cropped_mask = predicted_mask[:original_height, :original_width]
    final_segmentation_2d = (cropped_mask * 255).astype(np.uint8)

    # Convert the 2D grayscale mask to a 3-channel image to match validation requirements
    final_segmentation_3d = np.stack([final_segmentation_2d] * 3, axis=-1)

    return final_segmentation_3d


def get_threshold_segmentation(img:np.ndarray, threshold:int) -> np.ndarray:
    return (img < threshold).astype(np.uint8)*255