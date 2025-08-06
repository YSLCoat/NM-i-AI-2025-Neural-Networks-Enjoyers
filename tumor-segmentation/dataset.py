import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

import config

class TumorDataset(Dataset):
    def __init__(self, image_paths, mask_paths=None, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.is_control = (mask_paths is None)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = np.array(Image.open(img_path).convert("L"), dtype=np.float32)

        if self.is_control:
            mask = np.zeros_like(image, dtype=np.float32)
        else:
            mask_path = self.mask_paths[idx]
            mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)

            if image.shape != mask.shape:
                h, w = image.shape
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

            mask[mask == 255.0] = 1.0

        if self.transform:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask

train_transform = A.Compose([
    # Replace LongestMaxSize and PadIfNeeded with a single Resize command
    A.Resize(height=config.IMG_HEIGHT, width=config.IMG_WIDTH, interpolation=cv2.INTER_AREA),
    A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT),
    A.HorizontalFlip(p=0.5),
    A.GridDistortion(p=0.2, border_mode=cv2.BORDER_CONSTANT),
    A.ElasticTransform(p=0.2, alpha=120, sigma=120 * 0.05, border_mode=cv2.BORDER_CONSTANT),
    A.Normalize(mean=[0.485], std=[0.229], max_pixel_value=255.0),
    ToTensorV2(),
])

val_transform = A.Compose([
    # Also update the validation transform to match
    A.Resize(height=config.IMG_HEIGHT, width=config.IMG_WIDTH, interpolation=cv2.INTER_AREA),
    A.Normalize(mean=[0.485], std=[0.229], max_pixel_value=255.0),
    ToTensorV2(),
])