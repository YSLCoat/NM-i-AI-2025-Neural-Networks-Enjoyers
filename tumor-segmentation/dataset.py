import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2  # Import OpenCV

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
    A.PadIfNeeded(min_height=config.IMG_HEIGHT, min_width=config.IMG_WIDTH, border_mode=0),
    A.Rotate(limit=15, p=0.5, border_mode=0),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2), # Add this
    A.GaussNoise(p=0.2), # And this
    A.Normalize(mean=[0.0], std=[1.0]),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.PadIfNeeded(min_height=config.IMG_HEIGHT, min_width=config.IMG_WIDTH, border_mode=0),
    A.Normalize(mean=[0.0], std=[1.0]),
    ToTensorV2(),
])