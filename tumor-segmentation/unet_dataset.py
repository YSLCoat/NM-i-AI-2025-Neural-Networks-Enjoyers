import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class TumorSegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, resize_shape=None, augment=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.resize_shape = resize_shape  # (height, width)
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img = cv2.imread(self.image_paths[idx], cv2.IMREAD_UNCHANGED)
        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        # gray = np.log1p(gray)  # log transform to compress dynamic range
        # gray = (gray - gray.mean()) / (gray.std() + 1e-5)  # z-score normalization

        # Load mask and convert to grayscale
        mask_rgb = cv2.imread(self.mask_paths[idx], cv2.IMREAD_UNCHANGED)
        mask = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        mask = (mask > 0.5).astype(np.float32) # ensure binary mask
        
        # Resize if specified
        if self.resize_shape:
            target_h, target_w = self.resize_shape
            gray = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        # Add channel dimension
        gray = np.expand_dims(gray, axis=0)
        mask = np.expand_dims(mask, axis=0)

        # Convert to tensors
        gray_tensor = torch.tensor(gray, dtype=torch.float32)
        mask_tensor = torch.tensor(mask, dtype=torch.float32)

        return gray_tensor, mask_tensor