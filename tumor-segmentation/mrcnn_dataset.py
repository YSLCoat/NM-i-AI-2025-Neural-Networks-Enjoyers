import os
import numpy as np
import cv2
from mrcnn import utils
import skimage

class TumorDataset(utils.Dataset):
    def load_tumor(self, dataset_dir):
        self.add_class("tumor", 1, "tumor")

        image_dir = os.path.join(dataset_dir, "imgs")
        label_dir = os.path.join(dataset_dir, "labels")

        for filename in os.listdir(image_dir):
            if filename.endswith(".png"):
                image_path = os.path.join(image_dir, filename)
                # Replace 'patient_' with 'segmentation_' to get corresponding mask filename
                mask_filename = filename.replace("patient_", "segmentation_")
                mask_path = os.path.join(label_dir, mask_filename)

                self.add_image(
                    source="tumor",
                    image_id=filename,
                    path=image_path,
                    mask_path=mask_path
                )


    def load_image(self, image_id):
        info = self.image_info[image_id]
        img = cv2.imread(info['path'], cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Failed to load image: {info['path']}")

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return rgb

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        mask_path = info['mask_path']
        mask = skimage.io.imread(mask_path)  # Binary image with all tumors = 255

        # Convert to boolean and ensure 2D
        if mask.ndim == 3:
            mask = mask[..., 0]  # Take one channel if RGB

        mask = mask.astype(np.bool_)

        # Label connected components (each tumor)
        labeled_mask = skimage.measure.label(mask)
        tumor_count = labeled_mask.max()

        if tumor_count == 0:
            # No tumors found, return dummy mask
            mask = np.zeros((mask.shape[0], mask.shape[1], 1), dtype=bool)
            class_ids = np.array([], dtype=np.int32)
            return mask, class_ids

        # Create separate mask for each instance
        instance_masks = [(labeled_mask == i).astype(bool) for i in range(1, tumor_count + 1)]
        mask_stack = np.stack(instance_masks, axis=-1)  # Shape: H x W x N
        class_ids = np.array([1] * tumor_count, dtype=np.int32)  # All are tumors

        return mask_stack, class_ids
