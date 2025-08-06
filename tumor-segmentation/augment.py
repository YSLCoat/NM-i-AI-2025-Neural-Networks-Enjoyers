import os
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# === Config ===
NUM_AUGMENTED_TOTAL = 5000
INPUT_IMG_DIR = "datasets/original/imgs"
INPUT_MASK_DIR = "datasets/original/labels"
OUT_IMG_DIR = "datasets/original_augmented_5000/imgs"
OUT_MASK_DIR = "datasets/original_augmented_5000/labels"

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_MASK_DIR, exist_ok=True)

def add_salt_and_pepper_noise(image, amount=0.004, salt_vs_pepper=0.5):
    """
    Add salt and pepper noise to an image.
    - amount: proportion of image pixels to alter
    - salt_vs_pepper: proportion of salt vs. pepper noise
    """
    noisy = image.copy()
    num_salt = np.ceil(amount * image.size * salt_vs_pepper).astype(int)
    num_pepper = np.ceil(amount * image.size * (1.0 - salt_vs_pepper)).astype(int)

    # Salt noise
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
    noisy[coords[0], coords[1]] = 255

    # Pepper noise
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]
    noisy[coords[0], coords[1]] = 0

    return noisy


# === Define your augmentation pipeline ===
augment = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(
        shift_limit=0.05,
        scale_limit=0.05,
        rotate_limit=10,
        p=0.5
    ),
    A.ElasticTransform(
        alpha=0.5,
        sigma=20,
        alpha_affine=10,
        p=0.2
    ),
    A.GridDistortion(num_steps=5, distort_limit=0.03, p=0.2),
    A.RandomBrightnessContrast(
        brightness_limit=0.1,
        contrast_limit=0.1,
        p=0.3
    ),
    A.GaussianBlur(blur_limit=(3, 3), p=0.2),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
], additional_targets={'mask': 'image'})


# === Load all originals ===
image_filenames = [f for f in os.listdir(INPUT_IMG_DIR) if f.endswith('.png')]
original_count = len(image_filenames)
assert original_count > 0, "No training images found!"

# === Calculate how many synthetic images are needed ===
needed_augmented = NUM_AUGMENTED_TOTAL - original_count
per_image_augments = needed_augmented // original_count + 1

print(f"Generating {NUM_AUGMENTED_TOTAL} total images.")
print(f"{needed_augmented} will be synthetic, about {per_image_augments} augmentations per original.")

augmented_count = 0  # Count only synthetic images
uid = 0  # Unique filename ID for all images

for img_file in tqdm(image_filenames, desc="Augmenting"):
    mask_file = img_file.replace("patient", "segmentation")

    img_path = os.path.join(INPUT_IMG_DIR, img_file)
    mask_path = os.path.join(INPUT_MASK_DIR, mask_file)

    image = cv2.imread(img_path)
    mask = cv2.imread(mask_path)

    if image is None or mask is None:
        print(f"Skipping {img_file} due to loading error.")
        continue

    # Save original image and mask
    cv2.imwrite(os.path.join(OUT_IMG_DIR, f"aug_patient_{uid:03d}.png"), image)
    cv2.imwrite(os.path.join(OUT_MASK_DIR, f"aug_segmentation_{uid:03d}.png"), mask)
    uid += 1  # We still increase uid to make filenames unique

    # Generate augmented samples
    for _ in range(per_image_augments):
        if augmented_count >= needed_augmented:
            break

        augmented = augment(image=image, mask=mask)
        aug_img = augmented['image']
        aug_mask = augmented['mask']

        # Apply salt and pepper noise to 10% of augmentations
        if np.random.rand() < 0.1:
            aug_img = add_salt_and_pepper_noise(aug_img)

        cv2.imwrite(os.path.join(OUT_IMG_DIR, f"aug_patient_{uid:03d}.png"), aug_img)
        cv2.imwrite(os.path.join(OUT_MASK_DIR, f"aug_segmentation_{uid:03d}.png"), aug_mask)
        uid += 1
        augmented_count += 1

    if augmented_count >= needed_augmented:
        break  # Exit outer loop once enough augmentations are made

print(f"\nDone. Final dataset contains {uid} total images (original + augmented).")
