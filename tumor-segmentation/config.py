import torch

DATA_DIR = "data"
PATIENTS_DIR = f"{DATA_DIR}/patients"
CONTROLS_DIR = f"{DATA_DIR}/controls"

IMG_HEIGHT = 992
IMG_WIDTH = 400


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
EPOCHS = 25
VALIDATION_SPLIT = 0.2

CHECKPOINT_DIR = "saved_models"
MODEL_NAME = "unet_tumor_segmentation.pth"