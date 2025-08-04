import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.python.client import device_lib
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from mrcnn import model as modellib
from mrcnn.utils import generate_pyramid_anchors
from mrcnn.config import Config
from mrcnn_dataset import TumorDataset
from mrcnn import visualize


# === DEVICE CHECK ===
def get_available_devices():
    devices = device_lib.list_local_devices()
    for d in devices:
        print(f"{d.name} - {d.device_type}")

get_available_devices()


# === CONFIGURATION ===
class TumorConfig(Config):
    NAME = "tumor"
    NUM_CLASSES = 1 + 1  # background + tumor
    STEPS_PER_EPOCH = 170
    VALIDATION_STEPS = 12
    LEARNING_RATE = 0.005
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    BACKBONE = "resnet50"
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    RPN_ANCHOR_SCALES = RPN_ANCHOR_SCALES = (8, 16, 32, 64, 80)
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]


# === ANCHOR VISUALIZATION ===
def visualize_anchors_on_image(config, dataset):
    image_id = dataset.image_ids[0]
    image = dataset.load_image(image_id)

    image_shape = image.shape
    backbone_shapes = modellib.compute_backbone_shapes(config, image_shape)

    anchors = generate_pyramid_anchors(
        config.RPN_ANCHOR_SCALES,
        config.RPN_ANCHOR_RATIOS,
        backbone_shapes,
        config.BACKBONE_STRIDES,
        config.RPN_ANCHOR_STRIDE
    )

    print(f"Generated {len(anchors)} anchors")

    sample_indices = np.random.choice(np.arange(anchors.shape[0]), size=200, replace=False)
    anchor_sample = anchors[sample_indices]

    visualize.draw_boxes(image, boxes=anchor_sample, captions=None, title="Sample Anchors")
    plt.show()

def dice_score(y_true, y_pred, smooth=1e-6):
    """
    y_true, y_pred: numpy arrays of shape [height, width] or [num_masks, height, width]
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

# === MAIN TRAINING FUNCTION ===
def main():
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    LOGS_DIR = os.path.join(ROOT_DIR, "logs", "train")
    os.makedirs(LOGS_DIR, exist_ok=True)

    print(f"ROOT_DIR: {ROOT_DIR}")
    print(f"LOGS_DIR: {LOGS_DIR}")

    config = TumorConfig()
    config.display()

    train_dataset = TumorDataset()
    train_dataset.load_tumor(os.path.join(ROOT_DIR, "datasets/train")) # train
    train_dataset.prepare()

    val_dataset = TumorDataset()
    val_dataset.load_tumor(os.path.join(ROOT_DIR, "datasets/val")) # val
    val_dataset.prepare()

        # === DEBUG: Visualize Loaded Image and Mask ===


    image_id = random.choice(train_dataset.image_ids)
    print(f"\n[DEBUG] Visualizing image_id: {image_id}")

    image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(
        train_dataset, config, image_id, use_mini_mask=False
    )

    print(f"[DEBUG] Image shape: {image.shape}")
    print(f"[DEBUG] Mask shape: {gt_mask.shape}")
    print(f"[DEBUG] BBox: {gt_bbox}")
    print(f"[DEBUG] Class IDs: {gt_class_id}")

    # Use visualize.display_instances to show the image + all GT masks and boxes
    visualize.display_instances(
        image, gt_bbox, gt_mask, gt_class_id, train_dataset.class_names,
        show_bbox=True, show_mask=True, title="Image with Ground Truth"
    )



    visualize_anchors_on_image(config, train_dataset)

    model = modellib.MaskRCNN(mode="training", config=config, model_dir=LOGS_DIR)
    print(f"Model log_dir: {model.log_dir}")

    COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mrcnn", "mask_rcnn_coco.h5")

    if os.path.exists(COCO_WEIGHTS_PATH):
        print(f"Found COCO weights at {COCO_WEIGHTS_PATH}")
        print("Loading COCO weights (excluding task-specific heads)...")
        
        # Load weights with logging
        model.load_weights(COCO_WEIGHTS_PATH, by_name=True,
                        exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                    "mrcnn_bbox", "mrcnn_mask"])
        
        print("COCO weights loaded successfully.")
    else:
        raise FileNotFoundError(f"COCO weights file not found at {COCO_WEIGHTS_PATH}")


    # === CALLBACKS ===
    # Checkpoints
    checkpoint_path = os.path.join(LOGS_DIR, "checkpoint_epoch_{epoch:02d}.h5")
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,
        save_freq='epoch'
    )

    # TensorBoard
    tensorboard_callback = TensorBoard(
        log_dir=LOGS_DIR,
        histogram_freq=1,
        write_graph=True,
        write_images=True,
        profile_batch=0
    )



    class DiceScoreCallback(Callback):
        def __init__(self, model, val_dataset, config, log_dir):
            super().__init__()
            self.model = model
            self.val_dataset = val_dataset
            self.config = config
            self.file_writer = tf.summary.create_file_writer(log_dir)

        def on_epoch_end(self, epoch, logs=None):
            dice_scores = []

            # Iterate over the validation dataset
            for image_id in self.val_dataset.image_ids:
                # Load image and ground truth data
                image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(
                    self.val_dataset, self.config, image_id, use_mini_mask=False)

                # Add batch dimension
                molded_images = np.expand_dims(modellib.mold_image(image, self.config), 0)

                # Run detection (inference) on the image
                results = self.model.detect([image], verbose=0)
                r = results[0]

                # If there are no predicted masks, dice=0
                if r['masks'].size == 0:
                    dice_scores.append(0)
                    continue

                # Compute DICE between ground truth mask and predicted masks
                # We assume binary masks, so you may need to handle multiple masks
                # For simplicity, compute DICE per mask and take max score

                # Combine all predicted masks into one mask
                pred_mask = np.any(r['masks'], axis=-1).astype(np.uint8)

                # Combine all ground truth masks into one mask
                gt_mask_combined = np.any(gt_mask, axis=-1).astype(np.uint8)

                dice = dice_score(gt_mask_combined, pred_mask)
                dice_scores.append(dice)

            mean_dice = np.mean(dice_scores)

            print(f"\nEpoch {epoch + 1}: Mean DICE score on validation set: {mean_dice:.4f}")

            # Log to TensorBoard
            with self.file_writer.as_default():
                tf.summary.scalar('val_dice_score', mean_dice, step=epoch)

    dice_callback = DiceScoreCallback(model, val_dataset, config, LOGS_DIR)


    # === TRAIN ===
    # === PHASE 1: Train heads ===
    print("\n--- Training heads (initial training phase) ---")
    model.train(
        train_dataset,
        val_dataset,
        learning_rate=config.LEARNING_RATE,
        epochs=120,
        layers='3+',
        custom_callbacks=[tensorboard_callback, checkpoint_callback]
        # custom_callbacks=[tensorboard_callback]
    )

    # === PHASE 2: Fine-tune deeper layers (3+) ===
    # print("\n--- Fine-tuning layers 3+ (resnet stages 3+) ---")
    # model.train(
    #     train_dataset,
    #     val_dataset,
    #     learning_rate=config.LEARNING_RATE / 10,  # lower LR helps fine-tuning
    #     epochs=60,  # total number of epochs, not relative
    #     layers='3+',
    #     initial_epoch=20,
    #     custom_callbacks=[tensorboard_callback, checkpoint_callback]
    # )

    # === SAVE FINAL MODEL ===
    timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M")
    final_path = os.path.join(LOGS_DIR, f"mask_rcnn_tumor_{timestamp}.h5")
    model.keras_model.save_weights(final_path)
    print(f"\nSaved final model weights to {final_path}")


if __name__ == "__main__":
    main()
