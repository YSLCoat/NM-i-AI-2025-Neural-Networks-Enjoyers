import os
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split, ConcatDataset
from tqdm import tqdm
from pathlib import Path

import config
from dataset import TumorDataset, train_transform, val_transform
from model import UNet
from loss import DiceBCELoss


from torch.utils.data import Sampler

class BalancedBatchSampler(Sampler):
    """
    A custom PyTorch sampler to create balanced batches.

    For each batch, it samples a specified number of items from the patient dataset
    and a specified number from the control dataset to ensure each batch has a
    consistent composition.
    """
    def __init__(self, dataset, batch_size, patient_ratio=0.5):
        """
        Args:
            dataset (Dataset): The full, concatenated dataset.
            batch_size (int): The total number of samples in a batch.
            patient_ratio (float): The proportion of the batch that should be patient samples.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.patient_ratio = patient_ratio

        # Determine the number of patient and control samples per batch
        self.num_patients_per_batch = int(batch_size * patient_ratio)
        self.num_controls_per_batch = batch_size - self.num_patients_per_batch

        # Get the indices for each dataset from the ConcatDataset
        # The first dataset added is the patient dataset
        self.patient_indices = list(range(dataset.cumulative_sizes[0]))
        # The second dataset added is the control dataset
        self.control_indices = list(range(dataset.cumulative_sizes[0], dataset.cumulative_sizes[1]))

        # Ensure we can form a batch with the given ratio
        if self.num_patients_per_batch == 0 or self.num_controls_per_batch == 0:
            raise ValueError(
                f"Batch size {batch_size} and patient_ratio {patient_ratio} "
                "results in zero samples for one of the classes. Adjust parameters."
            )

    def __iter__(self):
        # Shuffle the indices at the beginning of each epoch
        np.random.shuffle(self.patient_indices)
        np.random.shuffle(self.control_indices)

        # Create iterators to pull indices from
        patient_iter = iter(self.patient_indices)
        control_iter = iter(self.control_indices)

        # Yield batches until the total dataset is traversed once
        for _ in range(len(self)):
            batch = []
            # Add patient indices to the batch
            for _ in range(self.num_patients_per_batch):
                try:
                    batch.append(next(patient_iter))
                except StopIteration:
                    # If we run out of patient samples, reshuffle and start over
                    np.random.shuffle(self.patient_indices)
                    patient_iter = iter(self.patient_indices)
                    batch.append(next(patient_iter))

            # Add control indices to the batch
            for _ in range(self.num_controls_per_batch):
                try:
                    batch.append(next(control_iter))
                except StopIteration:
                    # If we run out of control samples, reshuffle and start over
                    np.random.shuffle(self.control_indices)
                    control_iter = iter(self.control_indices)
                    batch.append(next(control_iter))
            
            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        # Returns the total number of batches per epoch
        return len(self.dataset) // self.batch_size



def check_accuracy(loader, model, device="cuda"):

    model.eval()

    val_loss = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0

    loss_fn = DiceBCELoss()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = model(x)

            loss = loss_fn(preds, y)
            val_loss += loss.item()

            preds_binary = (torch.sigmoid(preds) > 0.5).float()

            tp = (preds_binary * y).sum()
            fp = (preds_binary * (1 - y)).sum()
            fn = ((1 - preds_binary) * y).sum()

            total_tp += tp.item()
            total_fp += fp.item()
            total_fn += fn.item()

    epsilon = 1e-8
    precision = total_tp / (total_tp + total_fp + epsilon)
    recall = total_tp / (total_tp + total_fn + epsilon)
    dice_score = 2 * (precision * recall) / (precision + recall + epsilon)
    iou = total_tp / (total_tp + total_fp + total_fn + epsilon)

    metrics = {
        "val_loss": val_loss / len(loader),
        "dice_score": dice_score,
        "iou": iou,
        "precision": precision,
        "recall": recall,
    }

    model.train() 
    return metrics


def train_one_epoch(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader, leave=True)
    running_loss = 0.0

    for _, (data, targets) in enumerate(loop):
        data = data.to(device=config.DEVICE)
        targets = targets.float().unsqueeze(1).to(device=config.DEVICE)

        with torch.amp.autocast(device_type='cuda'):
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return running_loss / len(loader)


def main():
    print(f"--- Starting training on {config.DEVICE} ---")
    model = UNet(in_channels=1, out_channels=1).to(config.DEVICE)
    loss_fn = DiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scaler = torch.amp.GradScaler()

    patient_img_dir = os.path.join(config.PATIENTS_DIR, "imgs")
    patient_mask_dir = os.path.join(config.PATIENTS_DIR, "labels")

    patient_image_files = sorted(
        [os.path.join(patient_img_dir, f) for f in os.listdir(patient_img_dir) if f.endswith('.png')]
    )
    patient_mask_files = [
        os.path.join(patient_mask_dir, f.name.replace("patient_", "segmentation_"))
        for f in [Path(p) for p in patient_image_files]
    ]

    patient_dataset = TumorDataset(
        image_paths=patient_image_files,
        mask_paths=patient_mask_files,
        transform=train_transform
    )

    num_patients = len(patient_dataset)
    num_val = int(num_patients * config.VALIDATION_SPLIT)
    num_train = num_patients - num_val
    train_patient_ds, val_ds = random_split(patient_dataset, [num_train, num_val])
    val_ds.dataset.transform = val_transform

    control_img_dir = os.path.join(config.CONTROLS_DIR, "imgs")
    control_image_files = sorted([os.path.join(control_img_dir, f) for f in os.listdir(control_img_dir) if f.endswith('.png')])
    control_ds = TumorDataset(
        image_paths=control_image_files,
        mask_paths=None,
        transform=train_transform
    )
    train_ds = ConcatDataset([train_patient_ds, control_ds])

    print(f"Total training samples: {len(train_ds)}")
    print(f"Total validation samples: {len(val_ds)}")

    balanced_sampler = BalancedBatchSampler(
        dataset=train_ds,
        batch_size=config.BATCH_SIZE,
        patient_ratio=0.5  # This means 50% of each batch will be patient images
    )

    # When using a batch_sampler, you must NOT specify batch_size, shuffle, sampler, or drop_last in the DataLoader
    train_loader = DataLoader(
        train_ds,
        batch_sampler=balanced_sampler,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    best_dice_score = -1.0
    for epoch in range(config.EPOCHS):
        train_loss = train_one_epoch(train_loader, model, optimizer, loss_fn, scaler)

        val_metrics = check_accuracy(val_loader, model, device=config.DEVICE)

        print(
            f"\n--- Epoch {epoch+1}/{config.EPOCHS} ---\n"
            f"  Train Loss: {train_loss:.4f}\n"
            f"  Val Loss:   {val_metrics['val_loss']:.4f}\n"
            f"  Dice Score: {val_metrics['dice_score']:.4f}\n"
            f"  IoU:        {val_metrics['iou']:.4f}\n"
            f"  Precision:  {val_metrics['precision']:.4f}\n"
            f"  Recall:     {val_metrics['recall']:.4f}"
        )

        current_dice = val_metrics['dice_score']
        if current_dice > best_dice_score:
            best_dice_score = current_dice
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, config.MODEL_NAME)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"-> Model Saved to {checkpoint_path} (Best Dice Score: {best_dice_score:.4f})\n")


if __name__ == "__main__":
    if not os.path.exists(config.CHECKPOINT_DIR):
        os.makedirs(config.CHECKPOINT_DIR)
    main()