import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from unet_model import get_unet_model
from unet_dataset import TumorSegmentationDataset
from segmentation_models_pytorch.losses import DiceLoss
from glob import glob
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import time

class CombinedLoss(nn.Module):
    def __init__(self, weight_dice=1.0, weight_bce=1.0):
        super().__init__()
        self.dice = DiceLoss(mode='binary')
        self.bce = nn.BCEWithLogitsLoss()

        self.weight_dice = weight_dice
        self.weight_bce = weight_bce

    def forward(self, preds, targets):
        # preds are raw logits, so:
        loss_dice = self.dice(torch.sigmoid(preds), targets)  # Dice expects probabilities
        loss_bce = self.bce(preds, targets)  # BCE with logits expects raw preds
        return self.weight_dice * loss_dice + self.weight_bce * loss_bce

def train():
    # Configure the training
    model_name = 'model_5_1'
    epochs = 10
    batch_size = 3
    resize_shape = (512, 512)  # (height, width)
    learning_rate = 1e-3

    # Enable CUDA if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load training images and masks
    train_images = sorted(glob("tumor-segmentation/datasets/train_augmented_500/imgs/*.png"))
    train_masks = sorted(glob("tumor-segmentation/datasets/train_augmented_500/labels/*.png"))

    # Dataset and DataLoader
    train_ds = TumorSegmentationDataset(
        image_paths=train_images,
        mask_paths=train_masks,
        resize_shape=resize_shape,
        augment=True
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Model, loss, optimizer
    model = get_unet_model(in_channels=1, out_classes=1).to(device)
    #loss_fn = DiceLoss(mode='binary')
    #loss_fn = nn.BCEWithLogitsLoss()
    loss_fn = CombinedLoss(weight_dice=0.7, weight_bce=0.3)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    log_dir = f"runs/unet_{model_name}_{time.strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(log_dir=log_dir)

    global_step = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_dice = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, masks in loop:
            images = images.to(device)
            masks = masks.to(device)

            preds = model(images)
            loss = loss_fn(preds, masks)

            # Calculate Dice score (threshold 0.5)
            #preds_bin = (preds > 0.5).float()
            preds_bin = (torch.sigmoid(preds) > 0.5).float()

            intersection = (preds_bin * masks).sum(dim=[1,2,3])
            union = preds_bin.sum(dim=[1,2,3]) + masks.sum(dim=[1,2,3])
            dice = ((2. * intersection + 1e-6) / (union + 1e-6)).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_dice += dice.item()

            
            writer.add_scalar('Loss/train', loss.item(), global_step)
            writer.add_scalar('Dice/train', dice.item(), global_step)

            loop.set_postfix(loss=running_loss / (loop.n + 1))
            global_step += 1

        avg_loss = running_loss / len(train_loader)
        avg_dice = running_dice / len(train_loader)
        writer.add_scalar('Loss/epoch_avg', avg_loss, epoch)
        writer.add_scalar('Dice/epoch_avg', avg_dice, epoch)
        print(f"Epoch {epoch+1} finished, average loss: {avg_loss:.4f}, average dice: {avg_dice:.4f}")
    # Save model checkpoint

    torch.save(model.state_dict(), f"tumor-segmentation/models/unet_{model_name}.pth")
    print(f"Model saved as unet_{model_name}.pth")
    writer.close()


if __name__ == "__main__":
    train()
