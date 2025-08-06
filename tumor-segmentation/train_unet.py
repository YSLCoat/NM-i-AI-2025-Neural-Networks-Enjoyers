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
import torchvision.utils as vutils

# Tversky Loss (PyTorch)
class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = preds.view(-1)
        targets = targets.view(-1)
        TP = (preds * targets).sum()
        FP = ((1 - targets) * preds).sum()
        FN = (targets * (1 - preds)).sum()
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        return 1 - tversky

class CombinedLossDiceTversky(nn.Module):
    def __init__(self, weight_dice=1.0, weight_tversky=1.0, alpha=0.7, beta=0.3):
        super().__init__()
        self.dice = DiceLoss(mode='binary')
        self.tversky = TverskyLoss(alpha=alpha, beta=beta)
        self.weight_dice = weight_dice
        self.weight_tversky = weight_tversky

    def forward(self, preds, targets):
        preds_sigmoid = torch.sigmoid(preds)
        loss_dice = self.dice(preds_sigmoid, targets)
        loss_tversky = self.tversky(preds_sigmoid, targets)
        loss = self.weight_dice * loss_dice + self.weight_tversky * loss_tversky
        return loss, loss_dice.detach(), loss_tversky.detach()

def soft_dice_score(preds, targets, eps=1e-6):
    probs = torch.sigmoid(preds)
    intersection = (probs * targets).sum(dim=[1, 2, 3])
    union = probs.sum(dim=[1, 2, 3]) + targets.sum(dim=[1, 2, 3])
    dice = (2. * intersection + eps) / (union + eps)
    return dice.mean()

@torch.no_grad()
def validate(model, val_loader, device):
    model.eval()
    dice_scores = []
    for images, masks in val_loader:
        images = images.to(device)
        masks = masks.to(device)
        preds = model(images)
        preds_bin = (torch.sigmoid(preds) > 0.5).float()

        intersection = (preds_bin * masks).sum(dim=[1, 2, 3])
        union = preds_bin.sum(dim=[1, 2, 3]) + masks.sum(dim=[1, 2, 3])
        dice = ((2. * intersection + 1e-6) / (union + 1e-6)).mean()
        dice_scores.append(dice.item())

    return sum(dice_scores) / len(dice_scores)

def train():
    model_name = 'model_6_4'
    epochs = 5
    batch_size = 4
    resize_shape = (512, 512)
    learning_rate = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_images = sorted(glob("tumor-segmentation/datasets/train_augmented_2000/imgs/*.png"))
    train_masks = sorted(glob("tumor-segmentation/datasets/train_augmented_2000/labels/*.png"))

    val_images = sorted(glob("tumor-segmentation/datasets/val_2/imgs/*.png"))
    val_masks = sorted(glob("tumor-segmentation/datasets/val_2/labels/*.png"))

    train_ds = TumorSegmentationDataset(
        image_paths=train_images,
        mask_paths=train_masks,
        resize_shape=resize_shape,
        augment=True
    )
    val_ds = TumorSegmentationDataset(
        image_paths=val_images,
        mask_paths=val_masks,
        resize_shape=resize_shape,
        augment=False
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    model = get_unet_model(in_channels=1, out_classes=1).to(device)
    loss_fn = CombinedLossDiceTversky(weight_dice=0.7, weight_tversky=0.3)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    log_dir = f"runs/unet_{model_name}_{time.strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(log_dir=log_dir)

    global_step = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_dice = 0.0
        running_soft_dice = 0.0
        running_loss_dice = 0.0
        running_loss_tversky = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, masks in loop:
            images = images.to(device)
            masks = masks.to(device)

            preds = model(images)
            loss, loss_dice, loss_tversky = loss_fn(preds, masks)

            preds_bin = (torch.sigmoid(preds) > 0.5).float()

            if global_step % 50 == 0:
                img = images[0].detach().cpu()
                mask = masks[0].detach().cpu()
                pred = torch.sigmoid(preds[0]).detach().cpu()
                pred_bin = (pred > 0.5).float()

                writer.add_image("Debug/Input_Image", vutils.make_grid(img.unsqueeze(0), normalize=True), global_step)
                writer.add_image("Debug/True_Mask", vutils.make_grid(mask.unsqueeze(0)), global_step)
                writer.add_image("Debug/Predicted_Mask_Sigmoid", vutils.make_grid(pred), global_step)
                writer.add_image("Debug/Predicted_Mask_Binary", vutils.make_grid(pred_bin), global_step)

            intersection = (preds_bin * masks).sum(dim=[1, 2, 3])
            union = preds_bin.sum(dim=[1, 2, 3]) + masks.sum(dim=[1, 2, 3])
            hard_dice = ((2. * intersection + 1e-6) / (union + 1e-6)).mean()

            soft_dice = soft_dice_score(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_dice += hard_dice.item()
            running_soft_dice += soft_dice.item()
            running_loss_dice += loss_dice.item()
            running_loss_tversky += loss_tversky.item()

            writer.add_scalar('Loss/combined', loss.item(), global_step)
            writer.add_scalar('Loss/Dice', loss_dice.item(), global_step)
            writer.add_scalar('Loss/Tversky', loss_tversky.item(), global_step)
            writer.add_scalar('Dice/hard', hard_dice.item(), global_step)
            writer.add_scalar('Dice/soft', soft_dice.item(), global_step)

            loop.set_postfix(loss=running_loss / (loop.n + 1))
            global_step += 1

        avg_loss = running_loss / len(train_loader)
        avg_dice = running_dice / len(train_loader)
        avg_soft_dice = running_soft_dice / len(train_loader)
        avg_loss_dice = running_loss_dice / len(train_loader)
        avg_loss_tversky = running_loss_tversky / len(train_loader)

        writer.add_scalar('Epoch/Loss_combined', avg_loss, epoch)
        writer.add_scalar('Epoch/Loss_Dice', avg_loss_dice, epoch)
        writer.add_scalar('Epoch/Loss_Tversky', avg_loss_tversky, epoch)
        writer.add_scalar('Epoch/Dice_hard', avg_dice, epoch)
        writer.add_scalar('Epoch/Dice_soft', avg_soft_dice, epoch)

        print(f"Epoch {epoch+1} finished:")
        print(f" - Avg Combined Loss: {avg_loss:.4f}")
        print(f" - Avg Dice (hard):   {avg_dice:.4f}")
        print(f" - Avg Dice (soft):   {avg_soft_dice:.4f}")

        val_dice = validate(model, val_loader, device)
        writer.add_scalar('Val/Dice_hard', val_dice, epoch)
        print(f" - Val Dice (hard):   {val_dice:.4f}")

        scheduler.step(val_dice)

    torch.save(model.state_dict(), f"tumor-segmentation/models/unet_{model_name}.pth")
    print(f"Model saved as unet_{model_name}.pth")
    writer.flush()
    writer.close()

if __name__ == "__main__":
    train()
