"""
Training script for the Task 3 CBCT teeth segmentation baseline.

Reads paired (CBCT volume, segmentation mask) NIfTI files and trains
the 3D UNet model with Dice loss.

Usage:
    python train.py --data_dir /path/to/train --output_dir ./checkpoints
"""

import argparse
import os
import random

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from model import UNet3D


# ---------------------------------------------------------------------------
# Dice loss
# ---------------------------------------------------------------------------

def dice_coeff(pred, target, smooth=1.0):
    """Compute Dice coefficient."""
    pred = torch.sigmoid(pred)
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


def dice_loss(pred, target, smooth=1.0):
    """Compute Dice loss (1 - Dice coefficient)."""
    return 1.0 - dice_coeff(pred, target, smooth)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CBCTSegmentationDataset(Dataset):
    """
    Loads paired NIfTI volumes for CBCT teeth segmentation.

    Expects:
        data_dir/images/  -- CBCT volumes (.nii.gz)
        data_dir/masks/   -- Binary segmentation masks (.nii.gz)

    File names must match between images/ and masks/.
    """

    def __init__(self, data_dir, patch_size=64, augment=True):
        self.patch_size = patch_size
        self.augment = augment

        image_dir = os.path.join(data_dir, "images")
        mask_dir = os.path.join(data_dir, "masks")

        self.volumes = []  # list of (image_path, mask_path)

        for fname in sorted(os.listdir(image_dir)):
            if not fname.endswith((".nii", ".nii.gz")):
                continue
            img_path = os.path.join(image_dir, fname)
            mask_path = os.path.join(mask_dir, fname)
            if not os.path.exists(mask_path):
                print(f"Warning: no matching mask for {fname}, skipping.")
                continue
            self.volumes.append((img_path, mask_path))

        print(f"Found {len(self.volumes)} volume pairs from {image_dir}")

    def __len__(self):
        return len(self.volumes)

    def _load_volume(self, path):
        vol = nib.load(path).get_fdata()
        return vol.astype(np.float32)

    def _random_crop_3d(self, img, mask):
        d, h, w = img.shape
        ps = self.patch_size

        # Pad if needed
        if d < ps or h < ps or w < ps:
            pad_d = max(0, ps - d)
            pad_h = max(0, ps - h)
            pad_w = max(0, ps - w)
            img = np.pad(img, ((0, pad_d), (0, pad_h), (0, pad_w)), mode="reflect")
            mask = np.pad(mask, ((0, pad_d), (0, pad_h), (0, pad_w)), mode="constant")
            d, h, w = img.shape

        z = random.randint(0, d - ps)
        y = random.randint(0, h - ps)
        x = random.randint(0, w - ps)

        return (img[z:z+ps, y:y+ps, x:x+ps],
                mask[z:z+ps, y:y+ps, x:x+ps])

    def _augment_3d(self, img, mask):
        # Random flip along each axis
        if random.random() > 0.5:
            img = np.flip(img, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()
        if random.random() > 0.5:
            img = np.flip(img, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
        if random.random() > 0.5:
            img = np.flip(img, axis=2).copy()
            mask = np.flip(mask, axis=2).copy()
        return img, mask

    def __getitem__(self, idx):
        img_path, mask_path = self.volumes[idx]
        img = self._load_volume(img_path)
        mask = self._load_volume(mask_path)

        # Normalize to [0, 1] per volume
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        # Binarize mask
        mask = (mask > 0).astype(np.float32)

        img, mask = self._random_crop_3d(img, mask)

        if self.augment:
            img, mask = self._augment_3d(img, mask)

        img = torch.from_numpy(img).unsqueeze(0)       # (1, D, H, W)
        mask = torch.from_numpy(mask).unsqueeze(0)      # (1, D, H, W)
        return img, mask


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    dataset = CBCTSegmentationDataset(args.data_dir, patch_size=args.patch_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True)

    # Model
    model = UNet3D(in_channels=1, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Loss: combination of BCE and Dice
    bce_criterion = nn.BCEWithLogitsLoss()

    os.makedirs(args.output_dir, exist_ok=True)
    best_dice = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_dice = 0.0

        for batch_idx, (images, masks) in enumerate(loader):
            images = images.to(device)
            masks = masks.to(device)

            preds = model(images)
            bce_loss = bce_criterion(preds, masks)
            d_loss = dice_loss(preds, masks)
            loss = bce_loss + d_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_dice += dice_coeff(preds, masks).item()

        scheduler.step()
        avg_loss = epoch_loss / len(loader)
        avg_dice = epoch_dice / len(loader)
        print(f"Epoch {epoch}/{args.epochs}  Loss: {avg_loss:.6f}  Dice: {avg_dice:.4f}  LR: {scheduler.get_last_lr()[0]:.2e}")

        # Save checkpoint
        if epoch % args.save_every == 0 or avg_dice > best_dice:
            ckpt_path = os.path.join(args.output_dir, f"epoch_{epoch}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "dice": avg_dice,
                "loss": avg_loss,
            }, ckpt_path)
            if avg_dice > best_dice:
                best_dice = avg_dice
                best_path = os.path.join(args.output_dir, "best_model.pth")
                torch.save(model.state_dict(), best_path)
                print(f"  -> New best model saved (Dice={best_dice:.4f})")

    print("Training complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Task 3 baseline")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root dir with images/ and masks/ subfolders")
    parser.add_argument("--output_dir", type=str, default="./checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patch_size", type=int, default=64,
                        help="3D patch size for training crops")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=10,
                        help="Save checkpoint every N epochs")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
