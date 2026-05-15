"""
Training script for the Task 1 CBCT teeth segmentation baseline.

Reads paired (image, mask) NIfTI volumes, extracts 2D slices,
and trains the UNet model with BCE + Dice loss.

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
from torch.utils.data import Dataset, DataLoader

from model import UNet


# ---------------------------------------------------------------------------
# Dice loss
# ---------------------------------------------------------------------------

def dice_loss(pred, target, smooth=1e-5):
    """Compute Dice loss for binary segmentation."""
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    return 1 - (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CBCTSliceDataset(Dataset):
    """
    Loads paired NIfTI volumes and yields random 2D axial slices.

    Expects:
        data_dir/images/ -- CBCT volumes with metal artifacts (.nii.gz)
        data_dir/masks/  -- Binary teeth segmentation masks (.nii.gz)

    File names must match between images/ and masks/.
    """

    def __init__(self, data_dir, patch_size=256, augment=True):
        self.patch_size = patch_size
        self.augment = augment

        image_dir = os.path.join(data_dir, "images")
        mask_dir = os.path.join(data_dir, "masks")

        self.slices = []  # list of (image_path, mask_path, slice_idx)

        for fname in sorted(os.listdir(image_dir)):
            if not fname.endswith((".nii", ".nii.gz")):
                continue
            img_path = os.path.join(image_dir, fname)
            msk_path = os.path.join(mask_dir, fname)
            if not os.path.exists(msk_path):
                print(f"Warning: no matching mask for {fname}, skipping.")
                continue

            # Count slices in this volume
            nii = nib.load(img_path)
            n_slices = nii.shape[2] if nii.ndim == 3 else nii.shape[3]
            for i in range(n_slices):
                self.slices.append((img_path, msk_path, i))

        print(f"Found {len(self.slices)} slices from {image_dir}")

    def __len__(self):
        return len(self.slices)

    def _load_slice(self, path, idx):
        vol = nib.load(path).get_fdata()
        if vol.ndim == 3:
            slc = vol[:, :, idx]
        else:
            slc = vol[:, :, idx, 0]
        return slc.astype(np.float32)

    def _random_crop(self, img, mask):
        h, w = img.shape
        ps = self.patch_size
        if h < ps or w < ps:
            img = np.pad(img, ((0, max(0, ps - h)), (0, max(0, ps - w))), mode="reflect")
            mask = np.pad(mask, ((0, max(0, ps - h)), (0, max(0, ps - w))), mode="reflect")
            h, w = img.shape
        y = random.randint(0, h - ps)
        x = random.randint(0, w - ps)
        return img[y:y+ps, x:x+ps], mask[y:y+ps, x:x+ps]

    def _augment(self, img, mask):
        if random.random() > 0.5:
            img = np.flip(img, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()
        if random.random() > 0.5:
            img = np.flip(img, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
        return img, mask

    def __getitem__(self, idx):
        img_path, msk_path, sidx = self.slices[idx]
        img = self._load_slice(img_path, sidx)
        mask = self._load_slice(msk_path, sidx)

        # Normalize image to [0, 1] per-slice
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        # Binarize mask (anything > 0 is teeth)
        mask = (mask > 0).astype(np.float32)

        img, mask = self._random_crop(img, mask)

        if self.augment:
            img, mask = self._augment(img, mask)

        img = torch.from_numpy(img).unsqueeze(0)       # (1, H, W)
        mask = torch.from_numpy(mask).unsqueeze(0)
        return img, mask


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    dataset = CBCTSliceDataset(args.data_dir, patch_size=args.patch_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True)

    # Model
    model = UNet(in_channels=1, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    bce = nn.BCEWithLogitsLoss()

    os.makedirs(args.output_dir, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (images, masks) in enumerate(loader):
            images = images.to(device)
            masks = masks.to(device)

            preds = model(images)
            loss = bce(preds, masks) + dice_loss(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch}/{args.epochs}  Loss: {avg_loss:.6f}  LR: {scheduler.get_last_lr()[0]:.2e}")

        # Save checkpoint
        if epoch % args.save_every == 0 or avg_loss < best_loss:
            ckpt_path = os.path.join(args.output_dir, f"epoch_{epoch}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            }, ckpt_path)
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = os.path.join(args.output_dir, "best_model.pth")
                torch.save(model.state_dict(), best_path)
                print(f"  -> New best model saved (loss={best_loss:.6f})")

    print("Training complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Task 1 baseline")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root dir with images/ and masks/ subfolders")
    parser.add_argument("--output_dir", type=str, default="./checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=10,
                        help="Save checkpoint every N epochs")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
