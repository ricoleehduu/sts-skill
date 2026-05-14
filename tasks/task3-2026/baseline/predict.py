"""
Inference script for the Task 3 CBCT teeth segmentation baseline.

Loads a trained 3D UNet model and processes input NIfTI volumes,
producing binary segmentation masks.

Usage:
    python predict.py \
        --input_dir /path/to/test_inputs \
        --output_dir /path/to/predictions \
        --checkpoint ./checkpoints/best_model.pth
"""

import argparse
import os

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from model import UNet3D


def normalize_volume(vol):
    """Normalize volume to [0, 1] using global min/max."""
    vmin, vmax = vol.min(), vol.max()
    return (vol - vmin) / (vmax - vmin + 1e-8)


def process_volume(model, volume, device, patch_size=64, overlap=16):
    """
    Process a 3D volume using sliding window with overlap.

    Args:
        model: Trained UNet3D model.
        volume: numpy array of shape (D, H, W).
        device: torch device.
        patch_size: size of 3D patches.
        overlap: overlap between adjacent patches.

    Returns:
        Binary segmentation mask of the same shape.
    """
    d, h, w = volume.shape
    stride = patch_size - overlap

    # Pad volume to ensure complete coverage
    pad_d = max(0, patch_size - d)
    pad_h = max(0, patch_size - h)
    pad_w = max(0, patch_size - w)

    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        volume = np.pad(volume, ((0, pad_d), (0, pad_h), (0, pad_w)), mode="reflect")

    # Normalize
    volume = normalize_volume(volume)

    # Initialize output and count arrays
    output = np.zeros_like(volume, dtype=np.float32)
    count = np.zeros_like(volume, dtype=np.float32)

    model.eval()
    with torch.no_grad():
        # Generate patch coordinates
        z_coords = list(range(0, volume.shape[0] - patch_size + 1, stride))
        y_coords = list(range(0, volume.shape[1] - patch_size + 1, stride))
        x_coords = list(range(0, volume.shape[2] - patch_size + 1, stride))

        # Ensure last patch covers the boundary
        if z_coords[-1] + patch_size < volume.shape[0]:
            z_coords.append(volume.shape[0] - patch_size)
        if y_coords[-1] + patch_size < volume.shape[1]:
            y_coords.append(volume.shape[1] - patch_size)
        if x_coords[-1] + patch_size < volume.shape[2]:
            x_coords.append(volume.shape[2] - patch_size)

        for z in z_coords:
            for y in y_coords:
                for x in x_coords:
                    # Extract patch
                    patch = volume[z:z+patch_size, y:y+patch_size, x:x+patch_size]
                    patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)

                    # Predict
                    pred = torch.sigmoid(model(patch_tensor)).squeeze().cpu().numpy()

                    # Accumulate predictions
                    output[z:z+patch_size, y:y+patch_size, x:x+patch_size] += pred
                    count[z:z+patch_size, y:y+patch_size, x:x+patch_size] += 1

    # Average overlapping predictions
    output = output / (count + 1e-8)

    # Crop padding
    output = output[:d, :h, :w]

    # Threshold to binary mask
    output = (output > 0.5).astype(np.float32)

    return output


def main():
    parser = argparse.ArgumentParser(description="Predict with Task 3 baseline")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing input NIfTI volumes")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save predicted segmentation masks")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--patch_size", type=int, default=64,
                        help="3D patch size for inference")
    parser.add_argument("--overlap", type=int, default=16,
                        help="Overlap between adjacent patches")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold for binary segmentation")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = UNet3D(in_channels=1, out_channels=1).to(device)

    # Support both full checkpoint dict and raw state_dict
    ckpt = torch.load(args.checkpoint, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Process each volume
    filenames = sorted([
        f for f in os.listdir(args.input_dir)
        if f.endswith((".nii", ".nii.gz"))
    ])

    if not filenames:
        print(f"No NIfTI files found in {args.input_dir}")
        return

    for fname in filenames:
        print(f"Processing: {fname}")
        fpath = os.path.join(args.input_dir, fname)
        nii = nib.load(fpath)
        volume = nii.get_fdata()

        if volume.ndim != 3:
            print(f"  Skipping {fname}: expected 3D volume, got shape {volume.shape}")
            continue

        # Process
        result = process_volume(model, volume, device, args.patch_size, args.overlap)

        # Save with same affine/header
        out_nii = nib.Nifti1Image(result, affine=nii.affine, header=nii.header)
        out_path = os.path.join(args.output_dir, fname)
        nib.save(out_nii, out_path)
        print(f"  Saved: {out_path}")

    print("Inference complete.")


if __name__ == "__main__":
    main()
