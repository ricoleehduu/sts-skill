"""
Inference script for the Task 1 metal artifact removal baseline.

Loads a trained UNet model and processes input NIfTI volumes slice by slice,
producing artifact-corrected outputs.

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

from model import UNet


def normalize_volume(vol):
    """Normalize volume to [0, 1] using global min/max."""
    vmin, vmax = vol.min(), vol.max()
    return (vol - vmin) / (vmax - vmin + 1e-8), vmin, vmax


def denormalize_volume(vol, vmin, vmax):
    """Restore original intensity range."""
    return vol * (vmax - vmin) + vmin


def process_volume(model, volume, device, patch_size=256):
    """
    Process a 3D volume slice by slice.

    Args:
        model: Trained UNet model.
        volume: numpy array of shape (H, W, D).
        device: torch device.
        patch_size: tile size for inference.

    Returns:
        Restored volume of the same shape.
    """
    h, w, d = volume.shape
    output = np.zeros_like(volume, dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for i in range(d):
            slc = volume[:, :, i].astype(np.float32)

            # Normalize
            slc_norm, vmin, vmax = normalize_volume(slc)

            # Pad to patch_size if needed
            pad_h = max(0, patch_size - h)
            pad_w = max(0, patch_size - w)
            if pad_h > 0 or pad_w > 0:
                slc_norm = np.pad(slc_norm, ((0, pad_h), (0, pad_w)), mode="reflect")

            # To tensor
            tensor = torch.from_numpy(slc_norm).unsqueeze(0).unsqueeze(0).to(device)

            # Predict
            pred = model(tensor).squeeze().cpu().numpy()

            # Crop padding
            pred = pred[:h, :w]

            # Denormalize
            pred = denormalize_volume(pred, vmin, vmax)

            output[:, :, i] = pred

    return output


def main():
    parser = argparse.ArgumentParser(description="Predict with Task 1 baseline")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing input NIfTI volumes")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save predicted volumes")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--patch_size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = UNet(in_channels=1, out_channels=1).to(device)

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
        result = process_volume(model, volume, device, args.patch_size)

        # Save with same affine/header
        out_nii = nib.Nifti1Image(result, affine=nii.affine, header=nii.header)
        out_path = os.path.join(args.output_dir, fname)
        nib.save(out_nii, out_path)
        print(f"  Saved: {out_path}")

    print("Inference complete.")


if __name__ == "__main__":
    main()
