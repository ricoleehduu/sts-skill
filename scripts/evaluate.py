"""
Evaluate segmentation predictions against ground truth masks.

Computes Dice score (mean_dice, std_dice) matching Codabench scoring,
plus IoU as a bonus metric.

Usage:
    python scripts/evaluate.py --pred <pred_dir> --gt <gt_dir>
    python scripts/evaluate.py --pred data/test/masks --gt data/test/gt_masks --verbose
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def load_mask(path: str) -> np.ndarray:
    """Load a mask image as binary numpy array (0 or 1)."""
    mask = np.array(Image.open(path))
    # Handle RGB masks (take first channel or convert to grayscale)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    # Binarize: any non-zero pixel is foreground
    return (mask > 0).astype(np.uint8)


def dice_coeff(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute Dice coefficient between two binary masks."""
    intersection = np.sum(pred * gt)
    total = np.sum(pred) + np.sum(gt)
    if total == 0:
        return 1.0  # Both empty = perfect match
    return 2.0 * intersection / total


def iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute Intersection over Union between two binary masks."""
    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt) - intersection
    if union == 0:
        return 1.0  # Both empty = perfect match
    return intersection / union


def evaluate(pred_dir: str, gt_dir: str, verbose: bool = False) -> dict:
    """
    Evaluate all predictions against ground truth.

    Args:
        pred_dir: Directory containing predicted mask PNGs
        gt_dir: Directory containing ground truth mask PNGs
        verbose: Print per-image scores

    Returns:
        Dictionary with mean_dice, std_dice, mean_iou, per-image scores
    """
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)

    # Find matching files
    pred_files = sorted(pred_dir.glob("*.png"))
    gt_files = sorted(gt_dir.glob("*.png"))

    if not pred_files:
        print(f"Error: No PNG files found in {pred_dir}")
        sys.exit(1)

    if not gt_files:
        print(f"Error: No PNG files found in {gt_dir}")
        sys.exit(1)

    # Build filename → path maps
    pred_map = {f.name: f for f in pred_files}
    gt_map = {f.name: f for f in gt_files}

    # Find common filenames
    common_names = sorted(set(pred_map.keys()) & set(gt_map.keys()))
    if not common_names:
        print(f"Error: No matching filenames between {pred_dir} and {gt_dir}")
        print(f"  Pred files: {[f.name for f in pred_files[:5]]}...")
        print(f"  GT files:   {[f.name for f in gt_files[:5]]}...")
        sys.exit(1)

    pred_only = set(pred_map.keys()) - set(gt_map.keys())
    gt_only = set(gt_map.keys()) - set(pred_map.keys())
    if pred_only:
        print(f"Warning: {len(pred_only)} pred files have no GT match (ignored)")
    if gt_only:
        print(f"Warning: {len(gt_only)} GT files have no pred match (counted as 0)")

    # Evaluate each pair
    dice_scores = []
    iou_scores = []
    results = []

    for name in common_names:
        pred_mask = load_mask(str(pred_map[name]))
        gt_mask = load_mask(str(gt_map[name]))

        # Ensure same shape
        if pred_mask.shape != gt_mask.shape:
            print(f"Warning: Shape mismatch for {name}: pred={pred_mask.shape}, gt={gt_mask.shape}")
            # Resize pred to match gt
            from PIL import Image as PILImage
            pred_pil = PILImage.fromarray(pred_mask * 255)
            pred_pil = pred_pil.resize((gt_mask.shape[1], gt_mask.shape[0]))
            pred_mask = (np.array(pred_pil) > 0).astype(np.uint8)

        d = dice_coeff(pred_mask, gt_mask)
        i = iou_score(pred_mask, gt_mask)
        dice_scores.append(d)
        iou_scores.append(i)
        results.append({"name": name, "dice": d, "iou": i})

        if verbose:
            print(f"  {name}: dice={d:.4f}, iou={i:.4f}")

    # Handle GT files with no prediction (count as 0)
    for name in gt_only:
        dice_scores.append(0.0)
        iou_scores.append(0.0)
        results.append({"name": name, "dice": 0.0, "iou": 0.0})

    # Compute aggregates
    dice_arr = np.array(dice_scores)
    iou_arr = np.array(iou_scores)

    summary = {
        "mean_dice": float(np.mean(dice_arr)),
        "std_dice": float(np.std(dice_arr)),
        "mean_iou": float(np.mean(iou_arr)),
        "std_iou": float(np.std(iou_arr)),
        "num_images": len(common_names) + len(gt_only),
        "num_matched": len(common_names),
        "num_gt_only": len(gt_only),
        "num_pred_only": len(pred_only),
        "per_image": results,
    }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate segmentation predictions")
    parser.add_argument("--pred", required=True, help="Directory containing predicted masks")
    parser.add_argument("--gt", required=True, help="Directory containing ground truth masks")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print per-image scores")
    args = parser.parse_args()

    print(f"Evaluating predictions...")
    print(f"  Pred: {args.pred}")
    print(f"  GT:   {args.gt}")
    print()

    result = evaluate(args.pred, args.gt, args.verbose)

    print()
    print("=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"  Images evaluated: {result['num_images']}")
    print(f"  Mean Dice:        {result['mean_dice']:.4f}")
    print(f"  Std Dice:         {result['std_dice']:.4f}")
    print(f"  Mean IoU:         {result['mean_iou']:.4f}")
    print(f"  Std IoU:          {result['std_iou']:.4f}")
    print("=" * 50)

    if result["num_pred_only"] > 0:
        print(f"\nNote: {result['num_pred_only']} predictions had no matching GT (ignored)")
    if result["num_gt_only"] > 0:
        print(f"\nNote: {result['num_gt_only']} GT files had no prediction (scored as 0)")


if __name__ == "__main__":
    main()
