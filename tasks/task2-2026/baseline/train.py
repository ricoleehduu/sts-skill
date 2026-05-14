"""
Training script for the Task 2 CBCT-IOS registration baseline.

Trains a PointNet-based model to predict rigid transformations that align
IOS point clouds to CBCT-derived tooth surface point clouds.

Data layout:
    data_dir/
        cbct/           # CBCT volumes (.nii.gz)
        ios/            # IOS meshes (.stl or .ply)
        transform.json  # Ground-truth transformation per case

transform.json format:
    {
        "case_001": {
            "rotation": [[r00,r01,r02],[r10,r11,r12],[r20,r21,r22]],
            "translation": [tx, ty, tz]
        },
        ...
    }

Usage:
    python train.py --data_dir /path/to/train --output_dir ./checkpoints
"""

import argparse
import json
import os
import random

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

from model import CBCTIOSRegistration, chamfer_distance


# ---------------------------------------------------------------------------
# Geometry utilities
# ---------------------------------------------------------------------------

def extract_surface_points(volume, iso_value=0.5, num_points=4096, spacing=(1, 1, 1)):
    """Extract surface point cloud from a binary/label volume using marching cubes.

    Falls back to random boundary voxel sampling if marching cubes is unavailable.

    Args:
        volume: 3D numpy array (H, W, D).
        iso_value: threshold for surface extraction.
        num_points: number of points to sample.
        spacing: voxel spacing.

    Returns:
        points: (num_points, 3) surface points.
    """
    try:
        from skimage.measure import marching_cubes
        verts, faces, _, _ = marching_cubes(volume, level=iso_value, spacing=spacing)
        # Subsample
        if len(verts) > num_points:
            idx = np.random.choice(len(verts), num_points, replace=False)
            verts = verts[idx]
        elif len(verts) < num_points:
            idx = np.random.choice(len(verts), num_points, replace=True)
            verts = verts[idx]
        return verts.astype(np.float32)
    except Exception:
        # Fallback: sample from non-zero boundary voxels
        binary = volume > iso_value
        # Simple boundary: erode and subtract
        from scipy.ndimage import binary_erosion
        eroded = binary_erosion(binary)
        boundary = binary & ~eroded
        coords = np.argwhere(boundary)
        if len(coords) == 0:
            coords = np.argwhere(binary)
        if len(coords) == 0:
            return np.zeros((num_points, 3), dtype=np.float32)
        if len(coords) > num_points:
            idx = np.random.choice(len(coords), num_points, replace=False)
        else:
            idx = np.random.choice(len(coords), num_points, replace=True)
        return coords[idx].astype(np.float32)


def load_mesh_points(mesh_path, num_points=4096):
    """Load point cloud from a mesh file (STL, PLY, OBJ).

    Args:
        mesh_path: path to mesh file.
        num_points: number of surface points to sample.

    Returns:
        points: (num_points, 3) array.
    """
    if HAS_TRIMESH:
        mesh = trimesh.load(mesh_path, process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(mesh.dump())
        points, _ = trimesh.sample.sample_surface(mesh, num_points)
        return points.astype(np.float32)
    else:
        raise ImportError(
            "trimesh is required to load mesh files. "
            "Install with: pip install trimesh"
        )


def random_rotation_matrix():
    """Generate a random 3x3 rotation matrix."""
    angles = np.random.uniform(-np.pi / 6, np.pi / 6, size=3)
    rx, ry, rz = angles
    Rx = np.array([[1, 0, 0],
                    [0, np.cos(rx), -np.sin(rx)],
                    [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                    [0, 1, 0],
                    [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                    [np.sin(rz), np.cos(rz), 0],
                    [0, 0, 1]])
    return Rz @ Ry @ Rx


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RegistrationDataset(Dataset):
    """Dataset for CBCT-IOS registration training.

    Each sample provides:
      - source_pts: IOS point cloud (randomly augmented with rigid transform)
      - target_pts: CBCT surface point cloud
      - gt_R, gt_t: ground-truth transformation (identity if no augmentation)
    """

    def __init__(self, data_dir, num_points=4096, augment=True):
        self.num_points = num_points
        self.augment = augment

        self.cbct_dir = os.path.join(data_dir, "cbct")
        self.ios_dir = os.path.join(data_dir, "ios")
        transform_path = os.path.join(data_dir, "transform.json")

        # Load ground-truth transforms
        if os.path.exists(transform_path):
            with open(transform_path, "r") as f:
                self.transforms = json.load(f)
        else:
            print("Warning: No transform.json found, using identity transforms.")
            self.transforms = {}

        # Discover cases
        self.cases = []
        for fname in sorted(os.listdir(self.ios_dir)):
            if fname.endswith((".stl", ".ply", ".obj")):
                case_id = os.path.splitext(fname)[0]
                # Find matching CBCT
                for ext in (".nii.gz", ".nii"):
                    cbct_path = os.path.join(self.cbct_dir, case_id + ext)
                    if os.path.exists(cbct_path):
                        self.cases.append({
                            "id": case_id,
                            "cbct": cbct_path,
                            "ios": os.path.join(self.ios_dir, fname),
                        })
                        break

        print(f"Found {len(self.cases)} cases in {data_dir}")

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case = self.cases[idx]

        # Load CBCT surface points
        nii = nib.load(case["cbct"])
        volume = nii.get_fdata().astype(np.float32)
        spacing = tuple(nii.header.get_zooms()[:3])
        target_pts = extract_surface_points(volume, num_points=self.num_points,
                                            spacing=spacing)

        # Load IOS points
        source_pts = load_mesh_points(case["ios"], num_points=self.num_points)

        # Center both point clouds
        src_center = source_pts.mean(axis=0)
        tgt_center = target_pts.mean(axis=0)
        source_pts = source_pts - src_center
        target_pts = target_pts - tgt_center

        # Normalize to unit sphere
        src_scale = np.max(np.linalg.norm(source_pts, axis=1))
        tgt_scale = np.max(np.linalg.norm(target_pts, axis=1))
        source_pts = source_pts / (src_scale + 1e-8)
        target_pts = target_pts / (tgt_scale + 1e-8)

        # Apply random augmentation during training
        gt_R = np.eye(3, dtype=np.float32)
        gt_t = np.zeros(3, dtype=np.float32)

        if self.augment:
            # Random rigid perturbation to source
            R_aug = random_rotation_matrix().astype(np.float32)
            t_aug = np.random.uniform(-0.1, 0.1, size=3).astype(np.float32)
            source_pts = (R_aug @ source_pts.T).T + t_aug

            # Ground truth: inverse of the augmentation
            gt_R = R_aug.T
            gt_t = -R_aug.T @ t_aug

        # Random point subsampling
        if len(source_pts) > self.num_points:
            idx_src = np.random.choice(len(source_pts), self.num_points, replace=False)
            source_pts = source_pts[idx_src]
        if len(target_pts) > self.num_points:
            idx_tgt = np.random.choice(len(target_pts), self.num_points, replace=False)
            target_pts = target_pts[idx_tgt]

        return (
            torch.from_numpy(source_pts),
            torch.from_numpy(target_pts),
            torch.from_numpy(gt_R),
            torch.from_numpy(gt_t),
        )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def rotation_loss(pred_R, gt_R):
    """Geodesic rotation loss: ||I - R_pred^T @ R_gt||_F."""
    R_diff = torch.bmm(pred_R.transpose(1, 2), gt_R)
    I = torch.eye(3, device=R_diff.device).unsqueeze(0).expand_as(R_diff)
    return (I - R_diff).pow(2).sum(dim=(-2, -1)).mean()


def translation_loss(pred_t, gt_t):
    """L2 translation loss."""
    return torch.norm(pred_t - gt_t, dim=-1).mean()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    dataset = RegistrationDataset(
        args.data_dir, num_points=args.num_points, augment=True
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Model
    model = CBCTIOSRegistration(
        feat_dim=args.feat_dim, num_points=args.num_points
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    os.makedirs(args.output_dir, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_rot_loss = 0.0
        epoch_trans_loss = 0.0
        epoch_cd_loss = 0.0

        for src, tgt, gt_R, gt_t in loader:
            src = src.to(device)
            tgt = tgt.to(device)
            gt_R = gt_R.to(device)
            gt_t = gt_t.to(device)

            pred_R, pred_t, aligned = model(src, tgt)

            # Multi-component loss
            l_rot = rotation_loss(pred_R, gt_R)
            l_trans = translation_loss(pred_t, gt_t)
            l_cd = chamfer_distance(aligned, tgt)

            loss = args.w_rot * l_rot + args.w_trans * l_trans + args.w_cd * l_cd

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_rot_loss += l_rot.item()
            epoch_trans_loss += l_trans.item()
            epoch_cd_loss += l_cd.item()

        scheduler.step()
        n = len(loader)
        print(
            f"Epoch {epoch}/{args.epochs}  "
            f"Loss: {epoch_loss/n:.6f}  "
            f"Rot: {epoch_rot_loss/n:.6f}  "
            f"Trans: {epoch_trans_loss/n:.6f}  "
            f"CD: {epoch_cd_loss/n:.6f}  "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )

        # Save checkpoint
        if epoch % args.save_every == 0 or epoch_loss / n < best_loss:
            ckpt_path = os.path.join(args.output_dir, f"epoch_{epoch}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": epoch_loss / n,
            }, ckpt_path)
            if epoch_loss / n < best_loss:
                best_loss = epoch_loss / n
                best_path = os.path.join(args.output_dir, "best_model.pth")
                torch.save(model.state_dict(), best_path)
                print(f"  -> New best model saved (loss={best_loss:.6f})")

    print("Training complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Task 2 CBCT-IOS registration baseline")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root dir with cbct/, ios/, and transform.json")
    parser.add_argument("--output_dir", type=str, default="./checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_points", type=int, default=4096,
                        help="Number of points per point cloud")
    parser.add_argument("--feat_dim", type=int, default=256,
                        help="PointNet feature dimension")
    parser.add_argument("--w_rot", type=float, default=1.0,
                        help="Weight for rotation loss")
    parser.add_argument("--w_trans", type=float, default=1.0,
                        help="Weight for translation loss")
    parser.add_argument("--w_cd", type=float, default=0.1,
                        help="Weight for Chamfer distance loss")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=10,
                        help="Save checkpoint every N epochs")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
