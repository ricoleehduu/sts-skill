"""
Inference script for the Task 2 CBCT-IOS registration baseline.

Loads a trained registration model, processes each test case (CBCT + IOS pair),
predicts the rigid transformation, and saves the aligned IOS mesh.

Usage:
    python predict.py \
        --input_dir /path/to/test \
        --output_dir /path/to/predictions \
        --checkpoint ./checkpoints/best_model.pth

Expected input layout:
    input_dir/
        cbct/    # CBCT volumes (.nii.gz)
        ios/     # IOS meshes (.stl or .ply)

Output:
    output_dir/
        <case_id>_aligned.ply   # Transformed IOS mesh
        transforms.json         # Predicted transforms for all cases
"""

import argparse
import json
import os

import nibabel as nib
import numpy as np
import torch

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

from model import CBCTIOSRegistration


# ---------------------------------------------------------------------------
# Geometry utilities (same as train.py)
# ---------------------------------------------------------------------------

def extract_surface_points(volume, iso_value=0.5, num_points=4096, spacing=(1, 1, 1)):
    """Extract surface point cloud from a binary/label volume."""
    try:
        from skimage.measure import marching_cubes
        verts, _, _, _ = marching_cubes(volume, level=iso_value, spacing=spacing)
        if len(verts) > num_points:
            idx = np.random.choice(len(verts), num_points, replace=False)
            verts = verts[idx]
        elif len(verts) < num_points:
            idx = np.random.choice(len(verts), num_points, replace=True)
            verts = verts[idx]
        return verts.astype(np.float32)
    except Exception:
        from scipy.ndimage import binary_erosion
        binary = volume > iso_value
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
    """Load point cloud from a mesh file."""
    if not HAS_TRIMESH:
        raise ImportError("trimesh is required. Install with: pip install trimesh")
    mesh = trimesh.load(mesh_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    return points.astype(np.float32), mesh


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_transform(model, source_pts, target_pts, device):
    """Predict rigid transformation from source (IOS) to target (CBCT).

    Args:
        model: trained registration model.
        source_pts: (N, 3) IOS points.
        target_pts: (M, 3) CBCT points.
        device: torch device.

    Returns:
        R: (3, 3) rotation matrix (numpy).
        t: (3,) translation vector (numpy).
    """
    model.eval()

    # Center and normalize (same preprocessing as training)
    src_center = source_pts.mean(axis=0)
    tgt_center = target_pts.mean(axis=0)
    src_centered = source_pts - src_center
    tgt_centered = target_pts - tgt_center

    src_scale = np.max(np.linalg.norm(src_centered, axis=1))
    tgt_scale = np.max(np.linalg.norm(tgt_centered, axis=1))
    src_norm = src_centered / (src_scale + 1e-8)
    tgt_norm = tgt_centered / (tgt_scale + 1e-8)

    # To tensors
    src_tensor = torch.from_numpy(src_norm).unsqueeze(0).to(device)
    tgt_tensor = torch.from_numpy(tgt_norm).unsqueeze(0).to(device)

    with torch.no_grad():
        R, t, _ = model(src_tensor, tgt_tensor)

    R = R.squeeze(0).cpu().numpy()
    t = t.squeeze(0).cpu().numpy()

    # Undo normalization: the predicted R, t are in normalized space.
    # Transform back to original space:
    #   x_aligned = R @ (x - src_center)/src_scale + t
    #   In original space:
    #   x_aligned_orig = src_scale * (R @ (x - src_center)/src_scale + t) + tgt_center
    #                  = R @ (x - src_center) + src_scale * t + tgt_center

    R_orig = R
    t_orig = src_scale * t + tgt_center - R @ src_center

    return R_orig, t_orig


def apply_transform_to_mesh(mesh, R, t):
    """Apply rigid transformation to a trimesh mesh.

    Args:
        mesh: trimesh.Trimesh object.
        R: (3, 3) rotation matrix.
        t: (3,) translation vector.

    Returns:
        transformed_mesh: new trimesh.Trimesh.
    """
    new_mesh = mesh.copy()
    new_mesh.vertices = (R @ mesh.vertices.T).T + t
    return new_mesh


def main():
    parser = argparse.ArgumentParser(
        description="Predict with Task 2 CBCT-IOS registration baseline"
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory with cbct/ and ios/ subfolders")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save aligned meshes and transforms")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--num_points", type=int, default=4096,
                        help="Number of points per point cloud")
    parser.add_argument("--feat_dim", type=int, default=256,
                        help="PointNet feature dimension")
    args = parser.parse_args()

    if not HAS_TRIMESH:
        print("Error: trimesh is required for inference. Install with: pip install trimesh")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = CBCTIOSRegistration(
        feat_dim=args.feat_dim, num_points=args.num_points
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Discover test cases
    cbct_dir = os.path.join(args.input_dir, "cbct")
    ios_dir = os.path.join(args.input_dir, "ios")

    if not os.path.isdir(cbct_dir) or not os.path.isdir(ios_dir):
        print(f"Error: Expected cbct/ and ios/ subfolders in {args.input_dir}")
        return

    all_transforms = {}

    ios_files = sorted([
        f for f in os.listdir(ios_dir)
        if f.endswith((".stl", ".ply", ".obj"))
    ])

    if not ios_files:
        print(f"No mesh files found in {ios_dir}")
        return

    for fname in ios_files:
        case_id = os.path.splitext(fname)[0]
        print(f"\nProcessing: {case_id}")

        # Find matching CBCT volume
        cbct_path = None
        for ext in (".nii.gz", ".nii"):
            candidate = os.path.join(cbct_dir, case_id + ext)
            if os.path.exists(candidate):
                cbct_path = candidate
                break

        if cbct_path is None:
            print(f"  Warning: No matching CBCT for {case_id}, skipping.")
            continue

        ios_path = os.path.join(ios_dir, fname)

        # Load data
        nii = nib.load(cbct_path)
        volume = nii.get_fdata().astype(np.float32)
        spacing = tuple(nii.header.get_zooms()[:3])
        target_pts = extract_surface_points(volume, num_points=args.num_points,
                                            spacing=spacing)
        source_pts, ios_mesh = load_mesh_points(ios_path, num_points=args.num_points)

        # Predict transformation
        R, t = predict_transform(model, source_pts, target_pts, device)

        # Apply to mesh
        aligned_mesh = apply_transform_to_mesh(ios_mesh, R, t)

        # Save aligned mesh
        out_name = f"{case_id}_aligned.ply"
        out_path = os.path.join(args.output_dir, out_name)
        aligned_mesh.export(out_path)
        print(f"  Saved: {out_path}")

        # Store transform
        all_transforms[case_id] = {
            "rotation": R.tolist(),
            "translation": t.tolist(),
        }

    # Save all transforms
    transforms_path = os.path.join(args.output_dir, "transforms.json")
    with open(transforms_path, "w") as f:
        json.dump(all_transforms, f, indent=2)
    print(f"\nTransforms saved to: {transforms_path}")
    print("Inference complete.")


if __name__ == "__main__":
    main()
