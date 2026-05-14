"""
PointNet-based rigid registration model for CBCT-IOS alignment.

Predicts a 6-DoF rigid transformation (3 rotation angles + 3 translation)
that aligns an IOS point cloud to a CBCT-derived tooth surface point cloud.

The architecture:
  1. Shared PointNet encoder extracts global features from both point clouds.
  2. Concatenated features are passed through an MLP to predict (rotation, translation).
  3. A differentiable SVD layer converts the predicted 3x3 matrix to a proper rotation.

Reference:
  - PointNet (Qi et al., CVPR 2017)
  - PRNet (Wang & Solomon, ICCV 2019)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetEncoder(nn.Module):
    """Shared PointNet feature encoder.

    Input:  (B, N, 3) point cloud
    Output: (B, feat_dim) global feature vector
    """

    def __init__(self, feat_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, feat_dim, 1),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
        )
        self.feat_dim = feat_dim

    def forward(self, x):
        """x: (B, N, 3) -> (B, feat_dim)"""
        x = x.transpose(1, 2)           # (B, 3, N)
        x = self.mlp(x)                 # (B, feat_dim, N)
        x = torch.max(x, dim=2)[0]      # (B, feat_dim)
        return x


class RegistrationHead(nn.Module):
    """MLP that predicts rotation (Euler angles) and translation from features."""

    def __init__(self, feat_dim=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feat_dim * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.fc_rot = nn.Linear(256, 3)    # Euler angles (radians)
        self.fc_trans = nn.Linear(256, 3)  # Translation vector

        # Initialize to predict identity transform
        nn.init.zeros_(self.fc_rot.weight)
        nn.init.zeros_(self.fc_rot.bias)
        nn.init.zeros_(self.fc_trans.weight)
        nn.init.zeros_(self.fc_trans.bias)

    def forward(self, feat_src, feat_tgt):
        """Predict transformation from source to target features.

        Args:
            feat_src: (B, feat_dim) source (IOS) features
            feat_tgt: (B, feat_dim) target (CBCT) features

        Returns:
            rotation: (B, 3, 3) rotation matrix
            translation: (B, 3) translation vector
        """
        feat = torch.cat([feat_src, feat_tgt], dim=-1)  # (B, feat_dim*2)
        feat = self.fc(feat)

        # Euler angles -> rotation matrix
        angles = self.fc_rot(feat)  # (B, 3)
        rotation = euler_to_rotation_matrix(angles)  # (B, 3, 3)

        translation = self.fc_trans(feat)  # (B, 3)

        return rotation, translation


class CBCTIOSRegistration(nn.Module):
    """Full registration network.

    Takes two point clouds (IOS source, CBCT target) and predicts
    a rigid transformation (R, t) that aligns source to target.
    """

    def __init__(self, feat_dim=256, num_points=1024):
        super().__init__()
        self.encoder = PointNetEncoder(feat_dim=feat_dim)
        self.head = RegistrationHead(feat_dim=feat_dim)
        self.num_points = num_points

    def forward(self, source, target):
        """Forward pass.

        Args:
            source: (B, N, 3) IOS point cloud
            target: (B, M, 3) CBCT surface point cloud

        Returns:
            rotation: (B, 3, 3) rotation matrix
            translation: (B, 3) translation vector
            transformed_source: (B, N, 3) aligned source points
        """
        feat_src = self.encoder(source)  # (B, feat_dim)
        feat_tgt = self.encoder(target)  # (B, feat_dim)

        rotation, translation = self.head(feat_src, feat_tgt)

        # Apply transformation to source points
        # transformed = R @ source^T + t
        transformed = torch.bmm(
            source, rotation.transpose(1, 2)
        ) + translation.unsqueeze(1)

        return rotation, translation, transformed


def euler_to_rotation_matrix(angles):
    """Convert Euler angles (rx, ry, rz) to 3x3 rotation matrices.

    Uses the ZYX convention (Tait-Bryan angles).

    Args:
        angles: (B, 3) Euler angles in radians

    Returns:
        R: (B, 3, 3) rotation matrices
    """
    rx, ry, rz = angles[:, 0], angles[:, 1], angles[:, 2]

    cos_x, sin_x = torch.cos(rx), torch.sin(rx)
    cos_y, sin_y = torch.cos(ry), torch.sin(ry)
    cos_z, sin_z = torch.cos(rz), torch.sin(rz)

    zeros = torch.zeros_like(rx)
    ones = torch.ones_like(rx)

    # Rotation around X
    Rx = torch.stack([
        torch.stack([ones, zeros, zeros], dim=-1),
        torch.stack([zeros, cos_x, -sin_x], dim=-1),
        torch.stack([zeros, sin_x, cos_x], dim=-1),
    ], dim=1)  # (B, 3, 3)

    # Rotation around Y
    Ry = torch.stack([
        torch.stack([cos_y, zeros, sin_y], dim=-1),
        torch.stack([zeros, ones, zeros], dim=-1),
        torch.stack([-sin_y, zeros, cos_y], dim=-1),
    ], dim=1)

    # Rotation around Z
    Rz = torch.stack([
        torch.stack([cos_z, -sin_z, zeros], dim=-1),
        torch.stack([sin_z, cos_z, zeros], dim=-1),
        torch.stack([zeros, zeros, ones], dim=-1),
    ], dim=1)

    # ZYX order: R = Rz @ Ry @ Rx
    R = torch.bmm(Rz, torch.bmm(Ry, Rx))
    return R


def chamfer_distance(pc1, pc2):
    """Compute Chamfer Distance between two point clouds.

    Args:
        pc1: (B, N, 3)
        pc2: (B, M, 3)

    Returns:
        Scalar Chamfer distance (mean over batch).
    """
    # (B, N, 1, 3) - (B, 1, M, 3) -> (B, N, M)
    diff = pc1.unsqueeze(2) - pc2.unsqueeze(1)
    dist = torch.sum(diff ** 2, dim=-1)

    min_dist_1, _ = dist.min(dim=2)  # (B, N) nearest in pc2 for each pc1
    min_dist_2, _ = dist.min(dim=1)  # (B, M) nearest in pc1 for each pc2

    cd = min_dist_1.mean(dim=1) + min_dist_2.mean(dim=1)  # (B,)
    return cd.mean()


if __name__ == "__main__":
    # Sanity check
    model = CBCTIOSRegistration(feat_dim=256, num_points=1024)
    src = torch.randn(2, 1024, 3)
    tgt = torch.randn(2, 1024, 3)
    R, t, aligned = model(src, tgt)

    print(f"Source shape:       {src.shape}")
    print(f"Target shape:       {tgt.shape}")
    print(f"Rotation shape:     {R.shape}")
    print(f"Translation shape:  {t.shape}")
    print(f"Aligned shape:      {aligned.shape}")

    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters:         {params:.2f}M")

    # Check rotation matrix validity (should be close to orthogonal)
    RTR = torch.bmm(R.transpose(1, 2), R)
    I = torch.eye(3).unsqueeze(0).expand(2, -1, -1)
    ortho_err = (RTR - I).abs().max().item()
    print(f"Orthogonality error: {ortho_err:.6f}")
