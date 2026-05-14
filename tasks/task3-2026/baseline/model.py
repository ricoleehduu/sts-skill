"""
UNet baseline model for 3D CBCT teeth segmentation.

A standard 3D UNet with skip connections, designed as a simple starting
point for the STS 2026 Task 3 challenge.
"""

import torch
import torch.nn as nn


class DoubleConv3D(nn.Module):
    """Two consecutive Conv3D-BN-ReLU blocks."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet3D(nn.Module):
    """
    3D UNet for volumetric segmentation.

    Args:
        in_channels: Number of input channels (default: 1 for grayscale CBCT).
        out_channels: Number of output channels (default: 1 for binary segmentation).
        features: List of feature counts for each encoder level.
    """

    def __init__(self, in_channels=1, out_channels=1, features=(32, 64, 128, 256)):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.pool = nn.MaxPool3d(2)

        # Encoder path
        for f in features:
            self.encoders.append(DoubleConv3D(in_channels, f))
            in_channels = f

        # Bottleneck
        self.bottleneck = DoubleConv3D(features[-1], features[-1] * 2)

        # Decoder path
        for f in reversed(features):
            self.decoders.append(
                nn.ConvTranspose3d(f * 2, f, kernel_size=2, stride=2)
            )
            self.decoders.append(DoubleConv3D(f * 2, f))

        # Final 1x1x1 convolution
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder
        for encoder in self.encoders:
            x = encoder(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        skip_connections = skip_connections[::-1]
        for i in range(0, len(self.decoders), 2):
            x = self.decoders[i](x)          # ConvTranspose3d
            skip = skip_connections[i // 2]

            # Handle size mismatch from odd dimensions
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:])

            x = torch.cat([skip, x], dim=1)
            x = self.decoders[i + 1](x)      # DoubleConv3D

        return self.final_conv(x)


if __name__ == "__main__":
    # Quick sanity check
    model = UNet3D(in_channels=1, out_channels=1)
    dummy = torch.randn(1, 1, 64, 64, 64)
    out = model(dummy)
    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {out.shape}")
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters:   {params:.1f}M")
