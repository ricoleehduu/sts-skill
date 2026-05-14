"""
UNet baseline model for CBCT metal artifact removal.

A standard 2D UNet with skip connections, designed as a simple starting
point for the STS 2026 Task 1 challenge.
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Two consecutive Conv-BN-ReLU blocks."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """
    2D UNet for image-to-image restoration.

    Args:
        in_channels: Number of input channels (default: 1 for grayscale CT).
        out_channels: Number of output channels (default: 1).
        features: List of feature counts for each encoder level.
    """

    def __init__(self, in_channels=1, out_channels=1, features=(64, 128, 256, 512)):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)

        # Encoder path
        for f in features:
            self.encoders.append(DoubleConv(in_channels, f))
            in_channels = f

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder path
        for f in reversed(features):
            self.decoders.append(
                nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
            )
            self.decoders.append(DoubleConv(f * 2, f))

        # Final 1x1 convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

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
            x = self.decoders[i](x)          # ConvTranspose2d
            skip = skip_connections[i // 2]

            # Handle size mismatch from odd dimensions
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:])

            x = torch.cat([skip, x], dim=1)
            x = self.decoders[i + 1](x)      # DoubleConv

        return self.final_conv(x)


if __name__ == "__main__":
    # Quick sanity check
    model = UNet(in_channels=1, out_channels=1)
    dummy = torch.randn(1, 1, 256, 256)
    out = model(dummy)
    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {out.shape}")
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters:   {params:.1f}M")
