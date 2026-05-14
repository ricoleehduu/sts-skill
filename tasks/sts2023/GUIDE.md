# STS 2023: Semi-supervised 2D Teeth Segmentation

## Overview

The STS 2023 Challenge focused on semi-supervised 2D teeth segmentation on panoramic X-ray (OPG/PXI) images. Semi-supervised learning leverages both labeled and unlabeled data, which is critical given the high cost of medical image annotation.

STS 2023 挑战赛聚焦于全景 X 光片（OPG/PXI）上的半监督二维牙齿分割。半监督学习利用标注和未标注数据，在医学影像标注成本高昂的背景下具有重要价值。

## Task

- **Goal**: Semi-supervised 2D teeth segmentation on OPG/PXI images
- **Input**: Panoramic dental X-ray images (with limited labels)
- **Output**: 2D segmentation masks for teeth

## Key Methods

- **Dual Teacher**: Dual teacher-student framework for robust semi-supervised learning
- **Mean Teacher**: Exponential moving average teacher model for stable pseudo-label generation

## Repository

- **GitHub**: https://github.com/ricoleehduu/STS-Challenge
- Reference implementation and baseline code are available in the repository

## Getting Started

1. Clone the repository
2. Follow the README for environment setup and data preparation
3. Run the baseline model using the provided scripts

## References

- [STS Challenge GitHub (2023)](https://github.com/ricoleehduu/STS-Challenge)
- [STS Challenge GitHub (Main)](https://github.com/ricoleehduu/STS-Challenge-2026)
