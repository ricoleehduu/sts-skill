# STS 2024: Instance Segmentation of Teeth

## Overview

The STS 2024 Challenge addressed instance-level teeth segmentation on panoramic X-ray (OPG) images. Unlike semantic segmentation, instance segmentation distinguishes each individual tooth as a separate object.

STS 2024 挑战赛针对全景 X 光片（OPG）上的牙齿实例分割。与语义分割不同，实例分割需要将每颗牙齿区分为独立的个体。

## Task

- **Goal**: Instance-level teeth segmentation on OPG/X-ray images
- **Input**: Panoramic dental X-ray images
- **Output**: Instance segmentation masks (each tooth labeled with a unique ID)

## Key Methods

- **Two-Stage Detection + Segmentation**: First detect individual teeth with a detection network, then perform segmentation within each detected bounding box

## Repository

- **GitHub**: https://github.com/ricoleehduu/STS-Challenge-2024
- Reference implementation and baseline code are available in the repository

## Getting Started

1. Clone the repository
2. Follow the README for environment setup and data preparation
3. Run the baseline model using the provided scripts

## References

- [STS Challenge GitHub (2024)](https://github.com/ricoleehduu/STS-Challenge-2024)
- [STS Challenge GitHub (Main)](https://github.com/ricoleehduu/STS-Challenge-2026)
