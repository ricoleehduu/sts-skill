# STS 2026 Task 1: Metal Artifact Removal in CBCT

STS 2026 任务 1：CBCT 图像金属伪影去除

## Overview

Task 1 focuses on removing metal artifacts from Cone-Beam CT (CBCT) dental images. Metal artifacts caused by dental implants, fillings, and orthodontic hardware degrade image quality and interfere with downstream diagnosis and analysis.

本任务聚焦于去除口腔 CBCT 图像中的金属伪影。种植体、充填物和正畸装置等金属物体会产生伪影，降低图像质量，干扰后续诊断与分析。

## Competition

- **Platform**: Codabench
- **Link**: https://www.codabench.org/competitions/16027/
- **Status**: Baseline code available

## Task Description

- **Input**: CBCT volumes containing metal artifacts
- **Output**: Artifact-corrected CBCT volumes
- **Metric**: To be announced

## How to Submit

1. Register on [Codabench](https://www.codabench.org/competitions/16027/)
2. Download the dataset from the competition page
3. Run your method on the test set
4. Upload predictions following the submission format guidelines on Codabench

## Baseline

A simple UNet-based baseline is provided in the [`baseline/`](baseline/) directory.

**Approach:** 2D UNet (4 encoder/decoder levels, skip connections) trained with L1 loss to map artifact-corrupted slices to clean slices.

基线采用 2D UNet 架构，使用 L1 损失训练从含伪影切片到干净切片的映射。

### Quick Start

```bash
# Install dependencies
pip install torch numpy nibabel

# Train
python baseline/train.py \
    --data_dir /path/to/train \
    --output_dir ./checkpoints \
    --epochs 100 --batch_size 4 --lr 1e-4

# Predict
python baseline/predict.py \
    --input_dir /path/to/test_inputs \
    --output_dir /path/to/predictions \
    --checkpoint ./checkpoints/best_model.pth
```

### Data Format

- Training data: `data_dir/input/` (artifact volumes) and `data_dir/target/` (clean volumes), paired NIfTI files (.nii.gz)
- Input/output: 3D NIfTI volumes, processed slice by slice along the axial axis

See [`baseline/README.md`](baseline/README.md) for full details.

## References

- [STS Challenge GitHub](https://github.com/ricoleehduu/STS-Challenge-2026)
- [STS 2026 Official Website](https://nixy495.github.io/miccai2026/index.html)
