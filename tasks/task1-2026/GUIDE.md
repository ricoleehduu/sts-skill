# STS 2026 Task 1: CBCT Segmentation under Metal Artifacts

STS 2026 任务 1：金属伪影影响下的 CBCT 分割

## Overview

Task 1 focuses on teeth segmentation from Cone-Beam CT (CBCT) images degraded by metal artifacts. Dental implants, fillings, and orthodontic hardware introduce severe artifacts that challenge conventional segmentation methods.

本任务聚焦于在金属伪影干扰下的 CBCT 图像牙齿分割。种植体、充填物和正畸装置等金属物体会产生严重伪影，对传统分割方法构成挑战。

## Competition

- **Platform**: Codabench
- **Link**: https://www.codabench.org/competitions/16027/
- **Status**: Baseline code available

## Task Description

- **Input**: CBCT volumes containing metal artifacts
- **Output**: Binary segmentation masks for teeth
- **Metric**: Dice Similarity Coefficient (DSC)

## How to Submit

1. Register on [Codabench](https://www.codabench.org/competitions/16027/)
2. Download the dataset from the competition page
3. Run your method on the test set
4. Upload predictions following the submission format guidelines on Codabench

## Baseline

A simple UNet-based baseline is provided in the [`baseline/`](baseline/) directory.

**Approach:** 2D UNet (4 encoder/decoder levels, skip connections) trained with BCE + Dice loss for binary teeth segmentation from artifact-corrupted CBCT slices.

基线采用 2D UNet 架构，使用 BCE + Dice 联合损失训练从含伪影 CBCT 切片到牙齿分割 mask 的映射。

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

- Training data: `data_dir/images/` (CBCT volumes with metal artifacts) and `data_dir/masks/` (binary segmentation masks), paired NIfTI files (.nii.gz)
- Input/output: 3D NIfTI volumes, processed slice by slice along the axial axis

See [`baseline/README.md`](baseline/README.md) for full details.

## References

- [STS Challenge GitHub](https://github.com/ricoleehduu/STS-Challenge-2026)
- [STS 2026 Official Website](https://nixy495.github.io/miccai2026/index.html)
