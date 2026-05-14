# STS 2026 Task 3: 3D CBCT Teeth Segmentation

STS 2026 任务 3：CBCT 三维牙齿分割

## Overview

Task 3 focuses on 3D volumetric segmentation of teeth from Cone-Beam CT (CBCT) images. Accurate teeth segmentation is essential for dental treatment planning, orthodontic analysis, and computer-aided diagnosis.

本任务聚焦于 CBCT 图像中牙齿的三维体积分割。精准的牙齿分割对口腔治疗规划、正畸分析和计算机辅助诊断至关重要。

## Competition

- **Platform**: Codabench
- **Link**: https://www.codabench.org/competitions/16117/
- **Status**: Baseline code available

## Task Description

- **Input**: CBCT volumes containing dental structures
- **Output**: Binary segmentation masks for teeth
- **Metric**: Dice Similarity Coefficient (DSC)

## How to Submit

1. Register on [Codabench](https://www.codabench.org/competitions/16117/)
2. Download the dataset from the competition page
3. Run your method on the test set
4. Upload predictions following the submission format guidelines on Codabench

## Baseline

A simple 3D UNet-based baseline is provided in the [`baseline/`](baseline/) directory.

**Approach:** 3D UNet (4 encoder/decoder levels, skip connections) trained with combined BCE + Dice loss for binary teeth segmentation from CBCT volumes.

基线采用 3D UNet 架构，使用 BCE + Dice 联合损失训练 CBCT 体数据的二值牙齿分割。

### Quick Start

```bash
# Install dependencies
pip install torch numpy nibabel

# Train
python baseline/train.py \
    --data_dir /path/to/train \
    --output_dir ./checkpoints \
    --epochs 100 --batch_size 2 --lr 1e-4

# Predict
python baseline/predict.py \
    --input_dir /path/to/test_inputs \
    --output_dir /path/to/predictions \
    --checkpoint ./checkpoints/best_model.pth
```

### Data Format

- Training data: `data_dir/images/` (CBCT volumes) and `data_dir/masks/` (binary segmentation masks), paired NIfTI files (.nii.gz)
- Input/output: 3D NIfTI volumes

See [`baseline/README.md`](baseline/README.md) for full details.

## References

- [STS Challenge GitHub](https://github.com/ricoleehduu/STS-Challenge-2026)
- [STS 2026 Official Website](https://nixy495.github.io/miccai2026/index.html)
