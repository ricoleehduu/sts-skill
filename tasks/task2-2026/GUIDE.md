# STS 2026 Task 2

STS 2026 任务 2

## Overview

Task 2 of the STS 2026 Challenge. Details will be updated as the competition progresses.

STS 2026 挑战赛任务 2。详细信息将随竞赛推进逐步更新。

## Competition

- **Platform**: Codabench
- **Link**: https://www.codabench.org/competitions/16042/
- **Status**: Baseline code available

## How to Submit

1. Register on [Codabench](https://www.codabench.org/competitions/16042/)
2. Download the dataset from the competition page
3. Run your method on the test set
4. Upload predictions following the submission format guidelines on Codabench

## Baseline

A PointNet-based baseline is available in the [`baseline/`](baseline/) directory.

### Quick Start

```bash
# Install dependencies
pip install torch numpy nibabel trimesh scikit-image scipy

# Training
python baseline/train.py \
    --data_dir /path/to/train \
    --output_dir ./checkpoints \
    --epochs 100 --batch_size 4

# Inference
python baseline/predict.py \
    --input_dir /path/to/test \
    --output_dir /path/to/predictions \
    --checkpoint ./checkpoints/best_model.pth
```

See [`baseline/README.md`](baseline/README.md) for full documentation.

### Baseline Approach

- **Architecture**: PointNet encoder + rigid transform regression head
- **Input**: IOS point cloud + CBCT surface point cloud (4096 points each)
- **Output**: 6-DoF rigid transformation (rotation + translation)
- **Loss**: Chamfer distance + rotation geodesic loss + translation L2 loss

### Data Format

```
data_dir/
    cbct/               # CBCT volumes (.nii.gz)
    ios/                # IOS meshes (.stl, .ply, .obj)
    transform.json      # Ground-truth transforms (training only)
```

基线代码已在 [`baseline/`](baseline/) 目录下发布。详见 [`baseline/README.md`](baseline/README.md)。

## References

- [STS Challenge GitHub](https://github.com/ricoleehduu/STS-Challenge-2026)
- [STS 2026 Official Website](https://nixy495.github.io/miccai2026/index.html)
