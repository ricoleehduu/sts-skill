# Task 1 Baseline: Metal Artifact Removal

A simple UNet-based baseline for CBCT metal artifact removal.

## Approach

This baseline uses a 2D UNet architecture to learn a mapping from metal-artifact-corrupted CBCT slices to clean CBCT slices. The model processes each slice independently.

**Key details:**
- Architecture: UNet with 4 encoder/decoder levels and skip connections
- Input/Output: Single-channel 2D slices (256x256 by default)
- Loss: L1 loss
- Framework: PyTorch

## Requirements

```
torch>=1.10
numpy
nibabel
```

## Usage

### Training

```bash
python train.py \
    --data_dir /path/to/train \
    --output_dir ./checkpoints \
    --epochs 100 \
    --batch_size 4 \
    --lr 1e-4
```

Expected data structure:
```
data_dir/
    input/    # CBCT volumes with metal artifacts (.nii.gz)
    target/   # Clean CBCT volumes (.nii.gz)
```

### Inference

```bash
python predict.py \
    --input_dir /path/to/test_inputs \
    --output_dir /path/to/predictions \
    --checkpoint ./checkpoints/best_model.pth
```

## Limitations

- Processes 2D slices independently; no 3D context
- No domain-specific preprocessing (e.g., metal mask, sinogram correction)
- Small receptive field may miss large-scale artifacts
