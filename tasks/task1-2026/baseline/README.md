# Task 1 Baseline: CBCT Segmentation under Metal Artifacts

A simple UNet-based baseline for teeth segmentation from metal-artifact-corrupted CBCT images.

## Approach

This baseline uses a 2D UNet architecture to learn a mapping from metal-artifact-corrupted CBCT slices to binary teeth segmentation masks. The model processes each slice independently.

**Key details:**
- Architecture: UNet with 4 encoder/decoder levels and skip connections
- Input/Output: Single-channel 2D slices (256x256 by default)
- Loss: BCE + Dice loss
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
    images/   # CBCT volumes with metal artifacts (.nii.gz)
    masks/    # Binary teeth segmentation masks (.nii.gz)
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
- No domain-specific preprocessing (e.g., metal mask handling)
- Small receptive field may struggle with large artifact regions
