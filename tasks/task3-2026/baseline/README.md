# Task 3 Baseline: 3D CBCT Teeth Segmentation

A simple 3D UNet baseline for volumetric teeth segmentation from CBCT images.

## Approach

This baseline uses a 3D UNet architecture to segment teeth from Cone-Beam CT (CBCT) volumes. The model processes 3D patches and uses sliding window inference with overlap for whole-volume prediction.

**Key details:**
- Architecture: 3D UNet with 4 encoder/decoder levels and skip connections
- Input/Output: Single-channel 3D patches (64x64x64 by default)
- Loss: Combined BCE + Dice loss
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
    --batch_size 2 \
    --lr 1e-4
```

Expected data structure:
```
data_dir/
    images/   # CBCT volumes (.nii.gz)
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

- Processes 3D patches; limited global context
- No domain-specific preprocessing (e.g., HU windowing, registration)
- Binary segmentation only; does not distinguish individual teeth
- Small patch size may miss large-scale anatomical structures
