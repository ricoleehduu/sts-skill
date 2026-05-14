# Task 2 Baseline: CBCT-IOS Registration

A PointNet-based baseline for aligning intraoral scan (IOS) crown surfaces with CBCT volumes.

## Approach

This baseline uses a PointNet-style encoder to extract global features from both point clouds, then predicts a 6-DoF rigid transformation (3 rotation angles + 3 translation) to align the IOS to the CBCT-derived tooth surface.

**Key details:**
- Architecture: PointNet encoder + regression head
- Input: Two point clouds (IOS source, CBCT target), each with 4096 points
- Output: Rigid transformation (R, t)
- Loss: Weighted combination of rotation loss, translation loss, and Chamfer distance
- Refinement: ICP-based post-processing (optional)
- Framework: PyTorch + trimesh

## Requirements

```
torch>=1.10
numpy
nibabel
trimesh
scikit-image
scipy
```

## Data Layout

```
data_dir/
    cbct/               # CBCT volumes (.nii.gz)
    ios/                # IOS meshes (.stl, .ply, or .obj)
    transform.json      # Ground-truth transforms (for training)
```

**transform.json format:**
```json
{
    "case_001": {
        "rotation": [[1,0,0],[0,1,0],[0,0,1]],
        "translation": [0.0, 0.0, 0.0]
    }
}
```

## Usage

### Training

```bash
python train.py \
    --data_dir /path/to/train \
    --output_dir ./checkpoints \
    --epochs 100 \
    --batch_size 4 \
    --lr 1e-4 \
    --num_points 4096
```

### Inference

```bash
python predict.py \
    --input_dir /path/to/test \
    --output_dir /path/to/predictions \
    --checkpoint ./checkpoints/best_model.pth
```

Output will contain:
- `<case_id>_aligned.ply` -- Transformed IOS meshes aligned to CBCT
- `transforms.json` -- Predicted rotation and translation per case

## Limitations

- Predicts global rigid transformation only; no non-rigid / local deformation
- PointNet encoder has limited capacity for capturing fine geometric details
- No iterative refinement built into the network (consider adding ICP post-processing)
- Normalization assumes both point clouds are centered and scaled similarly
- Random augmentation uses small rotation range (+-30 degrees); may not cover all cases

## Tips for Participants

1. **ICP Refinement**: Use the predicted transform as initialization for ICP to improve accuracy
2. **Better Architectures**: Consider DCP, PRNet, or GeoTransformer for improved feature matching
3. **Multi-scale Features**: Use hierarchical or attention-based point cloud encoders
4. **Non-rigid Registration**: Dental scans may require non-rigid alignment for patients with soft tissue differences
5. **Tooth-level Registration**: Register individual teeth rather than full arches for finer control
