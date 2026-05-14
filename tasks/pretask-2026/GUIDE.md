# Pre-Task 2026: 2D Image Segmentation Guide

This guide walks you through the complete Pre-Task workflow. Expected time: ~1 hour (including training).

## Overview

Pre-Task is a prerequisite for STS 2026. It's a simple 2D image segmentation task using UNet. Upon completion, you'll be eligible to access the main competition dataset.

**Competition page:** https://www.codabench.org/competitions/16040/

**Contact:** SemiTeethSegChallenge@outlook.com

---

## Step 1: Environment Setup

### Prerequisites
- Python 3.8+
- PyTorch 1.13+
- CUDA (recommended, CPU works but very slow)

### Check your environment

```bash
python --version
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU')"
```

### Install dependencies

```bash
pip install -r tasks/pretask-2026/requirements.txt
```

If you don't have CUDA, install CPU PyTorch:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

## Step 2: Download Data

### Option A: Auto-download (recommended)

```bash
python scripts/download_data.py --task pretask-2026 --output ./data
```

This will try Huggingface, Modelscope, Google Drive in order.

### Option B: Manual download

| Source | Link |
|--------|------|
| Huggingface | https://huggingface.co/datasets/Ricoooo/MICCAI-STS26-Challenge-Pre-Task |
| Modelscope | https://modelscope.cn/datasets/lizhii/MICCAI-STS26-Challenge-Pre-Task |
| Baidu Netdisk | https://pan.baidu.com/s/1U090bZnMGEJQaD3jwqaQuA (code: bm2u) |
| Google Drive | https://drive.google.com/drive/folders/1lER9eIavr99g28aTO0kuxIcos_k9FBSx |

After download, extract to get this structure:
```
data/
├── imgs/          # Training images
├── masks/         # Training masks
└── test/
    └── imgs/      # Test images (50 samples)
```

### Verify data

```bash
python -c "
from pathlib import Path
imgs = list(Path('data/imgs').glob('*.png'))
masks = list(Path('data/masks').glob('*.png'))
test = list(Path('data/test/imgs').glob('*.png'))
print(f'Training images: {len(imgs)}')
print(f'Training masks:  {len(masks)}')
print(f'Test images:     {len(test)}')
assert len(imgs) == len(masks), 'Image/mask count mismatch!'
assert len(test) == 50, f'Expected 50 test images, got {len(test)}'
print('✅ Data looks good!')
"
```

---

## Step 3: Train Model

### Basic training (recommended)

```bash
python tasks/pretask-2026/train.py --amp --data-dir ./data
```

This trains UNet for 50 epochs with mixed precision. Takes ~30-60 min on a modern GPU.

### Custom training options

```bash
# More epochs
python tasks/pretask-2026/train.py --amp --data-dir ./data --epochs 100

# Larger batch size (if GPU memory allows)
python tasks/pretask-2026/train.py --amp --data-dir ./data --batch-size 20

# Different learning rate
python tasks/pretask-2026/train.py --amp --data-dir ./data --learning-rate 1e-4

# CPU training (slow, not recommended)
python tasks/pretask-2026/train.py --data-dir ./data --epochs 10
```

### Training output

Checkpoints are saved to `./checkpoints/` after each epoch:
```
checkpoints/
├── checkpoint_epoch1.pth
├── checkpoint_epoch2.pth
├── ...
└── checkpoint_epoch50.pth
```

### Monitor training

The training shows loss and validation Dice score. Look for:
- Loss decreasing over epochs
- Validation Dice score increasing
- No sudden spikes (indicates learning rate too high)

---

## Step 4: Evaluate Locally (Optional)

If you have ground truth masks for validation, evaluate locally before submitting:

```bash
# First, run inference on validation set
python tasks/pretask-2026/predict.py \
    -m checkpoints/checkpoint_epoch50.pth \
    -i data/imgs \
    -o data/pred_masks

# Then evaluate
python scripts/evaluate.py --pred data/pred_masks --gt data/masks --verbose
```

This gives you `mean_dice` and `std_dice` matching the platform's scoring.

---

## Step 5: Run Inference on Test Set

```bash
python tasks/pretask-2026/predict.py \
    -m checkpoints/checkpoint_epoch50.pth \
    -i data/test/imgs \
    -o data/test/masks
```

### Options

```bash
# Adjust threshold (if predictions are too conservative)
python tasks/pretask-2026/predict.py \
    -m checkpoints/checkpoint_epoch50.pth \
    -i data/test/imgs \
    -o data/test/masks \
    -t 0.3

# Use bilinear upsampling (slightly different quality)
python tasks/pretask-2026/predict.py \
    -m checkpoints/checkpoint_epoch50.pth \
    -i data/test/imgs \
    -o data/test/masks \
    --bilinear
```

### Verify output

```bash
python -c "
from pathlib import Path
masks = list(Path('data/test/masks').glob('*.png'))
print(f'Generated {len(masks)} mask files')
assert len(masks) == 50, f'Expected 50 masks, got {len(masks)}'
print('✅ Ready to submit!')
"
```

---

## Step 6: Package and Submit

### Create submission zip

```bash
python scripts/prepare_submit.py --masks data/test/masks --task pretask-2026
```

This creates `task_pre_validation_data.zip`.

### Submit to Codabench

1. Go to https://www.codabench.org/competitions/16040/
2. Click "Submit" in the Validation phase
3. Upload `task_pre_validation_data.zip`
4. Wait for scoring (usually < 1 minute)

### Check your score

The leaderboard shows `mean_dice` and `std_dice`. A good score is typically > 0.8 mean_dice.

---

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
python tasks/pretask-2026/train.py --amp --data-dir ./data --batch-size 5

# Or use gradient checkpointing (code modification needed)
```

### Training loss not decreasing

- Check data loading: `python -c "from tasks.pretask_2026.utils.data_loading import *; print('OK')"`
- Reduce learning rate: `--learning-rate 1e-6`
- Check masks are correct (visualize a few)

### Poor Dice score

- Train for more epochs
- Try different threshold: `-t 0.3` or `-t 0.7`
- Ensure data is correctly loaded (image-mask pairs match)

### Submission failed

- Check zip structure: should contain `data/test/masks/*.png`
- Verify PNG format (not JPEG or other)
- Check file count (should be ~50)

---

## Next Steps

After completing Pre-Task, you're eligible for the main STS 2026 dataset!

- **Task 1:** Metal artifact removal → `tasks/task1-2026/GUIDE.md`
- **Task 2:** → `tasks/task2-2026/GUIDE.md`
- **Task 3:** → `tasks/task3-2026/GUIDE.md`

---

# Pre-Task 2026 完整流程指南

本指南将带你完成 Pre-Task 的全部流程。预计时间：约 1 小时（含训练）。

## 概述

Pre-Task 是 STS 2026 的前置任务，使用 UNet 进行 2D 图像分割。完成后将有资格获取正赛数据集。

**竞赛页面：** https://www.codabench.org/competitions/16040/

**联系方式：** SemiTeethSegChallenge@outlook.com

## 流程概览

1. 环境检查 → Python/PyTorch/CUDA
2. 下载数据 → `python scripts/download_data.py --task pretask-2026`
3. 训练模型 → `python tasks/pretask-2026/train.py --amp --data-dir ./data`
4. 本地评估 → `python scripts/evaluate.py --pred <pred> --gt <gt>`
5. 推理测试 → `python tasks/pretask-2026/predict.py -m checkpoints/checkpoint_epoch50.pth -i data/test/imgs -o data/test/masks`
6. 打包提交 → `python scripts/prepare_submit.py --masks data/test/masks --task pretask-2026`

详细步骤请参考上方英文版指南。
