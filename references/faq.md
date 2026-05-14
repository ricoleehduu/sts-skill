# Frequently Asked Questions (FAQ)

Common questions from STS Challenge participants.

---

## How do I install dependencies?

We provide a `requirements.txt` (or `environment.yml`) in the starter kit. Typical setup:

```bash
# Create a virtual environment (recommended)
python -m venv sts-env
source sts-env/bin/activate        # Linux/Mac
# sts-env\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

Core dependencies usually include: `torch`, `numpy`, `nibabel` (for NIfTI), `SimpleITK`, `monai` or `nnunet`, `albumentations`, `scikit-learn`.

If you use **nnUNet**, follow its dedicated installation guide: https://github.com/MIC-DKFZ/nnUNet

---

## What if CUDA is not available?

CUDA (GPU) is strongly recommended but not strictly required. If `torch.cuda.is_available()` returns `False`:

- **CPU training works** but is extremely slow -- expect 10-50x longer training times depending on the model and input size.
- Reduce batch size to 1 and use smaller patch sizes to fit in CPU memory.
- Consider using Google Colab or Kaggle for free GPU access if you do not have a local GPU.

The code will automatically fall back to CPU if no GPU is detected.

---

## How do I improve my Dice score?

Common strategies ranked by typical impact:

1. **Train longer.** Many participants undertrain. For UNet on dental data, 200-300 epochs is a reasonable range. Monitor validation Dice -- it often continues to improve after the training loss plateaus.
2. **Use data augmentation.** Random rotations, flips, elastic deformations, and brightness/contrast jitter. See [algorithms.md](algorithms.md) for the recommended augmentation pipeline.
3. **Tune the learning rate.** Try LR = {1e-3, 5e-4, 1e-4, 5e-5} with cosine annealing. Too high causes divergence, too low causes slow convergence.
4. **Use a better loss function.** Switch from pure Cross Entropy to Combo Loss (Dice + CE) or Focal Dice Loss.
5. **Ensemble predictions.** Train 3-5 models with different seeds and average their predictions. This consistently boosts Dice by 1-3%.
6. **Test-time augmentation (TTA).** Apply horizontal/vertical flips at inference and average the outputs. Simple and effective.

---

## How do I handle OOM (Out of Memory) errors?

GPU memory issues are common with 3D CBCT volumes. Solutions in order of ease:

1. **Reduce batch size.** The most direct fix. Batch size of 1 or 2 is normal for 3D segmentation.
2. **Use AMP (mixed precision).** Halves memory usage with minimal quality loss. See [algorithms.md](algorithms.md).
3. **Reduce patch/crop size.** Smaller input patches require less memory. For CBCT, try 96x96x96 instead of 128x128x128.
4. **Use gradient checkpointing.** Trades compute for memory by recomputing activations during backward pass:
   ```python
   from torch.utils.checkpoint import checkpoint
   ```
5. **Use gradient accumulation.** Simulate larger batch sizes by accumulating gradients over multiple forward passes:
   ```python
   loss = loss / accumulation_steps
   loss.backward()
   if (step + 1) % accumulation_steps == 0:
       optimizer.step()
       optimizer.zero_grad()
   ```
6. **Free unused memory explicitly:**
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

---

## What is the submission format?

Submission format for each task:

- **Pre-Task (Artifact Removal):** Submit a ZIP file containing the denoised CBCT volumes in NIfTI format (`.nii.gz`). File names must match the original input file names.
- **Task 1/2/3 (Segmentation):** Submit a ZIP file containing predicted segmentation masks. Each mask must be in the same format and naming convention as the provided ground truth.

Detailed format specifications are provided on each task's Codabench page. Read the "Evaluation" and "Submission" tabs carefully before submitting.

**Important:**
- Ensure the predicted masks have the **same spatial dimensions and orientation** as the input images.
- Do not include extra files or folders in the ZIP.
- Double-check that class labels match the provided label map (e.g., 0 = background, 1-32 = tooth classes).

---

## How do I contact the organizers?

- **Email:** SemiTeethSegChallenge@outlook.com
- **GitHub Issues:** https://github.com/ricoleehduu/STS-Challenge-2026/issues

For technical questions about the dataset, evaluation metrics, or submission format, please check the GitHub repository and competition page first. Many common questions are already answered in the documentation.

For partnership or workshop inquiries, email us directly.

---

## What evaluation metrics are used?

The primary metric is **Dice Similarity Coefficient (DSC)**, averaged across all tooth classes. Secondary metrics typically include:

- **HD95:** 95th percentile Hausdorff Distance (boundary accuracy)
- **IoU:** Intersection over Union
- **Sensitivity / Specificity**

Check each task's Codabench evaluation page for the exact metric definitions and weighting.

---

## Can I use external data?

External pre-trained weights (e.g., ImageNet, medical imaging pre-trained models) are generally allowed. External **labeled dental datasets** for training are **not** allowed unless explicitly stated per task.

Check the specific task rules on Codabench or the competition website for the definitive policy.

---

## Can I use nnUNet?

Yes. nnUNet is an excellent baseline for medical image segmentation. It automatically configures the architecture, training scheme, and augmentation pipeline for your dataset.

Setup:
```bash
pip install nnunetv2
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
nnUNetv2_train DATASET_ID 3d_fullres 0 --npz
```

Many top-ranked teams in previous STS editions used nnUNet as their foundation and built custom modifications on top.
