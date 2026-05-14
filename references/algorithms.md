# Algorithm Reference

Quick reference on core algorithms, loss functions, and training techniques relevant to the STS Challenge.

## UNet Architecture

UNet is the de facto baseline for medical image segmentation. It follows an **encoder-decoder** structure with **skip connections** that bridge corresponding layers.

- **Encoder** (contracting path): successive Conv + ReLU + MaxPool layers that capture *what* is in the image (semantic features).
- **Decoder** (expanding path): UpConv + concatenation with encoder features that recover *where* things are (spatial detail).
- **Skip connections**: concatenate encoder feature maps with decoder feature maps at each resolution, preserving fine-grained spatial information lost during pooling.

**Why it works well for medical imaging:**
Medical images have strong spatial priors (organs/tissues occupy predictable locations) and require precise boundary delineation. Skip connections allow the decoder to access high-resolution edge information from early encoder layers, producing sharper segmentation masks than a pure encoder-decoder without skips.

```
Input --> [Enc1] --> [Enc2] --> [Enc3] --> [Bottleneck]
              |          |          |            |
              +----skip--+----skip--+-----skip---+
              v          v          v            v
Output <-- [Dec1] <-- [Dec2] <-- [Dec3] <-- [UpConv]
```

Common variants used in dental imaging: **UNet++** (nested dense skip connections), **Attention UNet** (attention gates on skip connections to suppress irrelevant features), and **nnUNet** (self-configuring UNet that auto-adapts architecture and training to the dataset).

## Semi-Supervised Learning

Semi-supervised learning leverages a small set of labeled images and a large pool of unlabeled images. This is central to the STS Challenge design.

### Mean Teacher

Maintains two models: a **student** (updated by gradient descent) and a **teacher** (updated via Exponential Moving Average of student weights).

```
theta_teacher = alpha * theta_teacher + (1 - alpha) * theta_student
```

- `alpha` typically 0.99 - 0.999.
- Training loss = supervised loss on labeled data + consistency loss (MSE or KL between teacher and student predictions on unlabeled data).
- The teacher produces more stable pseudo-targets than the student because EMA smooths out noisy updates.

### FixMatch

A simple yet powerful approach combining **pseudo-labeling** with **confidence thresholding**:

1. Generate pseudo-labels from weakly-augmented unlabeled images (e.g., standard flip/crop).
2. Apply strong augmentation (e.g., RandAugment, CutOut) to the same images.
3. Train the student on strong-augmented images using pseudo-labels, but **only when the teacher's confidence exceeds a threshold** (typically 0.95).
4. Supervised loss on labeled data is added normally.

Low-confidence predictions are ignored, preventing confirmation bias from propagating incorrect labels.

### CPS (Cross Pseudo Supervision)

Trains **two networks** simultaneously. Each network generates pseudo-labels for the other:

```
L_cps = CE(pred_A, argmax(pred_B)) + CE(pred_B, argmax(pred_A))
```

- Both networks share the same labeled data with standard supervised loss.
- On unlabeled data, each network is trained to match the other's predictions.
- The two networks encourage each other to improve, reducing confirmation bias compared to self-training with a single model.

## Loss Functions

| Loss | Formula | Strengths | Weaknesses |
|------|---------|-----------|------------|
| **Cross Entropy** | `-sum(y * log(p))` | Well-calibrated gradients, standard baseline | Class-imbalanced data leads to dominated gradients |
| **Dice Loss** | `1 - 2*sum(y*p) / (sum(y) + sum(p))` | Directly optimizes Dice score, handles class imbalance | Gradient can be unstable when denominator is near 0 |
| **Focal Loss** | `-alpha * (1-p)^gamma * log(p)` | Down-weights easy examples, focuses on hard pixels | Requires tuning gamma (typically 2.0) |
| **Combo Loss** | `alpha * CE + (1-alpha) * Dice` | Combines CE's calibration with Dice's overlap metric | Extra hyperparameter alpha |

**Practical tip for STS:** Combo Loss (Dice + CE, weighted equally or 0.5/0.5) is the most common choice among top-ranked teams. Add Focal Loss if the dataset has severe class imbalance (e.g., rare tooth classes).

## Training Tips

### AMP (Automatic Mixed Precision)

Trains with FP16 where safe and FP32 where needed, roughly halving GPU memory usage and speeding up training 1.5-3x.

```python
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    output = model(input)
    loss = criterion(output, target)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

Key: use `GradScaler` to prevent underflow in FP16 gradients.

### Learning Rate Scheduling

- **Cosine Annealing**: LR decays following a cosine curve from initial LR to near 0. Smooth, avoids sudden drops.
- **Polynomial Decay**: `lr = lr_init * (1 - epoch/max_epoch)^power`. Common in segmentation.
- **Warmup**: start with a small LR (1/10 of target) for the first 5-10 epochs, then ramp up. Stabilizes early training when gradients are noisy.

**Recommended:** Cosine annealing with 10-epoch warmup, base LR = 1e-4 for AdamW or 0.01 for SGD.

### Gradient Clipping

Prevents exploding gradients, especially with deep UNet variants:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

`max_norm=1.0` is a safe default. Monitor gradient norms during training -- if they frequently hit the clip threshold, consider reducing the learning rate.

### Data Augmentation

Essential for small medical imaging datasets. Recommended pipeline:

| Augmentation | Purpose |
|-------------|---------|
| Random rotation (90/180/270) | Orientation invariance |
| Random flip (horizontal/vertical) | Laterality invariance |
| Random elastic deformation | Simulates anatomical variation |
| Random brightness/contrast | Robustness to scanner settings |
| Random crop / resize | Scale invariance |
| Gaussian noise | Robustness to imaging noise |

Libraries: `torchio` for 3D CBCT volumes, `albumentations` for 2D images. Use `nnunet`'s built-in augmentation for an out-of-the-box strong baseline.
