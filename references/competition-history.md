# STS Challenge History (2023-2026)

A brief history of the Semi-supervised Teeth Segmentation Challenge from its inception to the current edition.

## STS 2023 -- 2D Teeth Segmentation

- **Edition:** 1st (MICCAI 2023)
- **Task:** 2D teeth segmentation on panoramic dental X-ray images
- **Modality:** OPG (Orthopantomogram) and PXI (Periapical X-ray)
- **Classes:** 33 classes (32 tooth types + background), using FDI notation
- **Setting:** Semi-supervised -- small labeled set + large unlabeled pool
- **Key challenge:** Fine-grained tooth type classification (distinguishing similar tooth classes) with limited labels
- **Top approaches:**
  - UNet/UNet++ with semi-supervised consistency training
  - Mean Teacher and FixMatch variants
  - Heavy use of test-time augmentation (TTA)
- **Notable:** Established the benchmark for semi-supervised dental segmentation; showed that semi-supervised methods significantly outperform supervised-only baselines with the same labeled data
- **Repository:** https://github.com/DentalSemSeg/STS2023

## STS 2024 -- Instance Segmentation

- **Edition:** 2nd (MICCAI 2024)
- **Task:** Instance-level teeth segmentation -- not just "which tooth type" but "which individual tooth"
- **Modality:** OPG (Orthopantomogram) and X-ray
- **Setting:** Two-stage pipeline: detection then segmentation
- **Key challenge:** Instance differentiation -- adjacent teeth of the same type must be assigned distinct instance IDs. Pure semantic segmentation cannot solve this
- **Top approaches:**
  - Detection-based: Mask R-CNN, Cascade Mask R-CNN for instance segmentation
  - Two-stage: separate tooth detection/localization, then per-instance segmentation
  - Topology-aware post-processing to split merged instances
- **Notable:** Shifted from semantic to instance segmentation, reflecting clinical needs (each tooth has its own treatment record). Highlighted the importance of instance-aware architectures
- **Repository:** https://github.com/DentalSemSeg/STS2024

## STS 2025 -- 3D CBCT Pulp Segmentation

- **Edition:** 3rd (MICCAI 2025)
- **Task:** 3D dental pulp segmentation from CBCT (Cone Beam Computed Tomography) volumes
- **Modality:** CBCT volumetric data (3D)
- **Setting:** Semi-supervised
- **Key challenge:** Transition from 2D to 3D; pulp is a small, low-contrast structure embedded within tooth anatomy; requires volumetric reasoning
- **Top approaches:**
  - **U-Mamba2** based architectures -- combining Mamba (state-space model) with UNet for efficient 3D feature extraction
  - nnUNet as the strong baseline with self-configuring architecture
  - 3D patch-based training with sliding-window inference
  - Semi-supervised consistency regularization in 3D
- **Notable:** First 3D edition of the challenge; Mamba-based architectures (U-Mamba, U-Mamba2) demonstrated competitive performance against transformer-based models at lower computational cost
- **Repository:** https://github.com/DentalSemSeg/STS2025

## STS 2026 -- Metal Artifact Removal + Segmentation

- **Edition:** 4th (MICCAI 2026)
- **Task:** Joint metal artifact removal and teeth segmentation on CBCT
- **Modality:** CBCT with metal artifacts (dental implants, braces, crowns)
- **Setting:** Semi-supervised, 4 subtasks

### Subtasks

| Task | Description | Codabench Link |
|------|-------------|----------------|
| **Pre-Task** | Metal artifact removal from CBCT | https://www.codabench.org/competitions/16040/ |
| **Task 1** | Teeth segmentation on clean CBCT | https://www.codabench.org/competitions/16027/ |
| **Task 2** | Teeth segmentation on artifact-affected CBCT | https://www.codabench.org/competitions/16042/ |
| **Task 3** | End-to-end: artifact removal then segmentation | https://www.codabench.org/competitions/16117/ |

### Key challenges

- Metal artifacts corrupt CBCT images with streaking and beam-hardening artifacts, severely degrading segmentation quality
- The competition explores whether artifact removal as a preprocessing step improves segmentation, or whether end-to-end approaches are superior
- Multi-task learning: participants can tackle tasks independently or jointly

### Competition platform and resources

- **GitHub:** https://github.com/ricoleehduu/STS-Challenge-2026
- **Website:** https://nixy495.github.io/miccai2026/index.html
- **Workshop:** ODIN 2026 (Oral and Dental Imaging meets AI), co-located with MICCAI 2026
- **Workshop submission:** https://openreview.net/group?id=MICCAI.org/2026/Workshop/ODIN
- **Workshop website:** https://odin-workshops.org/2026
