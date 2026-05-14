# STS-Skill

![GitHub stars](https://img.shields.io/github/stars/ricoleehduu/sts-skill?style=social)
![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-ee4c2c?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)
![MICCAI 2026](https://img.shields.io/badge/MICCAI-2026-orange)

[English](./README.md) | [中文](#简介)

A Claude Code Skill for MICCAI STS Challenge participants. Provides end-to-end workflows, evaluation tools, and baseline code for all STS tasks (2023-2026).

## What is this?

STS-Skill is your one-stop toolkit for the [MICCAI Semi-supervised Teeth Segmentation Challenge](https://github.com/ricoleehduu/STS-Challenge). Install it as a Claude Code Skill and get guided workflows for:

- **Pre-Task 2026**: Complete train → evaluate → submit pipeline (~1 hour)
- **Task 1/2/3 2026**: Baseline code and submission guides
- **Historical tasks (2023-2025)**: Reference implementations and resources

## How it Works

1. Install the skill in Claude Code
2. Tell Claude what you want to do (e.g., "Start the STS 2026 Pre-Task")
3. Claude reads the SKILL.md and routes you to the right workflow
4. Follow the guided steps to complete your task

## Quick Start

### Install as Claude Code Skill

```bash
# Clone the skill
git clone https://github.com/ricoleehduu/sts-skill.git

# In Claude Code, reference the skill path
# Or install via plugin system
```

### Use in Claude Code

Once installed, just tell Claude what you want to do:

```
> I want to start the STS 2026 Pre-Task
> Help me evaluate my model predictions
> Show me the 2025 challenge baseline
> Download the Pre-Task data
```

The skill automatically routes your request to the right workflow.

## Task Coverage

| Year | Task | Modality | Status | Guide |
|:----:|:-----|:---------|:------:|:-----:|
| **2026** | Pre-Task (2D Segmentation) | OPG | ✅ Full workflow | [Guide](tasks/pretask-2026/GUIDE.md) |
| **2026** | Task 1 (Metal Artifact Removal) | CBCT | 📋 Baseline | [Guide](tasks/task1-2026/GUIDE.md) |
| **2026** | Task 2 | CBCT | 📋 Baseline | [Guide](tasks/task2-2026/GUIDE.md) |
| **2026** | Task 3 | CBCT | 📋 Baseline | [Guide](tasks/task3-2026/GUIDE.md) |
| **2025** | 3D CBCT Pulp Segmentation | CBCT | 📖 Reference | [Guide](tasks/sts2025/GUIDE.md) |
| **2024** | Instance Segmentation | OPG | 📖 Reference | [Guide](tasks/sts2024/GUIDE.md) |
| **2023** | 2D Teeth Segmentation | OPG | 📖 Reference | [Guide](tasks/sts2023/GUIDE.md) |

## Project Structure

```
sts-skill/
├── SKILL.md                    # Claude Code entry point (router)
├── scripts/
│   ├── evaluate.py             # Dice, IoU evaluation
│   ├── download_data.py        # Multi-source data downloader
│   └── prepare_submit.py       # Codabench submission packager
├── tasks/
│   ├── pretask-2026/           # Pre-Task: full UNet workflow
│   ├── task1-2026/             # Task 1 baseline
│   ├── task2-2026/             # Task 2 baseline
│   ├── task3-2026/             # Task 3 baseline
│   ├── sts2025/                # 2025 reference
│   ├── sts2024/                # 2024 reference
│   └── sts2023/                # 2023 reference
├── references/
│   ├── algorithms.md           # Algorithm reference
│   ├── competition-history.md  # Challenge history
│   └── faq.md                  # FAQ
└── configs/                    # Training configurations
```

## Pre-Task 2026 Quick Start

```bash
# 1. Download data
python scripts/download_data.py --task pretask-2026 --output ./data

# 2. Train UNet
python tasks/pretask-2026/train.py --amp --data-dir ./data

# 3. Run inference
python tasks/pretask-2026/predict.py -m checkpoints/checkpoint_epoch50.pth -i data/test/imgs -o data/test/masks

# 4. Evaluate locally
python scripts/evaluate.py --pred data/test/masks --gt data/test/gt_masks

# 5. Package for submission
python scripts/prepare_submit.py --masks data/test/masks --task pretask-2026
```

## Pre-Task Workflow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Download   │ →  │   Train     │ →  │  Evaluate   │ →  │  Inference  │ →  │   Submit    │
│    Data     │    │   UNet      │    │  (Dice)     │    │  (predict)  │    │  (Codabench)│
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
     ~5 min            ~30 min             ~2 min             ~5 min             ~1 min
```

## Competition Links

| Resource | Link |
|:---------|:-----|
| Pre-Task | https://www.codabench.org/competitions/16040/ |
| Task 1 | https://www.codabench.org/competitions/16027/ |
| Task 2 | https://www.codabench.org/competitions/16042/ |
| Task 3 | https://www.codabench.org/competitions/16117/ |
| ODIN Workshop | https://odin-workshops.org/2026 |
| Challenge Website | https://nixy495.github.io/miccai2026/index.html |

## Data Access

Pre-Task data is available from multiple sources:

| Source | Link |
|:-------|:-----|
| Huggingface | https://huggingface.co/datasets/Ricoooo/MICCAI-STS26-Challenge-Pre-Task |
| Modelscope | https://modelscope.cn/datasets/lizhii/MICCAI-STS26-Challenge-Pre-Task |
| Baidu Netdisk | https://pan.baidu.com/s/1U090bZnMGEJQaD3jwqaQuA (code: bm2u) |
| Google Drive | https://drive.google.com/drive/folders/1lER9eIavr99g28aTO0kuxIcos_k9FBSx |

## Related Repositories

| Repository | Description |
|:-----------|:------------|
| [STS-Challenge](https://github.com/ricoleehduu/STS-Challenge) | STS 2023 (2D segmentation) |
| [STS-Challenge-2024](https://github.com/ricoleehduu/STS-Challenge-2024) | STS 2024 (instance segmentation) |
| [STS-Challenge-2025](https://github.com/ricoleehduu/STS-Challenge-2025) | STS 2025 (3D CBCT) |
| [STS-Challenge-2026](https://github.com/ricoleehduu/STS-Challenge-2026) | STS 2026 official repo |

---

## 简介

STS-Skill 是一个为 MICCAI STS Challenge 参赛者设计的 Claude Code Skill。提供端到端工作流、评估工具和所有 STS 任务（2023-2026）的 baseline 代码。

### 快速开始

安装为 Claude Code Skill 后，直接告诉 Claude 你想做什么：

```
> 我想开始 STS 2026 的 Pre-Task
> 帮我评估模型预测结果
> 给我看 2025 的 baseline
> 下载 Pre-Task 数据
```

### Pre-Task 2026 流程

1. 下载数据 → `python scripts/download_data.py --task pretask-2026`
2. 训练模型 → `python tasks/pretask-2026/train.py --amp`
3. 推理测试 → `python tasks/pretask-2026/predict.py -m checkpoints/checkpoint_epoch50.pth -i data/test/imgs`
4. 本地评估 → `python scripts/evaluate.py --pred data/test/masks --gt data/test/gt_masks`
5. 打包提交 → `python scripts/prepare_submit.py --masks data/test/masks --task pretask-2026`

---

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=ricoleehduu/sts-skill&type=Date)](https://star-history.com/#ricoleehduu/sts-skill&Date)

---

<p align="center">
  <strong>If this project helps you, please give it a ⭐ Star!</strong>
</p>
