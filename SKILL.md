---
name: sts-skill
description: >
  Claude Code Skill for MICCAI STS Challenge (2023-2026). Provides end-to-end workflows for
  teeth segmentation tasks including Pre-Task, Task 1/2/3, and historical challenges.
  Use when: starting STS challenge tasks, training UNet, evaluating segmentation masks,
  downloading challenge data, preparing submissions, or asking about STS competition workflow.
  Covers semi-supervised learning, CBCT processing, metal artifact removal, and instance segmentation.
---

# STS-Skill

Your entry point for all MICCAI STS Challenge tasks. This skill routes you to the right workflow based on what you want to do.

## How to Use

When the user mentions STS challenge, teeth segmentation, Pre-Task, or any competition task, follow this routing logic:

### Step 1: Identify User Intent

Parse the user's message for these keywords and route accordingly:

| User says | Route to |
|-----------|----------|
| "pre-task", "pretask", "pre task", "Pre-Task", "2026 pre" | `tasks/pretask-2026/GUIDE.md` |
| "task 1", "task1", "metal artifact", "2026 task1" | `tasks/task1-2026/GUIDE.md` |
| "task 2", "task2", "2026 task2" | `tasks/task2-2026/GUIDE.md` |
| "task 3", "task3", "2026 task3" | `tasks/task3-2026/GUIDE.md` |
| "2025", "sts2025", "last year", "CBCT pulp" | `tasks/sts2025/GUIDE.md` |
| "2024", "sts2024", "instance segmentation" | `tasks/sts2024/GUIDE.md` |
| "2023", "sts2023", "semi-supervised" | `tasks/sts2023/GUIDE.md` |
| "evaluate", "评估", "dice", "score" | Use `scripts/evaluate.py` |
| "download", "下载", "data", "数据" | Use `scripts/download_data.py` |
| "submit", "提交", "package", "打包" | Use `scripts/prepare_submit.py` |
| "help", "帮助", "what can you do", "菜单" | Show menu (below) |
| Unclear / general | Show menu (below) |

### Step 2: Load the Guide

Once you identify the task, read the corresponding GUIDE.md file. The guide contains step-by-step instructions that you should follow to help the user complete their task.

### Step 3: Execute

Follow the guide's instructions. For scripts, use the appropriate tool (Bash) to run them. For code tasks, write/modify code as instructed.

---

## Menu (show when intent is unclear)

When the user's intent is unclear, present this menu:

```
🏆 STS Challenge Skill - What would you like to do?

📋 2026 Tasks:
  1. Pre-Task    - 2D segmentation (UNet, ~1 hour)
  2. Task 1      - CBCT segmentation under metal artifacts
  3. Task 2      - CBCT-IOS registration (PointNet)
  4. Task 3      - 3D CBCT teeth segmentation (3D UNet)

📚 Historical:
  5. STS 2025    - 3D CBCT pulp segmentation
  6. STS 2024    - Instance segmentation
  7. STS 2023    - 2D teeth segmentation (semi-supervised)

🔧 Tools:
  8. Download data
  9. Evaluate predictions
  10. Prepare submission

Just tell me the number or describe what you need!
```

---

## Shared Workflows

These patterns are used across multiple tasks. Reference them from the guides.

### Environment Check

Before any task, verify the user's environment:

```bash
python --version          # Need 3.8+
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Data Download

Use `scripts/download_data.py` to download data from the fastest available source:

```bash
python scripts/download_data.py --task <task-name> --output ./data
```

Supported tasks: `pretask-2026`

### Local Evaluation

Use `scripts/evaluate.py` to check prediction quality before submitting:

```bash
python scripts/evaluate.py --pred <predicted_masks_dir> --gt <ground_truth_dir>
```

### Submission Packaging

Use `scripts/prepare_submit.py` to create a Codabench-compatible zip:

```bash
python scripts/prepare_submit.py --masks <masks_dir> --task <task-name>
```

---

## Data Sources

Pre-Task data is available from:

| Source | Link |
|--------|------|
| Huggingface | https://huggingface.co/datasets/Ricoooo/MICCAI-STS26-Challenge-Pre-Task |
| Modelscope | https://modelscope.cn/datasets/lizhii/MICCAI-STS26-Challenge-Pre-Task |
| Baidu Netdisk | https://pan.baidu.com/s/1U090bZnMGEJQaD3jwqaQuA (code: bm2u) |
| Google Drive | https://drive.google.com/drive/folders/1lER9eIavr99g28aTO0kuxIcos_k9FBSx |

## Competition Links

| Resource | Link |
|----------|------|
| Pre-Task | https://www.codabench.org/competitions/16040/ |
| Task 1 | https://www.codabench.org/competitions/16027/ |
| Task 2 | https://www.codabench.org/competitions/16042/ |
| Task 3 | https://www.codabench.org/competitions/16117/ |
| ODIN Workshop | https://odin-workshops.org/2026 |
| Challenge Website | https://nixy495.github.io/miccai2026/index.html |

## References

For deeper understanding, read:
- `references/algorithms.md` — UNet, semi-supervised learning, loss functions
- `references/competition-history.md` — STS 2023-2026 overview
- `references/faq.md` — Common questions and troubleshooting
