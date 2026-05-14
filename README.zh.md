# STS-Skill

![GitHub stars](https://img.shields.io/github/stars/ricoleehduu/sts-skill?style=social)
![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-ee4c2c?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)
![MICCAI 2026](https://img.shields.io/badge/MICCAI-2026-orange)

[English](./README.md) | [中文](./README.zh.md)

一个为 MICCAI STS Challenge 参赛者设计的 Claude Code Skill。提供端到端工作流、评估工具和所有 STS 任务（2023-2026）的 baseline 代码。

## 这是什么？

STS-Skill 是 [MICCAI 半监督牙齿分割挑战赛](https://github.com/ricoleehduu/STS-Challenge) 的一站式工具包。安装为 Claude Code Skill 后，你可以获得：

- **Pre-Task 2026**：完整的训练 → 评估 → 提交流程（约1小时）
- **Task 1/2/3 2026**：Baseline 代码和提交指南
- **历史任务（2023-2025）**：参考实现和资源

## 工作原理

1. 在 Claude Code 中安装 Skill
2. 告诉 Claude 你想做什么（例如"开始 STS 2026 的 Pre-Task"）
3. Claude 读取 SKILL.md 并引导你到正确的工作流
4. 按照引导步骤完成任务

## 快速开始

### 安装为 Claude Code Skill

```bash
# 克隆仓库
git clone https://github.com/ricoleehduu/sts-skill.git

# 在 Claude Code 中引用 Skill 路径
# 或通过插件系统安装
```

### 在 Claude Code 中使用

安装后，直接告诉 Claude 你想做什么：

```
> 我想开始 STS 2026 的 Pre-Task
> 帮我评估模型预测结果
> 给我看 2025 的 baseline
> 下载 Pre-Task 数据
```

Skill 会自动将你的请求路由到正确的工作流。

## 任务覆盖

| 年份 | 任务 | 模态 | 状态 | 指南 |
|:----:|:-----|:-----|:----:|:----:|
| **2026** | Pre-Task（2D 分割） | OPG | ✅ 完整流程 | [指南](tasks/pretask-2026/GUIDE.md) |
| **2026** | Task 1（金属伪影去除） | CBCT | 📋 Baseline | [指南](tasks/task1-2026/GUIDE.md) |
| **2026** | Task 2 | CBCT | 📋 Baseline | [指南](tasks/task2-2026/GUIDE.md) |
| **2026** | Task 3 | CBCT | 📋 Baseline | [指南](tasks/task3-2026/GUIDE.md) |
| **2025** | 3D CBCT 牙髓分割 | CBCT | 📖 参考 | [指南](tasks/sts2025/GUIDE.md) |
| **2024** | 实例分割 | OPG | 📖 参考 | [指南](tasks/sts2024/GUIDE.md) |
| **2023** | 2D 牙齿分割 | OPG | 📖 参考 | [指南](tasks/sts2023/GUIDE.md) |

## 项目结构

```
sts-skill/
├── SKILL.md                    # Claude Code 入口（路由器）
├── scripts/
│   ├── evaluate.py             # Dice、IoU 评估
│   ├── download_data.py        # 多源数据下载器
│   └── prepare_submit.py       # Codabench 提交打包
├── tasks/
│   ├── pretask-2026/           # Pre-Task：完整 UNet 流程
│   ├── task1-2026/             # Task 1 baseline
│   ├── task2-2026/             # Task 2 baseline
│   ├── task3-2026/             # Task 3 baseline
│   ├── sts2025/                # 2025 参考
│   ├── sts2024/                # 2024 参考
│   └── sts2023/                # 2023 参考
├── references/
│   ├── algorithms.md           # 算法参考
│   ├── competition-history.md  # 赛事历史
│   └── faq.md                  # 常见问题
└── configs/                    # 训练配置
```

## Pre-Task 2026 快速开始

```bash
# 1. 下载数据
python scripts/download_data.py --task pretask-2026 --output ./data

# 2. 训练 UNet
python tasks/pretask-2026/train.py --amp --data-dir ./data

# 3. 运行推理
python tasks/pretask-2026/predict.py -m checkpoints/checkpoint_epoch50.pth -i data/test/imgs -o data/test/masks

# 4. 本地评估
python scripts/evaluate.py --pred data/test/masks --gt data/test/gt_masks

# 5. 打包提交
python scripts/prepare_submit.py --masks data/test/masks --task pretask-2026
```

## Pre-Task 工作流程

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   下载数据   │ →  │   训练模型   │ →  │  本地评估    │ →  │  推理测试    │ →  │   打包提交   │
│  (download) │    │  (train)    │    │ (evaluate)  │    │ (predict)   │    │  (submit)   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
     ~5 分钟           ~30 分钟            ~2 分钟             ~5 分钟            ~1 分钟
```

## 竞赛链接

| 资源 | 链接 |
|:-----|:-----|
| Pre-Task | https://www.codabench.org/competitions/16040/ |
| Task 1 | https://www.codabench.org/competitions/16027/ |
| Task 2 | https://www.codabench.org/competitions/16042/ |
| Task 3 | https://www.codabench.org/competitions/16117/ |
| ODIN Workshop | https://odin-workshops.org/2026 |
| 赛事官网 | https://nixy495.github.io/miccai2026/index.html |

## 数据获取

Pre-Task 数据可从多个来源获取：

| 来源 | 链接 |
|:-----|:-----|
| Huggingface | https://huggingface.co/datasets/Ricoooo/MICCAI-STS26-Challenge-Pre-Task |
| Modelscope | https://modelscope.cn/datasets/lizhii/MICCAI-STS26-Challenge-Pre-Task |
| 百度网盘 | https://pan.baidu.com/s/1U090bZnMGEJQaD3jwqaQuA（提取码：bm2u） |
| Google Drive | https://drive.google.com/drive/folders/1lER9eIavr99g28aTO0kuxIcos_k9FBSx |

## 相关仓库

| 仓库 | 说明 |
|:-----|:-----|
| [STS-Challenge](https://github.com/ricoleehduu/STS-Challenge) | STS 2023（2D 分割） |
| [STS-Challenge-2024](https://github.com/ricoleehduu/STS-Challenge-2024) | STS 2024（实例分割） |
| [STS-Challenge-2025](https://github.com/ricoleehduu/STS-Challenge-2025) | STS 2025（3D CBCT） |
| [STS-Challenge-2026](https://github.com/ricoleehduu/STS-Challenge-2026) | STS 2026 官方仓库 |

---

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=ricoleehduu/sts-skill&type=Date)](https://star-history.com/#ricoleehduu/sts-skill&Date)

---

<p align="center">
  <strong>如果这个项目对你有帮助，请给它一个 ⭐ Star！</strong>
</p>
