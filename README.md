# MedRegLab

**Medical Image Registration Laboratory**  
融合 Learn2Reg 挑战赛三大冠军方案的跨模态医学图像配准工具箱

---

## 🎯 核心特性

| 方案 | 比赛 | 年份 | 排名 | 贡献 |
|------|------|------|------|------|
| **ConvexAdam** | Learn2Reg Task1 (MR-CT) | 2021 | 🥇 冠军 | 凸优化 + 物理先验正则化 |
| **DINO-Reg** | Learn2Reg 2023 | 2023 | 🥇 冠军 | DINOv2 自监督特征 + ConvexAdam |
| **IMPACT** | Learn2Reg 2023 | 2023 | 🥇 冠军 | 语义感知损失函数 |

## 🏗️ 架构设计

```
MedRegLab/
├── medreglab/
│   ├── core/              # 核心优化器与配准引擎
│   │   ├── convex_optimizer.py   # ConvexAdam 优化器
│   │   ├── registration_engine.py  # 配准主流程
│   │   └── loss_builder.py        # 损失函数构建器
│   ├── modules/
│   │   ├── impact_loss.py         # IMPACT 语义损失
│   │   ├── gradicon.py            # uniGradICON 特征网络
│   │   └── convex_regularizer.py  # 凸正则化器
│   ├── models/
│   │   ├── dinov2_extractor.py    # DINOv2 特征提取器
│   │   ├── reg_model.py           # 配准模型封装
│   │   └── deformation_field.py   # 变形场生成
│   ├── data/
│   │   ├── dataset.py             # 数据加载
│   │   └── augmentation.py        # 数据增强
│   └── utils/
│       ├── metrics.py             # 评估指标
│       ├── visualization.py       # 可视化工具
│       └── io_utils.py            # IO 工具
├── configs/                # 配置文件
├── experiments/           # 实验脚本
└── docs/                  # 文档
```

## 🚀 快速开始

### 安装

```bash
git clone https://github.com/LDT-von/MedRegLab.git
cd MedRegLab
pip install -e .
```

### 基础用法

```python
import medreglab as mrl

# 加载预训练模型
model = mrl.load_model("dino_reg_mrct")

# 单例配准
result = model.register(
    moving="path/to/moving.nii.gz",
    fixed="path/to/fixed.nii.gz",
    output="path/to/output/"
)

# 批量配准
model.register_batch(
    pairs=[("img1_moving.nii.gz", "img1_fixed.nii.gz"), ...],
    output_dir="results/"
)
```

### 使用 IMPACT 语义损失

```python
from medreglab.modules import IMPACTLoss

impact = IMPACTLoss(
    feature_dim=1024,
    temperature=0.1,
    semantic_weight=0.5
)
```

### 使用 ConvexAdam 优化器

```python
from medreglab.core import ConvexAdam

optimizer = ConvexAdam(
    learning_rate=0.5,
    momentum=0.9,
    max_iter=200
)
```

## 📊 支持的模态

- **MR → CT** (主要)
- CT → MR
- 其他跨模态对（需微调）

## 📋 依赖

- Python >= 3.8
- PyTorch >= 1.10
- MONAI >= 0.9
- SimpleITK
- NumPy, SciPy

## 📖 文档

详细文档见 [docs/](docs/)

## 📜 许可证

MIT License

## 🙏 致谢

本项目整合了以下开源工作：
- [convexAdam](https://github.com/multimodalscanning/convexAdam)
- [DINO-Reg](https://github.comLDT-von/DINO-Reg)
- [ImpactLoss](https://github.com/liuqu321/ImpactLoss)
- [uniGradICON](https://github.com/uncbiag/uniGradICON)
- [UTSRMorph](https://github.com/xmuyulong/UTSRMorph)
- [DGMIR](https://github.com/zhiyan114/DGMIR)
