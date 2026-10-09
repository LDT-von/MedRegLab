# MedRegLab

融合 Learn2Reg 挑战赛三大冠军方案的跨模态医学图像配准工具箱

## 快速开始

### 安装

```bash
git clone https://github.com/LDT-von/MedRegLab.git
cd MedRegLab
pip install -e .
```

### 单例配准

```python
from medreglab import load_model

# 加载模型
model = load_model("dino_reg_mrct")

# 执行配准
result = model.register(
    moving="path/to/moving.nii.gz",
    fixed="path/to/fixed.nii.gz",
)

# 查看结果
print(f"Loss: {result['loss']}")
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

## 支持的方法

| 方法 | 论文 | 年份 | 排名 |
|------|------|------|------|
| ConvexAdam | ConvexAdam: Bayesian Inverse Problems with Convex Penalty | 2021 | 🥇 |
| DINO-Reg | DINO-Reg: Self-supervised Deformable Image Registration | 2023 | 🥇 |
| IMPACT | IMPACT: Implicit Neural Penalty based Adversarial Cross-domain Transform | 2023 | 🥇 |

## 项目结构

```
MedRegLab/
├── medreglab/
│   ├── core/           # 核心优化器
│   ├── modules/        # 模型模块
│   ├── models/         # 配准模型
│   ├── data/           # 数据处理
│   └── utils/          # 工具函数
├── configs/            # 配置文件
├── experiments/        # 实验脚本
└── docs/               # 文档
```

## 许可证

MIT License
