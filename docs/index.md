# MedRegLab

跨模态（MR-CT）医学图像配准工具箱

## 快速开始

```bash
cd /data1/MedRegLab
source scripts/activate_medreg_env.sh
python -m pytest tests/ -q
bash selfcheck.sh                   # 默认仅使用合成数据
```

### 配准

```python
from medreglab import load_model

model = load_model("convex_3d")   # 目前唯一支持的模型
result = model(moving, fixed)      # moving/fixed: [1, C, D, H, W]

print("phi:", result["phi"].shape)  # [1, 3, D, H, W]
print("loss:", result["loss"])
```

### 推理

```bash
python experiments/infer_competition.py \
    --data Datasets --split test \
    --output competition_output --model convex_3d
```

### 单轮逐例诊断

```bash
python experiments/train_competition.py \
    --config configs/competition_config.yaml --data Datasets \
    --output outputs/NEW_DIAGNOSTIC
```

## 组件

| 组件 | 位置 | 说明 |
|---|---|---|
| `Simple3DRegistration` | `medreglab/core/convex_3d_working.py` | 3D 配准主实现（逐图优化） |
| `CompetitionDataset` | `medreglab/data/dataset.py` | train / test / unpaired_ct / unpaired_mr |
| `CompetitionEvaluator` | `medreglab/evaluation/dsc_evaluator.py` | DSC / DSC30 |
| `DeformationField` / `JacobianField` | `medreglab/models/deformation_field.py` | 变形场与折叠检查 |
| `LossBuilder` | `medreglab/core/loss_builder.py` | MSE / NCC 组合 |

## 支持的方法

| 方法 | 论文 | 年份 | 状态 |
|------|------|------|------|
| Simple3DRegistration | 无训练权重的逐例 Adam + MSE/NCC + TV 基线 | — | ✅ 可用（`convex_3d`；不是论文原版 ConvexAdam） |
| DINO-Reg | DINO-Reg: Self-supervised Deformable Image Registration | 2023 | ❌ 后端已损坏，代码已清理 |
| IMPACT | IMPACT: Implicit Neural Penalty based Adversarial Cross-domain Transform | 2023 | ❌ 后端已损坏，代码已清理 |

> 清理原因与细节见 [ENVIRONMENT.md](ENVIRONMENT.md) 第 6–7 节。

## 项目结构

```
MedRegLab/
├── medreglab/
│   ├── core/           # 核心配准与损失
│   ├── models/         # 变形场工具
│   ├── data/           # 数据处理
│   ├── evaluation/     # 评估
│   └── utils/          # 工具函数
├── configs/            # 配置文件
├── experiments/        # train_competition / infer_competition
└── docs/
```

## 许可证

MIT License
