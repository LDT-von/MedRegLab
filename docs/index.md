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

## 文档

| 文档 | 说明 |
|---|---|
| [`docs/SUBMISSION_LEDGER.md`](docs/SUBMISSION_LEDGER.md) | **提交包完整台账**：分数、路径、脚本、算法代码版本、复现方式、目录清理记录 |
| [`docs/SCORE_IMPROVEMENT_PLAN.md`](docs/SCORE_IMPROVEMENT_PLAN.md) | 提分实验全记录（31 条结论，含否证项）；顶部有**提交包标记速查表** |
| [`docs/FINAL_METHOD.md`](docs/FINAL_METHOD.md) | 2026-11-01 定版的交付清单（已被超越，见文首提示） |
| [`docs/EVIDENCE_BASELINE.md`](docs/EVIDENCE_BASELINE.md) | 历史基线与 6.18 轴序缺陷的证据核查记录 |
| [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md) | 环境与依赖说明 |

### 提交包标记速查

| 标签 | 榜单 DSC / DSC30 | 开发集 DSC / DSC30 | 提交包目录 | 生成脚本 |
|---|---|---|---|---|
| **`LB-0.9073`** | **0.9073 / 0.8977** | 0.9072 / 0.8958 | `outputs/submission_organ_best_20261102` | `/data/ctmr/run_submit.sh` |
| **`LB-0.9325`** | 待提交 | **0.9325 / 0.9252** | `outputs/submission_organ_best_20261103_final` | `/data/ctmr/run_submit_final.sh` |

> ⚠️ 只有 `LB-0.9073` 实际上过榜并出分；`LB-0.9325` 是**开发集**分。
> 每个提交包目录内有 `TAG_*.txt` 标记文件（位于 ZIP 外，不影响 `submission.zip` 哈希）。

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
