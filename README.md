# MedRegLab

**Medical Image Registration Laboratory**  
跨模态（MR-CT）医学图像配准工具箱

---

## 🎯 当前状态

| 项 | 说明 |
|---|---|
| 可用模型 | `convex_3d`（3D 配准，逐图优化 + TV 正则） |
| 逐例诊断 | `experiments/train_competition.py`（不训练网络权重） |
| 推理 / 提交 | `experiments/infer_competition.py` / `experiments/package_submission.py` |
| 数据 | `Datasets/`：train 24 例配对、test 8 例、unpaired CT 50 / MR 40 |
| 自检 | `bash selfcheck.sh`（默认仅合成数据；可选 `--data Datasets` 检查目录） |

> **说明**：仓库最初整合了 Learn2Reg 的 ConvexAdam / DINO-Reg / IMPACT 三条线，
> 但实测发现 DINO-Reg 与 IMPACT 所依赖的 `ConvexAdam` 后端存在多处
> 致命缺陷（变形场缺 batch 维、缺 `requires_grad`、`permute` 维度错误、
> DINOv2 特征提取器不支持 3D），导致 `dino_reg_mrct` / `impact_mrct` /
> `convex_mrct` 三个模型在 2D 与 3D 下**全部无法运行**。
> 这些代码已清理，详见 [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) 第 6–7 节。
> `convex_3d` 是逐例优化的 Adam + MSE/NCC + TV 基线，**不是**论文原版 ConvexAdam，也不是已训练的深度网络。

---

## 🏗️ 架构设计

```
MedRegLab/
├── medreglab/
│   ├── core/
│   │   ├── convex_3d_working.py   # Simple3DRegistration（3D 配准主实现）
│   │   └── loss_builder.py        # 损失函数构建器
│   ├── models/
│   │   └── deformation_field.py   # 变形场与 Jacobian 工具
│   ├── data/
│   │   ├── dataset.py             # CompetitionDataset
│   │   ├── transforms.py          # 数据增强
│   │   └── sampler.py             # 采样器
│   ├── evaluation/
│   │   └── dsc_evaluator.py       # DSC / DSC30 评估
│   └── utils/
│       ├── metrics.py             # 评估指标
│       └── io_utils.py            # IO 工具
├── configs/competition_config.yaml
├── experiments/
│   ├── train_competition.py       # 单轮逐例诊断
│   ├── infer_competition.py       # 推理与原始 CT 网格物理位移
│   └── package_submission.py      # 官方结构打包及自检
└── tests/
```

---

## 🚀 快速开始

### 安装（Linux / RTX 5090）

```bash
cd /data1/MedRegLab
source scripts/activate_medreg_env.sh  # 检查并激活现有 .venv
```

仅在没有现有环境、确实需要新建时：

```bash
bash scripts/setup_medreg_env.sh
```

> 在 RTX 5090 `sm_120` 上应使用支持该设备的 PyTorch CUDA 构建（已知可用 `torch 2.10.0+cu128`）。
> base 环境里的 `torch 2.6.0+cu124` 会报
> `CUDA error: no kernel image is available for execution on the device`
> （而 `torch.cuda.is_available()` 仍返回 `True`，极易误判）。

### 基础用法

```python
import medreglab as mrl

# 加载模型
model = mrl.load_model("convex_3d")

# 配准（moving / fixed 均为 [1, C, D, H, W] 张量）
result = model(moving_tensor, fixed_tensor)

print("phi:", result["phi"].shape)     # [1, 3, D, H, W]
print("loss:", result["loss"])
```

### 推理并生成提交文件

```bash
MEDREG_RUN_ID="$(date +%Y%m%d_%H%M%S)"
MEDREG_INFERENCE="outputs/test_${MEDREG_RUN_ID}"
MEDREG_SUBMISSION="outputs/submission_${MEDREG_RUN_ID}"
python experiments/infer_competition.py \
    --data Datasets \
    --split test \
    --output "$MEDREG_INFERENCE" \
    --model convex_3d
```

按测试集提交规范生成最终压缩包（输出目录必须是新目录）：

```bash
python experiments/package_submission.py \
    --inference "$MEDREG_INFERENCE" \
    --data Datasets \
    --output "$MEDREG_SUBMISSION"
```

生成的 `"$MEDREG_SUBMISSION"/submission.zip` 包含
`submission/025/displacement_field.nii.gz` 至 `submission/032/displacement_field.nii.gz`。
最终文件经 SimpleITK `TransformToDisplacementField` 生成，为原始 CT 网格上的
float32 三分量物理位移（毫米），含义为 `p_mr = p_ct + D(p_ct)`。
本地格式校验报告 `format_check.json` 保存在压缩包外；测试集 DSC/DSC30 由主办方计算。
当前 Windows 工作副本的数据在仓库旁的 `../Datasets`，两条命令的 `--data` 均应改为该路径。

### 单轮逐例诊断

```bash
python experiments/train_competition.py \
    --config configs/competition_config.yaml --data Datasets \
    --output outputs/NEW_DIAGNOSTIC
```

> `convex_3d` 没有可学习的网络参数；逐例 Adam 更新位移场 `phi`。
> 诊断脚本不保存网络 checkpoint，且默认不计算验证分数。若指定独立的 `--val`，
> 其结果是缩放网格上的诊断分数；原始空间验证请使用推理入口的 `--split val`。
> 缺少独立验证集时不会自动回退到训练集。

---

## 📊 支持的模态

- 比赛数据加载器固定 CT、浮动 MR；其他数据需自行核对模态预处理与几何关系。

---

## 📋 依赖

- Python >= 3.10（5090 环境曾验证 3.12）
- PyTorch >= 2.7（本机 2.10.0+cu128）
- MONAI, SimpleITK, NumPy, SciPy

---

## 📖 文档

- [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) — 环境、已知坑、清理记录
- [docs/index.md](docs/index.md)

---

## 📜 许可证

MIT License

## 🙏 致谢

本项目最初整合了以下开源工作（相关代码已清理，但思路来源在此致谢）：

- [convexAdam](https://github.com/multimodalscanning/convexAdam)
- [DINO-Reg](https://github.com/LDT-von/DINO-Reg)
- [ImpactLoss](https://github.com/liuqu321/ImpactLoss)
- [uniGradICON](https://github.com/uncbiag/uniGradICON)
