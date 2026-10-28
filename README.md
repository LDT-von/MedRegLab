# MedRegLab

**Medical Image Registration Laboratory**  
跨模态（MR-CT）医学图像配准工具箱

---

## 🎯 当前状态

| 项 | 说明 |
|---|---|
| 可用模型 | `convex_3d`（3D 配准，逐图优化 + TV 正则） |
| 训练 | `experiments/train_competition.py` |
| 推理 / 提交 | `experiments/infer_competition.py` |
| 数据 | `Datasets/`：train 24 例配对、test 8 例、unpaired CT 50 / MR 40 |
| 自检 | `bash selfcheck.sh`（9 项） |

> **说明**：仓库最初整合了 Learn2Reg 的 ConvexAdam / DINO-Reg / IMPACT 三条线，
> 但实测发现 DINO-Reg 与 IMPACT 所依赖的 `ConvexAdam` 后端存在多处
> 致命缺陷（变形场缺 batch 维、缺 `requires_grad`、`permute` 维度错误、
> DINOv2 特征提取器不支持 3D），导致 `dino_reg_mrct` / `impact_mrct` /
> `convex_mrct` 三个模型在 2D 与 3D 下**全部无法运行**。
> 这些代码已清理，详见 [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) 第 6–7 节。

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
│   ├── train_competition.py       # 训练
│   └── infer_competition.py       # 推理 + 生成提交文件
└── tests/
```

---

## 🚀 快速开始

### 安装（Linux / RTX 5090）

```bash
source /data1/MedRegLab/activate_env.sh   # 激活仓库自带 .venv
```

从零重建环境：

```bash
bash /data1/MedRegLab/setup_env.sh
```

> ⚠️ 本机 GPU 是 Blackwell `sm_120`，**必须**用 `torch>=2.7` 的 cu128 构建。
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
python experiments/infer_competition.py \
    --data Datasets \
    --split test \
    --output competition_output \
    --model convex_3d
```

### 训练

```bash
python experiments/train_competition.py \
    --data Datasets \
    --model convex_3d \
    --epochs 50 \
    --output competition_outputs
```

> `convex_3d` **没有可学习参数**（逐图优化 Adam on `phi`），
> 因此训练脚本只执行「配准 + 评估」，不做梯度更新、不保存 checkpoint。

---

## 📊 支持的模态

- **MR → CT**（主要）
- CT → MR
- 其他跨模态对（需微调）

---

## 📋 依赖

- Python >= 3.8（本机 3.12）
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
