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

### ⚠️ 历史效果基线（证据受限，详见 [docs/EVIDENCE_BASELINE.md](docs/EVIDENCE_BASELINE.md)）

历史提交 `336047c`（备份于 `backup/local-master-336047c` / `backup/pre-sync-336047c`）
的文档记录过一组数字：在 `Datasets/train` 前若干例、64³ 缩放网格上，

| 配置 | mean DSC (liver/spleen/kidney) |
|---|---|
| identity（零变形场 = 什么都不做） | 0.396 |
| `convex_3d`（200 iter） | 0.400 |

**这两组数字相差 0.004。** 请注意三点限制：

1. **无法复核**。产生它们的逐例结果、配置与日志已不存在，
   所在会话记录也不在本机（核查过程见证据文档 §1.1）。
   文档里写过数字，**不等于**数字已被独立验证。
2. **不是独立验证集，也不是测试集**。历史代码在缺少 `val/` 目录时
   会静默回退到 `train/`，而 `Datasets/` 下确实没有 `val/`。
3. **与当前提交包不可直接对比**。历史评分在 64³ 缩放网格上做、
   左右肾分别计分（4 项平均）；当前实现评在原始 CT 网格上做、
   双肾合并（3 项平均）。口径不同。

因此**不应**据此声称该模型已能有效完成 CT-MR 配准，
也**不应**把 0.004 说成「已证明有提升」。

> 历史缺陷「标签形变坐标轴顺序错误」（曾使 identity DSC 被严重低估）
> 的分析已恢复，见 [docs/ENVIRONMENT.md 第 6.18 节](docs/ENVIRONMENT.md)；
> 该缺陷本身已用合成数据**复算确认**，且在当前 `6933cfe` 中已修复。

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
│   │   ├── dsc_evaluator.py       # DSC / DSC30 评估
│   │   ├── dev_eval.py            # 开发集评估（配置对比）
│   │   └── prealign.py            # 图像驱动的刚性预对齐
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

#### 备选：基线 B 提交包（不跑配准）

若配准结果在开发集上并未优于基线，可以提交**基线 B** 的位移场。
基线 B 是模型**什么都没学到**时的退化解：缩放网格上的恒等场 `phi = 0`，
经 `displacement_to_fixed_grid` 转到原始 CT 网格。它**不是**原始物理空间零位移——
它已经包含了 CT 与 MR 之间的几何映射（尺寸 / spacing / origin / direction 差异），
因此在几何不一致的病例上明显优于零位移基线 A。

基线 B **不需要标签**，因此可以为隐藏测试集生成：

```bash
# 1) 生成 8 例测试集的基线 B 位移场（纯几何计算，不跑迭代、不读标签）
python scripts/baseline_test_fields.py \
    --data Datasets \
    --output outputs/baseline_B_test_<RUN_ID> \
    --split test

# 2) 按同一规范打包（与上面完全相同的命令形式）
python experiments/package_submission.py \
    --inference outputs/baseline_B_test_<RUN_ID> \
    --data Datasets \
    --output outputs/submission_B_<RUN_ID>
```

生成器的输出目录布局与 `infer_competition.py` 完全一致，
因此 `package_submission.py` 无需改动即可打包。
任一病例失败时**不写** `inference_summary.json`，打包步骤会拒绝执行。

> **交付的 C1 与 B 两个包的用途不同**：
> C1（`outputs/submission_20261028_065511`）是**已配准**结果；
> B（`outputs/submission_B_20261028`）是**几何对齐基线**，不含任何学习到的形变。
> 两者都是**合法格式**的提交包，但隐藏测试集得分未知，
> 主办方评分前无法判断哪个更高。


> **DSC30 取整口径**：`ceil(total × 0.3)`，即 24 例取最低 **8** 例、
> 8 例取最低 **3** 例（**向上取整**，不是向下取整）。
> 实现见 `medreglab/evaluation/dsc_evaluator.py::DSC30_Scorer.compute`，
> 回归测试见 `tests/test_evaluation_protocol.py::test_dsc30_rounds_case_count_up`。

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

### 开发集评估工具（配置对比）

用于在**同一批**开发集病例上比较若干**事先固定**的注册配置，
并与两个不同的基线对照。详见
[docs/EVIDENCE_BASELINE.md §5](docs/EVIDENCE_BASELINE.md)。

```bash
# 1) 只读盘点：病例数、几何差异、标签直方图（不跑配准）
python scripts/dev_eval_preflight.py --data Datasets --split train

# 2) 逐配置跑同一批病例，输出 manifest.json + comparison.json
python scripts/dev_eval_run.py --data Datasets --output outputs/dev_eval_<RUN_ID>
```

对比的配置固定为：

| 配置 | 含义 |
|---|---|
| `A_zero_physical` | 基线 A：原始物理空间零位移 `D(p) = 0` |
| `B_zero_model_identity` | 基线 B：模型缩放网格 `phi=0` 转原始空间后的映射 |
| `C1_current` | `max_iter=200, lr=0.1, tv=0.01, mse`（当前配置） |
| `C2_tv005` | `tv=0.05` |
| `C3_tv005_lr005` | `tv=0.05, lr=0.05` |
| `C4_iter100` | `max_iter=100` |

另有两个**可选**的刚性预对齐配置（默认不在固定集合内，需显式 `--configs` 指定）：

| 配置 | 含义 |
|---|---|
| `P1_prealign` | 仅刚性配准（图像驱动 Mattes MI，不读标签） |
| `P2_prealign_C1` | 刚性预对齐后，再按 C1 参数做形变优化 |

要点：

- **A 与 B 不是同一个东西**。CT 与 MR 的尺寸 / spacing / origin / direction
  不同时，B 通常不为零（它包含了两者之间的几何映射）。
- **DSC30 向上取整**：`ceil(total × 0.3)`，24 例 → 8 例，8 例 → 3 例。
- 评分在**原始 CT 网格**上做，三个器官（liver / spleen / 合并双肾），
  CT 侧空器官记 1.0，与提交口径一致。
- 报告保留逐例结果、病例 ID、输入与标签 SHA256、配置、**实际生效参数**、
  代码 commit、运行错误与汇总；缺病例 / 缺标签 / 几何不匹配 / 运行失败
  会**直接报错**，不生成看似完整的汇总。
- `--split test` 会被拒绝：测试集 `025–032` 不用于配置选择。
- **结果只能称为开发集结果**。在开发集上选配置**不构成**独立验证。
- 「差异 < 0.01」只是**预先约定的工程阈值**，
  不能单凭它断言模型之间无实质差异；结论应基于**逐例配对差值**。

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
