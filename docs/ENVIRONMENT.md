# MedRegLab 环境说明（Linux / RTX 5090）

本文件记录本仓库在 `/data1/MedRegLab` 上的实际运行环境、已知坑和复现方式。

> 本文第 2–10 节主要保留 2026-10-28 环境迁移和旧版自检记录，不代表后续修改已在 RTX 5090 上重新验证。当前命令以 [README.md](../README.md) 为准；早期 64³ 输出不能据此认定符合原始 CT 网格提交规范。

---

## 1. 快速开始

```bash
cd /data1/MedRegLab
source scripts/activate_medreg_env.sh
python -m pytest tests/ -q
```

仅当目标 .venv 不存在且需要从零创建时：

```bash
bash scripts/setup_medreg_env.sh
```

---

## 2. 硬件与驱动

| 项 | 值 |
|---|---|
| GPU | 2 × NVIDIA GeForce RTX 5090 (32 GB) |
| 计算能力 | sm_120 (Blackwell) |
| 驱动 | 570.86.10 |
| 磁盘 | `/data1` 10 TB (8.5 TB 可用) / 根分区 97 GB (~8 GB 可用) |

---

## 3. 关键结论：必须用 cu128 的 PyTorch

系统 base 环境（`/data/env/data/env/anaconda`）里的 `torch 2.6.0+cu124`
**无法在本机 5090 上执行任何 CUDA kernel**：

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

因为 torch 2.6 编译时不含 `sm_120` SASS/PTX。`torch.cuda.is_available()`
仍返回 `True`，所以**这个错误容易被误判为"GPU 可用"**。

本仓库使用独立 venv：

| 包 | 版本 |
|---|---|
| Python | 3.12.4 |
| torch | 2.10.0+cu128 |
| torchvision | 0.25.0+cu128 |
| monai | 1.6.1 |
| SimpleITK | 2.5.6 |
| nibabel | 5.4.2 |
| timm | 1.0.30 |
| numpy / scipy | 1.26.4 / 1.18.1 |
| scikit-image | 0.26.0 |

已验证 GPU 计算正常：

```bash
.venv/bin/python -c "
import torch
x = torch.randn(2048, 2048, device='cuda', dtype=torch.float16)
print('gpu matmul OK', float((x @ x).sum()))"
```

---

## 4. 安装位置与磁盘注意事项

* venv 放在**仓库内**：`/data1/MedRegLab/.venv`（约 7 GB）。
  根分区只剩 ~8 GB，装不下 torch + CUDA 依赖。
* `setup_env.sh` 里把 `TMPDIR` 指向 `/data1/pipcache`，
  避免 pip 解包 wheel 时撑爆根分区。
* 安装过程中 pip 缓存曾占用 4.8 GB，可用 `pip cache purge` 释放。

---

## 5. HuggingFace 镜像（本机必需）

`huggingface.co` 直连**不通**（curl 30 s 超时、无响应）。
必须走国内镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1   # 关键
```

`HF_HUB_DISABLE_XET=1` 不能省：走 xet 协议时下载会在 10 MB 左右
**卡死不动**；关掉后 ViT-S 39 s、ViT-L 578 s 下载完成。

已缓存权重（`/data1/MedRegLab/.cache/huggingface/hub`）：

| 模型 | 大小 |
|---|---|
| `timm/vit_small_patch14_dinov2.lvd142m` | ~88 MB |
| `timm/vit_large_patch14_dinov2.lvd142m` | ~1.16 GB |

`activate_env.sh` 已自动设置 `HF_ENDPOINT`、`HF_HOME`（指向仓库内 `.cache/`）
与 `HF_HUB_DISABLE_XET=1`。权重放在 `/data1` 分区，避免占满根分区。

> **注意**：清理后仓库已不再包含任何 DINOv2 / IMPACT 相关代码
> （见第 6 节），因此 timm 与 HF 权重目前**不是运行必需**，
> 保留它们只是为了将来若要重建 DINO-Reg 方案时不必重新下载。
> 完整依赖里 `timm` 仍是可选依赖。

---

## 6. 本次修复的问题

以下问题在 Windows 环境下不可见，迁到 Linux 后逐一暴露：

### 6.1 硬编码 Windows 绝对路径

`E:\MedRegLab\...` 在 Linux 上必然失败。已改为基于仓库根目录解析：

| 文件 | 修改 |
|---|---|
| `configs/competition_config.yaml` | `data.train_dir` 等改为 `Datasets/train` 等相对路径 |
| `smoke_test_train.py` | `DATA_ROOT` / `OUTPUT_DIR` 改为 `Path(__file__).parent` 相对路径 |
| `medreglab/data/dataset.py` | 仅文档注释中的路径 |

> 6.2 – 6.6 记录的是 **DINOv2 / IMPACT 相关代码**（`dinov2_extractor.py`、
> `train_impact.py`）的问题。这些代码已在第 7 节清理中删除，
> 此处保留记录，便于将来重建该方案时避免重蹈覆辙。

### 6.2 DINOv2 `img_size` 不匹配（真实功能 bug）

timm 的 DINOv2 预训练权重默认 `img_size=518`，
而 `DINOv2Slices3D` 把切片缩放到 224×224，导致：

```
AssertionError: Input height (224) doesn't match model (518).
```

修复：`DINOv2Extractor` 新增 `img_size` 参数（默认 `(224, 224)`），
并在 `timm.create_model(..., img_size=self.img_size)` 中显式传入。
`DINOv2Slices3D` 改为沿用 `extractor.img_size`，不再硬编码。

### 6.3 DINOv2 懒加载模型留在 CPU

`DINOv2Extractor` 用 `self._model = None` 懒加载 timm 模型，
`.to(device)` 作用在 extractor 上时子模型还没创建，结果留在 CPU：

```
RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same
```

修复：新增 `_ensure_device(device)`，在 `forward` 中先触发懒加载再对齐设备。

### 6.4 `num_classes=0` 导致拿到池化特征而非 patch token

`model(x)` 在 `num_classes=0` 时返回 `[B, D]`（池化后），
但代码需要 `[B, N, D]`（patch token）来还原 2D 特征图。

修复：改用 `model.forward_features(x)`，返回 `[B, 1+N, D]`（含 1 个 CLS token），
并在 `_upsample_features` 与 `DINOv2Slices3D` 中去掉 CLS token
（判据：`N-1` 是完全平方数即为含 CLS）。

### 6.5 `train_impact.py` 中 `phi` 多了一维

`batch["mr"]` 是 `[B, 1, D, H, W]`，原代码用 `mr.shape[1:]` 得到 4 个尺寸，
拼出 6 维 `phi`，与 `warp_volume` 期望的 5 维 `[B, 3, D, H, W]` 不符：

```
RuntimeError: permute(sparse_coo): number of dimensions in the tensor input
does not match the length of the desired ordering of dimensions
```

修复：改用 `mr.shape[2:]`。

### 6.6 灰度切片通道数

DINOv2 的 patch embed 是 3 通道 `nn.Conv2d`，
`DINOv2Slices3D` 直接把 1 通道切片喂进去会报通道不匹配。
修复：在切片 reshape 后把 1 通道复制为 3 通道。

### 6.7 测试断言漏了通道维

`tests/test_medreglab.py` 断言 `jac.shape == (1, 63, 63)`，
但 `JacobianField._compute_2d` 实际返回 `[B, 1, H-1, W-1]`。
实现与所有调用方（`train_competition.py`、`infer_competition.py`，
均只用 `.mean()/.var()/.sum()`）都是自洽的，
因此修正的是**测试断言**，不是实现。

### 6.8 `ConvexAdam._init_identity_field` 缺 batch 维与 `requires_grad`

返回的是 `(3, D, H, W)`，但下游全部按 `(B, 3, D, H, W)` 使用；
且张量 `requires_grad=False`，`torch.autograd.grad` 必然失败：

```
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
```

同时 `_warp_image` 里的 `phi.permute(0, 1, 3, 4, 2)` 只在 5D 成立，
4D 输入会报 `permute(sparse_coo): input.dim() = 3 is not equal to len(dims) = 4`。

**后果**：`ConvexAdam` 在 2D 与 3D 下均完全不可用。

### 6.9 `DINOv2Extractor.forward` 只支持 2D

输入 5D 时 `x.repeat(1, 3, 1, 1)` 报
`Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor`。

**后果**：`DINORegModel` 家族（`dino_reg_mrct` / `impact_mrct` / `convex_mrct`）
在 2D 与 3D 下**全部崩溃**。注意 `convex_mrct` 名字有误导性——
它同样挂了 `DINOv2Extractor`，并非纯 ConvexAdam。

### 6.10 `infer_competition.py` 默认模型指向已崩溃的 `dino_reg_mrct`

原默认 `model_name="dino_reg_mrct"`，实测 **8/8 样例全部失败**，
一个变形场都没产出。

### 6.11 `uniGradICON` 通道数不匹配

`GradientComputationModule.matching_layer` 期望拼接后 512 通道
（`feature_dim * 2`），但 `MultiScaleFeatureExtractor` 末层输出只有 256 通道：

```
RuntimeError: Given groups=1, weight of size [256, 512, 1, 1],
expected input[1, 256, 16, 16] to have 512 channels
```

### 6.12 `train_impact.py` 的对比损失恒为 0

`ContrastiveHead` 计算 `cross_entropy(similarity[B, B], labels)`，
`batch_size=1` 时 softmax 只有一个类别，loss 恒等于 0。
实测 `{'semantic': 0.000163, 'contrastive': 0.0, 'adversarial': 0.0}`。
所谓"训练"实际只优化了 semantic encoder 的 MSE 项。

### 6.13 数据增强 `_apply_displacement` 的 grid 维度错误

网格被构造成 `[3, D, H, W]` 后接 `permute(0, 3, 4, 1, 2)`（5 维），
必然报 `input.dim() = 4 is not equal to len(dims) = 5`。

**修复**：改为与 `convex_3d_working.Simple3DRegistration._warp` 完全一致的
约定（`F.grid_sample` 3D 要求 grid 为 `[B, D, H, W, 3]`，最后一维是 `(x, y, z)`）。
修复后与 `convex_3d._warp` 逐元素比对，三个轴向的最大误差均为 `0.00e+00`。

### 6.14 CT/MR 数据 dtype 不一致

`np.percentile` 返回 `float64`，把 MR 数组整体提升为 float64，
而 CT 仍是 float32。两者在 `grid_sample` 中混合会报：

```
RuntimeError: expected scalar type Double but found Float
```

**修复**：MR 归一化后显式 `arr.astype(np.float32)`；
`_apply_displacement` 中额外做 `warped_grid.to(dtype=image.dtype)` 兜底。

### 6.15 `DataLoader` 无法 collate SimpleITK 图像对象

`CompetitionDataset` 的 sample 里带 `fixed_sitk` / `moving_sitk`，
默认 collate 报 `found <class 'SimpleITK.SimpleITK.Image'>`。
训练循环并不需要这些元信息。

**修复**：`experiments/train_competition.py` 增加 `_collate_training`，
丢弃 SITK 对象并把 `seg_moving` 统一到 `segmentations`（供 DSC 计算用）。

### 6.16 训练/验证阶段张量维度

数据集每样本已是 `[1, C, D, H, W]`，`DataLoader` 再加一维会变成 6D，
而 `Simple3DRegistration.forward` 期望 5D。
`infer_competition.py` 原本已有 `squeeze(0)` 处理，训练侧缺失。
**修复**：训练与验证阶段统一压回 5D。

### 6.17 `convex_3d` 无可学习参数导致的训练崩溃

`torch.optim.Adam(model.parameters())` 遇空参数列表会抛
`ValueError: optimizer got an empty parameter list`；
验证阶段的 `torch.no_grad()` 也会让 `convex_3d` 内部的 autograd 优化失效。

**修复**：训练脚本按 `trainable_params` 是否为空决定是否建优化器与反向传播；
验证阶段改用 `torch.enable_grad()`；无参数时不再保存空 checkpoint。

### 6.18 标签形变的坐标轴顺序与归一化除数（曾使 DSC 被严重低估）

> 本节恢复自提交 `336047c`（备份：`backup/local-master-336047c`）。
> 缺陷本身已用合成数据**复算确认**（见 [EVIDENCE_BASELINE.md §3.2](EVIDENCE_BASELINE.md)）。
> 提交信息中「identity 0.15 → 0.40」这一对具体数字**未能复现**，属待核实。

DSC 计算走 `infer_competition._apply_deformation_to_label`，
它把 moving 分割按 `phi` 形变到 fixed 空间。该函数与参考实现
`convex_3d_working.Simple3DRegistration._warp` 有两处不一致：

| 项 | 错误写法 | 正确写法 |
|---|---|---|
| grid 最后一维 | `(D, H, W)` | `(W, H, D)` = `(x, y, z)` |
| 归一化除数 | `x/D, y/H, z/W` | `x/W, y/H, z/D` |

`F.grid_sample` 的 3D 约定要求 grid 为 `[B, D, H, W, 3]` 且最后一维是
`(x, y, z) = (W, H, D)`。写反会让 **D 与 W 轴互换**。

**复算结果**（identity 位移场下，正确实现的 Dice 必然为 1.0）：

| 网格 | 器官 | 错误版 Dice | 修复版 Dice |
|---|---|---|---|
| 64³（真实 mask） | liver / spleen / kidney | 0.5756 / **0.0000** / 0.0443 | 1.0 / 1.0 / 1.0 |
| 原生 CT 192×160×192 | liver | 0.5811 | 1.0 |

即该缺陷**在立方体网格上同样成立**，并非只在非立方体数据上出现。
（用「居中立方体」造出的对称 mask 测试会漏掉这个 bug，
必须用**非对称**的真实器官形状才能暴露。）

**影响**：修复前的 DSC 数字被测量方式严重低估，并非模型效果。

**修复**（提交 `a2226a4`）：`infer_competition._apply_deformation_to_label` 与
`train_competition._warp_label` 同步改正；新增回归测试
`tests/test_medreglab.py::test_label_warp_axis_order` 逐轴验证形变方向。

> ⚠️ **当前状态**：该缺陷在 `6933cfe` 中**已修复**
> （两处 warp 均改为 `z, y, x = torch.meshgrid(...)` +
> `torch.stack((x, y, z), dim=-1)`，除数 `2/(width-1), 2/(height-1), 2/(depth-1)`）。
> `336047c` 引入的 `test_label_warp_axis_order` 虽被上游重写时删除，
> 但**逐轴验证能力已由上游新用例覆盖**：
> `tests/test_inference_contract.py::test_label_warp_identity_and_axis_translations`
> 用逐体素递增的非对称 label 与非立方网格 4×5×6，
> 断言 identity 与 x/y/z 三方向平移；
> `tests/test_train_pipeline.py::test_label_warp_identity_translation_and_outside_use_xyz_axes`
> 覆盖 `train_competition._warp_label` 的同类行为。
> 实测把错误版 warp 代入这些断言后**全部失败**，
> 即可确认它们能捕获该缺陷。详见
> [EVIDENCE_BASELINE.md §3.5](EVIDENCE_BASELINE.md)。

**本轮补充的物理空间回归测试**：
`tests/test_inference_contract.py::test_native_dsc_resamples_each_physical_axis_to_the_expected_voxel`
把原先只覆盖 **x 方向**的物理空间评分测试扩展为 x/y/z 三个方向，
并覆盖**立方等距网格**、**非立方非等距 spacing**、**非单位方向矩阵**三种几何，
使用**非对称**合成标签，断言**预期重采样体素位置**，
同时断言沿**错误**物理轴拉取**不能**复现解剖。

---

## 7. 清理记录：删除了什么、为什么

### 7.1 已删除的代码（均因不可用或无人引用）

| 文件 | 原因 |
|---|---|
| `medreglab/core/convex_optimizer.py` | `ConvexAdam` 2D/3D 均崩溃（6.8） |
| `medreglab/core/registration_engine.py` | 依赖 `ConvexAdam`，3D/2D 均崩溃 |
| `medreglab/models/reg_model.py` | `DINORegModel` 全家族崩溃（6.9、6.10） |
| `medreglab/models/dinov2_extractor.py` | 仅 2D 可用，且 3D 崩溃 |
| `medreglab/modules/impact_loss.py` | 仅服务于已删除的 `DINORegModel` |
| `medreglab/modules/gradicon.py` | 通道数 bug，且无任何预设会用到 |
| `medreglab/modules/convex_regularizer.py` | 仅服务于已删除的路径 |
| `medreglab/utils/visualization.py` | 仅服务于已删除的 `train.py`/`eval.py`/`infer.py` |
| `experiments/train.py`、`eval.py` | `ImportError`（引用不存在的 `RegistrationDataset`） |
| `experiments/infer.py` | 仅支持单对文件，且默认模型已崩溃 |
| `experiments/train_impact.py` | 对比损失恒为 0，训练信号退化（6.12） |
| `configs/l2r2021_convex.yaml`、`l2r2023_dino_reg.yaml`、`test_config.yaml` | 指向不存在的 `data/l2r*` 路径，无人引用 |

### 7.2 已删除的产物

- `competition_outputs_all/`（2.3 GB，24 例变形场 + warped 图，Windows 时代产物，路径写死 `E:\MedRegLab`）
- `outputs/train_smoke*`（4.4 MB 调试残留）
- 各处 `__pycache__`、`.pytest_cache`、`medreglab.egg-info`、空的 `medreglab/modules/`

### 7.3 清理后现状

**唯一可用模型是 `convex_3d`**（`Simple3DRegistration`）：
逐图优化（Adam on `phi`）+ MSE/NCC + TV 正则，**没有可学习参数**。

`load_model()` 现在只接受 `"convex_3d"`，其余名字会抛 `ValueError`。
`train_competition.py` / `infer_competition.py` 的默认模型已同步改为 `convex_3d`。

### 7.4 备份

删除前已打包两份备份（均在 `/data1`）：

- `/data1/MedRegLab_code_backup_20261028.tar.gz` — 全部代码（含未提交的修复）
- `/data1/MedRegLab_untracked_backup_20261028.tar.gz` — 未纳入 git 的新文件

git 远端 `https://github.com/LDT-von/MedRegLab.git` 保留了初始提交 `67b7fc4`。

---

## 8. 已验证可用的功能

| 检查 | 结果 |
|---|---|
| `pytest tests/` | **6 passed** |
| `import medreglab` | OK，`__all__` 全部可导入 |
| `medreglab.modules` | 已删除，import 正确报 `ModuleNotFoundError` |
| `load_model("convex_3d")` | OK；`load_model("dino_reg_mrct")` 正确抛 `ValueError` |
| 数据集 train split | 24 例配对，CT=fixed / MR=moving，标签可加载，dtype 一致 |
| 数据集 test split | 8 例（025–032） |
| 数据集 unpaired_ct / unpaired_mr | 50 / 40 例 |
| 3D 配准 `convex_3d`（GPU） | 64³ 用时 1.7 s，loss 0.0217 |
| 完整推理 `infer_competition.py` | 8/8 成功，0 错误 |
| 变形场输出 | 合法 3 分量 NIfTI，`[64,64,64,3]` float32 |
| 数据增强 + DSC/DSC30 评估器 | 正常，DSC30 取最低 30% 病例均值（病例数**向上取整**：`ceil(total×0.3)`，24 例→8 例，8 例→3 例） |
| `train_competition.py` 训练 | 1 epoch 跑通 24 例，产出 history.json |
| `selfcheck.sh` | **9/9 全部通过** |

**注意**：`convex_3d`（`Simple3DRegistration`）**没有可学习参数**，
它是逐图优化（Adam on `phi`）。因此 `train_competition.py` 在该模型下
只做「配准 + 评估」，不会更新网络参数或保存 checkpoint；逐例位移场仍会通过梯度优化更新——
这是预期行为，不是错误。

---

## 9. 数据去重记录

`Datasets/` 下共 572 个文件，其中 **278 个是浏览器下载重名副本**
（`X(1).nii.gz` 与 `X.nii.gz` 字节完全相同，md5 一致）。
已按"同目录下存在去掉 `(N)` 后缀的同名文件且 md5 相同"这一条件
逐个校验后删除，**释放 4.26 GB**（8.7 GB → 4.4 GB）。

删除前后文件数：572 → 294。数据内容未做任何改动。

---

## 10. 目录结构

```
/data1/MedRegLab/
├── .venv/                 # Python 3.12 + torch 2.10.0+cu128（约 7 GB）
├── activate_env.sh        # 激活环境
├── setup_env.sh           # 从零重建环境
├── selfcheck.sh           # 9 项交付前自检
├── Datasets/              # train(24) / test(8) / extra(50 CT + 40 MR)，已去重
├── medreglab/             # 核心包
│   ├── core/              # convex_3d_working.py / loss_builder.py
│   ├── models/            # deformation_field.py
│   ├── data/              # dataset.py / transforms.py / sampler.py
│   ├── evaluation/        # dsc_evaluator.py
│   └── utils/             # metrics.py / io_utils.py
├── experiments/           # train_competition.py / infer_competition.py
├── configs/               # competition_config.yaml
├── outputs/               # 运行产物（gitignore）
└── tests/                 # 6 个测试
```

`.venv/`、`.cache/`、`Datasets/`、`outputs/`、`competition_outputs_all/`
均已加入 `.gitignore`，不会被误提交。
