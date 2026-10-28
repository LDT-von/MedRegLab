# MedRegLab 环境说明（Linux / RTX 5090）

本文件记录本仓库在 `/data1/MedRegLab` 上的实际运行环境、已知坑和复现方式。

---

## 1. 快速开始

```bash
source /data1/MedRegLab/activate_env.sh   # 激活 .venv
cd /data1/MedRegLab
python -m pytest tests/ -q                 # 6 passed
```

从零重建环境：

```bash
bash /data1/MedRegLab/setup_env.sh
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

---

## 6. 本次修复的问题

以下问题在 Windows 环境下不可见，迁到 Linux 后逐一暴露：

### 6.1 硬编码 Windows 绝对路径

`E:\MedRegLab\...` 在 Linux 上必然失败。已改为基于仓库根目录解析：

| 文件 | 修改 |
|---|---|
| `configs/competition_config.yaml` | `data.train_dir` 等改为 `Datasets/train` 等相对路径 |
| `smoke_test_train.py` | `DATA_ROOT` / `OUTPUT_DIR` 改为 `Path(__file__).parent` 相对路径 |
| `experiments/train_impact.py` | `CT_DIR` 改为 `REPO_ROOT / "Datasets" / "train"`；`--output` 默认值同理 |
| `medreglab/data/dataset.py` | 仅文档注释中的路径 |

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
实现与所有调用方（`train_competition.py`、`train_impact.py`、`eval.py`、
`infer_competition.py`，均只用 `.mean()/.var()/.sum()`）都是自洽的，
因此修正的是**测试断言**，不是实现。

---

## 7. 已验证可用的功能

| 检查 | 结果 |
|---|---|
| `pytest tests/` | **5 passed** |
| `import medreglab` | OK，`__all__` 全部可导入 |
| 数据集 train split | 24 例配对，CT=fixed / MR=moving，标签可加载 |
| 数据集 test split | 8 例（025–032） |
| 数据集 unpaired_ct / unpaired_mr | 50 / 40 例 |
| 3D 配准 `convex_3d`（GPU） | 64³ 用时 1.7 s，loss 0.0217 |
| 完整推理 `infer_competition.py` | 8/8 成功，0 错误 |
| 变形场输出 | 合法 3 分量 NIfTI，`[64,64,64,3]` float32 |
| 数据增强 + DSC/DSC30 评估器 | 正常，DSC30 取最低 30% 病例均值 |
| `train_impact.py` 训练 | 1 epoch 完成，产出 checkpoint + history.json |

**注意**：`convex_3d`（`Simple3DRegistration`）**没有可学习参数**，
它是逐图优化（Adam on `phi`），因此**不能**用于 `train_competition.py`
（该脚本会构造 `torch.optim.Adam(model.parameters())`，
空参数列表会抛 `ValueError: optimizer got an empty parameter list`）。
`train_competition.py` 需用 `dino_reg_mrct` / `impact_mrct`。
`train_impact.py` 只训练 IMPACT 的 558,464 个参数（DINOv2 冻结）。

---

## 8. 数据去重记录

`Datasets/` 下共 572 个文件，其中 **278 个是浏览器下载重名副本**
（`X(1).nii.gz` 与 `X.nii.gz` 字节完全相同，md5 一致）。
已按"同目录下存在去掉 `(N)` 后缀的同名文件且 md5 相同"这一条件
逐个校验后删除，**释放 4.26 GB**（8.7 GB → 4.4 GB）。

删除前后文件数：572 → 294。数据内容未做任何改动。

---

## 9. 目录结构

```
/data1/MedRegLab/
├── .venv/                 # Python 3.12 + torch 2.10.0+cu128（约 7 GB）
├── activate_env.sh        # 激活环境
├── setup_env.sh           # 从零重建环境
├── Datasets/              # train(24) / test(8) / extra(50 CT + 40 MR)，已去重
├── competition_outputs_all/  # 历史推理产物（变形场 1.8 GB + warped 457 MB）
├── medreglab/             # 核心包
├── experiments/           # train_competition / train_impact / infer_competition / eval
├── configs/               # competition_config.yaml 等
├── outputs/               # 运行产物（gitignore）
└── tests/                 # 5 个测试
```

`.venv/`、`.cache/`、`Datasets/`、`outputs/`、`competition_outputs_all/`
均已加入 `.gitignore`，不会被误提交。
