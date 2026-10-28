# 效果基线与历史缺陷：证据核查记录

本文件记录对历史实测数字（identity / `convex_3d` 的 mean DSC）与历史缺陷
（标签形变坐标轴顺序）的**证据核查**结果，以及下一轮验证方案。

核查日期：2026-10-28。核查方式：只读检查 git 备份对象、磁盘残留物、
会话记录与合成数据复算。

> **结论先行**
>
> 1. `0.396` 与 `0.400` 两个数字**只能追溯到文档文字本身**。
>    产生它们的运行产物（逐例结果、配置、日志）**已不存在**，
>    所在会话的记录也不在本机。因此这两个数字目前**无法复核**，
>    只能作为「历史记录」引用，不能当作已验证的独立验证集/测试集结果。
> 2. 6.18 所述的坐标轴缺陷**是真的**，且比我此前理解更严重：
>    它在**立方体网格（64³）上同样会把 identity 形变的 DSC 打到接近 0**。
>    这一点已用合成数据复算确认。
> 3. 历史上写下的「identity 0.15 → 修复后 0.40」这一对数字，
>    其**机制方向**与复算一致，但**具体数值未复现**，仍属待核实。

---

## 1. 证据来源清单

| # | 证据 | 位置 | 状态 |
|---|---|---|---|
| E1 | 文档基线文字（README + ENVIRONMENT 第 9 节） | commit `336047c` | ✅ 存在 |
| E2 | 6.18 轴序缺陷分析文字 | commit `336047c`（引用于 `a2226a4`） | ✅ 存在 |
| E3 | 轴序修复的代码 diff | commit `a2226a4` | ✅ 存在 |
| E4 | 轴序回归测试 `test_label_warp_axis_order` | commit `a2226a4` | ✅ 存在（上游 `6933cfe` 已**删除**） |
| E5 | 备份分支 / 标签 | `backup/local-master-336047c`、`backup/pre-sync-336047c` | ✅ 存在 |
| E6 | 历史配置 `configs/competition_config.yaml` | commit `336047c` | ✅ 存在 |
| E7 | 历史 DSC 计算代码 | `336047c:experiments/infer_competition.py` | ✅ 存在 |
| E8 | 历史数据集加载器 | `336047c:medreglab/data/dataset.py` | ✅ 存在 |
| E9 | **原始逐例 DSC 结果文件** | — | ❌ **不存在** |
| E10 | **产生 0.396/0.400 的运行日志** | — | ❌ **不存在** |
| E11 | **原始运行输出目录** | `competition_outputs*` | ❌ **不存在**（已被 `git reset` 后的工作区覆盖，且 `outputs/` 被 gitignore） |
| E12 | 产生该数字的会话记录 | — | ❌ **不在本机** |

### 1.1 产物缺失的排查过程

- `git fsck --lost-found`：无游离对象。
- `git stash list`：空。
- `git log --all --diff-filter=A`：整个历史中从未提交过任何结果 JSON / 日志。
- 磁盘全盘搜索 `*.json` / `*.log` / `history*` / `*result*` / `*dsc*`：
  仓库内只有源码，无运行产物。
- `competition_outputs_all/`（`.gitignore` 中列出的历史目录）：不存在。
- `/data1/MedRegLab_code_backup_20261028.tar.gz`、
  `/data1/MedRegLab_untracked_backup_20261028.tar.gz`：只含**源码**，无结果文件。
- 会话记录（`agent-transcripts`）：仅两个，均**不覆盖**测量时段
  （`ffadef55` 结束于 05:49，`47a25cd4` 开始于 06:47），
  测量发生在 05:49–06:00 之间，其会话记录不在本机。
- `.pytest_cache/v/cache/nodeids`：仅 7 个测试名，无实验痕迹。

**因此 `0.396` / `0.400` 无法复核。** 下文一律标注「待核实」。

---

## 2. 基线数字 0.396 / 0.400 的核查

### 2.1 文档原始表述（E1，逐字）

`336047c:README.md`：

> 在 `Datasets/train` 前 6 例、64³ 分辨率下测得：
>
> | 配置 | mean DSC (liver/spleen/kidney) |
> |---|---|
> | identity（零变形场 = 什么都不做） | 0.396 |
> | `convex_3d`（200 iter） | 0.400 |

`336047c:docs/ENVIRONMENT.md` 第 9 节（标题「实际效果基线（诚实记录）」）额外列出：

| 配置 | mean DSC |
|---|---|
| identity（零变形场，即什么都不做） | **0.396** |
| `convex_3d`（100 iter） | 0.430 |
| `convex_3d`（200 iter, lr=0.1） | **0.400** |

并写有「当前模型相对"什么都不做"只有约 +0.4% 的提升，实质上等同基线」。

### 2.2 逐项核查结果

| 核查项 | 结论 | 依据 |
|---|---|---|
| 具体代码版本 | **可确定**：`336047c` 树（父提交 `a2226a4` 已含轴序修复） | E1/E3 |
| 数据集划分 | **文档自述为** `Datasets/train` 前 6 例，**但无法核实** | E1，无产物 |
| 病例数量 | **文档自述为 6 例，病例 ID 待核实** | E1 |
| 病例 ID | **待核实**（文档未列 ID） | — |
| 两组是否同病例/同标签/同预处理 | **待核实**（无产物） | — |
| 数值来自原始空间还是缩放网格 | **缩放网格**（文档自述 64³；历史代码亦在缩放网格上算） | E1/E7/E8 |
| 轴序缺陷是否影响这两组 | **是，且影响极大**（见 §3） | 复算 |
| 原始逐例结果 / 配置 / 日志 | **均不存在** | E9/E10/E11 |

### 2.3 可以支持什么

- 「在 `336047c` 所记录的代码版本下，曾在 `Datasets/train` 的前若干例、
  64³ 缩放网格上，测得 identity 与 `convex_3d` 的 mean DSC 分别为
  0.396 与 0.400」——**这一句可引用**，因为它复述的是文档记载，
  不是本次新做的测量。
- 「两者相差 0.004」——可引用（纯算术）。

### 2.4 不能支持什么

- ❌ **不能**称这两个数字是「独立验证集」或「测试集」结果。
  历史代码在 `split="val"` 时，若不存在 `Datasets/val/` 目录，
  会**回退到 `train/`** 并发出 warning
  （`336047c:medreglab/data/dataset.py:98-107`）：

  ```python
  elif self.split in ("val", "validation"):
      val_dir = self.root / "val"
      if val_dir.exists():
          samples.extend(self._scan_paired_dir(val_dir))
      else:
          warnings.warn("No val/ directory; using train/ as val.")
          train_dir = self.root / "train"
          if train_dir.exists():
              samples.extend(self._scan_paired_dir(train_dir))
  ```

  而 `Datasets/` 下**只有** `train/`、`test/`、`extra/`，**没有 `val/`**。
  历史配置 E6 中也明写 `val_dir: "Datasets/train"`。
  因此若当时走的是 `--split val`，它**实际就是训练集本身**。
- ❌ **不能**说「模型比什么都不做提升了 0.004，因此有效」。
  0.396 → 0.400 的差值落在单次测量的噪声范围内，
  且缺少逐例结果与重复测量，无法做显著性判断。
  正确表述为：**相差 0.004**。
- ❌ **不能**说这些数字来自原始 CT/MR 空间。它们来自 64³ 缩放网格，
  与比赛要求的原始 CT 网格不是同一空间。
- ❌ **不能**把它们与 `6933cfe` 生成的提交包直接比较——
  评分空间（缩放网格 vs 原始 CT 网格）、器官集合定义、
  双肾合并方式全部不同。

### 2.5 历史评分代码与当前实现的差异（重要）

历史与当前的 DSC 计算**不是同一套口径**：

| 方面 | 历史（`336047c`） | 当前（`6933cfe`） |
|---|---|---|
| 计算空间 | 缩放网格（`target_size`，如 64³/128³） | 原始 CT 网格（物理 mm 位移场 + SimpleITK 重采样） |
| 器官集合 | `["liver","spleen","kidney_left","kidney_right"]`（**4 项**） | `["liver","spleen","kidney"]`（**3 项**，双肾合并） |
| 平均方式 | 4 个器官的 DSC 取均值 | 3 个器官的 DSC 取均值 |
| 双肾 | 左右肾**分别**评分 | `labels==3 \| labels==4` **合并**为一个 kidney |
| 空器官 | 缺失时该项记 0 | CT 中该器官为空 → 该项记 **1.0** |
| CT 侧标签来源 | `seg_fixed`（缩放后的张量） | 原始 `seg*_CT.nii.gz` 文件 |

历史 `infer_competition._compute_dsc_for_sample` 里的双肾合并分支
（`if organ == "kidney"`）**永远不会触发**，因为传入的
`organ_labels` 来自 `ORGANS_FOR_DSC`（值为 `kidney_left`/`kidney_right`），
不含 `"kidney"`。所以历史数字确实是**左右肾分别计分**的。

结论：历史数字与当前实现**不可直接对比**。这一点在补回文档时会写明。

---

## 3. 6.18 轴序缺陷：核查与复算

### 3.1 缺陷本身（E3，真实且已修复）

`a2226a4` 修改了 `experiments/infer_competition.py::_apply_deformation_to_label`
与 `experiments/train_competition.py::_warp_label` 两处：

| 项 | 错误写法 | 正确写法 |
|---|---|---|
| grid 最后一维 | `(D, H, W)` | `(W, H, D)` = `(x, y, z)` |
| 归一化除数 | `x/D, y/H, z/W` | `x/W, y/H, z/D` |

`F.grid_sample` 的 3D 约定要求 grid 为 `[B, D, H, W, 3]` 且最后一维是
`(x, y, z) = (W, H, D)`。写反会让 **D 与 W 轴互换**。

### 3.2 复算（合成数据 + 真实标签，只读）

复算脚本逻辑：把「错误版」与「修复版」两个 warp 函数并排，
在 **identity 位移场**（`phi = 0`）下对同一 mask 求 Dice。
identity 场下正确实现必然给出 Dice = 1.0，因此任何 < 1 的结果
都是缺陷造成的。

**（a）立方体网格 64³，真实器官 mask（`Datasets/train/001`，CT 标签）**

| 器官 | 错误版 identity Dice | 修复版 identity Dice |
|---|---|---|
| liver | 0.5756 | 1.0000 |
| spleen | **0.0000** | 1.0000 |
| kidney | 0.0443 | 1.0000 |

**（b）原生 CT 网格（`001` 号病例，192×160×192，liver）**

| 器官 | 错误版 | 修复版 |
|---|---|---|
| liver | 0.5811 | 1.0000 |

**（c）原始 CT 体素间距与立方体缩放的差别**：错误版把 grid 的
最后一维按 `(D, H, W)` 排列，等价于把「z 坐标」喂给了 `x` 分量、
把「x 坐标」喂给了 `z` 分量。轴互换在**任何**网格上都发生，
与是否立方体无关；立方体只让**数值幅度**看起来没那么极端
（错误版在对称立方体上的某些对称 mask 上可能偶然接近正确）。

> 注意：早期用「居中立方体 mask」的合成测试会**低估**该缺陷
> （对称形状下错误版偶然给出 Dice ≈ 1.0）。用**真实器官形状**复算才暴露出来。
> 本文引用的数字均来自真实器官 mask。

### 3.3 对历史 0.396 / 0.400 的影响

- 历史基线是在 `a2226a4` **之后**测的（`336047c` 是 `a2226a4` 的子提交），
  所以 **0.396 / 0.400 这两个数字本身不含该缺陷的影响**。
- 但缺陷**确实**存在于 `a2226a4` 之前的实现中；
  `a2226a4` 提交信息里的「identity 基线 DSC 从 0.15 提升到 0.40」
  描述的是**修复前后的对比**，方向与 §3.2 复算一致
  （缺陷会把 identity 打到接近 0），但**具体数值 0.15 / 0.40 未复现**，
  属**待核实**。
- 复算表明缺陷的影响幅度**强烈依赖网格形状与 mask 形状**
  （本例中 spleen 被打到 0.0000，liver 为 0.58），
  因此**不能**用复算结果去推断或替代历史的具体数值。

### 3.4 已修复 / 仍待确认

**已修复（当前 `6933cfe`）：**

- 两处 warp 的轴序与归一化除数已改正（`6933cfe` 中
  `experiments/infer_competition.py::_apply_deformation_to_label`、
  `experiments/train_competition.py::_warp_label`，
  均为 `z, y, x = torch.meshgrid(...)` + `torch.stack((x, y, z), dim=-1)`，
  除数为 `2/(width-1), 2/(height-1), 2/(depth-1)`）。
- 上游 `6933cfe` 还把 DSC 评分整体迁到**原始 CT 网格**，
  用物理 mm 位移场 + `sitk.Resample`，
  与提交包走的是**同一条路径**（`experiments/infer_competition.py::_compute_dsc_for_sample`）。
- 双肾合并、空器官记 1.0、哈希校验等均已在 `6933cfe` 中实现。

**仍待确认：**

- `336047c` 引入的回归测试 `tests/test_medreglab.py::test_label_warp_axis_order`
  在上游 `6933cfe` 中**被删除**（该文件在 `6933cfe` 里仍有其他 6 个测试，
  但轴序那条不在其中）。当前 `6933cfe` 的测试集中在
  `tests/test_field_regression.py` 等文件，**未见逐轴验证形变方向的用例**。
  → 见 §6 的建议。
- 历史 0.396 / 0.400 无产物，永久无法复核。

---

## 4. 当前实现的已知风险点（供下一轮参考）

本轮 `6933cfe` 在 8 例测试集上实际运行的结果
（`outputs/test_20261028_065511/inference_summary.json`）：

| case | loss | fold_ratio | min_jacobian | fixed_size_xyz |
|---|---|---|---|---|
| 025 | 0.035940 | 0.0943 | **-102.56** | 512×512×45 |
| 026 | 0.015097 | 0.0529 | -42.27 | 512×512×43 |
| 027 | 0.053895 | 0.1124 | -42.21 | 512×512×109 |
| 028 | 0.022047 | 0.0715 | -48.01 | 512×512×45 |
| 029 | 0.022159 | 0.0765 | -51.03 | 512×512×46 |
| 030 | 0.027047 | 0.0821 | -61.46 | 512×512×130 |
| 031 | 0.027098 | 0.0739 | -46.36 | 512×512×142 |
| 032 | 0.010245 | 0.0412 | -42.41 | 512×512×35 |

- `fold_ratio`（`det(I + ∂u/∂x) ≤ 0` 的体素占比）为 **4.1%–11.2%**，
  `min_jacobian` 低至 **-102.6**。
- 该指标由 `medreglab/utils/submission.py::grid_folding_statistics` 计算，
  是**诊断指标**，不是官方得分项。
- 折叠意味着位移场不再是单射，重采样会把器官拉成非物理形状，
  通常**不利于** DSC 与 DSC30。但**不能**据此断言
  「增大 TV 权重就一定提高 DSC」——TV 与 DSC 的关系需要实测，
  见 §5。

---

## 5. 下一轮验证方案（未执行，仅方案）

### 5.1 目标

在**不使用隐藏测试标签**的前提下，量化：
1. identity 基线在**原始 CT 网格**上的 mean DSC / DSC30；
2. 当前配置（`max_iter=200`, `lr=0.1`, `tv_weight=0.01`, `similarity=mse`）的表现；
3. 少量**预先确定**的正则化配置对 DSC 与折叠率的影响。

### 5.2 病例划分

- **不新建**独立验证集：`Datasets/` 下没有 `val/`，
  当前代码在 `--split val` 时会**直接报错**
  （`6933cfe:medreglab/data/dataset.py:135-140`：
  `raise FileNotFoundError(f"Validation directory not found: {val_dir}")`）。
  这与历史的「静默回退到 train」不同，是一个**改进**。
- 本方案使用 **`Datasets/train` 中带完整 CT+MR 标签的病例**，
  作为**开发集（development set）**。
  所有配置在**同一批病例**上评估，配置之间**逐例配对比较**。
- ⚠️ 这**不是**独立验证集，也不是测试集。测试集 `025–032` **无标签**，
  DSC/DSC30 仍须主办方用隐藏标签评分。
- ⚠️ 因此**不得**用开发集上的 DSC 反向挑选提交用的正则化配置后再声称
  「验证通过」。挑选配置本身就会在开发集上过拟合。
  建议：先用开发集**诊断**（含折叠率），选定 1–2 个配置后，
  再生成一份新的提交包交主办方评分。

### 5.3 病例清单（须由使用者确认后固定）

`Datasets/train` 24 例中，CT 标签的取值分布为：

- `001`、`002`：标签 `{0,1,2,3}`（只有一个肾标签）
- `003`–`008`：标签 `{0,1,2,3,4}`（双肾分开）
- `009`–`024`：主办方格式

按 `6933cfe` 的 `competition_masks_from_labels`：
编号病例 **001–008** 双肾合并（`labels==3 | labels==4`）；
编号 ≥ 009 的病例若出现标签 4 会**直接报错**。

> ⚠️ **待核实**：`009`–`024` 的实际标签取值需在运行前用只读脚本确认
> （见 §5.6 的 `list_labels.py`）。若发现 `009+` 存在标签 4，
> 当前 `competition_masks_from_labels` 会抛
> `Case {id} has unsupported label 4 under competition rules`，
> 需要先澄清规则再跑。

**建议的固定评估集合**（若 24 例全部带标签且标签分布合规）：
**全部 24 例**，即 `Datasets/train/001 … 024`。
理由：样本量最大，且 `DSC30` 取最低 30% 时
24 例 → 7 例（`floor(0.3*24)=7`），8 例 → 2 例。
样本更多时 DSC30 更稳定。

### 5.4 评估口径（必须与提交包一致）

- **原始 CT 网格**：用 `6933cfe` 的 `infer_competition.py --split val` 路径，
  它直接复用提交用的物理位移场与 `sitk.Resample`，
  评分与提交**同源**。
- **器官集合**：`liver`、`spleen`、`kidney`（**双肾合并**，共 3 项）。
- **双肾合并**：`competition_masks_from_labels`，
  编号 001–008 → `labels==3 | labels==4`；≥009 → `labels==3`。
- **空器官**：`6933cfe:medreglab/evaluation/dsc_evaluator.py::dice_coefficient`
  中，**CT 侧该器官为空 → 该项记 1.0**（含 warped 侧出现假阳性的情形）。
  注意这与历史实现（记 0）不同。
- **DSC 公式**：`2|A∩B| / (|A|+|B|)`，**不加平滑项**。
- **DSC30**：所有评估病例 mean DSC 中**最低 30%** 的均值。
- **identity 基线**：零位移场。这**不是**现成 CLI 选项，
  需要一个只做零形变的路径 —— 方案见 §5.6 的 `identity_baseline.py`。
  它同样走原始 CT 网格 + `sitk.Resample`，
  以保证与配准组**完全同口径**。

### 5.5 预先确定的配置清单

所有配置**共用**：同一批病例、同一标签文件、同一预处理
（CT HU 裁剪 `[-200,500]`、MR 百分位 `[0.5,99.5]`）、
同一 `target_size`、同一评分代码、同一随机种子 `42`。
`seed` 只影响数据增强顺序；`6933cfe` 的 `configs/competition_config.yaml`
已把增强全部关闭，故本轮**无增强**，结果应可复现。

| # | 配置 | `max_iter` | `lr` | `tv_weight` | `similarity` | 目的 |
|---|---|---|---|---|---|---|
| C0 | identity | 0（零位移场） | — | — | — | 下界基线 |
| C1 | 当前配置 | 200 | 0.1 | 0.01 | mse | 与已生成提交包**同参数**，可对照 |
| C2 | 更强 TV | 200 | 0.1 | **0.05** | mse | 观察折叠率是否下降、DSC 是否变化 |
| C3 | 更强 TV + 更低 lr | 200 | **0.05** | **0.05** | mse | 降低步长，减小过冲 |
| C4 | 更少迭代 | **100** | 0.1 | 0.01 | mse | 迭代数是否过冲 |

> 只做 **5 个配置**，且**在运行前固定**，不根据结果追加。
> ⚠️ **不假设**「TV 越大 DSC 越高」——这是待检验的假设，不是前提。
> 同理，**不假设**折叠率下降就一定 DSC 上升。
> 折叠率与 DSC 的关系由本次实验给出答案。
>
> ⚠️ `tv_weight`/`lr` **当前没有 CLI 参数**，
> 需要一个薄封装脚本或复用 `train_competition.py` 的配置路径。
> §5.6 给出两种做法，建议用「薄封装脚本」，不改动仓库现有 CLI。

### 5.6 需要新增的脚本（**尚未创建**，本次不运行）

| 脚本 | 作用 | 是否触及患者数据 |
|---|---|---|
| `scripts/list_labels.py` | 只读打印每个病例的标签取值、图像尺寸、几何信息 | 只读 |
| `scripts/identity_baseline.py` | 生成零位移场，按原始 CT 网格算 DSC/DSC30 | 是（只读） |
| `scripts/run_eval_configs.py` | 按 §5.5 表格逐配置跑同一批病例，输出对比表 | 是（只读） |

> 这三个脚本是**方案的一部分**，本次**不创建、不运行**，
> 等你确认后再落地。

### 5.7 预期输出路径

```
outputs/eval_<RUN_ID>/
├── label_inventory.json          # list_labels.py 产物
├── identity/
│   ├── eval_results/eval_report.json
│   └── per_case.json
├── c1_current/                   # max_iter=200 lr=0.1 tv=0.01 mse
│   ├── deformation_fields/<case>_deformation_field.nii.gz
│   ├── eval_results/eval_report.json
│   └── per_case.json
├── c2_tv005/                     # tv=0.05
├── c3_tv005_lr005/               # tv=0.05 lr=0.05
├── c4_iter100/                   # max_iter=100
└── comparison.json               # 逐例配对对比表 + 折叠率汇总
```

`comparison.json` 建议字段：

```json
{
  "case_ids": ["001", "..."],
  "configs": {
    "identity":  {"mean_dsc": 0.0, "dsc30": 0.0, "fold_ratio": null, "per_case": {}},
    "c1_current":{"mean_dsc": 0.0, "dsc30": 0.0, "fold_ratio": 0.0, "per_case": {}}
  },
  "per_case_delta_vs_identity": {"c1_current": {"001": 0.0}}
}
```

### 5.8 判定规则（预先写死，避免事后挑结果）

1. **主判据**：原始 CT 网格上的 mean DSC 与 DSC30。
2. **诊断判据**：`fold_ratio`、`min_jacobian`。
3. 只有当某配置在 mean DSC **和** DSC30 上都**不低于 C1**，
   且 `fold_ratio` 明显低于 C1 时，才考虑把它用于生成新提交包。
4. 若各配置 DSC 差异 < 0.01，判定为「**在当前模型能力下无实质差异**」，
   结论转为「需要换相似性度量/多尺度，而不是调 TV」。
5. **不得**用测试集 `025–032` 的隐藏标签做任何选择。

### 5.9 预期命令（**本次不执行**）

```bash
cd /data1/MedRegLab &&
source scripts/activate_medreg_env.sh &&

MEDREG_RUN_ID="$(date +%Y%m%d_%H%M%S)" &&
MEDREG_EVAL="outputs/eval_${MEDREG_RUN_ID}" &&

# 1) 只读盘点标签与几何（不跑配准）
python scripts/list_labels.py --data Datasets --output "${MEDREG_EVAL}/label_inventory.json" &&

# 2) identity 基线（零位移场，原始 CT 网格）
python scripts/identity_baseline.py \
    --data Datasets --output "${MEDREG_EVAL}/identity" &&

# 3) 逐配置跑同一批病例
python scripts/run_eval_configs.py \
    --data Datasets \
    --output "${MEDREG_EVAL}" \
    --configs c1_current c2_tv005 c3_tv005_lr005 c4_iter100 &&

# 4) 汇总对比
python scripts/compare_eval_configs.py \
    --eval "${MEDREG_EVAL}" \
    --output "${MEDREG_EVAL}/comparison.json" &&

printf '评估结果目录：%s\n' "${MEDREG_EVAL}"
```

> 上面的脚本名是**建议**。落地时需要先确认 §5.6 的实现方式
> （尤其是「零位移场 identity 基线」在当前代码里没有现成入口）。

---

## 6. 建议补的回归测试

上游 `6933cfe` 删除了 `336047c` 引入的
`tests/test_medreglab.py::test_label_warp_axis_order`。
建议在 `6933cfe` 基础上重新补一条**逐轴**验证：
对**非对称** mask 分别沿 x/y/z 施加已知位移，
断言形变后 mask 的质心位移方向正确。

> 注意：用**居中立方体** mask 的测试会漏掉这个 bug（见 §3.2 的脚注），
> 必须用非对称形状。

---

## 7. 证据边界小结

| 结论 | 证据强度 |
|---|---|
| 轴序缺陷真实存在，且在立方体网格上也会把 identity DSC 打到接近 0 | **已复算验证**（§3.2） |
| `6933cfe` 已修复轴序，并把 DSC 评分迁到原始 CT 网格 | **已核对代码**（§3.4） |
| 历史文档记录过 0.396 / 0.400 两个数字 | **已核对文档**（E1） |
| 0.396 / 0.400 的具体测量条件（病例 ID、逐例结果、日志） | **待核实**（无产物） |
| 0.396 / 0.400 来自 `Datasets/train` 前若干例的 64³ 缩放网格 | 文档自述，**待核实** |
| 0.396 与 0.400 相差 0.004 | **算术事实** |
| 「0.15 → 0.40」这一对修复前后数字 | **待核实**（机制方向已复算支持） |
| 历史数字与当前提交包可比 | **否**（评分空间/器官集合/双肾/空器官规则均不同，§2.5） |
| 当前折叠率 4.1%–11.2%、min_jacobian 低至 -102.6 | **已核对本轮运行产物**（§4） |
| 增大 TV 是否提高 DSC | **未知**，由 §5 实验检验 |
