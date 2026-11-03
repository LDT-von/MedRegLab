# 提交包台账（Submission Ledger）

> 建立时间：2026-11-03
> 用途：区分**已提交榜单并取得分数**的包，与仅在本地生成/待提交的包。
> 权威口径：榜单分数由主办方在隐藏测试集（025–032）上计算；开发集分数（017–024，域内口径）
> 由本机 `SCORE_IMPROVEMENT_PLAN.md` 的受控消融实验给出。**两者不可混用。**
>
> 📌 本文件是 `outputs/SUBMISSION_LEDGER.md` 的**入库副本**（`outputs/` 已被 `.gitignore`
> 忽略，内容会被清理）。以本文件为准；`outputs/` 下那份若存在则为同步副本。

---

## 0. 快速索引（一眼区分）

| 标签 | 榜单分数 (DSC / DSC30) | 开发集 (DSC / DSC30) | 提交包路径 | 状态 |
|---|---|---|---|---|
| **`LB-0.9073`** ⭐ | **0.9073 / 0.8977** | 0.9072 / 0.8958 | `outputs/submission_organ_best_20261102/submission.zip` | **已提交，已出分** |
| **`LB-0.9325`** ⭐ | 待提交 | 0.9325 / 0.9252 | `outputs/submission_organ_best_20261103_final/submission.zip` | **待提交** |
| `LB-0.8832` | 未提交 | 0.8832 / 0.8518 | `outputs/submission_organ_best_20261029/submission.zip` | 早期版本，未提交 |
| `DEV-0.9200` | 未提交 | 0.9200 / 0.9117 | `outputs/submission_organ_best_20261103/submission.zip` | 中间版本，未提交 |
| `DEV-0.9230` | 未提交 | 0.9230 / 0.9152 | `outputs/submission_organ_best_20261103_v2/submission.zip` | 中间版本，未提交 |
| `DEV-0.9321` | 未提交 | 0.9321 / 0.9248 | `outputs/submission_organ_best_20261103_v5/submission.zip` | 中间版本，未提交 |

> ⚠️ **重要**：只有 `LB-0.9073` 一个包**实际上过榜并拿到分数**。
> `LB-0.9325` 的 0.9325 是**开发集**分数，不是榜单分数——不要写成"上次 0.9073 对应 0.9325"，
> 两者分属不同口径。

### 0.1 当前 outputs 目录状态（2026-11-03 清理后）

```
outputs/                                    共 6.9G
├── submission_organ_best_20261102/          LB-0.9073  ⭐ 提交包（已出分）
├── submission_organ_best_20261103_final/    LB-0.9325  ⭐ 提交包（待提交）
├── submission_organ_best_20261029/          LB-0.8832  备选包
├── submission_organ_best_20261103/          DEV-0.9200 备选包
├── submission_organ_best_20261103_v2/       DEV-0.9230 备选包
├── submission_organ_best_20261103_v5/       DEV-0.9321 备选包
├── organ_guided_best_test_20261102/         LB-0.9073 推理溯源（8 场文件 + 参数快照）
├── organ_guided_best_test_20261103_final/   LB-0.9325 推理溯源（8 场文件 + 参数快照）
├── _config_snapshots/                       中间版本配置快照（params/summary/per_case，100K）
├── _algo_snapshots/                         两版算法代码（v1/v2）
└── SUBMISSION_LEDGER.md                     本文件
```

**已清理的冗余**（均为经逐字节验证的纯重复副本，不含任何信息损失）：

| 项 | 释放 | 验证依据 |
|---|---|---|
| `submission/*/submission/` 解压目录 | 4.3G | 与 ZIP 内文件 8/8 逐字节相同 |
| `*/engine/fields/` | 5.2G | 与同目录 `deformation_fields/` 8/8 逐字节相同 |
| 4 个中间版本的推理目录 | 3.5G | 配置快照已存 `_config_snapshots/`；ZIP 全部保留 |
| **合计** | **约 13G** | |

> **为什么只保留 2 个推理目录**：ZIP 内的场文件是经官方
> `SimpleITK.TransformToDisplacementField` **重新栅格化**的产物，与引擎原始输出
> `deformation_fields/` **不完全字节相同**（实测 030 病例有 1 个体素、2.8e-14 的
> 浮点舍入差）。原始输出是溯源链的**源头**，保留已出分包与待提交包对应的两个目录，
> 即可完整回溯；其余 4 个从未上榜、参数已完整记录，可安全删除。
>
> ⚠️ **6 个提交包 ZIP 全部保留**——若新包上��表现不及预期，它们是现成备选。

---

## 1. ⭐ `LB-0.9073` —— 上一次实际提交的包（0.9073 / 0.8977）

### 1.1 提交物

| 项 | 值 |
|---|---|
| **提交包** | `outputs/submission_organ_best_20261102/submission.zip` |
| SHA256 | `a2304ae471d31109f6a291a2165a06d270630061c30cad386695fd4be85d9052` |
| 大小 | 919,460,747 字节（877.1 MiB） |
| 生成时间 | 2026-11-02 17:40 |
| 独立校验 | `status: PASS`（`independent_verification.json`，8/8 病例） |
| 格式校验 | `format_check.json`，`status: PASS` |
| 推理目录 | `outputs/organ_guided_best_test_20261102/` |
| **榜单分数** | **DSC 0.9073 / DSC30 0.8977**（主办方在隐藏测试集 025–032 计算） |

### 1.2 生成脚本（算法入口）

| 项 | 路径 |
|---|---|
| **一键提交脚本** | `/data/ctmr/run_submit.sh` |
| 配准推理驱动 | `experiments/organ_guided_inference.py` |
| 打包器 | `experiments/package_submission.py`（当时版本 SHA256 前缀 `25badeb189e49338`） |
| 独立校验 | `experiments/verify_submission.py` |

> 注：`/data/ctmr/run_submit_organ.sh` **不存在**（历史对话中曾提及，但磁盘上从未创建）。
> 实际生成 0.9073 这个包的脚本是 **`run_submit.sh`**。

### 1.3 生效配置（取自 `organ_guided_best_test_20261102/engine/params.json`，运行时快照）

```
seg_source=pred  pred_dir=/home/ubuntu/ctmr/preds/ens_fall   init=seg
seg_weight=8     io_lambda=1     io_iters=500    reg_sp=1.5    grid=4
affine_mode=none affine_penalty=0.1
pred_hard=True   pred_sigma=1.0    seg_dice=0.0
```

概率图来源：`preds/ens_fall` = CT/MR 各 100ep 模型（tag `v1`+`v1s1`）集成，**无 TTA**。

### 1.4 算法代码版本 ⭐（这是区分两次提交的关键）

`/data/ctmr/code` 是独立 git 仓库，其 `HEAD = 1244d29`（提交时间 2026-11-02 15:11）
**正好是 0.9073 这个包生成时的算法版本**。

| 文件 | SHA256 | 说明 |
|---|---|---|
| `ctmr/v2/register.py` | `f8f71e77857b8ab5808b64264991ebb0e4b153ea001c2bce2186fc08cf387bf8` | 配准引擎（11-02 打包时版本） |
| `ctmr/v2/seg.py` | `3fb5ad0321dcf2c774e167e8dea65a8fa40b7f96731c3ce795690f562601b789` | 分割推理（11-02 打包时版本） |

**已冻结快照**：`outputs/_algo_snapshots/v1_20261102_lb09073/`
（`git show HEAD:...` 导出，哈希已复核一致）

### 1.5 与后续版本的差异（为什么分数会涨）

| 旋钮 | 0.9073 版 | 0.9325 版 | 效果 |
|---|---|---|---|
| 概率图 | `ens_fall`（100ep，无 TTA） | `ens_v3v3_ms`（CT/MR 各 300ep + 多尺度 TTA） | 分割更准 |
| `seg_weight` | 8 | 768 | 引导强度右移（结论 22） |
| `pred_sigma` | 1.0（**有**高斯平滑） | 0.0（**不**平滑） | 平滑有害（结论 20） |
| `seg_dice` | 0.0（无） | 4.0（概率余弦匹配） | +0.0025（结论 22） |
| `reg_sp` | 1.5 | 3.0（位移场变粗） | +0.0044（结论 25/27） |
| `grid` | 4 | 3 | 控制网格粗细解耦 |
| `io_iters` | 500 | 1000 | 精修迭代更多 |

---

## 2. ⭐ `LB-0.9325` —— 最新最优包（待提交）

### 2.1 提交物

| 项 | 值 |
|---|---|
| **提交包** | `outputs/submission_organ_best_20261103_final/submission.zip` |
| SHA256 | `56f878b20425b53232aafd0989054a97dd26ec61c04e0ee86c4878cd97ffaebd` |
| 大小 | 908,991,827 字节（866.7 MiB） |
| 生成时间 | 2026-11-03 06:19 |
| 独立校验 | `status: PASS`（8/8 病例，`mapped_inside_mr_extent: True`） |
| 推理目录 | `outputs/organ_guided_best_test_20261103_final/` |
| **开发集分数** | **DSC 0.9325 / DSC30 0.9252**（017–024 域内口径，配对 Δ=+0.0309, t=+6.88, p=0.000, 8胜0负） |
| 榜单分数 | **待提交后由主办方计算** |

### 2.2 生成脚本（算法入口）

| 项 | 路径 |
|---|---|
| **一键提交脚本** | `/data/ctmr/run_submit_final.sh` |
| 配准推理驱动 | `experiments/organ_guided_inference.py`（2026-10-28 起未变） |
| 打包器 | `experiments/package_submission.py`（当时版本 SHA256 前缀 `6aa8f791516bd16f`） |
| 独立校验 | `experiments/verify_submission.py` |

### 2.3 生效配置（取自 `organ_guided_best_test_20261103_final/engine/params.json`，运行时快照）

```
seg_source=pred  pred_dir=/home/ubuntu/ctmr/preds/ens_v3v3_ms   init=seg
seg_weight=768   io_lambda=1     io_iters=1000   reg_sp=3.0    grid=3
affine_mode=none affine_penalty=0.3
pred_hard=True   pred_sigma=0.0     seg_dice=4.0
```

概率图来源：`preds/ens_v3v3_ms` = CT 300ep(seed3) + MR 300ep(seed3)，多尺度 TTA (scales 1.0 + 1.25)。

### 2.4 算法代码版本

| 文件 | SHA256 |
|---|---|
| `ctmr/v2/register.py` | `bab7ec506b2e8e15dab51330660d6e8326f11c21bf7377660464cd0b40b36a9f` |
| `ctmr/v2/seg.py` | `7236b993d2cd6d720fbfca70b7a04024750467292fb99d26f1ff62e261f4e6fe` |

**已冻结快照**：`outputs/_algo_snapshots/v2_20261103_final/`

**git commit**：`/data/ctmr/code` 的 `ec788be`（2026-11-03）——
"feat: 提分所需全部算法改动（对应提交包 LB-0.9325）"。
工作区、冻结快照、git 对象三者哈希**完全一致** → 最优包已完全可复现。

---

## 3. 完整包清单（含未提交版本）

> 全部 6 个提交包 ZIP **均保留**。其中仅 `20261102` 与 `20261103_final`
> 保留了对应的推理溯源目录；其余 4 个的配置快照存于 `outputs/_config_snapshots/`。

| 目录 | 标签 | 概率图 | seg_weight | reg_sp | grid | io_iters | affine_pen | pred_sigma | seg_dice | 开发集 DSC |
|---|---|---|---|---|---|---|---|---|---|---|
| `20261029` | `LB-0.8832` | `ens` | 2 | 1.5 | 4 | 500 | — | — | — | 0.8832 |
| `20261102` | **`LB-0.9073`** ⭐ | `ens_fall` | 8 | 1.5 | 4 | 500 | 0.1 | 1.0 | 0 | **0.9072** |
| `20261103` | `DEV-0.9200` | `ens_bothv2_ms` | 24 | 1.5 | 4 | 500 | 0.3 | 0.0 | 0 | 0.9200 |
| `20261103_v2` | `DEV-0.9230` | `ens_bothv2_ms` | 96 | 1.5 | 4 | 500 | 0.3 | 0.0 | 4 | 0.9230 |
| `20261103_v5` | `DEV-0.9321` | `ens_v3v3_ms` | 768 | 3.0 | 4 | 500 | 0.3 | 0.0 | 4 | 0.9321 |
| `20261103_final` | **`LB-0.9325`** ⭐ | `ens_v3v3_ms` | 768 | 3.0 | 3 | 1000 | 0.3 | 0.0 | 4 | **0.9325** |

> 说明：`run_submit_v3.sh` / `run_submit_v4.sh` 都指向 `organ_guided_best_test_20261103`（同名覆盖），
> 因此磁盘上该目录只保留了**最后一次**运行的参数快照（`ens_bothv2_ms` / sw24）。
> `run_submit_v6.sh` 指向的 `20261103_v3`、`run_submit_v7.sh` 指向的 `20261103_v4`
> 这两个目录**在磁盘上不存在**（已被清理或从未生成）。

---

## 4. 复现方式

```bash
# ---- LB-0.9073（上次已提交的版本）----
cd /data/ctmr
bash run_submit.sh
# → outputs/organ_guided_best_test_20261102/ + outputs/submission_organ_best_20261102/

# ---- LB-0.9325（最新最优，待提交）----
cd /data/ctmr
bash run_submit_final.sh
# → outputs/organ_guided_best_test_20261103_final/ + outputs/submission_organ_best_20261103_final/
```

⚠️ **两个脚本共用同一个推理入口**（`experiments/organ_guided_inference.py`），
但依赖**不同的算法代码版本**（`register.py` / `seg.py` 的 SHA256 见上表）。

**当前 `/data/ctmr/code` 工作区 = v2 版本**（commit `ec788be`），对应 `LB-0.9325`。

若要严格复现 `LB-0.9073`（v1 版本），二选一：

```bash
# 方式 A：从 git 直接取出 v1 版本（推荐，可追溯）
cd /data/ctmr/code
git checkout 1244d29 -- ctmr/v2/register.py ctmr/v2/seg.py
bash /data/ctmr/run_submit.sh
git checkout ec788be -- ctmr/v2/register.py ctmr/v2/seg.py   # 用完还原

# 方式 B：从冻结快照恢复
cp outputs/_algo_snapshots/v1_20261102_lb09073/{register.py,seg.py} /data/ctmr/code/ctmr/v2/
```

---

## 5. 提交规范提醒（来自官方规范 6.1 节）

```
1. 打包   → submission.zip（已完成）
2. 校验   → 提交前自检（已完成）
3. 上传   → 华为云控制台（分配的子账号）→ OBS → 创建桶 → 上传 submission.zip
4. 提交   → 算法命题赛道 QQ 群 1101385600 的在线文档中填该文件的 URL
```

- 提交平台：华为云 OBS
- 频率：**每天最多 1 次**；决赛最终提交截止 **2026-11-10 24:00**
- ⚠️ 本服务器**没有** `obsutil` / 桶名 / AK / SK / Endpoint，无法代传，需由你在本地账号操作。
