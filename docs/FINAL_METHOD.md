# 最终方案交付清单（2026-11-01 定版）

## 结论

最终采用 **器官概率引导的跨模态配准**（`opt_c` 配置），作为本赛题的最终方法。
其余候选方法与其结果文件已清理，仅保留本方案所需的模型、代码与提交产物。

---

## 一、最终方法定义

| 项 | 内容 |
|---|---|
| 方案名 | `organ_guided_registration`（器官概率引导配准） |
| 引擎 | `ctmr.v2.register`（`/data/ctmr/code/ctmr/v2/register.py`） |
| 驱动脚本 | `MedRegLab/experiments/organ_guided_inference.py` |
| 核心思路 | 单模态器官分割网络 → 器官概率质心给初值平移 → MIND 由粗到细搜索 → 稠密形变场优化 |
| 语义约定 | pull：`p_mr = p_ct + D(p_ct)`，`D` 单位为物理毫米，落在原始 CT 网格 |

### 最终超参数（`runs/opt_c/params.json`，与测试集运行逐字段一致）

```json
{
  "reg_sp": 1.5, "io_iters": 500, "io_lambda": 2.0,
  "seg_weight": 2.0, "seg_sigma": 2.0,
  "init": "seg", "seg_source": "pred",
  "pred_dir": "/home/ubuntu/ctmr/preds/ens",
  "affine_mode": "none", "grid": 4, "radius": 6
}
```

---

## 二、开发集表现（**不是**独立验证，**不是**测试集成绩）

来源：`/data/ctmr/runs/opt_c/summary.json`，017–024（主办方自采集组，与测试集 025–032 同源），
用 **真实标注（GT）** 评分，非预测 mask 代理分。

| 指标 | 数值 |
|---|---|
| mean DSC | **0.8832** |
| DSC30（最低 3 例） | **0.8518** |

对照历史基线（同一批 8 例，`C1_current` 口径）：DSC 0.2279 → 0.8832，DSC30 0.1138 → 0.8518。

> 口径提醒：以上为开发集成绩。隐藏测试集标签不可用，正式 DSC / DSC30 由主办方计算。

---

## 三、保留资产

### 1. 算法代码（已纳入 git，commit `8ad2831`）

```
/data/ctmr/code/
├── ctmr/            # 算法实现（v1 + v2 两套）
│   ├── v2/seg.py        # 器官分割训练与折外预测
│   ├── v2/register.py   # 配准引擎（最终方案）
│   ├── v2/prep.py       # 预处理
│   ├── utils/submit.py  # 提交打包
│   └── ...
├── configs/         # 配置
├── scripts/         # 训练/预测/打包/评估脚本
├── docs/            # 含官方提交规范原文与实验记录
└── requirements.txt
```

另有 `/data/ctmr/run_final.sh`（最终方案一键脚本，在 `code/` 之外）。

### 2. 模型权重（**不可重建，必须保留**）

```
/data/ctmr/runs/seg_{ct,mr}_f{0,1,2,3,all}_{v1,v1s1}/last.pt   # 20 个，各 63MB，合计 1.3G
```

- 2 模态 × 5 折（4 折 + 全量）× 2 种子
- 集成预测 `preds/ens` 由 `--tag v1 v1s1` 按折平均概率得到
- **重新训练才能再生**，是本方案的唯一不可重建资产

### 3. 提交产物

```
/data1/MedRegLab/outputs/submission_organ_best_20261029/
├── submission.zip          # 909,650,880 字节 (867.5 MiB)
│                            # SHA256 cb56922aa4eed8548eefa70ab3039f40e9640b0659f1d2d4d29e42f1b2ead635
├── submission/             # 025..032/displacement_field.nii.gz
├── format_check.json       # 官方规范 14/14 项通过
├── independent_verification.json
├── delivery_summary.json
└── README.txt
```

### 4. 辅助资产

| 路径 | 说明 |
|---|---|
| `/data/ctmr/preds/ens/` | 器官概率图（20M），可由权重 + cache 重建；**重建非逐位可复现**（见 §5.1） |
| `/data/ctmr/cache/seg_1.5x1.5x3.0/` | 预处理缓存（958M），可由 Datasets 重建 |
| `/data/ctmr/run_final.sh` | 最终方案一键脚本（缓存 → 概率 → 配准）；本仓库 `scripts/run_final.sh` 为同内容副本 |
| `/data/ctmr/Datasets` | **符号链接** → `/data1/MedRegLab/Datasets`（原为重复副本，已删） |
| `/data/ctmr/env/` | 独立虚拟环境（7.4G）。⚠️ 其 torch 不支持 RTX 5090，实际用 MedRegLab 的 `.venv` |

---

## 四、已清理内容

| 类别 | 内容 | 释放 |
|---|---|---|
| 旧方法提交包 | `submission_20261028_065511`、`submission_B_20261028`、`submission_rigid_*`、`submission_organ_guided_*`、`submission_organ_opt_*` | ~12G |
| 旧方法推理产物 | `test_20261028_065511`、`rigid_test_*`、`organ_guided_test_*`、`organ_guided_opt_test_*` | ~6G |
| 开发集对照实验 | `dev_eval_20261028_160000`（A/B/C1–C4 的位移场）、`dev_eval_robust_full`、`prealign_diffgroup_*`、`probe_*` | ~21G |
| ctmr 实验运行 | `runs/` 下除 20 个分割权重外的全部实验目录 | ~5.5G |
| 临时文件 | `ctmr/tmp/`、`ctmr/code/outputs/` | ~1.7G |
| 日志 | 各次运行的 `.log`（保留在 git 提交记录中） | 少量 |

> 已清理项均属"其他方法/其他结果"。历史 0.396 / 0.400 基线与 6.18 轴序缺陷分析
> 的**结论**保留在 `MedRegLab/docs/EVIDENCE_BASELINE.md`（已入库），
> 被删除的只是可重新生成的位移场中间产物。

---

## 五、复现最终方案的步骤

> **一键脚本**：`bash scripts/run_final.sh`（本仓库内，含缓存检查 → 器官概率 → 配准三步）。
> 同一脚本在 `/data/ctmr/run_final.sh` 与 `/data/ctmr/code/scripts/run_final.sh` 也有副本。
>
> ⚠️ **环境注意**：`/data/ctmr/env` 的 torch 2.6.0+cu124 只编译到 `sm_90`，
> **不支持本机 RTX 5090（sm_120）**，直接运行会报
> `no kernel image is available for execution on the device`。
> 请改用 `/data1/MedRegLab/.venv`（torch 2.10.0+cu128，支持 sm_120）。
> `run_final.sh` 已默认使用后者，可用 `PY=` 覆盖。

依赖：`Datasets/`、20 个分割权重（`/data/ctmr/runs/seg_*/last.pt`）。

```bash
# 1. 预处理缓存（若已删除）
cd /data/ctmr/code
../env/bin/python -m ctmr.v2.prep --config configs/base.yaml

# 2. 器官概率（折外集成，测试集用全量模型）
../env/bin/python -m ctmr.v2.seg predict --modality ct --out ~/ctmr/preds/ens --gpu 0 --tag v1 v1s1
../env/bin/python -m ctmr.v2.seg predict --modality mr --out ~/ctmr/preds/ens --gpu 0 --tag v1 v1s1

# 3. 配准（测试集 8 例）
cd /data1/MedRegLab
.venv/bin/python experiments/organ_guided_inference.py \
    --data Datasets --split test \
    --output outputs/organ_guided_best_test_20261029 \
    --predictions /home/ubuntu/ctmr/preds/ens \
    --engine-root /data/ctmr/code \
    --settings "seg_weight=2 io_iters=500 io_lambda=2 reg_sp=1.5"

# 4. 打包
.venv/bin/python experiments/package_submission.py \
    --inference outputs/organ_guided_best_test_20261029 \
    --data Datasets --output outputs/submission_organ_best_20261029
```

### 5.1 可复现性：实测结论（2026-11-01）

**重跑能得到同等质量的场，但不是逐位相同。** 实测数据：

| 环节 | 复现情况 |
|---|---|
| 分割概率图（同一权重、同一命令跑两次） | **不可复现**：2.77% 体素概率值不同（最大差 255/255） |
| 由此产生的位移场 | 与交付包差异 **11.8–19.5 mm**（场均值差异 < 0.1 mm） |
| 器官概率固定后重跑配准 | case 030 体素最大差异 **0.0 mm**（完全一致） |

根因：GPU 上 `sliding_window` 滑窗推理的浮点累加顺序不确定，导致概率图末位差异；
该差异经器官质心与搜索初值放大为毫米级场差异。

**因此**：

- **交付的 `submission.zip` 是唯一权威产物**，其 SHA256 与 8 例场的哈希链完整、校验通过；
- 重跑得到的包**不应**用来替换交付包（哈希会变，且质量仅有概率性等价保证）；
- 若需重新提交，应以本包为准；重跑仅用于验证流程连通性。

---

## 六、提交方式（官方规范）

1. 登录华为云控制台（分配的子账号）→ OBS → 创建桶 → 上传 `submission.zip`
2. 加入算法命题赛道 QQ 群 **1101385600**，在在线文档中提交 OBS 中压缩包的 **URL**
3. 每天最多提交 1 次；决赛最终提交截止 **2026-11-10 24:00**
4. 赛事咨询：文老师 13022388326

规范硬约束（本包已 14/14 通过）：包结构 `submission/<case_id>/displacement_field.nii.gz`、
float32、3 分量、毫米相对位移、几何与原始 CT 完全一致、≤ 4GB。
