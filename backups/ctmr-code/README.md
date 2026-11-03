# ctmr 算法代码备份（`/data/ctmr/code`）

> 创建时间：2026-11-03
> 目的：`/data/ctmr/code` 是**独立 git 仓库且没有远端**，其内容仅存在于这台 Linux 机器上。
> 本目录是该仓库的完整备份，随 MedRegLab 一起推送到 GitHub。

## 备份的是什么

| 文件 | 内容 |
|---|---|
| `ctmr-code-ec788be.bundle` | **完整 Git 仓库打包**（含全部 5 个提交的历史） |
| `ctmr-code-ec788be.tar.gz` | `ec788be` 的**纯源码快照**（72 个文件，无 `.git`） |
| `commit.txt` | `ec788be` 的完整提交信息（作者、时间、消息） |
| `bundle-verify.txt` | `git bundle verify` 的输出 |
| `status.txt` | 打包时的工作区状态（**空 = 工作区干净**） |
| `worktree.diff` | 未暂存改动（**空 = 无未暂存改动**） |
| `index.diff` | 已暂存改动（**空 = 无已暂存改动**） |
| `SHA256SUMS` | 上述所有文件的 SHA256 校验 |

## 对应的算法版本

| 项 | 值 |
|---|---|
| commit | `ec788be34fe9f50b3511a08ce2a23aaee22caacf` |
| 提交信息 | `feat: 提分所需全部算法改动（对应提交包 LB-0.9325）` |
| `ctmr/v2/register.py` | `bab7ec506b2e8e15dab51330660d6e8326f11c21bf7377660464cd0b40b36a9f` |
| `ctmr/v2/seg.py` | `7236b993d2cd6d720fbfca70b7a04024750467292fb99d26f1ff62e261f4e6fe` |

这两个 SHA256 与提交包 `outputs/submission_organ_best_20261103_final/submission.zip`
（榜单标签 `LB-0.9325`）生成时所用的算法代码**完全一致**。

## 恢复方法

### 方式 A：从 bundle 恢复（保留完整 Git 历史，推荐）

```bash
git clone backups/ctmr-code/ctmr-code-ec788be.bundle ctmr-code
cd ctmr-code
git log --oneline          # 应看到 ec788be 在 HEAD
sha256sum ctmr/v2/register.py ctmr/v2/seg.py
```

### 方式 B：从 tar.gz 恢复（仅源码快照）

```bash
tar -xzf backups/ctmr-code/ctmr-code-ec788be.tar.gz
# 得到 ctmr-code-ec788be/ 目录（纯源码，无 .git）
```

### 校验

```bash
cd backups/ctmr-code
sha256sum -c SHA256SUMS
```

## 恢复演练记录（2026-11-03 已实测）

| 项 | 结果 |
|---|---|
| 从 bundle 克隆 | ✅ HEAD = `ec788be`，5 个提交，`register.py`/`seg.py` 哈希与期望一致 |
| 从 tar.gz 解包 | ✅ 72 个文件，两文件哈希与期望一致，无 `.git` |
| `sha256sum -c SHA256SUMS` | ✅ 7/7 全部 OK |

## 未包含的内容（有意排除）

| 项 | 原因 |
|---|---|
| `submission.zip`（908MB） | 提交包太大，不应入库；它在 `outputs/` 下，且另有备份 |
| 分割模型权重（28 个 `last.pt`，1.8G） | 体积过大；`/data/ctmr/runs/` 下的权重是**不可重建资产**，需另行备份 |
| `/data/ctmr/env/`（7.4G 虚拟环境） | 可由 `requirements.txt` 重建 |
| `Datasets/`、`cache/` | 可从原始数据重建 |

> ⚠️ **仍需注意**：模型权重（`/data/ctmr/runs/seg_*/last.pt`）**不在本备份内**。
> 那是重新训练才能再生的唯一不可重建资产。`SCORE_IMPROVEMENT_PLAN.md` 中记录的
> 概率图（`ens_v3v3_ms` 等）也依赖这些权重。
