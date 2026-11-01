#!/usr/bin/env bash
# 最终方案：器官概率引导配准（opt_c 配置）
#
#   seg_source=pred  pred_dir=~/ctmr/preds/ens  init=seg  seg_weight=2
#   io_iters=500     io_lambda=2                 reg_sp=1.5   affine_mode=none
#
# 三步：
#   1) 预处理缓存（cache/seg_1.5x1.5x3.0）
#   2) 器官概率折外集成预测（preds/ens）
#   3) 配准（测试集 025-032）并保存原生 CT 网格位移场
#
# 注意：/data/ctmr/env 的 torch 2.6.0+cu124 不支持 RTX 5090 (sm_120)。
# 在该卡上请改用 /data1/MedRegLab/.venv（torch 2.10.0+cu128）。
set -euo pipefail

PY="${PY:-/data1/MedRegLab/.venv/bin/python}"
CTMR="${CTMR:-/data/ctmr/code}"
PRED="${PRED:-$HOME/ctmr/preds/ens}"
SET="seg_source=pred pred_dir=$PRED init=seg seg_weight=2 io_lambda=2 io_iters=500 reg_sp=1.5 affine_mode=none"

export PYTHONNOUSERSITE=1
export PYTHONPATH="$CTMR"
cd "$CTMR"

# 1) 预处理缓存（已存在则跳过）
if [ ! -d "$HOME/ctmr/cache/seg_1.5x1.5x3.0" ]; then
    echo "==> 1/3 生成预处理缓存"
    "$PY" -m ctmr.v2.prep --out "$HOME/ctmr/cache" --spacing 1.5 1.5 3.0 --workers 8
else
    echo "==> 1/3 跳过（缓存已存在）"
fi

# 2) 器官概率：折外集成（两个种子 v1 / v1s1 按折平均）
echo "==> 2/3 生成器官概率预测"
"$PY" -m ctmr.v2.seg predict --modality ct --out "$PRED" --gpu 0 --tag v1 v1s1
"$PY" -m ctmr.v2.seg predict --modality mr --out "$PRED" --gpu 0 --tag v1 v1s1

# 3) 配准
echo "==> 3/3 配准测试集 025-032"
"$PY" -m ctmr.v2.register run --split test --cases 025-032 \
    --out "$HOME/ctmr/runs/final_test" --gpu 0 --save-fields \
    --set $SET

echo
echo "完成。位移场：$HOME/ctmr/runs/final_test/fields/<case>/displacement_field.nii.gz"
echo "下一步（在 MedRegLab 侧打包）："
cat <<'EOF'
  cd /data1/MedRegLab && .venv/bin/python experiments/organ_guided_inference.py \
      --data Datasets --split test --output outputs/organ_guided_best_test_20261029 \
      --predictions /home/ubuntu/ctmr/preds/ens --engine-root /data/ctmr/code \
      --settings "seg_weight=2 io_iters=500 io_lambda=2 reg_sp=1.5"
EOF
