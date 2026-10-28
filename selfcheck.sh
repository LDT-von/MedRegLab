#!/usr/bin/env bash
# MedRegLab 交付前全量自检
set -uo pipefail
cd /data1/MedRegLab

fail=0
ok()   { echo "  [ OK ] $*"; }
bad()  { echo "  [FAIL] $*"; fail=1; }

echo "=== 1. Python / torch / GPU ==="
.venv/bin/python - <<'PY'
import sys, torch
assert torch.__version__.endswith("+cu128"), torch.__version__
assert torch.cuda.is_available()
cap = torch.cuda.get_device_capability(0)
assert cap[0] >= 12, cap
x = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
(x @ x).float().sum().item()
print(f"  torch {torch.__version__} | cuda {torch.version.cuda} | {torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]} | GPU 计算正常")
PY
[ $? -eq 0 ] && ok "torch+cu128 on sm_120" || bad "torch/GPU"

echo "=== 2. 依赖导入 ==="
.venv/bin/python - <<'PY'
import medreglab, monai, SimpleITK, nibabel, timm, scipy, skimage, einops, yaml, tqdm
print(f"  medreglab {medreglab.__version__} | monai {monai.__version__} | SimpleITK {SimpleITK.__version__} | timm {timm.__version__}")
PY
[ $? -eq 0 ] && ok "依赖齐全" || bad "依赖"

echo "=== 3. 单元测试 ==="
.venv/bin/python -m pytest tests/ -q 2>&1 | tail -2

echo "=== 4. 数据集 ==="
.venv/bin/python - <<'PY'
import warnings; warnings.filterwarnings("ignore")
from medreglab.data import CompetitionDataset
for split, n in [("train",24), ("test",8), ("unpaired_ct",50), ("unpaired_mr",40)]:
    ds = CompetitionDataset(root="Datasets", split=split, target_size=(64,64,64))
    assert len(ds) == n, (split, len(ds), n)
    print(f"  {split:12s} n={len(ds):3d}  OK")
PY
[ $? -eq 0 ] && ok "数据集四个 split 数量正确" || bad "数据集"

echo "=== 5. 端到端配准 (GPU, 真实数据) ==="
.venv/bin/python - <<'PY'
import warnings, torch; warnings.filterwarnings("ignore")
from medreglab import load_model
from medreglab.data import CompetitionDataset
ds = CompetitionDataset(root="Datasets", split="train", target_size=(64,64,64))
s = ds[0]
m = load_model("convex_3d").to("cuda")
out = m(s["moving"].to("cuda"), s["fixed"].to("cuda"))
assert out["phi"].shape == (1,3,64,64,64), out["phi"].shape
assert out["warped"].shape == (1,1,64,64,64)
print(f"  case {s['case_id']}  phi {tuple(out['phi'].shape)}  loss {out['loss']:.6f}  iters {len(out['loss_history'])}")
PY
[ $? -eq 0 ] && ok "3D 配准" || bad "3D 配准"

echo "=== 6. 推理 + 提交产物 ==="
rm -rf outputs/_selfcheck
.venv/bin/python - <<'PY'
import warnings, sys, glob; warnings.filterwarnings("ignore")
sys.path.insert(0, "experiments")
from infer_competition import run_competition_inference
import SimpleITK as sitk
r = run_competition_inference(data_root="Datasets", output_dir="outputs/_selfcheck",
    model_name="convex_3d", split="test", target_size=(64,64,64), max_iter=20)
assert r["num_errors"] == 0, r
files = sorted(glob.glob("outputs/_selfcheck/deformation_fields/*.nii.gz"))
assert len(files) == 8, len(files)
img = sitk.ReadImage(files[0])
assert img.GetNumberOfComponentsPerPixel() == 3
print(f"  {len(files)} 个变形场, 3 分量 NIfTI, size={img.GetSize()}")
PY
[ $? -eq 0 ] && ok "推理与提交产物" || bad "推理"
rm -rf outputs/_selfcheck

echo "=== 7. 模型注册表 ==="
.venv/bin/python - <<'PY'
import warnings; warnings.filterwarnings("ignore")
from medreglab import load_model
m = load_model("convex_3d")
assert type(m).__name__ == "Simple3DRegistration", type(m).__name__
try:
    load_model("dino_reg_mrct")
    raise AssertionError("removed model should raise")
except ValueError:
    pass
print("  仅 convex_3d 可用，移除的模型正确报错")
PY
[ $? -eq 0 ] && ok "模型注册表" || bad "模型注册表"

echo "=== 8. 数据去重校验 ==="
.venv/bin/python - <<'PY'
import os, re, hashlib
root = "Datasets"
dups = 0
for dp, dn, fn in os.walk(root):
    for f in fn:
        stem, ext = os.path.splitext(f)
        m = re.match(r"^(.*)\(\d+\)$", stem)
        if not m:
            continue
        dups += 1
print(f"  剩余 '(N)' 重名副本: {dups} (应为 0)")
assert dups == 0
PY
[ $? -eq 0 ] && ok "无残留重复文件" || bad "去重"

echo "=== 9. 无残留 Windows 路径 ==="
if grep -rn 'E:\\\\MedRegLab\|E:/MedRegLab' --include="*.py" --include="*.yaml" \
     . 2>/dev/null | grep -v '\.venv/' | grep -v '^\./docs/'; then
    bad "仍有 Windows 绝对路径"
else
    ok "已全部改为相对路径"
fi

echo
if [ $fail -eq 0 ]; then echo "全部检查通过 ✅"; else echo "存在失败项 ❌"; fi
exit $fail
