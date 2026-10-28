#!/usr/bin/env bash
# Portable local checks. Default runs synthetic tests, not patient registration.
set -uo pipefail
medreg_repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$medreg_repo_dir" || exit 1
medreg_python="${MEDREG_PYTHON:-${MEDREG_VENV_PATH:-${medreg_repo_dir}/.venv}/bin/python}"
medreg_data_root=""
if [[ ${1:-} == "--data" && $# -eq 2 ]]; then
    medreg_data_root="$2"
elif [[ $# -ne 0 ]]; then
    printf 'Usage: bash selfcheck.sh [--data DATASET_ROOT]\n' >&2
    exit 2
fi
if [[ ! -x "$medreg_python" ]]; then
    printf 'Python not found: %s (set MEDREG_PYTHON or MEDREG_VENV_PATH)\n' "$medreg_python" >&2
    exit 1
fi
export PYTHONDONTWRITEBYTECODE=1
fail=0
ok() { printf '  [ OK ] %s\n' "$*"; }
bad() { printf '  [FAIL] %s\n' "$*"; fail=1; }

printf '=== 1. Python, dependencies and GPU ===\n'
if "$medreg_python" - <<'PY'
import importlib
import os
import torch
for name in ("medreglab", "monai", "SimpleITK", "nibabel", "numpy", "scipy", "skimage", "yaml", "tqdm", "pytest"):
    importlib.import_module(name)
if not torch.cuda.is_available():
    if os.environ.get("MEDREG_ALLOW_CPU") != "1":
        raise SystemExit("CUDA unavailable; intentional CPU use requires MEDREG_ALLOW_CPU=1")
    print("CPU-only mode explicitly requested")
else:
    cap = torch.cuda.get_device_capability()
    cuda = torch.version.cuda
    if cap[0] >= 12 and (cuda is None or tuple(map(int, cuda.split(".")[:2])) < (12, 8)):
        raise SystemExit(f"Incompatible CUDA wheel {cuda} for capability {cap}")
    assert (torch.ones(2, device="cuda") + 1).sum().item() == 4
    print(f"torch {torch.__version__} | CUDA {cuda} | {torch.cuda.get_device_name()} | capability {cap}")
PY
then ok 'imports and device'; else bad 'imports or device'; fi

printf '=== 2. Synthetic unit tests ===\n'
if "$medreg_python" -m pytest tests/ -q -p no:cacheprovider; then
    ok 'unit tests'
else
    bad 'unit tests'
fi

printf '=== 3. Registered model ===\n'
if "$medreg_python" - <<'PY'
from medreglab import load_model
model = load_model("convex_3d")
assert not any(p.requires_grad for p in model.parameters())
try:
    load_model("dino_reg_mrct")
except ValueError:
    pass
else:
    raise AssertionError("Removed model unexpectedly available")
print("convex_3d: per-pair displacement optimization; no learned checkpoint")
PY
then ok 'model registry'; else bad 'model registry'; fi

printf '=== 4. Synthetic per-pair smoke ===\n'
if "$medreg_python" smoke_test_train.py --device cpu; then
    ok 'synthetic registration'
else
    bad 'synthetic registration'
fi

if [[ -n "$medreg_data_root" ]]; then
    printf '=== 5. Dataset inventory (directory metadata only) ===\n'
    if "$medreg_python" - "$medreg_data_root" <<'PY'
import sys
from medreglab.data import CompetitionDataset
for split, expected in (("train", 24), ("test", 8), ("unpaired_ct", 50), ("unpaired_mr", 40)):
    dataset = CompetitionDataset(root=sys.argv[1], split=split)
    if len(dataset) != expected:
        raise AssertionError((split, len(dataset), expected))
    print(f"{split}: {len(dataset)}")
PY
    then ok 'dataset inventory'; else bad 'dataset inventory'; fi
fi

if [[ $fail -eq 0 ]]; then printf 'All selected checks passed.\n'; else printf 'Some checks failed.\n'; fi
exit "$fail"
