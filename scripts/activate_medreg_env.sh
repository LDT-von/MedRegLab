#!/usr/bin/env bash
# Source this file; check the existing environment before activation.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    printf 'Use: source scripts/activate_medreg_env.sh\n' >&2
    exit 2
fi

_medreg_repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
_medreg_venv_dir="${MEDREG_VENV_PATH:-${_medreg_repo_dir}/.venv}"
if [[ ! -f "${_medreg_venv_dir}/bin/activate" || ! -x "${_medreg_venv_dir}/bin/python" ]]; then
    printf 'Existing virtual environment not found at %s\n' "$_medreg_venv_dir" >&2
    return 1
fi
if ! "${_medreg_venv_dir}/bin/python" - <<'PY'
import importlib
import os
import torch

for name in ("monai", "SimpleITK", "nibabel", "numpy", "scipy", "skimage", "yaml", "tqdm", "pytest"):
    importlib.import_module(name)
if not torch.cuda.is_available():
    if os.environ.get("MEDREG_ALLOW_CPU") != "1":
        raise SystemExit("CUDA unavailable; set MEDREG_ALLOW_CPU=1 only for intentional CPU use")
    print("CPU-only mode explicitly requested")
else:
    cap = torch.cuda.get_device_capability()
    cuda = torch.version.cuda
    if cap[0] >= 12 and (cuda is None or tuple(map(int, cuda.split(".")[:2])) < (12, 8)):
        raise SystemExit(f"GPU capability {cap} requires a compatible CUDA wheel; found {cuda}")
    # A real kernel catches unsupported architecture even if CUDA is reported available.
    assert (torch.ones(2, device="cuda") + 1).sum().item() == 4
    print(f"torch {torch.__version__}; CUDA {cuda}; {torch.cuda.get_device_name()} capability {cap}")
PY
then
    printf 'Environment verification failed; environment was not activated.\n' >&2
    return 1
fi

source "${_medreg_venv_dir}/bin/activate"
export HF_HOME="${HF_HOME:-${_medreg_repo_dir}/.cache/huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
printf 'Activated %s\n' "$_medreg_venv_dir"
unset _medreg_repo_dir _medreg_venv_dir
