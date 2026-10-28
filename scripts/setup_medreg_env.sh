#!/usr/bin/env bash
# Create a NEW Linux environment. Never overwrite an existing venv or HF cache.
set -euo pipefail

medreg_repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
medreg_venv_dir="${MEDREG_VENV_PATH:-${medreg_repo_dir}/.venv}"
if [[ -e "$medreg_venv_dir" || -L "$medreg_venv_dir" ]]; then
    printf 'Refusing to modify existing environment: %s\n' "$medreg_venv_dir" >&2
    printf 'For an existing environment use: source scripts/activate_medreg_env.sh\n' >&2
    exit 1
fi

command -v python3 >/dev/null || { printf 'python3 not found\n' >&2; exit 1; }
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else "Python >= 3.10 is required")'
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${medreg_repo_dir}/.cache/pip}"
export TMPDIR="${TMPDIR:-${medreg_repo_dir}/.cache/tmp}"
mkdir -p -- "$PIP_CACHE_DIR" "$TMPDIR"
python3 -m venv "$medreg_venv_dir"
medreg_python="${medreg_venv_dir}/bin/python"
"$medreg_python" -m pip install --upgrade pip
# Official pairing for PyTorch 2.10.0 / CUDA 12.8; compatible with sm_120.
"$medreg_python" -m pip install 'torch==2.10.0' 'torchvision==0.25.0' \
    --index-url https://download.pytorch.org/whl/cu128
"$medreg_python" -m pip install -r "${medreg_repo_dir}/requirements.txt"
"$medreg_python" -m pip install --no-deps -e "$medreg_repo_dir"
printf 'Created %s. Next: source scripts/activate_medreg_env.sh\n' "$medreg_venv_dir"
