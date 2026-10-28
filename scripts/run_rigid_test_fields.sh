#!/usr/bin/env bash
# Generate rigid pre-alignment fields for the eight test cases.
#
# The eight workers are independent processes; each one gets two SimpleITK
# threads so that the total thread count stays at the core count.  Without the
# limit every worker opens 16 ITK threads and the machine ends up with 128
# runnable threads on 16 cores, which slows every case down instead of
# speeding it up.
set -euo pipefail
cd /data1/MedRegLab

export OMP_NUM_THREADS=2
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

OUT="outputs/rigid_test_20261028_2130"

exec .venv/bin/python scripts/rigid_test_fields.py \
  --data Datasets \
  --output "$OUT" \
  --jobs 8
