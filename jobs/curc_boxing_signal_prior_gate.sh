#!/bin/bash
set -euo pipefail
source /projects/paco0228/miniconda3/etc/profile.d/conda.sh
conda activate ace
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
cd "${ACE_CODE_ROOT:?}"
test "$(git rev-parse HEAD)" = "${ACE_SOURCE_REVISION:?}"
python -u scripts/research/fit_boxing_signal_prior_gate.py \
  --public "${PUBLIC_INPUT:?}" \
  --protocol docs/development/guidance/protocol_boxing_signal_prior_gate_dev_2026-09-28.json \
  --output "${FIT_OUTPUT:?}" --source-revision "$ACE_SOURCE_REVISION"
