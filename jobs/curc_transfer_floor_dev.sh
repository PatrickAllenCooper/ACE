#!/bin/bash
set -euo pipefail
source /projects/paco0228/miniconda3/etc/profile.d/conda.sh
conda activate ace
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
cd "${ACE_CODE_ROOT:?}"
test "$(git rev-parse HEAD)" = "${ACE_SOURCE_REVISION:?}"
: "${FLOOR:?}" "${CELL_OUTPUT:?}"
case "$FLOOR" in 5|6) ;; *) echo 'invalid frozen floor' >&2; exit 2;; esac
python -u scripts/research/transfer_adaptive_prequential_dev.py \
  --floor "$FLOOR" \
  --uniform-reference results/local_transfer_prequential_dev_20260926/node_metrics.csv \
  --output "$CELL_OUTPUT"
