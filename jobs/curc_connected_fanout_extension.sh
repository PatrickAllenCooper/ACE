#!/bin/bash
# One short CPU job for six frozen fanout development cells.
set -euo pipefail
source /projects/paco0228/miniconda3/etc/profile.d/conda.sh
conda activate ace
: "${ACE_CODE_ROOT:?}" "${ACE_SOURCE_REVISION:?}" "${CELL_OUTPUT:?}"
cd "$ACE_CODE_ROOT"
test "$(git rev-parse HEAD)" = "$ACE_SOURCE_REVISION"
test -z "$(git status --porcelain --untracked-files=no)"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
python -u scripts/research/connected_fanout_dev.py \
  --protocol docs/development/guidance/protocol_connected_fanout_extension_2026-09-28.json \
  --output "$CELL_OUTPUT"
