#!/bin/bash
# One connected-SCM development cell; numerical CPU work only.
set -euo pipefail
source /projects/paco0228/miniconda3/etc/profile.d/conda.sh
conda activate ace
cd /projects/paco0228/ACE
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
: "${ACE_SOURCE_REVISION:?}" "${CELL_SEED:?}" "${NODES:?}" "${MOTIFS:?}" "${CELL_OUTPUT:?}"
test "$(git rev-parse HEAD)" = "$ACE_SOURCE_REVISION"
python -u scripts/research/connected_acquisition.py --seed "$CELL_SEED" \
    --nodes "$NODES" --motifs "$MOTIFS" --root-sd 0.15 --penalty 4 \
    --budget 400 --output "$CELL_OUTPUT"
python scripts/research/validate_cell.py --kind connected_acquisition --directory "$CELL_OUTPUT"
