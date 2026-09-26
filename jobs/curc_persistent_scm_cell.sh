#!/bin/bash
# One CPU-only persistent SCM campaign; no closed-model services.
set -euo pipefail
source /projects/paco0228/miniconda3/etc/profile.d/conda.sh
conda activate ace
cd /projects/paco0228/ACE
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
: "${ACE_SOURCE_REVISION:?}" "${FAMILY:?}" "${METHOD:?}" "${SEED:?}" "${CELL_OUTPUT:?}"
test "$(git rev-parse HEAD)" = "$ACE_SOURCE_REVISION"
python -u scripts/research/persistent_scm.py \
    --family "$FAMILY" --method "$METHOD" --seed "$SEED" \
    --budget "${BUDGET:-2000}" --epochs "${EPOCHS:-20}" \
    --ensemble-size "${ENSEMBLE_SIZE:-3}" --output "$CELL_OUTPUT"
python scripts/research/validate_cell.py --kind persistent --directory "$CELL_OUTPUT"
