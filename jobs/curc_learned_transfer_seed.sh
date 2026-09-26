#!/bin/bash
# One six-cell numerical transfer screen. No model API calls.
set -euo pipefail
source /projects/paco0228/miniconda3/etc/profile.d/conda.sh
conda activate ace
cd /projects/paco0228/ACE
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
: "${ACE_SOURCE_REVISION:?}" "${SEED:?}" "${CELL_OUTPUT:?}"
test "$(git rev-parse HEAD)" = "$ACE_SOURCE_REVISION"
for change_type in family coefficient; do
    for changed in 1 3 10; do
        output="$CELL_OUTPUT/$change_type/changed_$changed"
        python -u scripts/research/learned_transfer.py --seed "$SEED" \
            --changed "$changed" --change-type "$change_type" --output "$output"
        python scripts/research/validate_cell.py --kind learned_transfer --directory "$output"
    done
done
