#!/bin/bash
set -euo pipefail
source /projects/paco0228/miniconda3/etc/profile.d/conda.sh
conda activate "${NEURON_ENV:?}"
cd "${ACE_CODE_ROOT:?}"
test "$(git rev-parse HEAD)" = "${ACE_SOURCE_REVISION:?}"
: "${UPSTREAM_ROOT:?}" "${CELL_OUTPUT:?}"
python -u scripts/research/neuronbench_custody_smoke.py \
  --upstream "$UPSTREAM_ROOT" \
  --protocol docs/development/guidance/protocol_neuronbench_custody_smoke_2026-09-27.json \
  --output "$CELL_OUTPUT" --source-revision "$ACE_SOURCE_REVISION"
python -u scripts/research/neuronbench_custody_targets.py \
  --upstream "$UPSTREAM_ROOT" \
  --protocol docs/development/guidance/protocol_neuronbench_custody_smoke_2026-09-27.json \
  --output "$CELL_OUTPUT"
