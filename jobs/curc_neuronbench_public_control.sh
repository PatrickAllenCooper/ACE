#!/bin/bash
set -euo pipefail
source /projects/paco0228/miniconda3/etc/profile.d/conda.sh
conda activate "${NEURON_ENV:?}"
cd "${ACE_CODE_ROOT:?}"
test "$(git rev-parse HEAD)" = "${ACE_SOURCE_REVISION:?}"
: "${UPSTREAM_ROOT:?}" "${CELL_OUTPUT:?}" "${PLAN_PATH:?}" "${WORLD:?}"
hashes=docs/development/guidance/protocol_neuronbench_custody_smoke_2026-09-27.json
python -u scripts/research/neuronbench_control_oracle.py --upstream "$UPSTREAM_ROOT" \
  --source-hashes "$hashes" --plan "$PLAN_PATH" --world "$WORLD" --seed 42 \
  --output "$CELL_OUTPUT" --source-revision "$ACE_SOURCE_REVISION"
python -u scripts/research/neuronbench_public_control.py forecast \
  --public "$CELL_OUTPUT/public" --ridge 1.0 --output "$CELL_OUTPUT/predictions.json"
python -u scripts/research/neuronbench_control_score.py --upstream "$UPSTREAM_ROOT" \
  --source-hashes "$hashes" --output "$CELL_OUTPUT"
