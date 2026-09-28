#!/bin/bash
set -euo pipefail
source /projects/paco0228/miniconda3/etc/profile.d/conda.sh
conda activate "${NEURON_ENV:?}"
cd "${ACE_CODE_ROOT:?}"
test "$(git rev-parse HEAD)" = "${ACE_SOURCE_REVISION:?}"
python -u scripts/research/neuronbench_export_public_questions.py \
  --upstream "${UPSTREAM_ROOT:?}" \
  --source-hashes docs/development/guidance/protocol_neuronbench_custody_smoke_2026-09-27.json \
  --protocol "${NEURON_PROTOCOL:-docs/development/guidance/protocol_neuronbench_public_controls_dev_v2_2026-09-28.json}" \
  --output "${EXPORT_OUTPUT:?}" --source-revision "$ACE_SOURCE_REVISION"
