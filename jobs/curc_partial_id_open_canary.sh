#!/bin/bash
set -euo pipefail
source /projects/paco0228/miniconda3/etc/profile.d/conda.sh
conda activate ace
cd "${ACE_CODE_ROOT:?}"
test "$(git rev-parse HEAD)" = "${ACE_SOURCE_REVISION:?}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
: "${PUBLIC_FILE:?}" "${REVEAL_FILE:?}" "${MODEL_ROOT:?}" "${CELL_OUTPUT:?}"
python -u scripts/research/run_partial_id_open_model.py \
  --public "$PUBLIC_FILE" --reveals "$REVEAL_FILE" \
  --ids pair_2100 pair_2101 pair_2102 pair_2103 \
  --model "$MODEL_ROOT" --source-revision "$ACE_SOURCE_REVISION" \
  --output "$CELL_OUTPUT"
