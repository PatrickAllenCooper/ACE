#!/bin/bash
set -euo pipefail
source /projects/paco0228/miniconda3/etc/profile.d/conda.sh
conda activate ace
cd "${ACE_CODE_ROOT:?}"
test "$(git rev-parse HEAD)" = "${ACE_SOURCE_REVISION:?}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
: "${SEED:?}" "${CONDITION:?}" "${PUBLIC_ROOT:?}" "${MODEL_ROOT:?}" "${CELL_OUTPUT:?}"
python -u scripts/research/run_boxing_lotka_open_proposal.py \
  --public "$PUBLIC_ROOT/seed_$SEED/public" --condition "$CONDITION" \
  --model "$MODEL_ROOT" --model-revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  --source-revision "$ACE_SOURCE_REVISION" --seed "$SEED" \
  --output "$CELL_OUTPUT"
