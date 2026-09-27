#!/bin/bash
set -euo pipefail
source /projects/paco0228/miniconda3/etc/profile.d/conda.sh
conda activate ace
cd "${ACE_CODE_ROOT:?}"
test "$(git rev-parse HEAD)" = "${ACE_SOURCE_REVISION:?}"
: "${SEED:?}" "${INPUT_ROOT:?}" "${CELL_OUTPUT:?}"
python -u scripts/research/boxing_lotka_fixed_data.py \
  --public "$INPUT_ROOT/seed_$SEED/public" --output "$CELL_OUTPUT"
python -u scripts/research/score_boxing_lotka_fixed_data.py \
  --models "$CELL_OUTPUT/models.json" \
  --private "$INPUT_ROOT/seed_$SEED/private" \
  --output "$CELL_OUTPUT/scores.csv"
python - "$CELL_OUTPUT" "$SEED" "$ACE_SOURCE_REVISION" <<'PY'
import csv,hashlib,json,sys
from pathlib import Path
out=Path(sys.argv[1]); seed=int(sys.argv[2]); revision=sys.argv[3]
rows=list(csv.DictReader((out/'scores.csv').open()))
assert len(rows)==3 and {r['arm'] for r in rows}=={'rbf','fourier','privileged_lotka_volterra'}
receipt={'seed':seed,'source_revision':revision,'n_train':8,'n_heldout':16,
         'models_sha256':hashlib.sha256((out/'models.json').read_bytes()).hexdigest(),
         'scores_sha256':hashlib.sha256((out/'scores.csv').read_bytes()).hexdigest(),
         'closed_model_calls':0}
(out/'complete.json').write_text(json.dumps(receipt,indent=2)+'\n')
PY
