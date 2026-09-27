#!/bin/bash
# Numerical-only external benchmark custody check. No LM/API calls.
set -euo pipefail
source /projects/paco0228/miniconda3/etc/profile.d/conda.sh
conda activate ace
cd "${ACE_CODE_ROOT:?}"
test "$(git rev-parse HEAD)" = "${ACE_SOURCE_REVISION:?}"
: "${BOXING_UPSTREAM_ROOT:?}" "${SEED:?}" "${CELL_OUTPUT:?}"
python -u scripts/research/boxing_lotka_smoke.py \
  --upstream "$BOXING_UPSTREAM_ROOT" \
  --upstream-revision b43e38cb03d09c13efa9cf4d9bae740d51157bfd \
  --expected-source-sha256 e86c35069c49988948737ca89f939cae8e0138682cc62abb99eb97857de9abba \
  --seed "$SEED" --observations 8 --holdout 16 --output "$CELL_OUTPUT"
