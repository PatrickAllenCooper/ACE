#!/bin/bash
# One numerical research-portfolio cell. No model API calls.
set -euo pipefail
source /projects/paco0228/miniconda3/etc/profile.d/conda.sh
conda activate ace
cd /projects/paco0228/ACE
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1

: "${CELL_KIND:?}" "${CELL_SEED:?}" "${CELL_OUTPUT:?}" "${ACE_SOURCE_REVISION:?}"
test "$(git rev-parse HEAD)" = "$ACE_SOURCE_REVISION"

if [ "$CELL_KIND" = agenda ]; then
    : "${AGENDA_TRACK:?}"
    args=(--track "$AGENDA_TRACK" --seed "$CELL_SEED" --output "$CELL_OUTPUT")
    case "$AGENDA_TRACK" in
        prior|prior_gate) ;;
        design) args+=(--budget 400 --background-sd "$BACKGROUND_SD" --actuator-penalty "$ACTUATOR_PENALTY") ;;
        transfer) args+=(--budget 400 --nodes 30 --changed "$CHANGED") ;;
        *) echo "Unknown agenda track" >&2; exit 2 ;;
    esac
    python -u scripts/research/agenda_runner.py "${args[@]}"
    python scripts/research/validate_cell.py --kind agenda --directory "$CELL_OUTPUT"
elif [ "$CELL_KIND" = pev ]; then
    : "${PEV_FAMILY:?}" "${PEV_ARM:?}"
    runner=scripts/runners/run_30node_baseline_seed.py
    extra=(--family large_scale --n_nodes 30)
    if [ "$PEV_FAMILY" = legacy5 ]; then
        runner=scripts/runners/run_5node_baseline_seed.py
        extra=()
    elif [ "$PEV_FAMILY" = hetero30 ]; then
        extra=(--family hetero --n_nodes 30)
    elif [ "$PEV_FAMILY" != hom30 ]; then
        echo "Unknown PEV family" >&2; exit 2
    fi
    method="$PEV_ARM"
    if [ "$PEV_ARM" = pev_var ]; then method=pev; extra+=(--pev_scoring var); fi
    python -u "$runner" --method "$method" --tag "$PEV_ARM" --seed "$CELL_SEED" \
        --episodes 1 --steps 8 --obs_train_interval 3 --obs_train_samples 40 \
        --train_epochs 20 --ensemble_size 3 --pev_values 5 --pev_sim 8 \
        --output "$CELL_OUTPUT" "${extra[@]}"
    run_dir="$CELL_OUTPUT/$PEV_ARM/seed_$CELL_SEED"
    python - "$run_dir" "$ACE_SOURCE_REVISION" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
d = Path(sys.argv[1])
metric = d / ('node_losses.csv' if (d / 'node_losses.csv').exists() else 'results.csv')
receipt = {'kind': 'pev_canary', 'steps': 8, 'source_revision': sys.argv[2],
           'job_id': os.environ.get('SLURM_JOB_ID'),
           'metrics_sha256': hashlib.sha256(metric.read_bytes()).hexdigest()}
tmp = d / 'complete.json.tmp'
tmp.write_text(json.dumps(receipt, indent=2) + '\n')
os.replace(tmp, d / 'complete.json')
PY
    python scripts/research/validate_cell.py --kind pev --directory "$run_dir" --steps 8
else
    echo "Unknown cell kind" >&2; exit 2
fi
