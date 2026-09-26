#!/bin/bash
# Submit the fixed-budget persistent-learner pilot on CURC. Safe to rerun.
set -euo pipefail
cd /projects/paco0228/ACE
REV=$(git rev-parse HEAD)
test "$REV" = "${REQUIRED_REVISION:?Specify tested commit}"
test -z "$(git status --porcelain --untracked-files=no)"
ROOT=${ROOT:-/scratch/alpine/paco0228/ACE/results/research_persistent_20260925}
mkdir -p "$ROOT/logs"
LEDGER="$ROOT/submitted.tsv"
if [ ! -e "$LEDGER" ]; then
    printf 'job_id\tjob_name\trevision\toutput\tsubmitted_at\n' > "$LEDGER"
fi
N=0
for FAMILY in ${FAMILIES:-legacy5 hom30 hetero30}; do
    for METHOD in ${METHODS:-nonleaf_random_ens nonleaf_coverage_ens pev pev_var}; do
        for SEED in ${SEEDS:-42 123 456}; do
            CELL_OUTPUT="$ROOT/$FAMILY/$METHOD/seed_$SEED"
            if python scripts/research/validate_cell.py --kind persistent --directory "$CELL_OUTPUT" >/dev/null 2>&1; then
                echo "SKIP verified: $FAMILY $METHOD $SEED"
                continue
            fi
            NAME="acer_persist_${FAMILY}_${METHOD}_s${SEED}"
            if squeue -h -u "$USER" -o '%j' | grep -Fxq "$NAME"; then
                echo "SKIP queued: $NAME"
                continue
            fi
            ID=$(sbatch --parsable --account=ucb736_asc1 --partition=acpu --qos=cpu-normal \
                --job-name="$NAME" --nodes=1 --ntasks=1 --cpus-per-task=4 \
                --mem=16G --time=02:00:00 \
                --output="$ROOT/logs/%x_%j.out" --error="$ROOT/logs/%x_%j.err" \
                --export="ALL,ACE_SOURCE_REVISION=$REV,FAMILY=$FAMILY,METHOD=$METHOD,SEED=$SEED,CELL_OUTPUT=$CELL_OUTPUT,BUDGET=${BUDGET:-2000},EPOCHS=${EPOCHS:-20},ENSEMBLE_SIZE=${ENSEMBLE_SIZE:-3}" \
                jobs/curc_persistent_scm_cell.sh)
            printf '%s\t%s\t%s\t%s\t%s\n' "$ID" "$NAME" "$REV" "$CELL_OUTPUT" "$(date -Is)" >> "$LEDGER"
            echo "Submitted: $NAME -> $ID"
            N=$((N+1))
        done
    done
done
echo "$N persistent campaigns submitted -> $ROOT"
