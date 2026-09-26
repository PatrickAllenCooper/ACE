#!/bin/bash
# Frozen Bayesian source-library confirmation; numerical CPU work only.
set -euo pipefail
cd "${ACE_CODE_ROOT:-/projects/paco0228/ACE}"
rev=$(git rev-parse HEAD)
test "$rev" = "${REQUIRED_REVISION:?Specify tested commit}"
test -z "$(git status --porcelain --untracked-files=no)"
root=${RESEARCH_ROOT:-/scratch/alpine/paco0228/ACE/results/research_transfer_bayes_v2}
mkdir -p "$root/logs"
manifest="$root/submitted.tsv"
if [ ! -e "$manifest" ]; then
    printf 'job_id\tjob_name\trevision\taccount\toutput\tsubmitted_at\n' > "$manifest"
fi
count=0
for SEED in {4000..4019}; do
    CELL_OUTPUT="$root/seed_$SEED"
    ready=1
    for type in family coefficient; do
        for k in 1 3 10; do
            if ! python scripts/research/validate_cell.py --kind learned_transfer \
                --directory "$CELL_OUTPUT/$type/changed_$k" >/dev/null 2>&1; then ready=0; fi
        done
    done
    if [ "$ready" = 1 ]; then echo "SKIP verified seed $SEED"; continue; fi
    name="acer_transferb_s${SEED}"
    if squeue -h -u "$USER" -o '%j' | grep -Fxq "$name"; then
        echo "SKIP queued $name"
        continue
    fi
    job=$(sbatch --parsable --account=ucb736_asc1 --partition=acpu --qos=cpu-normal \
        --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=2G --time=00:30:00 \
        --job-name="$name" --output="$root/logs/${name}_%j.out" \
        --error="$root/logs/${name}_%j.err" \
        --export="ALL,ACE_CODE_ROOT=$PWD,ACE_SOURCE_REVISION=$rev,TRANSFER_BAYES_MIXTURE=1,SEED=$SEED,CELL_OUTPUT=$CELL_OUTPUT" \
        jobs/curc_learned_transfer_seed.sh)
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$job" "$name" "$rev" ucb736_asc1 "$CELL_OUTPUT" "$(date -Is)" >> "$manifest"
    echo "SUBMITTED $name -> $job"
    count=$((count+1))
done
echo "$count jobs submitted; revision $rev; manifest $manifest"
