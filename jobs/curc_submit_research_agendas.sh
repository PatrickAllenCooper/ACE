#!/bin/bash
# Submit staged CPU cells for the Sept 25 SCM research portfolio.
# All runs are numerical. No Azure or other closed-model API calls.
# RUN_NUMERICAL=0 or RUN_PEV=0 can scope a submission wave.
set -euo pipefail
cd /projects/paco0228/ACE
rev=$(git rev-parse HEAD)
if [ -n "${REQUIRED_REVISION:-}" ] && [ "$rev" != "$REQUIRED_REVISION" ]; then
    echo "Remote revision $rev differs from required $REQUIRED_REVISION" >&2
    exit 2
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Tracked remote checkout is dirty; use a clean explicit revision" >&2
    exit 2
fi
root="${RESEARCH_ROOT:-/scratch/alpine/paco0228/ACE/results/research_portfolio_20260925}"
mkdir -p "$root/logs"
manifest="$root/submitted.tsv"
if [ ! -e "$manifest" ]; then printf 'job_id\tjob_name\trevision\toutput\tsubmitted_at\n' > "$manifest"; fi
count=0

submit_cell() {
    local name=$1 kind=$2 output=$3 time_limit=$4 mem=$5 cpus=$6 exports=$7 valid_kind=$8
    if [ "$valid_kind" = pev ]; then
        local check_dir="$output/$PEV_ARM/seed_$CELL_SEED"
    else
        local check_dir="$output"
    fi
    if python scripts/research/validate_cell.py --kind "$valid_kind" --directory "$check_dir" >/dev/null 2>&1; then
        echo "SKIP verified: $name"
        return
    fi
    if [ -n "$(squeue -h -u "$USER" -n "$name" -o '%i')" ]; then
        echo "SKIP in flight: $name"
        return
    fi
    local job
    job=$(sbatch --parsable --account=ucb736_asc1 --partition=acpu --qos=cpu-normal \
        --nodes=1 --ntasks=1 --cpus-per-task="$cpus" --mem="$mem" --time="$time_limit" \
        --job-name="$name" --output="$root/logs/${name}_%j.out" \
        --error="$root/logs/${name}_%j.err" \
        --export="ALL,CELL_KIND=$kind,CELL_OUTPUT=$output,CELL_SEED=$CELL_SEED,ACE_SOURCE_REVISION=$rev,$exports" \
        jobs/curc_research_cell.sh)
    printf '%s\t%s\t%s\t%s\t%s\n' "$job" "$name" "$rev" "$check_dir" "$(date -Is)" >> "$manifest"
    count=$((count + 1))
    echo "SUBMITTED $name -> $job"
}

if [ "${RUN_NUMERICAL:-1}" = 1 ]; then
    for CELL_SEED in ${NUMERICAL_SEEDS:-42 123 456}; do
        submit_cell "acer_prior_s${CELL_SEED}" agenda "$root/numerical/prior/seed_$CELL_SEED" \
            00:30:00 2G 1 'AGENDA_TRACK=prior' agenda
        for BACKGROUND_SD in 0.0 0.15; do
            for ACTUATOR_PENALTY in 0 1; do
                sd_tag=${BACKGROUND_SD/./p}
                name="acer_design_b${sd_tag}_c${ACTUATOR_PENALTY}_s${CELL_SEED}"
                output="$root/numerical/design/background_$sd_tag/cost_$ACTUATOR_PENALTY/seed_$CELL_SEED"
                submit_cell "$name" agenda "$output" 00:30:00 2G 1 \
                    "AGENDA_TRACK=design,BACKGROUND_SD=$BACKGROUND_SD,ACTUATOR_PENALTY=$ACTUATOR_PENALTY" agenda
            done
        done
        for CHANGED in 1 3 10; do
            submit_cell "acer_transfer_k${CHANGED}_s${CELL_SEED}" agenda \
                "$root/numerical/transfer/changed_$CHANGED/seed_$CELL_SEED" \
                00:30:00 2G 1 "AGENDA_TRACK=transfer,CHANGED=$CHANGED" agenda
        done
    done
fi

if [ "${RUN_PEV:-1}" = 1 ]; then
    CELL_SEED="${PEV_SEED:-42}"
    for PEV_FAMILY in legacy5 hom30 hetero30; do
        for PEV_ARM in random_ens nonleaf_coverage_ens pev pev_var; do
            output="$root/pev_canary/$PEV_FAMILY"
            submit_cell "acer_pev_${PEV_FAMILY}_${PEV_ARM}_s${CELL_SEED}" pev "$output" \
                02:00:00 16G 4 "PEV_FAMILY=$PEV_FAMILY,PEV_ARM=$PEV_ARM" pev
        done
    done
fi
echo "$count new jobs; revision $rev; manifest $manifest"
