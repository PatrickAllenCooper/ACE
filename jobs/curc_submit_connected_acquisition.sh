#!/bin/bash
# Frozen 15-cell connected-motif development scale grid.
set -euo pipefail
cd /projects/paco0228/ACE
rev=$(git rev-parse HEAD)
test "$rev" = "${REQUIRED_REVISION:?Specify tested commit}"
test -z "$(git status --porcelain --untracked-files=no)"
root=${RESEARCH_ROOT:-/scratch/alpine/paco0228/ACE/results/research_connected_acquisition_dev_v0}
mkdir -p "$root/logs"
manifest="$root/submitted.tsv"
if [ ! -e "$manifest" ]; then
    printf 'job_id\tjob_name\trevision\taccount\toutput\tsubmitted_at\n' > "$manifest"
fi
count=0
for CELL_SEED in 200 201 202; do
    for setting in 15:3 30:3 100:3 30:1 30:10; do
        NODES=${setting%:*}
        MOTIFS=${setting#*:}
        CELL_OUTPUT="$root/nodes_${NODES}/motifs_${MOTIFS}/seed_${CELL_SEED}"
        if python scripts/research/validate_cell.py --kind connected_acquisition \
            --directory "$CELL_OUTPUT" >/dev/null 2>&1; then
            echo "SKIP verified $CELL_OUTPUT"
            continue
        fi
        name="acer_conn_n${NODES}_k${MOTIFS}_s${CELL_SEED}"
        if squeue -h -u "$USER" -o '%j' | grep -Fxq "$name"; then
            echo "SKIP queued $name"
            continue
        fi
        job=$(sbatch --parsable --account=ucb736_asc1 --partition=acpu --qos=cpu-normal \
            --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=2G --time=00:30:00 \
            --job-name="$name" --output="$root/logs/${name}_%j.out" \
            --error="$root/logs/${name}_%j.err" \
            --export="ALL,ACE_SOURCE_REVISION=$rev,CELL_SEED=$CELL_SEED,NODES=$NODES,MOTIFS=$MOTIFS,CELL_OUTPUT=$CELL_OUTPUT" \
            jobs/curc_connected_acquisition_seed.sh)
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$job" "$name" "$rev" ucb736_asc1 "$CELL_OUTPUT" "$(date -Is)" >> "$manifest"
        echo "SUBMITTED $name -> $job"
        count=$((count+1))
    done
done
echo "$count jobs submitted; revision $rev; manifest $manifest"
