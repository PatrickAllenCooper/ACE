#!/bin/bash
# nodes50 Phase 2 only (10 jobs). Use when anon30 Phase 1 already submitted.
# Full 20-job flow: jobs/curc_submit_30node_followup.sh
#
# cd /projects/paco0228/ACE && git pull && bash jobs/curc_submit_30node_followup_phase2_only.sh

set -euo pipefail

cd /projects/paco0228/ACE

source /projects/paco0228/miniconda3/etc/profile.d/conda.sh
conda activate ace

OUT="/projects/paco0228/ACE/results/curc_30node_followup"
mkdir -p "$OUT/logs"

SEEDS="42 123 456 789 1011"

echo ">>> Phase 2 only: nodes50 (50-node SCM) <<<"
for SEED in $SEEDS; do
    JOB=$(sbatch --parsable \
        --job-name="ace_n50_s${SEED}" \
        --partition=aa100 --qos=normal \
        --nodes=1 --ntasks=1 --gres=gpu:1 \
        --cpus-per-task=8 --mem=78G \
        --time=12:00:00 \
        --output="$OUT/logs/ace_nodes50_seed${SEED}_%j.out" \
        --error="$OUT/logs/ace_nodes50_seed${SEED}_%j.err" \
        --export=ALL,CONDITION=nodes50,METHOD=ace,SEED=$SEED,OUT=$OUT \
        jobs/curc_30node_followup_seed.sh)
    echo "  Submitted: ACE nodes50 seed=$SEED -> Job $JOB"
done

for SEED in $SEEDS; do
    JOB=$(sbatch --parsable \
        --job-name="zsl_n50_s${SEED}" \
        --partition=aa100 --qos=normal \
        --nodes=1 --ntasks=1 --gres=gpu:1 \
        --cpus-per-task=8 --mem=78G \
        --time=10:00:00 \
        --output="$OUT/logs/zsl_nodes50_seed${SEED}_%j.out" \
        --error="$OUT/logs/zsl_nodes50_seed${SEED}_%j.err" \
        --export=ALL,CONDITION=nodes50,METHOD=zero_shot_lm,SEED=$SEED,OUT=$OUT \
        jobs/curc_30node_followup_seed.sh)
    echo "  Submitted: Zero-shot LM nodes50 seed=$SEED -> Job $JOB"
done

echo "Phase 2 done (10 jobs). Logs: $OUT/logs/"
