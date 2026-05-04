#!/bin/bash
# =============================================================================
# Submit 2 more ACE 30-node seeds to reach N=5.
#
# Existing results: seeds 42, 123, 456 (best-loss 2.79, 1.80, 1.27)
# New seeds: 789, 1011 to get N=5 total matching the baseline set
#
# Usage (from /projects/paco0228/ACE):
#   cd /projects/paco0228/ACE
#   bash jobs/curc_submit_ace_30node_extra_seeds.sh
#
# Results land in: results/curc_30node_baselines/ace/seed_{seed}/
# (co-located with baselines so aggregate_30node_baselines.py can pick them up)
#
# Resources per job:
#   partition : aa100 (A100 80GB)
#   QoS       : normal
#   time      : 08:00:00
#   GPUs      : 1
#   CPUs      : 8
#   MEM       : 32G
# =============================================================================

set -euo pipefail

cd /projects/paco0228/ACE

source /projects/paco0228/miniconda3/etc/profile.d/conda.sh
conda activate ace

OUT="/projects/paco0228/ACE/results/curc_30node_ace_extra"
mkdir -p "$OUT/logs"

echo "================================================================"
echo " ACE 30-node extra seeds (to reach N=5)"
echo "================================================================"
echo " Output : $OUT"
echo " Started: $(date)"
echo "================================================================"

for SEED in 789 1011; do
    JOB=$(sbatch --parsable \
        --job-name="ace30_s${SEED}" \
        --partition=aa100 \
        --qos=normal \
        --nodes=1 --ntasks=1 --gres=gpu:1 \
        --cpus-per-task=8 \
        --mem=32G \
        --time=08:00:00 \
        --output="$OUT/logs/ace30_seed${SEED}_%j.out" \
        --error="$OUT/logs/ace30_seed${SEED}_%j.err" \
        --export=ALL,SEED=$SEED,OUT=$OUT \
        jobs/curc_large_scale_seed.sh)

    echo "  Submitted: ACE seed=$SEED -> Job $JOB"
done

echo ""
echo "2 jobs submitted."
echo "Monitor with:  squeue -u \$USER"
echo "Logs in:       $OUT/logs/"
echo ""
echo "When complete, pull locally:"
echo "  scp -r paco0228@login.rc.colorado.edu:$OUT ./results/"
