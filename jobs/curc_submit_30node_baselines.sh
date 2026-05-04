#!/bin/bash
# =============================================================================
# Submit 30-node MLP-learner baselines: Random, Round-Robin, Max-Variance
# 3 methods x 5 seeds = 15 independent CPU jobs (all run in parallel)
#
# Usage (from /projects/paco0228/ACE):
#   cd /projects/paco0228/ACE
#   bash jobs/curc_submit_30node_baselines.sh
#
# Results land in: results/curc_30node_baselines/
#   results/curc_30node_baselines/{method}/seed_{seed}/summary.csv
#   results/curc_30node_baselines/{method}/seed_{seed}/per_episode.csv
#
# After all jobs finish, aggregate with:
#   python scripts/runners/aggregate_30node_baselines.py
#
# SLURM resources per job:
#   partition : amilan (CPU only; no GPU needed)
#   QoS       : normal
#   time      : 08:00:00  (~150 episodes x 25 steps x 1.5s/step x safety margin)
#   CPUs      : 4
#   MEM       : 8G
# =============================================================================

set -euo pipefail

cd /projects/paco0228/ACE

source /projects/paco0228/miniconda3/etc/profile.d/conda.sh
conda activate ace

OUT="/projects/paco0228/ACE/results/curc_30node_baselines"
mkdir -p "$OUT/logs"

echo "================================================================"
echo " 30-node Baselines -- CURC SLURM (15 jobs)"
echo "================================================================"
echo " Output : $OUT"
echo " Started: $(date)"
echo "================================================================"

METHODS="random round_robin max_variance"
SEEDS="42 123 456 789 1011"

for METHOD in $METHODS; do
    for SEED in $SEEDS; do

        JOB=$(sbatch --parsable \
            --job-name="30bl_${METHOD:0:3}_s${SEED}" \
            --partition=amilan \
            --qos=normal \
            --nodes=1 --ntasks=1 \
            --cpus-per-task=4 \
            --mem=8G \
            --time=08:00:00 \
            --output="$OUT/logs/${METHOD}_seed${SEED}_%j.out" \
            --error="$OUT/logs/${METHOD}_seed${SEED}_%j.err" \
            --export=ALL,METHOD=$METHOD,SEED=$SEED,OUT=$OUT \
            jobs/curc_30node_baseline_seed.sh)

        echo "  Submitted: method=$METHOD seed=$SEED -> Job $JOB"
    done
done

echo ""
echo "All 15 jobs submitted."
echo "Monitor with:  squeue -u \$USER"
echo "Logs in:       $OUT/logs/"
echo ""
echo "When complete, pull results locally with:"
echo "  scp -r paco0228@login.rc.colorado.edu:$OUT ./results/"
echo ""
echo "Then aggregate:"
echo "  python scripts/runners/aggregate_30node_baselines.py"
