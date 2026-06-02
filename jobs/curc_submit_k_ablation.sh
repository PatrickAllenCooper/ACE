#!/bin/bash
# =============================================================================
# Phase 4 ablation: lookahead breadth K at N=50.
#
# Scaling principle under test: the action space (~N x value-grid) grows with
# N, so a fixed K=4 samples a shrinking fraction of candidate interventions.
# This sweeps K in {4, 8, 16} at N=50 (compact prompt held fixed) to measure
# whether more candidates per step buy lower best-MSE -- and at what wallclock
# cost (cost ~ prompt_len x K x steps).
#
# Output: results/scaling_kablation/nodes50/ace/seed_<seed>/...   (one tree per K
# via the K-tagged OUT root so cells do not collide).
#
# Usage:
#   git pull
#   bash jobs/curc_submit_k_ablation.sh
# =============================================================================

set -euo pipefail

cd /projects/paco0228/ACE
source /projects/paco0228/miniconda3/etc/profile.d/conda.sh
conda activate ace

BASE="/projects/paco0228/ACE/results/scaling_kablation"
mkdir -p "$BASE/logs"

SEEDS="${SEEDS:-42 123 456}"
KS="${KS:-4 8 16}"
WORKER="jobs/curc_scaling_seed.sh"

echo "================================================================"
echo " K (lookahead breadth) ablation at N=50  --  Ks=$KS  Seeds=$SEEDS"
echo " Started: $(date)"
echo "================================================================"

for K in $KS; do
    OUT="$BASE/K${K}"
    mkdir -p "$OUT/logs"
    for SEED in $SEEDS; do
        name="k${K}_n50_s${SEED}"
        # aa100 normal-QOS caps wall-time at 24h, so every K uses the 24h
        # ceiling. Higher K is slower per episode, so larger-K cells simply
        # need more resume windows (re-run this script; STABLE dirs continue
        # from checkpoint) rather than a longer single window.
        WALL=24:00:00
        JOB=$(sbatch --parsable \
            --job-name="$name" \
            --partition=aa100 --qos=normal \
            --nodes=1 --ntasks=1 --gres=gpu:1 \
            --cpus-per-task=8 --mem=78G \
            --time="$WALL" \
            --output="$BASE/logs/${name}_%j.out" \
            --error="$BASE/logs/${name}_%j.err" \
            --export=ALL,SCALE=50,METHOD=ace,SEED=$SEED,OUT=$OUT,PROMPT_STRATEGY=compact,PROMPT_TOP_M=8,CANDIDATES=$K,EPISODES=40 \
            "$WORKER")
        echo "  Submitted: $name (K=$K) -> Job $JOB"
    done
done

echo ""
echo "Monitor: squeue -u \$USER ; logs in $BASE/logs/"
