#!/bin/bash
# =============================================================================
# Node-importance ablation on the 5-node benchmark: full (cov_bonus=60, paper
# default) vs no_node_importance (cov_bonus=0), isolating w(V_i, {L_j}) from
# the information-gain and diversity reward terms. Complements the existing
# no-diversity row in the component-ablation table (Table 3) and addresses
# reviewer wZrW's request for a node-importance ablation.
#
# 2 configs x 3 seeds = 6 jobs (matches the N=3-seed pilot used for the other
# component-ablation rows).
#
# Usage (from /projects/paco0228/ACE):
#   cd /projects/paco0228/ACE
#   git pull
#   bash jobs/curc_submit_node_importance_ablation.sh
#
# Output: written to /scratch/alpine1/paco0228 (NOT /projects, which is at
# 229G/250G quota) -- see curc-quota before adding further large result trees.
#   /scratch/alpine1/paco0228/ACE/results/curc_node_importance_ablation/{config}/seed_{seed}/
#
# SLURM resources per job:
#   partition : aa100 (A100 GPU)
#   time      : 08:00:00 (same config as the main 5-node ACE runs)
# =============================================================================

set -euo pipefail

cd /projects/paco0228/ACE

source /projects/paco0228/miniconda3/etc/profile.d/conda.sh
conda activate ace

OUT="/scratch/alpine1/paco0228/ACE/results/curc_node_importance_ablation"
mkdir -p "$OUT/logs"

echo "================================================================"
echo " Node-importance ablation -- CURC SLURM (6 jobs)"
echo "================================================================"
echo " Output : $OUT"
echo " Started: $(date)"
echo "================================================================"

CONFIGS="full no_node_importance"
SEEDS="42 123 456"

for CONFIG in $CONFIGS; do
    for SEED in $SEEDS; do
        JOB=$(sbatch --parsable \
            --job-name="nodeimp_${CONFIG:0:6}_s${SEED}" \
            --partition=aa100 --qos=normal \
            --nodes=1 --ntasks=1 --gres=gpu:1 \
            --cpus-per-task=8 --mem=32G \
            --time=08:00:00 \
            --output="$OUT/logs/${CONFIG}_seed${SEED}_%j.out" \
            --error="$OUT/logs/${CONFIG}_seed${SEED}_%j.err" \
            --export=ALL,CONFIG=$CONFIG,SEED=$SEED,OUT=$OUT \
            jobs/curc_node_importance_ablation_seed.sh)
        echo "  Submitted: config=$CONFIG seed=$SEED -> Job $JOB"
    done
done

echo ""
echo "6 jobs submitted."
echo "Monitor with:  squeue -u \$USER"
echo "Logs in:       $OUT/logs/"
echo ""
echo "When complete, pull results locally with:"
echo "  scp -r paco0228@login.rc.colorado.edu:$OUT ./results/"
