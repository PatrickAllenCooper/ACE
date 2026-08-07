#!/bin/bash
# =============================================================================
# E4: free-lookahead candidate-breadth (K) scaling at N=30.
#
# The existing K-ablation (jobs/curc_submit_k_ablation.sh, N=50) sweeps K
# under STANDARD lookahead, where every extra candidate is an extra query
# against the ground-truth environment -- so higher K is not free, and its
# benefit is confounded with its extra query cost.
#
# Under --lookahead_on_student, candidates are scored against the student's
# own current beliefs (StudentSCM.forward), so lookahead makes zero oracle
# queries: K candidates cost exactly the same reported query budget as K=1.
# This isolates the pure selection-quality question -- does scoring more
# candidates per step improve best-MSE when doing so is genuinely free -- from
# the query-budget question the standard-lookahead K-ablation is entangled
# with.
#
# K in {4, 8, 16, 32}, N=30, 3 seeds, default Qwen2.5-1.5B / full prompt.
#
# Usage (from /projects/paco0228/ACE):
#   cd /projects/paco0228/ACE
#   git pull
#   bash jobs/curc_submit_k_scaling_student.sh
#
# Output: results/curc_k_scaling_student/K{k}/nodes30/ace/seed_{seed}/...
#
# GPU targeting: default RTX Pro 6000 (K=32 is ~8x the per-step LM calls of
# the paper's K=4 default, but still comfortably fits a single 1.5B policy).
#   GPU_PARTITION=<partition> GPU_QOS=<qos> GPU_GRES=<gres> \
#       bash jobs/curc_submit_k_scaling_student.sh
# =============================================================================

set -euo pipefail

GPU_PARTITION="${GPU_PARTITION:-artxpro6000}"
GPU_QOS="${GPU_QOS:-gpu-normal}"
GPU_GRES="${GPU_GRES:-gpu:rtx_pro_6000:1}"

cd /projects/paco0228/ACE

source /projects/paco0228/miniconda3/etc/profile.d/conda.sh
conda activate ace

BASE="/scratch/alpine/paco0228/ACE/results/curc_k_scaling_student"
mkdir -p "$BASE/logs"
WORKER="jobs/curc_scaling_seed.sh"

SEEDS="${SEEDS:-42 123 456}"
KS="${KS:-4 8 16 32}"

echo "================================================================"
echo " Free-lookahead K scaling at N=30 (student-mode) -- Ks=$KS Seeds=$SEEDS"
echo " Started: $(date)"
echo "================================================================"

for K in $KS; do
    OUT="$BASE/K${K}"
    mkdir -p "$OUT/logs"
    # Per-step cost scales roughly linearly with K (one LM forward pass per
    # candidate); budget wall-time accordingly.
    case "$K" in
        4)  WALL=10:00:00 ;;
        8)  WALL=14:00:00 ;;
        16) WALL=20:00:00 ;;
        32) WALL=30:00:00 ;;
        *)  WALL=24:00:00 ;;
    esac
    for SEED in $SEEDS; do
        name="ksK${K}_n30_s${SEED}"
        JOB=$(sbatch --parsable \
            --job-name="$name" \
            --partition="$GPU_PARTITION" --qos="$GPU_QOS" \
            --nodes=1 --ntasks=1 --gres="$GPU_GRES" \
            --cpus-per-task=8 --mem=48G \
            --time="$WALL" \
            --output="$BASE/logs/${name}_%j.out" \
            --error="$BASE/logs/${name}_%j.err" \
            --export=ALL,SCALE=30,METHOD=ace,SEED=$SEED,OUT=$OUT,CANDIDATES=$K,LOOKAHEAD_STUDENT=1,EPISODES=120 \
            "$WORKER")
        echo "  Submitted: $name (K=$K) -> Job $JOB"
    done
done

echo ""
echo "Monitor: squeue -u \$USER ; logs in $BASE/logs/"
