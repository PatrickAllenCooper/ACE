#!/bin/bash
# =============================================================================
# Rebuttal experiments for 30-node SCM (per docs/REBUTTAL_PREP.md)
#
# Submits 3 method classes x 5 seeds = 15 jobs total:
#
#   1. PPO on 30-node                 (5 seeds, ~6-8h each, GPU)
#   2. Zero-shot LM (--no_dpo)        (5 seeds, ~6h each, GPU)
#   3. Bayesian OED on 30-node        (5 seeds, ~12-16h each, CPU OK)
#
# Each addresses a specific reviewer concern:
#   PPO          -> isolates DPO contribution at scale (same reward as ACE)
#   Zero-shot LM -> tests whether the pretrained LM prior alone is enough
#   Bayesian OED -> the strongest principled baseline at 30 nodes
#
# Usage (from /projects/paco0228/ACE):
#   cd /projects/paco0228/ACE
#   bash jobs/curc_submit_30node_rebuttal.sh
#
# Output structure:
#   results/curc_30node_rebuttal/
#     ppo/seed_{seed}/
#     zero_shot_lm/seed_{seed}/job_{jobid}/
#     bayesian_oed/seed_{seed}/
#
# Aggregate after completion:
#   python scripts/runners/aggregate_30node_rebuttal.py
# =============================================================================

set -euo pipefail

cd /projects/paco0228/ACE

source /projects/paco0228/miniconda3/etc/profile.d/conda.sh
conda activate ace

OUT="/projects/paco0228/ACE/results/curc_30node_rebuttal"
mkdir -p "$OUT/logs"

echo "================================================================"
echo " 30-node Rebuttal Experiments -- CURC SLURM (15 jobs)"
echo "================================================================"
echo " Output : $OUT"
echo " Started: $(date)"
echo "================================================================"

SEEDS="42 123 456 789 1011"

# -----------------------------------------------------------------------------
# (1) PPO on 30-node: GPU, same MLP learner as ACE, 8h wall time
# -----------------------------------------------------------------------------
echo ""
echo ">>> Phase 1: PPO on 30-node <<<"
for SEED in $SEEDS; do
    JOB=$(sbatch --parsable \
        --job-name="ppo30_s${SEED}" \
        --partition=aa100 --qos=normal \
        --nodes=1 --ntasks=1 --gres=gpu:1 \
        --cpus-per-task=8 --mem=32G \
        --time=08:00:00 \
        --output="$OUT/logs/ppo_seed${SEED}_%j.out" \
        --error="$OUT/logs/ppo_seed${SEED}_%j.err" \
        --export=ALL,METHOD=ppo,SEED=$SEED,OUT=$OUT \
        jobs/curc_30node_rebuttal_seed.sh)
    echo "  Submitted: PPO seed=$SEED -> Job $JOB"
done

# -----------------------------------------------------------------------------
# (2) Zero-shot LM (ACE --no_dpo): GPU, 8h
# -----------------------------------------------------------------------------
echo ""
echo ">>> Phase 2: Zero-shot LM (no DPO) on 30-node <<<"
for SEED in $SEEDS; do
    JOB=$(sbatch --parsable \
        --job-name="zsl30_s${SEED}" \
        --partition=aa100 --qos=normal \
        --nodes=1 --ntasks=1 --gres=gpu:1 \
        --cpus-per-task=8 --mem=64G \
        --time=08:00:00 \
        --output="$OUT/logs/zero_shot_lm_seed${SEED}_%j.out" \
        --error="$OUT/logs/zero_shot_lm_seed${SEED}_%j.err" \
        --export=ALL,METHOD=zero_shot_lm,SEED=$SEED,OUT=$OUT \
        jobs/curc_30node_rebuttal_seed.sh)
    echo "  Submitted: Zero-shot LM seed=$SEED -> Job $JOB"
done

# -----------------------------------------------------------------------------
# (3) Bayesian OED: CPU partition (no GPU needed; per-step compute is MLP
#     training of M=3 cloned learners x 10 candidates), 16h wall.
# -----------------------------------------------------------------------------
echo ""
echo ">>> Phase 3: Bayesian OED on 30-node <<<"
for SEED in $SEEDS; do
    JOB=$(sbatch --parsable \
        --job-name="boed30_s${SEED}" \
        --partition=amilan --qos=normal \
        --nodes=1 --ntasks=1 \
        --cpus-per-task=8 --mem=16G \
        --time=16:00:00 \
        --output="$OUT/logs/bayesian_oed_seed${SEED}_%j.out" \
        --error="$OUT/logs/bayesian_oed_seed${SEED}_%j.err" \
        --export=ALL,METHOD=bayesian_oed,SEED=$SEED,OUT=$OUT \
        jobs/curc_30node_rebuttal_seed.sh)
    echo "  Submitted: Bayesian OED seed=$SEED -> Job $JOB"
done

echo ""
echo "All 15 rebuttal jobs submitted."
echo "Monitor with:  squeue -u \$USER"
echo "Logs in:       $OUT/logs/"
echo ""
echo "When complete, pull locally:"
echo "  scp -r paco0228@login.rc.colorado.edu:$OUT ./results/"
