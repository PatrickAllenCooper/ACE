#!/bin/bash
# =============================================================================
# 30/50-node follow-up experiments to test the LM-prior-vs-DPO contribution
# claim raised by the zero-shot-LM ablation.
#
# Submits 2 conditions x 2 methods x 5 seeds = 20 jobs:
#
#   Condition  Method        N  Wall  Tests
#   ---------  ------------  -  ----  ----------------------------------------
#   anon30     ACE           5   8h   ACE on 30-node, anonymised node names
#   anon30     Zero-shot LM  5   8h   ACE --no_dpo on 30-node, anonymised
#   nodes50    ACE           5  12h   ACE on 50-node, canonical names
#   nodes50    Zero-shot LM  5  10h   ACE --no_dpo on 50-node, canonical names
#
# Hypothesis: in BOTH conditions, ACE > Zero-shot LM by a wider margin than
# at the canonical 30-node setting (1.95 vs 1.73, statistically tied), because
# the LM prior is weakened (anonymisation removes semantic handles; 50 nodes
# stresses prompt length/context).
#
# Usage (from /projects/paco0228/ACE):
#   cd /projects/paco0228/ACE
#   git pull   # ensure latest --anonymize_nodes flag and 50-node SCM support
#   bash jobs/curc_submit_30node_followup.sh
#
# If Phase 1 (anon30) already submitted successfully and only Phase 2 failed,
# git pull the mem fix then run:
#   bash jobs/curc_submit_30node_followup_phase2_only.sh
#
# Output structure:
#   results/curc_30node_followup/
#     anon30/{ace,zero_shot_lm}/seed_{seed}/job_{jobid}/
#     nodes50/{ace,zero_shot_lm}/seed_{seed}/job_{jobid}/
#
# Pull locally when complete:
#   scp -r paco0228@login.rc.colorado.edu:/projects/paco0228/ACE/results/curc_30node_followup ./results/
# =============================================================================

set -euo pipefail

cd /projects/paco0228/ACE

source /projects/paco0228/miniconda3/etc/profile.d/conda.sh
conda activate ace

OUT="/projects/paco0228/ACE/results/curc_30node_followup"
mkdir -p "$OUT/logs"

echo "================================================================"
echo " 30/50-node Follow-up Experiments -- CURC SLURM (20 jobs)"
echo "================================================================"
echo " Output : $OUT"
echo " Started: $(date)"
echo "================================================================"

SEEDS="42 123 456 789 1011"

# -----------------------------------------------------------------------------
# Phase 1: anon30 (anonymised 30-node SCM)
#   ACE: 8h GPU, 120 episodes
#   Zero-shot LM: 8h GPU, 40 episodes (fixed policy, asymptote reached early)
# -----------------------------------------------------------------------------
echo ""
echo ">>> Phase 1: anon30 (30 nodes, anonymised names) <<<"
for SEED in $SEEDS; do
    JOB=$(sbatch --parsable \
        --job-name="ace_a30_s${SEED}" \
        --partition=aa100 --qos=normal \
        --nodes=1 --ntasks=1 --gres=gpu:1 \
        --cpus-per-task=8 --mem=64G \
        --time=08:00:00 \
        --output="$OUT/logs/ace_anon30_seed${SEED}_%j.out" \
        --error="$OUT/logs/ace_anon30_seed${SEED}_%j.err" \
        --export=ALL,CONDITION=anon30,METHOD=ace,SEED=$SEED,OUT=$OUT \
        jobs/curc_30node_followup_seed.sh)
    echo "  Submitted: ACE anon30 seed=$SEED -> Job $JOB"
done

for SEED in $SEEDS; do
    JOB=$(sbatch --parsable \
        --job-name="zsl_a30_s${SEED}" \
        --partition=aa100 --qos=normal \
        --nodes=1 --ntasks=1 --gres=gpu:1 \
        --cpus-per-task=8 --mem=64G \
        --time=08:00:00 \
        --output="$OUT/logs/zsl_anon30_seed${SEED}_%j.out" \
        --error="$OUT/logs/zsl_anon30_seed${SEED}_%j.err" \
        --export=ALL,CONDITION=anon30,METHOD=zero_shot_lm,SEED=$SEED,OUT=$OUT \
        jobs/curc_30node_followup_seed.sh)
    echo "  Submitted: Zero-shot LM anon30 seed=$SEED -> Job $JOB"
done

# -----------------------------------------------------------------------------
# Phase 2: nodes50 (50-node SCM, canonical names)
#   ACE: 12h GPU, 120 episodes (longer per-episode at 50 nodes)
#   Zero-shot LM: 10h GPU, 40 episodes
#
# Memory: CURC aa100 rejects --mem=80G for 1 GPU (81920 MiB > 80640 MiB cap).
# Use 78G (79872 MiB) to stay under the per-GPU RAM limit.
# -----------------------------------------------------------------------------
echo ""
echo ">>> Phase 2: nodes50 (50-node SCM, canonical names) <<<"
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

echo ""
echo "All 20 follow-up jobs submitted. Monitor with:  squeue -u \$USER"
echo "Logs in:  $OUT/logs/"
