#!/bin/bash
# 30/50-node follow-up worker: dispatches one (CONDITION, METHOD, SEED) tuple.
#
# Conditions tested (test the LM-prior-vs-DPO claim from the rebuttal):
#   anon30  -> 30-node SCM with anonymised node names (n_xxxx) instead of X1..X30
#              Hypothesis: ACE strongly outperforms zero-shot LM here, because
#              DPO must compensate for the missing semantic prior.
#   nodes50 -> 50-node hierarchical SCM (canonical X-names)
#              Hypothesis: at scale, both methods degrade but ACE less so.
#
# Methods:
#   ace          -> full ACE pipeline (LM + lookahead + DPO)
#   zero_shot_lm -> ACE with --no_dpo (LM proposes, lookahead selects, no DPO)
#
# SLURM env vars expected:
#   CONDITION : anon30 | nodes50
#   METHOD    : ace | zero_shot_lm
#   SEED      : integer seed
#   OUT       : absolute results root

set -euo pipefail

source /projects/paco0228/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate ace 2>/dev/null || true

export HF_HOME="/projects/paco0228/cache/huggingface"
export MPLCONFIGDIR="/projects/paco0228/cache/matplotlib"
mkdir -p "$HF_HOME" "$MPLCONFIGDIR"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /projects/paco0228/ACE
echo "30/50-node followup CONDITION=$CONDITION METHOD=$METHOD seed=$SEED started at $(date)"
echo "  SLURM_JOB_ID=${SLURM_JOB_ID:-none}"
echo "  SLURMD_NODENAME=${SLURMD_NODENAME:-unknown}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

JOB_TAG="${SLURM_JOB_ID:-local}"

# Per-condition flags + episode budget. zero-shot LM gets a smaller cap
# because the LM forward pass is slow (~10 min/episode at 30 nodes,
# ~14 min/episode at 50 nodes) and the policy is fixed.
case "$CONDITION" in
    anon30)
        SCALE=30
        ANON_FLAG="--anonymize_nodes"
        ;;
    nodes50)
        SCALE=50
        ANON_FLAG=""
        ;;
    *)
        echo "ERROR: unknown CONDITION=$CONDITION"
        exit 1
        ;;
esac

# Episode budget: zero_shot_lm caps at 40 (fixed policy converges fast).
# ACE caps at 120 by default, but at 50 nodes the working-DPO double-forward
# costs ~57 min/episode (verified May 26 ace_n50c_s42), so 120 episodes is
# infeasible inside CURC's 24h wall-time cap. Honour an override env var
# EPISODES_ACE if present, else use a per-condition default.
if [ "$METHOD" = "zero_shot_lm" ]; then
    EPISODES=${EPISODES_ZSL:-40}
    NO_DPO_FLAG="--no_dpo"
else
    if [ -n "${EPISODES_ACE:-}" ]; then
        EPISODES="$EPISODES_ACE"
    elif [ "$CONDITION" = "nodes50" ]; then
        EPISODES=30
    else
        EPISODES=120
    fi
    NO_DPO_FLAG=""
fi

# By default each job lands in its own job_<id> dir (fresh start). Set
# STABLE_DIR=1 to drop the job-id suffix so ace_experiments.py resumes from the
# prior run_* checkpoint on resubmission (used by the anon30 ACE convergence
# rerun, which needs several wall-time windows to reach its best-MSE plateau).
if [ "${STABLE_DIR:-0}" = "1" ]; then
    OUT_DIR="$OUT/${CONDITION}/${METHOD}/seed_${SEED}"
else
    OUT_DIR="$OUT/${CONDITION}/${METHOD}/seed_${SEED}/job_${JOB_TAG}"
fi
mkdir -p "$OUT_DIR"

python -u ace_experiments.py \
    --large_scale "$SCALE" \
    $ANON_FLAG \
    $NO_DPO_FLAG \
    --episodes "$EPISODES" \
    --seed "$SEED" \
    --use_dedicated_root_learner \
    --obs_train_interval 3 \
    --obs_train_samples 200 \
    --obs_train_epochs 100 \
    --output "$OUT_DIR"

echo "30/50-node followup CONDITION=$CONDITION METHOD=$METHOD seed=$SEED finished at $(date)"
