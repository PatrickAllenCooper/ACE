#!/bin/bash
# 30-node rebuttal worker: dispatches one (METHOD, SEED) pair.
#
# Methods supported:
#   ppo            -> PPO baseline on 30-node (same MLP learner as ACE)
#   bayesian_oed   -> Bayesian OED (random subset of 10 candidates per step,
#                     M=3 posterior MC samples per candidate)
#   zero_shot_lm   -> ACE with --no_dpo (LM proposals + lookahead, no DPO updates)
#
# SLURM env vars expected:
#   METHOD : ppo | bayesian_oed | zero_shot_lm
#   SEED   : integer seed
#   OUT    : absolute results root

source /projects/paco0228/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate ace 2>/dev/null || true

export HF_HOME="/projects/paco0228/cache/huggingface"
export MPLCONFIGDIR="/projects/paco0228/cache/matplotlib"
mkdir -p "$HF_HOME" "$MPLCONFIGDIR"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /projects/paco0228/ACE
echo "30-node rebuttal method=$METHOD seed=$SEED started at $(date)"
echo "  SLURM_JOB_ID=${SLURM_JOB_ID:-none}"
echo "  SLURMD_NODENAME=${SLURMD_NODENAME:-unknown}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

JOB_TAG="${SLURM_JOB_ID:-local}"

case "$METHOD" in
    ppo)
        # PPO uses GPU for fast actor-critic updates; episodes match baseline budget.
        python -u scripts/runners/run_30node_baseline_seed.py \
            --method   ppo \
            --seed     "$SEED" \
            --episodes 150 \
            --steps    25 \
            --obs_train_interval 3 \
            --obs_train_samples  200 \
            --output   "$OUT/ppo"
        ;;

    bayesian_oed)
        # Bayesian OED is compute-heavy; reduce to 50 episodes for tractability.
        python -u scripts/runners/run_30node_baseline_seed.py \
            --method   bayesian_oed \
            --seed     "$SEED" \
            --episodes 50 \
            --steps    25 \
            --obs_train_interval 3 \
            --obs_train_samples  200 \
            --output   "$OUT/bayesian_oed"
        ;;

    zero_shot_lm)
        # ACE LM policy with --no_dpo (tests the pretrained LM prior alone,
        # no DPO updates so policy is FIXED). At 30 nodes the LM forward pass
        # is ~10 min/episode, so we cap episodes at 50 to fit safely in the
        # 8h wall time. Since the policy never updates, asymptotic performance
        # is reached in well under 50 episodes; longer runs only re-sample
        # the same fixed strategy.
        python -u ace_experiments.py \
            --large_scale 30 \
            --episodes 50 \
            --seed "$SEED" \
            --no_dpo \
            --use_dedicated_root_learner \
            --obs_train_interval 3 \
            --obs_train_samples 200 \
            --obs_train_epochs 100 \
            --output "$OUT/zero_shot_lm/seed_${SEED}/job_${JOB_TAG}"
        ;;

    *)
        echo "ERROR: unknown METHOD=$METHOD"
        exit 1
        ;;
esac

echo "30-node rebuttal method=$METHOD seed=$SEED finished at $(date)"
