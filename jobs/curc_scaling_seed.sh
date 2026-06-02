#!/bin/bash
# Scaling-sweep worker: one (SCALE, METHOD, SEED) tuple on the consistent
# LargeScaleSCM hierarchical family. Supports the scaling analysis described in
# Guidance_Documents (principles for scaling 30 -> 50 -> 100+ nodes).
#
# The scaling story is "ACE scales to larger N without architectural change":
# we report PER-NODE best MSE so absolute totals (which grow mechanically with
# N) do not make larger graphs look spuriously worse.
#
# SLURM env vars expected:
#   SCALE           : node count (consistent family: 15 | 30 | 50; >=10)
#   METHOD          : ace | zero_shot_lm | random | round_robin | max_variance
#   SEED            : integer seed
#   OUT             : absolute results root
# Optional env vars (scaling/ablation knobs):
#   PROMPT_STRATEGY : full | compact     (default: full; compact recommended >=50)
#   PROMPT_TOP_M    : top-m failing nodes surfaced by compact prompt (default 8)
#   CANDIDATES      : lookahead breadth K (default: ace_experiments.py default 4)
#   EPISODES        : episode budget (default: per-method below)
#   ANON            : 1 to anonymise node names (default 0)
#
# Output structure:
#   $OUT/nodes${SCALE}/${METHOD}/seed_${SEED}/job_${JOB_TAG}/

set -euo pipefail

source /projects/paco0228/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate ace 2>/dev/null || true

export HF_HOME="/projects/paco0228/cache/huggingface"
export MPLCONFIGDIR="/projects/paco0228/cache/matplotlib"
mkdir -p "$HF_HOME" "$MPLCONFIGDIR"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /projects/paco0228/ACE
echo "scaling worker SCALE=$SCALE METHOD=$METHOD seed=$SEED started at $(date)"
echo "  PROMPT_STRATEGY=${PROMPT_STRATEGY:-full} CANDIDATES=${CANDIDATES:-default} EPISODES=${EPISODES:-default} ANON=${ANON:-0}"
echo "  SLURM_JOB_ID=${SLURM_JOB_ID:-none} NODE=${SLURMD_NODENAME:-unknown}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

# Stable per-(scale,method,seed) directory (NO job-id suffix) so that
# ace_experiments.py's checkpoint-resume logic finds the prior run_* dir and
# its checkpoint on resubmission -- essential for 50-node ACE which needs
# several wall-time windows to reach its best-MSE plateau.
OUT_DIR="$OUT/nodes${SCALE}/${METHOD}/seed_${SEED}"
mkdir -p "$OUT_DIR"

ANON_FLAG=""
if [ "${ANON:-0}" = "1" ]; then ANON_FLAG="--anonymize_nodes"; fi

PROMPT_FLAGS="--prompt_strategy ${PROMPT_STRATEGY:-full}"
if [ -n "${PROMPT_TOP_M:-}" ]; then PROMPT_FLAGS="$PROMPT_FLAGS --prompt_top_m $PROMPT_TOP_M"; fi

CAND_FLAG=""
if [ -n "${CANDIDATES:-}" ]; then CAND_FLAG="--candidates $CANDIDATES"; fi

case "$METHOD" in
    ace|zero_shot_lm)
        # LM policy methods go through ace_experiments.py.
        if [ "$METHOD" = "zero_shot_lm" ]; then
            NO_DPO_FLAG="--no_dpo"
            EP=${EPISODES:-40}
        else
            NO_DPO_FLAG=""
            # ACE budget defaults: best-MSE plateaus early (~ep 20 at 30 nodes),
            # but allow room. Larger N is slower per episode, so the submit
            # script sets EPISODES explicitly per scale.
            if [ "$SCALE" -ge 50 ]; then EP=${EPISODES:-40}; else EP=${EPISODES:-120}; fi
        fi
        python -u ace_experiments.py \
            --large_scale "$SCALE" \
            $ANON_FLAG \
            $NO_DPO_FLAG \
            $PROMPT_FLAGS \
            $CAND_FLAG \
            --episodes "$EP" \
            --seed "$SEED" \
            --use_dedicated_root_learner \
            --obs_train_interval 3 \
            --obs_train_samples 200 \
            --obs_train_epochs 100 \
            --output "$OUT_DIR"
        ;;
    random|round_robin|max_variance|ppo|bayesian_oed)
        # Passive / non-LM baselines via the MLP-learner runner, which itself
        # appends <method>/seed_<seed> under --output, so pass the scale root.
        EP=${EPISODES:-150}
        python -u scripts/runners/run_30node_baseline_seed.py \
            --method "$METHOD" \
            --n_nodes "$SCALE" \
            --seed "$SEED" \
            --episodes "$EP" \
            --output "$OUT/nodes${SCALE}"
        ;;
    *)
        echo "ERROR: unknown METHOD=$METHOD"
        exit 1
        ;;
esac

echo "scaling worker SCALE=$SCALE METHOD=$METHOD seed=$SEED finished at $(date)"
