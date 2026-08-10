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
#   MODEL           : HF model name override (default: ace_experiments.py's
#                     Qwen/Qwen2.5-1.5B). Used by the model-scale sweep
#                     (curc_submit_model_scale_sweep.sh) to run larger priors.
#   GRAD_CKPT       : 1 to force --gradient_checkpointing regardless of SCALE/
#                     ANON (needed for larger policy models doing DPO)
#   POLICY_DTYPE    : float32 | bfloat16 | float16, forces the policy LM's
#                     dtype regardless of SCALE (default: auto bf16 at >=50)
#   LOOKAHEAD_STUDENT: 1 to add --lookahead_on_student (zero-oracle-query
#                     lookahead; keeps the query budget honest at any scale)
#
# Output structure:
#   $OUT/nodes${SCALE}/${METHOD}/seed_${SEED}/job_${JOB_TAG}/

set -euo pipefail

source /projects/paco0228/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate ace 2>/dev/null || true

export HF_HOME="/projects/paco0228/cache/huggingface"
export MPLCONFIGDIR="/projects/paco0228/cache/matplotlib"
mkdir -p "$HF_HOME" "$MPLCONFIGDIR"
# HF_TOKEN (if set in the submitting shell / --export) raises Hub rate limits
# and is required for first-time downloads. After
# scripts/runners/prefetch_qwen_models.py has populated HF_HOME, ACE loads
# from local snapshots and does not need the Hub at all.
if [ -n "${HF_TOKEN:-}${HUGGING_FACE_HUB_TOKEN:-}" ]; then
    echo "HF_TOKEN is set (authenticated Hub access)"
else
    echo "HF_TOKEN unset -- relying on local HF_HOME snapshots only"
fi
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

MODEL_FLAG=""
if [ -n "${MODEL:-}" ]; then MODEL_FLAG="--model $MODEL"; fi

GRAD_CKPT_FLAG=""
if [ "${GRAD_CKPT:-0}" = "1" ]; then GRAD_CKPT_FLAG="--gradient_checkpointing"; fi

POLICY_DTYPE_FLAG=""
if [ -n "${POLICY_DTYPE:-}" ]; then POLICY_DTYPE_FLAG="--policy_dtype $POLICY_DTYPE"; fi

LOOKAHEAD_STUDENT_FLAG=""
if [ "${LOOKAHEAD_STUDENT:-0}" = "1" ]; then LOOKAHEAD_STUDENT_FLAG="--lookahead_on_student"; fi

# LargeScaleSCM (experiments/large_scale_scm.py) requires n_nodes >= 10 (its
# fixed-size layers alone sum to 8). For SCALE=5, omit --large_scale entirely
# so ace_experiments.py falls back to its bespoke 5-node GroundTruthSCM (the
# paper's original diagnostic SCM) instead of erroring out.
LARGE_SCALE_FLAG="--large_scale $SCALE"
if [ "$SCALE" -lt 10 ]; then LARGE_SCALE_FLAG=""; fi

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
            $LARGE_SCALE_FLAG \
            $ANON_FLAG \
            $NO_DPO_FLAG \
            $PROMPT_FLAGS \
            $CAND_FLAG \
            $MODEL_FLAG \
            $GRAD_CKPT_FLAG \
            $POLICY_DTYPE_FLAG \
            $LOOKAHEAD_STUDENT_FLAG \
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
