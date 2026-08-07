#!/bin/bash
# =============================================================================
# Model-scale sweep, enabled by the Aug 2026 Alpine expansion (H200 141GB,
# RTX Pro 6000 96GB). Two phases sharing one worker (jobs/curc_scaling_seed.sh):
#
#   Phase A (E1 -- LM-prior capability scaling law):
#     Zero-shot (--no_dpo, i.e. LM prior + lookahead-selection only, no
#     preference-pair training) across Qwen2.5 {0.5B,1.5B,3B,7B,14B,32B} at
#     N in {5,30}, with --lookahead_on_student so every arm's reported query
#     budget is honest regardless of model size. Tests whether mechanism
#     recovery improves monotonically with prior capability, and preempts
#     "why 1.5B?" by showing the paper's chosen size is not cherry-picked.
#
#   Phase B (E2 -- DPO's marginal contribution vs. prior strength):
#     Adds the DPO arm (method=ace) at N=30 for the three sizes DPO is
#     tractable at ({0.5B,1.5B,3B}; DPO's optimizer state + reference model
#     make 7B+ impractical even on 141GB). Phase A's zero_shot_lm@30 runs for
#     these same sizes/seeds are reused as the no-DPO comparison point, so
#     Phase B only submits the extra "ace" jobs.
#
# Together: does the ICLR reframing's "LM prior does the heavy lifting, DPO
# calibrates" claim hold as the prior itself gets stronger, or does DPO's
# contribution shrink towards zero?
#
# Seeds: 5 for <=3B (cheap), 3 for >=7B (Phase A only; DPO is not run there).
#
# Usage (from /projects/paco0228/ACE):
#   cd /projects/paco0228/ACE
#   git pull
#   bash jobs/curc_submit_model_scale_sweep.sh              # both phases
#   PHASE=A bash jobs/curc_submit_model_scale_sweep.sh      # capability sweep only
#   PHASE=B bash jobs/curc_submit_model_scale_sweep.sh      # DPO-vs-scale only
#
# GPU targeting: each model size has a sensible default partition/GRES below
# (RTX Pro 6000 for <=7B, H200 for 14B/32B, since 32B in bf16 is ~64GB of
# weights alone). Override per-run if needed, e.g. to move everything onto
# H200 if RTX Pro 6000 is congested:
#   GPU_PARTITION=ah200 GPU_QOS=gpu-normal GPU_GRES=gpu:h200:1 \
#       bash jobs/curc_submit_model_scale_sweep.sh
#
# Output:
#   results/curc_model_scale_sweep/phaseA/{model_tag}/nodes{N}/zero_shot_lm/seed_{seed}/
#   results/curc_model_scale_sweep/phaseB/{model_tag}/nodes30/ace/seed_{seed}/
# =============================================================================

set -euo pipefail

cd /projects/paco0228/ACE

source /projects/paco0228/miniconda3/etc/profile.d/conda.sh
conda activate ace

OUT="/scratch/alpine/paco0228/ACE/results/curc_model_scale_sweep"
mkdir -p "$OUT/logs"
WORKER="jobs/curc_scaling_seed.sh"

PHASE="${PHASE:-AB}"
SEEDS_SMALL="${SEEDS_SMALL:-42 123 456 789 1011}"   # <=3B
SEEDS_LARGE="${SEEDS_LARGE:-42 123 456}"            # >=7B

echo "================================================================"
echo " Model-scale sweep -- CURC SLURM (phase=$PHASE)"
echo " Output : $OUT   Started: $(date)"
echo "================================================================"

# model_config <model_tag> -> sets MODEL, PARTITION, GRES, MEM, DTYPE, GC
# via the MC_* globals below. GPU_PARTITION/GPU_QOS/GPU_GRES env vars (if
# exported before calling this script) override every size's default.
model_config() {
    local tag=$1
    MC_GC="0"
    case "$tag" in
        0.5B)
            MC_MODEL="Qwen/Qwen2.5-0.5B"; MC_DTYPE=""
            MC_PARTITION="artxpro6000"; MC_GRES="gpu:rtx_pro_6000:1"; MC_MEM=24G ;;
        1.5B)
            MC_MODEL="Qwen/Qwen2.5-1.5B"; MC_DTYPE=""
            MC_PARTITION="artxpro6000"; MC_GRES="gpu:rtx_pro_6000:1"; MC_MEM=32G ;;
        3B)
            MC_MODEL="Qwen/Qwen2.5-3B"; MC_DTYPE="bfloat16"; MC_GC="1"
            MC_PARTITION="artxpro6000"; MC_GRES="gpu:rtx_pro_6000:1"; MC_MEM=48G ;;
        7B)
            MC_MODEL="Qwen/Qwen2.5-7B"; MC_DTYPE="bfloat16"
            MC_PARTITION="artxpro6000"; MC_GRES="gpu:rtx_pro_6000:1"; MC_MEM=64G ;;
        14B)
            MC_MODEL="Qwen/Qwen2.5-14B"; MC_DTYPE="bfloat16"
            MC_PARTITION="ah200"; MC_GRES="gpu:h200:1"; MC_MEM=96G ;;
        32B)
            MC_MODEL="Qwen/Qwen2.5-32B"; MC_DTYPE="bfloat16"
            MC_PARTITION="ah200"; MC_GRES="gpu:h200:1"; MC_MEM=140G ;;
        *)
            echo "ERROR: unknown model tag $tag" >&2; exit 1 ;;
    esac
    # Env overrides win over the per-size default (e.g. to force everything
    # onto one partition if the other is congested).
    MC_PARTITION="${GPU_PARTITION:-$MC_PARTITION}"
    MC_GRES="${GPU_GRES:-$MC_GRES}"
    MC_QOS="${GPU_QOS:-gpu-normal}"
}

submit_a() {
    # submit_a <tag> <scale> <seed> <time>
    local tag=$1 scale=$2 seed=$3 time=$4
    model_config "$tag"
    local outdir="$OUT/phaseA/${tag}/"
    mkdir -p "$outdir/logs"
    local name="msA_${tag}_n${scale}_s${seed}"
    JOB=$(sbatch --parsable \
        --job-name="$name" \
        --partition="$MC_PARTITION" --qos="$MC_QOS" \
        --nodes=1 --ntasks=1 --gres="$MC_GRES" \
        --cpus-per-task=8 --mem="$MC_MEM" \
        --time="$time" \
        --output="$outdir/logs/${name}_%j.out" \
        --error="$outdir/logs/${name}_%j.err" \
        --export=ALL,SCALE=$scale,METHOD=zero_shot_lm,SEED=$seed,OUT=$outdir,MODEL="$MC_MODEL",POLICY_DTYPE=$MC_DTYPE,LOOKAHEAD_STUDENT=1,EPISODES=40 \
        "$WORKER")
    echo "  [A] Submitted: $name -> Job $JOB ($MC_PARTITION/$MC_GRES)"
}

submit_b() {
    # submit_b <tag> <seed> <time>
    local tag=$1 seed=$2 time=$3
    model_config "$tag"
    local outdir="$OUT/phaseB/${tag}/"
    mkdir -p "$outdir/logs"
    local name="msB_${tag}_n30_s${seed}"
    JOB=$(sbatch --parsable \
        --job-name="$name" \
        --partition="$MC_PARTITION" --qos="$MC_QOS" \
        --nodes=1 --ntasks=1 --gres="$MC_GRES" \
        --cpus-per-task=8 --mem="$MC_MEM" \
        --time="$time" \
        --output="$outdir/logs/${name}_%j.out" \
        --error="$outdir/logs/${name}_%j.err" \
        --export=ALL,SCALE=30,METHOD=ace,SEED=$seed,OUT=$outdir,MODEL="$MC_MODEL",POLICY_DTYPE=$MC_DTYPE,GRAD_CKPT=$MC_GC,LOOKAHEAD_STUDENT=1,EPISODES=120 \
        "$WORKER")
    echo "  [B] Submitted: $name -> Job $JOB ($MC_PARTITION/$MC_GRES)"
}

if [[ "$PHASE" == *A* ]]; then
    echo ""
    echo ">>> Phase A: zero-shot LM-prior capability sweep <<<"
    for TAG in 0.5B 1.5B 3B; do
        for SCALE in 5 30; do
            for SEED in $SEEDS_SMALL; do
                submit_a "$TAG" "$SCALE" "$SEED" 08:00:00
            done
        done
    done
    for TAG in 7B 14B 32B; do
        for SCALE in 5 30; do
            for SEED in $SEEDS_LARGE; do
                submit_a "$TAG" "$SCALE" "$SEED" 12:00:00
            done
        done
    done
fi

if [[ "$PHASE" == *B* ]]; then
    echo ""
    echo ">>> Phase B: DPO marginal gain vs. model scale (N=30 only) <<<"
    for TAG in 0.5B 1.5B; do
        for SEED in $SEEDS_SMALL; do
            submit_b "$TAG" "$SEED" 12:00:00
        done
    done
    # 3B DPO does 4 forward/backward passes through a 3B model with a long
    # 30-node prompt -- generously budgeted, gradient checkpointing forced on.
    for SEED in $SEEDS_SMALL; do
        submit_b "3B" "$SEED" 20:00:00
    done
fi

echo ""
echo "Monitor: squeue -u \$USER"
echo "Logs in: $OUT/phase{A,B}/<model_tag>/logs/"
echo ""
echo "Per-user GPU caps on gpu-normal are 4 concurrent per partition; jobs"
echo "beyond that queue (PD) and drain automatically."
