#!/bin/bash
# =============================================================================
# Model-scale sweep, enabled by the Aug 2026 Alpine expansion (H200 141GB,
# RTX Pro 6000 96GB). Two phases sharing one worker (jobs/curc_scaling_seed.sh):
#
#   Phase A (E1 -- LM-prior capability scaling law):
#     Zero-shot (--no_dpo) across Qwen2.5 {0.5B,1.5B,3B,7B,14B,32B} at
#     N in {5,30}, with --lookahead_on_student.
#
#   Phase B (E2 -- DPO's marginal contribution vs. prior strength):
#     DPO arm (method=ace) at N=30 for {0.5B,1.5B,3B}.
#
# Usage (from /projects/paco0228/ACE):
#   bash jobs/curc_submit_model_scale_sweep.sh              # both phases
#   PHASE=A bash jobs/curc_submit_model_scale_sweep.sh
#   PHASE=B bash jobs/curc_submit_model_scale_sweep.sh
#   SKIP_COMPLETED=1 bash jobs/curc_submit_model_scale_sweep.sh  # resubmit only gaps
#
# GPU overrides (apply to every size):
#   GPU_PARTITION=artxpro6000 GPU_QOS=gpu-normal GPU_GRES=gpu:rtx_pro_6000:1 \
#       bash jobs/curc_submit_model_scale_sweep.sh
# =============================================================================

set -euo pipefail

cd /projects/paco0228/ACE

source /projects/paco0228/miniconda3/etc/profile.d/conda.sh
conda activate ace

OUT="/scratch/alpine/paco0228/ACE/results/curc_model_scale_sweep"
mkdir -p "$OUT/logs"
WORKER="jobs/curc_scaling_seed.sh"

PHASE="${PHASE:-AB}"
SKIP_COMPLETED="${SKIP_COMPLETED:-0}"
SEEDS_SMALL="${SEEDS_SMALL:-42 123 456 789 1011}"
SEEDS_LARGE="${SEEDS_LARGE:-42 123 456}"

echo "================================================================"
echo " Model-scale sweep -- CURC SLURM (phase=$PHASE skip_completed=$SKIP_COMPLETED)"
echo " Output : $OUT   Started: $(date)"
echo "================================================================"

# Returns 0 if this cell already has a usable node_losses.csv (completed run).
cell_done() {
    local root=$1 method=$2 scale=$3 seed=$4
    local seed_dir="$root/nodes${scale}/${method}/seed_${seed}"
    [[ -d "$seed_dir" ]] || return 1
    find "$seed_dir" -name node_losses.csv 2>/dev/null | grep -q .
}

model_config() {
    local tag=$1
    MC_GC="0"
    case "$tag" in
        0.5B)
            MC_MODEL="Qwen/Qwen2.5-0.5B"; MC_DTYPE=""
            MC_PARTITION="artxpro6000"; MC_GRES="gpu:rtx_pro_6000:1"; MC_MEM=64G ;;
        1.5B)
            MC_MODEL="Qwen/Qwen2.5-1.5B"; MC_DTYPE=""
            MC_PARTITION="artxpro6000"; MC_GRES="gpu:rtx_pro_6000:1"; MC_MEM=64G ;;
        3B)
            MC_MODEL="Qwen/Qwen2.5-3B"; MC_DTYPE="bfloat16"; MC_GC="1"
            MC_PARTITION="artxpro6000"; MC_GRES="gpu:rtx_pro_6000:1"; MC_MEM=80G ;;
        7B)
            MC_MODEL="Qwen/Qwen2.5-7B"; MC_DTYPE="bfloat16"
            MC_PARTITION="artxpro6000"; MC_GRES="gpu:rtx_pro_6000:1"; MC_MEM=80G ;;
        14B)
            # 14B bf16 ~28GB weights -- fits RTX Pro 6000; keep off contested H200.
            MC_MODEL="Qwen/Qwen2.5-14B"; MC_DTYPE="bfloat16"
            MC_PARTITION="artxpro6000"; MC_GRES="gpu:rtx_pro_6000:1"; MC_MEM=90G ;;
        32B)
            # 32B bf16 ~64GB weights -- needs H200 headroom.
            MC_MODEL="Qwen/Qwen2.5-32B"; MC_DTYPE="bfloat16"
            MC_PARTITION="ah200"; MC_GRES="gpu:h200:1"; MC_MEM=140G ;;
        *)
            echo "ERROR: unknown model tag $tag" >&2; exit 1 ;;
    esac
    MC_PARTITION="${GPU_PARTITION:-$MC_PARTITION}"
    MC_GRES="${GPU_GRES:-$MC_GRES}"
    MC_QOS="${GPU_QOS:-gpu-normal}"
}

# Build a Slurm --export list. Never emit empty KEY= values: Slurm can glue
# the next KEY into the previous value (e.g. POLICY_DTYPE=,LOOKAHEAD_STUDENT=1
# -> POLICY_DTYPE="LOOKAHEAD_STUDENT=1"), which then breaks --policy_dtype.
build_export() {
    local export_list="ALL"
    local kv
    for kv in "$@"; do
        local key="${kv%%=*}"
        local val="${kv#*=}"
        if [ -n "$val" ]; then
            export_list="${export_list},${key}=${val}"
        fi
    done
    printf '%s' "$export_list"
}

submit_a() {
    local tag=$1 scale=$2 seed=$3 time=$4
    model_config "$tag"
    local outdir="$OUT/phaseA/${tag}/"
    mkdir -p "$outdir/logs"
    local name="msA_${tag}_n${scale}_s${seed}"

    if [ "$SKIP_COMPLETED" = "1" ] && cell_done "$outdir" zero_shot_lm "$scale" "$seed"; then
        echo "  [A] SKIP (done): $name"
        return 0
    fi

    local export_vars
    export_vars=$(build_export \
        "SCALE=$scale" \
        "METHOD=zero_shot_lm" \
        "SEED=$seed" \
        "OUT=$outdir" \
        "MODEL=$MC_MODEL" \
        "POLICY_DTYPE=$MC_DTYPE" \
        "LOOKAHEAD_STUDENT=1" \
        "EPISODES=40")

    JOB=$(sbatch --parsable \
        --job-name="$name" \
        --partition="$MC_PARTITION" --qos="$MC_QOS" \
        --nodes=1 --ntasks=1 --gres="$MC_GRES" \
        --cpus-per-task=8 --mem="$MC_MEM" \
        --time="$time" \
        --output="$outdir/logs/${name}_%j.out" \
        --error="$outdir/logs/${name}_%j.err" \
        --export="$export_vars" \
        "$WORKER")
    echo "  [A] Submitted: $name -> Job $JOB ($MC_PARTITION/$MC_GRES mem=$MC_MEM)"
}

submit_b() {
    local tag=$1 seed=$2 time=$3
    model_config "$tag"
    # DPO keeps a reference copy of the policy -- bump host RAM one tier.
    case "$tag" in
        0.5B|1.5B) MC_MEM=80G ;;
        3B)        MC_MEM=96G ;;
    esac
    local outdir="$OUT/phaseB/${tag}/"
    mkdir -p "$outdir/logs"
    local name="msB_${tag}_n30_s${seed}"

    if [ "$SKIP_COMPLETED" = "1" ] && cell_done "$outdir" ace 30 "$seed"; then
        echo "  [B] SKIP (done): $name"
        return 0
    fi

    local export_vars
    export_vars=$(build_export \
        "SCALE=30" \
        "METHOD=ace" \
        "SEED=$seed" \
        "OUT=$outdir" \
        "MODEL=$MC_MODEL" \
        "POLICY_DTYPE=$MC_DTYPE" \
        "GRAD_CKPT=$MC_GC" \
        "LOOKAHEAD_STUDENT=1" \
        "EPISODES=120")

    JOB=$(sbatch --parsable \
        --job-name="$name" \
        --partition="$MC_PARTITION" --qos="$MC_QOS" \
        --nodes=1 --ntasks=1 --gres="$MC_GRES" \
        --cpus-per-task=8 --mem="$MC_MEM" \
        --time="$time" \
        --output="$outdir/logs/${name}_%j.out" \
        --error="$outdir/logs/${name}_%j.err" \
        --export="$export_vars" \
        "$WORKER")
    echo "  [B] Submitted: $name -> Job $JOB ($MC_PARTITION/$MC_GRES mem=$MC_MEM)"
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
    for SEED in $SEEDS_SMALL; do
        submit_b "3B" "$SEED" 20:00:00
    done
fi

echo ""
echo "Monitor: squeue -u \$USER -n msA_0.5B_n5_s42,msA_1.5B_n5_s42 2>/dev/null; squeue -u \$USER | grep -E 'msA_|msB_'"
echo "Logs in: $OUT/phase{A,B}/<model_tag>/logs/"
