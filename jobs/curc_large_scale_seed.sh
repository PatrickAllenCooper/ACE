#!/bin/bash
# 30-node large-scale ACE worker -- SEED and OUT passed via env

if command -v module &>/dev/null; then
    module purge 2>/dev/null || true
    module load cuda 2>/dev/null || true
fi
source /projects/paco0228/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate ace 2>/dev/null || true

export HF_HOME="/projects/paco0228/cache/huggingface"
export MPLCONFIGDIR="/projects/paco0228/cache/matplotlib"
mkdir -p "$HF_HOME" "$MPLCONFIGDIR"

# Memory-fragmentation mitigation (recommended by torch OOM error message).
# Critical when SLURM places us on a 40GB A100 instead of 80GB; the active +
# reference Qwen2.5-1.5B + cloned learners + KV cache + grads come close to 40GB.
# Newer PyTorch deprecated PYTORCH_CUDA_ALLOC_CONF in favor of PYTORCH_ALLOC_CONF;
# we set both for forward/backward compatibility across PyTorch versions.
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /projects/paco0228/ACE
echo "30-node ACE seed $SEED started at $(date)"
echo "  SLURM_JOB_ID=${SLURM_JOB_ID:-none}"
echo "  SLURMD_NODENAME=${SLURMD_NODENAME:-unknown}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

# Job-specific output dir so that retry submissions for the same seed do not
# race or overwrite each other. Falls back to plain seed dir if not under SLURM.
JOB_TAG="${SLURM_JOB_ID:-local}"
OUTPUT_DIR="$OUT/large_scale/seed_${SEED}/job_${JOB_TAG}"

python -u ace_experiments.py \
    --large_scale 30 \
    --episodes 300 \
    --seed "$SEED" \
    --use_dedicated_root_learner \
    --obs_train_interval 3 \
    --obs_train_samples 200 \
    --obs_train_epochs 100 \
    --output "$OUTPUT_DIR"

echo "30-node ACE seed $SEED finished at $(date)"
