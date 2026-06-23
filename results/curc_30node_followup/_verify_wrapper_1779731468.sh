#!/bin/bash
set -euo pipefail
source /projects/paco0228/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate ace 2>/dev/null || true
export HF_HOME="/projects/paco0228/cache/huggingface"
export MPLCONFIGDIR="/projects/paco0228/cache/matplotlib"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /projects/paco0228/ACE
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
JOB_TAG="${SLURM_JOB_ID:-local}"
OUT_DIR="${OUT}/anon30/ace/seed_42_VERIFY/job_${JOB_TAG}"
mkdir -p "$OUT_DIR"
python -u ace_experiments.py \
    --large_scale 30 \
    --anonymize_nodes \
    --episodes 5 \
    --seed 42 \
    --use_dedicated_root_learner \
    --obs_train_interval 3 \
    --obs_train_samples 200 \
    --obs_train_epochs 100 \
    --output "$OUT_DIR"
