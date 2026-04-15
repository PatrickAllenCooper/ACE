#!/bin/bash
# CPU-only experiment suite: Bayesian OED, K ablation, Duffing, Phillips
# OUT passed via env

source /projects/paco0228/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate ace 2>/dev/null || true
export HF_HOME="/projects/paco0228/cache/huggingface"
export MPLCONFIGDIR="/projects/paco0228/cache/matplotlib"
mkdir -p "$HF_HOME" "$MPLCONFIGDIR"

cd /projects/paco0228/ACE
echo "CPU suite started at $(date)"

python -u scripts/runners/run_reviewer_experiments.py \
    --bayesian-baseline \
    --k-ablation \
    --duffing-baselines \
    --phillips-baselines \
    --seeds 42 123 456 789 1011 314 271 577 618 141 \
    --episodes 171 \
    --output "$OUT/suite"

echo "CPU suite finished at $(date)"
