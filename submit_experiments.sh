#!/bin/bash
# ACE Paper Experiments - HPC Submission
# Usage: ./submit_experiments.sh [--quick] [--skip-ppo]

EPISODES=${ACE_EPISODES:-500}
BASELINE_EP=${BASELINE_EPISODES:-100}
[[ "$1" == "--quick" ]] && EPISODES=10 && BASELINE_EP=10 && shift
SKIP_PPO=$([[ "$1" == "--skip-ppo" ]] && echo "--all" || echo "--all_with_ppo")

TS=$(date +%Y%m%d_%H%M%S)
OUT="results/paper_${TS}"
mkdir -p "$OUT" logs

export HF_HOME="/projects/$USER/cache/huggingface"
export MPLCONFIGDIR="/projects/$USER/cache/matplotlib"

# Job 1: ACE
JOB1=$(sbatch --parsable --job-name=ace_$TS --partition=aa100 --gres=gpu:1 \
  --mem=32G --time=12:00:00 --output=logs/ace_$TS.out --error=logs/ace_$TS.err \
  --wrap="source /projects/\$USER/miniconda3/etc/profile.d/conda.sh && conda activate ace && \
          python ace_experiments.py --episodes $EPISODES --output $OUT/ace")

# Job 2: Baselines
JOB2=$(sbatch --parsable --job-name=base_$TS --partition=aa100 --gres=gpu:1 \
  --mem=32G --time=08:00:00 --output=logs/base_$TS.out --error=logs/base_$TS.err \
  --wrap="source /projects/\$USER/miniconda3/etc/profile.d/conda.sh && conda activate ace && \
          python baselines.py $SKIP_PPO --episodes $BASELINE_EP --output $OUT/baselines")

# Job 3: Analysis (after 1 & 2)
JOB3=$(sbatch --parsable --dependency=afterok:$JOB1:$JOB2 --job-name=cmp_$TS \
  --partition=short --mem=4G --time=00:30:00 --output=logs/cmp_$TS.out \
  --wrap="source /projects/\$USER/miniconda3/etc/profile.d/conda.sh && conda activate ace && \
          python visualize.py $OUT/ace/run_* && python visualize.py $OUT/baselines/baselines_*")

echo "Submitted: ACE=$JOB1 Baselines=$JOB2 Compare=$JOB3"
echo "Monitor: squeue -u \$USER | Output: $OUT"
