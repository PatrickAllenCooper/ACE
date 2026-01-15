#!/bin/bash
# ACE Paper Experiments - HPC Submission
# Usage: ./submit_experiments.sh [--quick] [--skip-ppo] [--scm-only]

EPISODES=${ACE_EPISODES:-500}
BASELINE_EP=${BASELINE_EPISODES:-100}
SCM_ONLY=false
[[ "$1" == "--quick" ]] && EPISODES=10 && BASELINE_EP=10 && shift
[[ "$1" == "--scm-only" ]] && SCM_ONLY=true && shift
SKIP_PPO=$([[ "$1" == "--skip-ppo" ]] && echo "--all" || echo "--all_with_ppo")

TS=$(date +%Y%m%d_%H%M%S)
OUT="results/paper_${TS}"
mkdir -p "$OUT" logs

export HF_HOME="/projects/$USER/cache/huggingface"
export MPLCONFIGDIR="/projects/$USER/cache/matplotlib"

# Job 1: ACE (Synthetic SCM)
JOB1=$(sbatch --parsable --job-name=ace_$TS --partition=aa100 --gres=gpu:1 \
  --mem=32G --time=12:00:00 --output=logs/ace_$TS.out --error=logs/ace_$TS.err \
  --wrap="source /projects/\$USER/miniconda3/etc/profile.d/conda.sh && conda activate ace && \
          python ace_experiments.py --episodes $EPISODES --output $OUT/ace")

# Job 2: Baselines
JOB2=$(sbatch --parsable --job-name=base_$TS --partition=aa100 --gres=gpu:1 \
  --mem=32G --time=08:00:00 --output=logs/base_$TS.out --error=logs/base_$TS.err \
  --wrap="source /projects/\$USER/miniconda3/etc/profile.d/conda.sh && conda activate ace && \
          python baselines.py $SKIP_PPO --episodes $BASELINE_EP --output $OUT/baselines")

echo "Submitted: ACE=$JOB1 Baselines=$JOB2"

if [ "$SCM_ONLY" = false ]; then
  # Job 3: Duffing Oscillators (Physics)
  JOB3=$(sbatch --parsable --job-name=duff_$TS --partition=short --mem=8G \
    --time=02:00:00 --output=logs/duff_$TS.out --error=logs/duff_$TS.err \
    --wrap="source /projects/\$USER/miniconda3/etc/profile.d/conda.sh && conda activate ace && \
            python -m experiments.duffing_oscillators --episodes $BASELINE_EP --output $OUT")

  # Job 4: Phillips Curve (Economics)
  JOB4=$(sbatch --parsable --job-name=phil_$TS --partition=short --mem=8G \
    --time=01:00:00 --output=logs/phil_$TS.out --error=logs/phil_$TS.err \
    --wrap="source /projects/\$USER/miniconda3/etc/profile.d/conda.sh && conda activate ace && \
            python -m experiments.phillips_curve --episodes $BASELINE_EP --output $OUT")

  echo "Submitted: Duffing=$JOB3 Phillips=$JOB4"
fi

echo "Monitor: squeue -u \$USER | Output: $OUT"
