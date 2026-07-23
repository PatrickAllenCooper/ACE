#!/bin/bash
# DPO-alternative worker -- POLICY_UPDATE, SEED, OUT passed via sbatch --export
#
# Runs the 5-node ACE config (same as curc_ace_seed.sh, the one that produced
# the 0.61 median headline result) with --policy_update swapped in for {dpo,
# sft_best, ranking}, isolating what preference-based learning contributes
# beyond simpler alternatives (reviewer d6tT: "supervised learning on the best
# candidate, a pairwise ranking loss... should be compared").
#
# SLURM vars expected:
#   POLICY_UPDATE : dpo | sft_best | ranking
#   SEED          : integer seed
#   OUT           : absolute path to results root

if command -v module &>/dev/null; then
    module purge 2>/dev/null || true
    module load cuda 2>/dev/null || true
fi
source /projects/paco0228/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate ace 2>/dev/null || true

export HF_HOME="/projects/paco0228/cache/huggingface"
export MPLCONFIGDIR="/projects/paco0228/cache/matplotlib"
mkdir -p "$HF_HOME" "$MPLCONFIGDIR"

cd /projects/paco0228/ACE
echo "DPO-alternative policy_update=$POLICY_UPDATE seed=$SEED started at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

python -u ace_experiments.py \
    --episodes 200 \
    --seed "$SEED" \
    --policy_update "$POLICY_UPDATE" \
    --use_dedicated_root_learner \
    --obs_train_interval 3 \
    --obs_train_samples 200 \
    --obs_train_epochs 100 \
    --output "$OUT/${POLICY_UPDATE}/seed_${SEED}"

echo "DPO-alternative policy_update=$POLICY_UPDATE seed=$SEED finished at $(date)"
