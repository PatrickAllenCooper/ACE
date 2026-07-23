#!/bin/bash
# Node-importance ablation worker -- CONFIG, SEED, OUT passed via sbatch --export
#
# Isolates the node-importance term w(V_i, {L_j}) (Eq. 4 in the main text;
# realized in code as --cov_bonus, the direct-child-impact weighting term)
# from the information-gain and diversity terms, addressing the reviewer
# request (wZrW) for a node-importance ablation alongside the existing
# no-diversity row in the component-ablation table.
#
# CONFIG values:
#   full             : --cov_bonus 60.0 (paper default; reproduces the main row)
#   no_node_importance : --cov_bonus 0.0 (node-importance term removed)
#
# SLURM vars expected:
#   CONFIG : full | no_node_importance
#   SEED   : integer seed
#   OUT    : absolute path to results root

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
echo "Node-importance ablation config=$CONFIG seed=$SEED started at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

if [ "$CONFIG" == "no_node_importance" ]; then
    COV_BONUS=0.0
else
    COV_BONUS=60.0
fi

python -u ace_experiments.py \
    --episodes 200 \
    --seed "$SEED" \
    --cov_bonus "$COV_BONUS" \
    --use_dedicated_root_learner \
    --obs_train_interval 3 \
    --obs_train_samples 200 \
    --obs_train_epochs 100 \
    --output "$OUT/${CONFIG}/seed_${SEED}"

echo "Node-importance ablation config=$CONFIG seed=$SEED finished at $(date)"
