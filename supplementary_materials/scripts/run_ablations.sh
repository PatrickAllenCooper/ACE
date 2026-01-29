#!/bin/bash
# Ablation Studies for ACE Components
# Tests impact of removing key architectural components

SEED=${1:-42}
OUTPUT_DIR=${2:-"results/ablations"}
EPISODES=${3:-200}

echo "Running ACE Ablation Studies"
echo "Seed: $SEED"
echo "Episodes: $EPISODES"
echo "Output: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

# Full ACE (baseline)
echo ""
echo "[1/4] Full ACE (all components)"
python -u ../code/ace_experiments.py \
    --seed $SEED \
    --episodes $EPISODES \
    --output "$OUTPUT_DIR/full_ace" \
    --diversity_constraint \
    --use_dedicated_root_learner \
    --use_per_node_convergence \
    --obs_train_interval 3

# No diversity reward
echo ""
echo "[2/4] Ablation: No diversity reward"
python -u ../code/ace_experiments.py \
    --seed $SEED \
    --episodes $EPISODES \
    --output "$OUTPUT_DIR/no_diversity" \
    --no_diversity_reward \
    --use_dedicated_root_learner \
    --use_per_node_convergence \
    --obs_train_interval 3

# No dedicated root learner
echo ""
echo "[3/4] Ablation: No dedicated root learner"
python -u ../code/ace_experiments.py \
    --seed $SEED \
    --episodes $EPISODES \
    --output "$OUTPUT_DIR/no_root_learner" \
    --diversity_constraint \
    --no_dedicated_root_learner \
    --use_per_node_convergence \
    --obs_train_interval 3

# No per-node convergence
echo ""
echo "[4/4] Ablation: No per-node convergence"
python -u ../code/ace_experiments.py \
    --seed $SEED \
    --episodes $EPISODES \
    --output "$OUTPUT_DIR/no_convergence" \
    --diversity_constraint \
    --use_dedicated_root_learner \
    --no_per_node_convergence \
    --obs_train_interval 3

echo ""
echo "Ablation studies complete. Results in $OUTPUT_DIR"
