#!/bin/bash
# ACE Training on 15-Node Complex SCM
# This script runs ACE on the challenging 15-node benchmark with 5 colliders

# Set experiment parameters
SEED=${1:-42}
OUTPUT_DIR=${2:-"results/ace_complex"}
EPISODES=${3:-300}

echo "Running ACE on 15-Node Complex SCM"
echo "Seed: $SEED"
echo "Episodes: $EPISODES"
echo "Output: $OUTPUT_DIR"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run ACE with full configuration for complex SCM
python -u ../code/experiments/run_ace_complex_full.py \
    --model "Qwen/Qwen2.5-1.5B" \
    --episodes $EPISODES \
    --steps 50 \
    --candidates 4 \
    --seed $SEED \
    --output "$OUTPUT_DIR" \
    --lr 1e-5 \
    --learner_lr 2e-3 \
    --learner_epochs 100 \
    --buffer_steps 50 \
    --pretrain_steps 500 \
    --cov_bonus 60.0 \
    --diversity_reward_weight 0.3 \
    --max_concentration 0.4 \
    --diversity_constraint \
    --diversity_threshold 0.60 \
    --smart_breaker \
    --obs_train_interval 3 \
    --obs_train_samples 200 \
    --obs_train_epochs 100 \
    --update_reference_interval 25 \
    --early_stopping \
    --use_per_node_convergence

echo "Training complete. Results saved to $OUTPUT_DIR"
