#!/bin/bash
# ACE Training on 5-Node SCM Benchmark
# This script runs the complete ACE method on the standard 5-node benchmark

# Set experiment parameters
SEED=${1:-42}
OUTPUT_DIR=${2:-"results/ace"}
EPISODES=${3:-200}

echo "Running ACE on 5-Node SCM"
echo "Seed: $SEED"
echo "Episodes: $EPISODES"
echo "Output: $OUTPUT_DIR"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run ACE with full configuration
python -u ../code/ace_experiments.py \
    --model "Qwen/Qwen2.5-1.5B" \
    --episodes $EPISODES \
    --steps 25 \
    --candidates 4 \
    --seed $SEED \
    --output "$OUTPUT_DIR" \
    --lr 1e-5 \
    --learner_lr 2e-3 \
    --learner_epochs 100 \
    --buffer_steps 50 \
    --pretrain_steps 200 \
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
    --use_per_node_convergence \
    --use_dedicated_root_learner \
    --dedicated_root_interval 3

echo "Training complete. Results saved to $OUTPUT_DIR"
