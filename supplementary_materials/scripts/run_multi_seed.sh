#!/bin/bash
# Multi-Seed Statistical Validation
# Runs ACE with 5 different random seeds for statistical significance

OUTPUT_DIR=${1:-"results/ace_multi_seed"}
EPISODES=${2:-200}

echo "Running ACE with Multiple Seeds for Statistical Validation"
echo "Seeds: 42, 123, 456, 789, 1011"
echo "Episodes: $EPISODES"
echo "Output: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

# Run with 5 different seeds
for seed in 42 123 456 789 1011; do
    echo ""
    echo "================================================"
    echo "Running seed $seed"
    echo "================================================"
    
    python -u ../code/ace_experiments.py \
        --seed $seed \
        --episodes $EPISODES \
        --output "$OUTPUT_DIR/seed_${seed}" \
        --diversity_constraint \
        --use_dedicated_root_learner \
        --use_per_node_convergence \
        --obs_train_interval 3
done

echo ""
echo "Multi-seed experiments complete."
echo "Analyze results with: python ../code/scripts/compute_statistics.py $OUTPUT_DIR"
