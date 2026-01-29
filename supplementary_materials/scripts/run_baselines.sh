#!/bin/bash
# Run Baseline Methods on 5-Node SCM
# Compares ACE against random, greedy, round-robin, and PPO baselines

SEED=${1:-42}
OUTPUT_DIR=${2:-"results/baselines"}
EPISODES=${3:-200}

echo "Running Baseline Comparisons on 5-Node SCM"
echo "Seed: $SEED"
echo "Episodes: $EPISODES"
echo "Output: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

# Run each baseline
for method in random greedy_collider round_robin max_variance ppo; do
    echo ""
    echo "Running baseline: $method"
    python -u ../code/baselines.py \
        --method $method \
        --episodes $EPISODES \
        --seed $SEED \
        --output "$OUTPUT_DIR/${method}_seed${SEED}"
done

echo ""
echo "All baselines complete. Results in $OUTPUT_DIR"
