#!/bin/bash
#
# Run ablation studies to validate ACE components
# Tests: no-DPO, no-convergence, no-root-learner, no-diversity
#
# Usage:
#   ./run_ablations.sh --seeds 3
#   ./run_ablations.sh --seeds 5 --episodes 100
#

set -e

# Default parameters
SEEDS=3
EPISODES=200
STEPS=25
OUTPUT_BASE="results/ablations_$(date +%Y%m%d_%H%M%S)"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --seeds)
            SEEDS="$2"
            shift 2
            ;;
        --episodes)
            EPISODES="$2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --seeds N --episodes M --steps K"
            exit 1
            ;;
    esac
done

echo "=================================================="
echo "ACE Ablation Studies"
echo "=================================================="
echo "Seeds: $SEEDS"
echo "Episodes: $EPISODES"
echo "Steps per episode: $STEPS"
echo "Output: $OUTPUT_BASE"
echo ""

mkdir -p "$OUTPUT_BASE"

# Array of seed values
SEED_ARRAY=(42 123 456 789 1011)

# Function to run ablation
run_ablation() {
    local name=$1
    local args=$2
    local seed=$3
    local output_dir="${OUTPUT_BASE}/${name}/seed_${seed}"
    
    echo "Running: $name (seed $seed)"
    
    mkdir -p "$output_dir"
    
    python ace_experiments.py \
        --episodes $EPISODES \
        --steps $STEPS \
        --seed $seed \
        --output "$output_dir" \
        $args \
        > "${output_dir}/run.log" 2>&1
    
    echo "  -> Complete: $output_dir"
}

echo "=================================================="
echo "1. ABLATION: No DPO (Custom/Simple Policy)"
echo "=================================================="
echo "Tests if DPO learning provides value over simple policy"
echo ""

for i in $(seq 0 $((SEEDS-1))); do
    seed=${SEED_ARRAY[$i]}
    # Use custom transformer policy instead of pretrained LLM
    # This disables DPO training on a pretrained LLM
    run_ablation "no_dpo" "--custom" $seed
done

echo ""
echo "=================================================="
echo "2. ABLATION: No Per-Node Convergence"
echo "=================================================="
echo "Tests if per-node early stopping improves efficiency"
echo ""

for i in $(seq 0 $((SEEDS-1))); do
    seed=${SEED_ARRAY[$i]}
    run_ablation "no_convergence" "--no_per_node_convergence" $seed
done

echo ""
echo "=================================================="
echo "3. ABLATION: No Dedicated Root Learner"
echo "=================================================="
echo "Tests if dedicated root learner improves root variable learning"
echo ""

for i in $(seq 0 $((SEEDS-1))); do
    seed=${SEED_ARRAY[$i]}
    run_ablation "no_root_learner" "--no_dedicated_root_learner" $seed
done

echo ""
echo "=================================================="
echo "4. ABLATION: No Diversity Reward"
echo "=================================================="
echo "Tests if diversity reward prevents policy collapse"
echo ""

for i in $(seq 0 $((SEEDS-1))); do
    seed=${SEED_ARRAY[$i]}
    run_ablation "no_diversity" "--no_diversity_reward" $seed
done

echo ""
echo "=================================================="
echo "ABLATION STUDIES COMPLETE"
echo "=================================================="
echo ""
echo "Results saved to: $OUTPUT_BASE"
echo ""
echo "To analyze:"
echo "  python scripts/analyze_ablations.py $OUTPUT_BASE"
echo ""
echo "Summary:"
echo "  - no_dpo: Random intervention selection"
echo "  - no_convergence: No early stopping"
echo "  - no_root_learner: No dedicated root training"
echo "  - no_diversity: No diversity reward"
echo ""
echo "Each ablation: $SEEDS seeds × $EPISODES episodes"
echo "Total runs: $((4 * SEEDS))"
echo ""
