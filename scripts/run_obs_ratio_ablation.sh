#!/bin/bash

# ============================================================================
# Observational Training Ratio Ablation Study
# ============================================================================
# Tests different observational training frequencies to find optimal ratio.
#
# Tests intervals: 2, 3, 4, 5 (corresponding to 50%, 33%, 25%, 20% obs)
#
# Usage:
#   ./scripts/run_obs_ratio_ablation.sh
# ============================================================================

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OBS_ABLATION_DIR="results/obs_ratio_ablation_${TIMESTAMP}"
mkdir -p "$OBS_ABLATION_DIR"

echo "========================================"
echo "Observational Training Ratio Ablation"
echo "========================================"
echo "Output: $OBS_ABLATION_DIR"
echo "Testing intervals: 2, 3, 4, 5"
echo "Started: $(date)"
echo ""

declare -a JOB_IDS

# Test different observational training intervals
for INTERVAL in 2 3 4 5; do
    
    # Calculate percentage
    PCT=$((100 / INTERVAL))
    
    echo "Testing obs_train_interval=$INTERVAL (~${PCT}% observational)"
    
    JOB=$(sbatch --parsable \
        --job-name=obs_int${INTERVAL} \
        --output=logs/obs_interval${INTERVAL}_${TIMESTAMP}_%j.out \
        --error=logs/obs_interval${INTERVAL}_${TIMESTAMP}_%j.err \
        --partition=aa100 --qos=normal --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=8:00:00 \
        --wrap="python ace_experiments.py \
            --episodes 200 \
            --early_stopping \
            --use_per_node_convergence \
            --use_dedicated_root_learner \
            --obs_train_interval $INTERVAL \
            --obs_train_samples 200 \
            --obs_train_epochs 100 \
            --output $OBS_ABLATION_DIR/obs_interval_${INTERVAL}")
    
    JOB_IDS+=($JOB)
    echo "  Job ID: $JOB"
    echo ""
done

echo "========================================"
echo "All Observational Ratio Ablations Submitted!"
echo "========================================"
echo "Jobs: ${JOB_IDS[*]}"
echo "Monitor: squeue -j $(IFS=,; echo "${JOB_IDS[*]}")"
echo ""
echo "Expected Results:"
echo "  - Too frequent (interval=2): May slow interventional learning"
echo "  - Current (interval=3): Baseline"
echo "  - Less frequent (interval=4-5): May speed convergence but risk forgetting"
echo ""
echo "Optimal should balance: fast convergence vs preventing forgetting"
echo ""

# Save info
cat > "$OBS_ABLATION_DIR/obs_ratio_ablation_info.txt" <<EOF
Observational Training Ratio Ablation
======================================
Submitted: $(date)
Directory: $OBS_ABLATION_DIR

Configurations Tested:
  Interval 2 (~50% obs): Job ${JOB_IDS[0]}
  Interval 3 (~33% obs): Job ${JOB_IDS[1]} [Current default]
  Interval 4 (~25% obs): Job ${JOB_IDS[2]}
  Interval 5 (~20% obs): Job ${JOB_IDS[3]}

Expected Pattern:
  - Too much obs (interval=2): Slower convergence, good X2
  - Balanced (interval=3-4): Best overall
  - Too little obs (interval=5): Faster convergence, worse X2

Monitor: squeue -j $(IFS=,; echo "${JOB_IDS[*]}")

Analyze After Completion:
  python scripts/analyze_obs_ratio_ablation.py $OBS_ABLATION_DIR
EOF

echo "Ablation info saved to: $OBS_ABLATION_DIR/obs_ratio_ablation_info.txt"
