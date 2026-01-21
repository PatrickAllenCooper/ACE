#!/bin/bash
# Quick test script for January 21, 2026 fixes
# Tests the critical changes without full HPC deployment

set -e  # Exit on error

echo "========================================"
echo "Testing January 21, 2026 Fixes"
echo "========================================"
echo ""

# Create test output directory
TEST_DIR="results/test_jan21_fixes_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TEST_DIR"

echo "Output directory: $TEST_DIR"
echo ""

# Test parameters
EPISODES=10
STEPS=10

echo "Running quick test:"
echo "  Episodes: $EPISODES"
echo "  Steps per episode: $STEPS"
echo ""

# Run with all fixes enabled
python ace_experiments.py \
    --episodes $EPISODES \
    --steps $STEPS \
    --output "$TEST_DIR" \
    \
    --early_stopping \
    --early_stop_patience 5 \
    --early_stop_min_episodes 3 \
    --use_per_node_convergence \
    --node_convergence_patience 5 \
    --zero_reward_threshold 0.92 \
    \
    --root_fitting \
    --use_dedicated_root_learner \
    --dedicated_root_interval 3 \
    \
    --undersampled_bonus 200.0 \
    --diversity_reward_weight 0.3 \
    --max_concentration 0.7 \
    --concentration_penalty 150.0 \
    \
    --pretrain_steps 50 \
    --pretrain_interval 10 \
    --smart_breaker

echo ""
echo "========================================"
echo "Test Complete!"
echo "========================================"
echo ""

# Check for expected improvements
echo "Checking results..."
echo ""

LOG_FILE="$TEST_DIR/experiment.log"

if [ -f "$LOG_FILE" ]; then
    echo "✓ Log file created"
    
    # Check for diversity scores
    echo ""
    echo "Diversity scores (should be > -10):"
    grep "diversity=" "$LOG_FILE" | tail -5 || echo "  [No diversity scores found]"
    
    # Check for gradient norms
    echo ""
    echo "Gradient checks (should be > 0):"
    grep "Gradient Check" "$LOG_FILE" || echo "  [No gradient checks found]"
    
    # Check for rewards
    echo ""
    echo "Recent rewards (should have some > 0):"
    grep "Reward:" "$LOG_FILE" | tail -10 || echo "  [No reward logs found]"
    
    # Check for early stopping
    echo ""
    echo "Early stopping status:"
    grep -E "Early stop|saturation|convergence" "$LOG_FILE" | tail -5 || echo "  [No early stopping triggered]"
    
    # Check for emergency retraining
    echo ""
    echo "Emergency retraining (if gradients were low):"
    grep "Emergency Re-training" "$LOG_FILE" || echo "  [No emergency retraining needed]"
    
    # Count zero rewards
    echo ""
    echo "Analyzing zero-reward fraction..."
    TOTAL_REWARDS=$(grep -c "Reward:" "$LOG_FILE" || echo "0")
    ZERO_REWARDS=$(grep "Reward: 0.00" "$LOG_FILE" | wc -l || echo "0")
    
    if [ "$TOTAL_REWARDS" -gt 0 ]; then
        ZERO_PCT=$(echo "scale=1; 100 * $ZERO_REWARDS / $TOTAL_REWARDS" | bc)
        echo "  Zero-reward steps: $ZERO_REWARDS / $TOTAL_REWARDS ($ZERO_PCT%)"
        echo "  Target: < 60%"
    fi
else
    echo "✗ Log file not found: $LOG_FILE"
    echo "  Test may have failed"
fi

echo ""
echo "========================================"
echo "Results saved to: $TEST_DIR"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Review $TEST_DIR/experiment.log"
echo "  2. Check $TEST_DIR/*.png for visualizations"
echo "  3. If successful, deploy to HPC with:"
echo "     sbatch jobs/run_ace_main.sh"
echo ""
