#!/bin/bash

# ============================================================================
# Quick ACE Test - Single Episode for Immediate Feedback
# ============================================================================
# Run 1 episode of ACE locally to verify:
# - Code works correctly
# - Early stopping logic functions
# - Node losses converge
# - No errors or crashes
#
# Usage: ./test_ace_quick.sh
# ============================================================================

set -e

echo "=========================================="
echo "  QUICK ACE TEST (1 Episode)"
echo "=========================================="
echo ""

OUTPUT_DIR="results/ace_quick_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "Running ACE for 1 episode with verbose logging..."
echo "Output: $OUTPUT_DIR"
echo ""

# Run with maximum verbosity
python ace_experiments.py \
    --episodes 1 \
    --seed 42 \
    --early_stopping \
    --use_per_node_convergence \
    --use_dedicated_root_learner \
    --obs_train_interval 3 \
    --obs_train_samples 200 \
    --obs_train_epochs 100 \
    --output "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/test_log.txt"

echo ""
echo "=========================================="
echo "  TEST COMPLETE"
echo "=========================================="
echo ""

# Show results
if [ -f "$OUTPUT_DIR/node_losses.csv" ]; then
    echo "Node losses (first 5 rows):"
    head -6 "$OUTPUT_DIR/node_losses.csv"
    echo ""
    echo "Node losses (last 5 rows):"
    tail -5 "$OUTPUT_DIR/node_losses.csv"
    echo ""
fi

if [ -f "$OUTPUT_DIR/metrics.csv" ]; then
    echo "Metrics (last 5 rows):"
    tail -5 "$OUTPUT_DIR/metrics.csv"
    echo ""
fi

echo "Full log: $OUTPUT_DIR/test_log.txt"
echo ""
echo "If this looks good, run 10-episode test:"
echo "  ./run_ace_only.sh --test"
