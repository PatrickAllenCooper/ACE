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
#
# IMPORTANT: Source setup_env.sh first if on HPC!
#   source setup_env.sh
#   ./test_ace_quick.sh
# ============================================================================

set -e

echo "=========================================="
echo "  QUICK ACE TEST (1 Episode)"
echo "=========================================="
echo ""

# Check if Python/PyTorch available
if ! python -c "import torch" 2>/dev/null; then
    echo "ERROR: PyTorch not found!"
    echo ""
    echo "On HPC, run this first:"
    echo "  source setup_env.sh"
    echo ""
    echo "This will:"
    echo "  1. Activate conda environment"
    echo "  2. Load CUDA modules"
    echo "  3. Set cache directories"
    echo ""
    exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="results/ace_quick_test_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo "Submitting quick ACE test job (1 episode)..."
echo "Output: $OUTPUT_DIR"
echo ""

# Submit as SLURM job (even for quick test - needs GPU)
JOB=$(sbatch --parsable \
    --nodes=1 --partition=aa100 --qos=normal \
    --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=0:30:00 \
    --job-name=ace_test \
    --output=logs/ace_quick_test_${TIMESTAMP}_%j.out \
    --error=logs/ace_quick_test_${TIMESTAMP}_%j.err \
    --wrap="python ace_experiments.py \
        --episodes 1 \
        --seed 42 \
        --early_stopping \
        --use_per_node_convergence \
        --use_dedicated_root_learner \
        --obs_train_interval 3 \
        --obs_train_samples 200 \
        --obs_train_epochs 100 \
        --output $OUTPUT_DIR")

echo "Job submitted: $JOB"
echo "Monitor: squeue -j $JOB"
echo "Log: logs/ace_quick_test_${TIMESTAMP}_${JOB}.out"
echo ""
echo "Wait ~5-10 minutes, then check results:"
echo "  tail -f logs/ace_quick_test_${TIMESTAMP}_${JOB}.out"

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
