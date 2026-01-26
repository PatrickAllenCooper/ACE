#!/bin/bash

#SBATCH --job-name=ace_stats
#SBATCH --partition=aa100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00

# Statistical Tests Job Script
# Generates formal significance tests for ACE vs baselines
# NOTE: This is CPU-only but aa100 partition requires GPU allocation

# --- Environment Setup ---
export MPLCONFIGDIR="${MPLCONFIGDIR:-/projects/$USER/cache/matplotlib}"
mkdir -p "$MPLCONFIGDIR" 2>/dev/null || true

if [ "$CONDA_DEFAULT_ENV" != "ace" ]; then
    if [ -f "/projects/$USER/miniconda3/etc/profile.d/conda.sh" ]; then
        source /projects/$USER/miniconda3/etc/profile.d/conda.sh
        conda activate ace
    fi
fi

# --- Parameters ---
ACE_DIR="${ACE_DIR:-results/ace_multi_seed_20260125_115453}"
BASELINE_DIR="${BASELINE_DIR:-results/baselines/baselines_20260124_182827}"
OUTPUT_FILE="${OUTPUT_FILE:-results/statistical_analysis_$(date +%Y%m%d_%H%M%S).txt}"

# --- Job Info ---
echo "========================================"
echo "Statistical Significance Tests"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "ACE dir: $ACE_DIR"
echo "Baseline dir: $BASELINE_DIR"
echo "Output: $OUTPUT_FILE"
echo "========================================"
echo ""

# --- Run Analysis ---
python scripts/statistical_tests.py \
    --ace "$ACE_DIR" \
    --baselines "$BASELINE_DIR" \
    --output "$OUTPUT_FILE"

# --- Summary ---
echo ""
echo "========================================"
echo "Statistical Tests Complete"
echo "========================================"
echo "Finished: $(date)"
echo "Results: $OUTPUT_FILE"
echo ""
echo "Add LaTeX table to paper supplement!"
