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
#SBATCH --output=logs/ace_stats_%j.out
#SBATCH --error=logs/ace_stats_%j.err

# Statistical Tests Job Script (using SCRATCH storage)
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

# --- Use SCRATCH for working ---
SCRATCH_DIR="/scratch/alpine1/$USER/ace_stats_$SLURM_JOB_ID"
mkdir -p "$SCRATCH_DIR"

# Copy results to scratch for analysis
echo "Copying results to scratch..."
cp -r results/ace_multi_seed_20260125_115453 "$SCRATCH_DIR/"
cp -r results/baselines/baselines_20260124_182827 "$SCRATCH_DIR/"

# --- Parameters ---
ACE_DIR="$SCRATCH_DIR/ace_multi_seed_20260125_115453"
BASELINE_DIR="$SCRATCH_DIR/baselines_20260124_182827"
SCRATCH_OUTPUT="$SCRATCH_DIR/statistical_analysis.txt"
FINAL_OUTPUT="results/statistical_analysis_$(date +%Y%m%d_%H%M%S).txt"

# --- Job Info ---
echo "========================================"
echo "Statistical Significance Tests"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "Scratch dir: $SCRATCH_DIR"
echo "ACE dir: $ACE_DIR"
echo "Baseline dir: $BASELINE_DIR"
echo "Final output: $FINAL_OUTPUT"
echo "========================================"
echo ""

# --- Run Analysis ---
cd "$SCRATCH_DIR"
python $SLURM_SUBMIT_DIR/scripts/statistical_tests.py \
    --ace "$ACE_DIR" \
    --baselines "$BASELINE_DIR" \
    --output "$SCRATCH_OUTPUT"

# --- Copy results back to projects ---
echo ""
echo "Copying results back to projects..."
cp "$SCRATCH_OUTPUT" "$SLURM_SUBMIT_DIR/$FINAL_OUTPUT"

# --- Cleanup scratch ---
echo "Cleaning up scratch..."
rm -rf "$SCRATCH_DIR"

# --- Summary ---
echo ""
echo "========================================"
echo "Statistical Tests Complete"
echo "========================================"
echo "Finished: $(date)"
echo "Results: $FINAL_OUTPUT"
echo "Scratch cleaned up"
echo ""
echo "Add LaTeX table to paper supplement!"
