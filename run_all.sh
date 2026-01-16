#!/bin/bash

# --- SLURM Directives ---
#SBATCH --job-name=ace_paper
#SBATCH --partition=aa100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/paper_%j.out
#SBATCH --error=logs/paper_%j.err

# --- Configuration ---
EPISODES=${ACE_EPISODES:-500}
BASELINE_EPISODES=${BASELINE_EPISODES:-100}
QUICK=${QUICK:-false}

if [ "$QUICK" = "true" ]; then
    EPISODES=10
    BASELINE_EPISODES=10
fi

# --- Environment Setup ---
if command -v module &> /dev/null; then
    module purge || true
    module load cuda || echo "Warning: Could not load cuda module. Running on CPU or system CUDA?"
fi

# Use existing env vars if set, otherwise default to projects dir
export HF_HOME="${HF_HOME:-/projects/$USER/cache/huggingface}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/projects/$USER/cache/matplotlib}"
# Try to create directories, suppress error if permission denied (e.g. running locally outside project structure)
mkdir -p logs "$HF_HOME" "$MPLCONFIGDIR" 2>/dev/null || true

# Only try to setup conda if not already in 'ace' or if specific path exists
if [ "$CONDA_DEFAULT_ENV" != "ace" ]; then
    if [ -f "/projects/$USER/miniconda3/etc/profile.d/conda.sh" ]; then
        source /projects/$USER/miniconda3/etc/profile.d/conda.sh
        conda activate ace
    else
        echo "Note: Not sourcing specific conda path. Assuming environment is set up."
    fi
fi

# --- Output Directory ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT="results/paper_${TIMESTAMP}"
mkdir -p "$OUT"

echo "========================================"
echo "ACE Paper Experiments"
echo "========================================"
if [ -n "$SLURM_JOB_ID" ]; then
    echo "Job ID: $SLURM_JOB_ID"
else
    echo "Running Locally (Not a SLURM job)"
fi
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "Output: $OUT"
echo "Episodes: $EPISODES (baselines: $BASELINE_EPISODES)"

if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "nvidia-smi not found. Running on CPU?"
fi
echo "========================================"

# --- 1. Synthetic SCM (Main ACE Experiment) ---
echo ""
echo "[1/4] Running ACE (DPO) on Synthetic SCM..."
python ace_experiments.py \
    --episodes $EPISODES \
    --output "$OUT/ace" \
    2>&1 | tee "$OUT/ace.log"

# --- 2. Baselines ---
echo ""
echo "[2/4] Running Baselines (Random, Round-Robin, Max-Variance, PPO)..."
python baselines.py \
    --all_with_ppo \
    --episodes $BASELINE_EPISODES \
    --output "$OUT/baselines" \
    2>&1 | tee "$OUT/baselines.log"

# --- 3. Duffing Oscillators (Physics) ---
echo ""
echo "[3/4] Running Duffing Oscillators Experiment..."
python -m experiments.duffing_oscillators \
    --episodes $BASELINE_EPISODES \
    --output "$OUT" \
    2>&1 | tee "$OUT/duffing.log"

# --- 4. Phillips Curve (Economics) ---
echo ""
echo "[4/4] Running Phillips Curve Experiment..."
python -m experiments.phillips_curve \
    --episodes $BASELINE_EPISODES \
    --output "$OUT" \
    2>&1 | tee "$OUT/phillips.log"

# --- Summary ---
echo ""
echo "========================================"
echo "All experiments complete"
echo "========================================"
echo "Finished: $(date)"
echo "Results: $OUT"
ls -la "$OUT"
echo "========================================"
