#!/bin/bash

# --- SLURM Directives ---
#SBATCH --job-name=ace_experiment
#SBATCH --partition=aa100              # Request A100 GPU partition
#SBATCH --qos=normal                   # Quality of Service
#SBATCH --nodes=1                      # Request 1 physical node
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1                   # Request 1 GPU (Crucial for Qwen!)
#SBATCH --cpus-per-task=8              # 8 CPU cores for data loading
#SBATCH --mem=32G                      # 32GB RAM
#SBATCH --time=12:00:00                # Run for up to 12 hours
#SBATCH --output=logs/ace_%j.out       # Standard output log (%j = Job ID)
#SBATCH --error=logs/ace_%j.err        # Standard error log

# --- 1. Environment Setup ---
module purge
module load cuda                       # Load CUDA drivers

# CRITICAL FIX: Redirect HuggingFace and Matplotlib caches to /projects
# This prevents the "No space left on device" error in your home directory
export HF_HOME="/projects/$USER/cache/huggingface"
export MPLCONFIGDIR="/projects/$USER/cache/matplotlib"

# Ensure these directories exist
mkdir -p logs
mkdir -p $HF_HOME
mkdir -p $MPLCONFIGDIR

# --- 2. Activate Conda ---
# Source conda (assuming installed in /projects as per guide)
source /projects/$USER/miniconda3/etc/profile.d/conda.sh
conda activate ace

# --- 3. Run Experiment ---
echo "Job started on $(hostname) at $(date)"
echo "--------------------------------------"
echo "GPU Debug Info:"
nvidia-smi                             # Verify GPU visibility
echo "--------------------------------------"

# Run your specific script
python ace_experiments.py --model "Qwen/Qwen2.5-1.5B" --episodes 100 --output "experiment_results"

echo "Job finished at $(date)"
