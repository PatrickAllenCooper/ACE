#!/bin/bash

# ============================================================================
# Environment Setup for ACE on HPC
# ============================================================================
# Activates Python environment for running scripts on login node
#
# Usage:
#   source setup_env.sh
#   OR
#   . setup_env.sh
#
# Note: Must use 'source' or '.' to activate conda in current shell
# Note: CUDA is NOT loaded here (login nodes don't have GPUs)
#       CUDA is loaded automatically on compute nodes by SLURM
# ============================================================================

echo "Setting up ACE Python environment..."

# Activate conda environment
# Try common environment names
if conda env list | grep -q "^ace "; then
    echo "  Activating conda env: ace"
    conda activate ace
elif conda env list | grep -q "^base "; then
    echo "  Using conda base environment"
    conda activate base
else
    echo "  Warning: No conda environment found"
    echo "  You may need to create one:"
    echo "    conda create -n ace python=3.10"
    echo "    conda activate ace"
    echo "    pip install torch transformers pandas matplotlib seaborn networkx scipy"
fi

# Verify PyTorch is available
python -c "import torch; print(f'  PyTorch {torch.__version__} ready')" 2>/dev/null || {
    echo "  ERROR: PyTorch not found!"
    echo "  Install with: pip install torch transformers"
    exit 1
}

# Set HPC-specific environment variables
export HF_HOME="/projects/$USER/cache/huggingface"
export MPLCONFIGDIR="/projects/$USER/cache/matplotlib"
export TRANSFORMERS_CACHE="/projects/$USER/cache/transformers"

# Create cache directories if they don't exist
mkdir -p "$HF_HOME" "$MPLCONFIGDIR" "$TRANSFORMERS_CACHE" 2>/dev/null

echo "  Environment ready!"
echo ""
echo "You can now run:"
echo "  ./test_ace_quick.sh"
echo "  ./run_ace_only.sh --test"
echo "  ./run_ace_only.sh --seeds 5"
