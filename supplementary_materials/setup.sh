#!/bin/bash
# Quick setup script for ACE experiments

echo "================================================"
echo "ACE - Active Causal Experimentation Setup"
echo "================================================"
echo ""

# Check Python version
echo "[1/5] Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

if ! python -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "ERROR: Python 3.8+ required"
    exit 1
fi

# Create virtual environment
echo ""
echo "[2/5] Creating virtual environment..."
if [ ! -d "ace_env" ]; then
    python -m venv ace_env
    echo "Virtual environment created: ace_env/"
else
    echo "Virtual environment already exists"
fi

# Activate and install dependencies
echo ""
echo "[3/5] Installing dependencies..."
source ace_env/bin/activate 2>/dev/null || source ace_env/Scripts/activate 2>/dev/null

pip install --upgrade pip
pip install -r requirements.txt

# Make scripts executable
echo ""
echo "[4/5] Making scripts executable..."
chmod +x scripts/*.sh

# Check CUDA availability
echo ""
echo "[5/5] Checking CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}' if torch.cuda.is_available() else 'Using CPU')"

echo ""
echo "================================================"
echo "Setup Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "  1. Activate environment: source ace_env/bin/activate"
echo "  2. (Optional) Login to Hugging Face: huggingface-cli login"
echo "  3. Run experiments: cd scripts && ./run_ace_5node.sh"
echo ""
echo "See README.md for detailed instructions"
