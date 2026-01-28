#!/usr/bin/env python3
"""
Fast local setup using pip instead of conda.
For when you need to start experiments immediately.
"""

import subprocess
import sys

print("="*70)
print("FAST LOCAL SETUP FOR ACE EXPERIMENTS")
print("="*70)
print()

# Check Python version
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print()

# Install essential packages via pip
packages = [
    "torch",
    "transformers",
    "pandas",
    "matplotlib",
    "seaborn",
    "networkx",
    "scipy",
    "numpy",
]

print("Installing packages...")
for pkg in packages:
    print(f"  Installing {pkg}...", end=" ", flush=True)
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", pkg, "--quiet"],
            capture_output=True,
            timeout=300
        )
        if result.returncode == 0:
            print("[OK]")
        else:
            print(f"[FAIL] (error: {result.stderr[:100]})")
    except Exception as e:
        print(f"[FAIL] (exception: {e})")

print()
print("="*70)
print("Testing installation...")
print("="*70)

# Test imports
try:
    import torch
    print(f"PyTorch: {torch.__version__} [OK]")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
except Exception as e:
    print(f"PyTorch: [FAIL] ({e})")

try:
    import transformers
    print(f"Transformers: {transformers.__version__} [OK]")
except Exception as e:
    print(f"Transformers: [FAIL] ({e})")

try:
    import pandas, matplotlib, networkx
    print(f"Data/viz packages: [OK]")
except Exception as e:
    print(f"Data/viz packages: [FAIL] ({e})")

print()
print("="*70)
print("SETUP COMPLETE")
print("="*70)
print()
print("Next steps:")
print("1. Run experiments:")
print("   python -u scripts/runners/run_all_experiments_local.py")
print()
print("2. Or run individually for parallel execution")
print("="*70)
