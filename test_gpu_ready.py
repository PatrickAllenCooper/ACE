#!/usr/bin/env python3
"""Quick test to verify GPU is ready for experiments."""

import sys

print("Testing GPU readiness...")
print("="*60)

try:
    import torch
    print(f"[OK] PyTorch {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"[OK] CUDA available")
        print(f"[OK] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[OK] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Quick tensor test
        x = torch.randn(1000, 1000).cuda()
        y = x @ x.T
        print(f"[OK] GPU computation works")
        
        print("="*60)
        print("GPU READY FOR EXPERIMENTS")
        print("="*60)
        sys.exit(0)
    else:
        print("[FAIL] CUDA not available - CPU only")
        print("="*60)
        print("Installing CUDA-enabled PyTorch...")
        print("Run: pip install torch --index-url https://download.pytorch.org/whl/cu118")
        sys.exit(1)
        
except ImportError as e:
    print(f"[FAIL] Import error: {e}")
    print("Run setup_local_fast.py first")
    sys.exit(1)
except Exception as e:
    print(f"[FAIL] Error: {e}")
    sys.exit(1)
