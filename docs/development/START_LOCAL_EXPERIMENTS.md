# Local Experiment Execution Guide

**Environment is being created now...**

## Step 1: Verify Environment (After Creation)

```powershell
# Activate environment
conda activate ace

# Verify GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Test imports
python -c "import transformers, pandas, numpy; print('All dependencies OK')"
```

**Expected output:**
```
CUDA available: True
GPU: NVIDIA GeForce RTX 3080
All dependencies OK
```

---

## Step 2: Start Experiments

**Option A: Run all experiments (30 hours total)**
```powershell
conda activate ace
python -u scripts/runners/run_all_experiments_local.py
```

**Option B: Run individually (more control)**
```powershell
# Terminal 1: Complex SCM with PPO (20h, PRIORITY)
conda activate ace
python -u scripts/runners/run_critical_experiments.py --complex-scm --seeds 42 123 456 789 1011 --output-dir results/critical_experiments_local

# Terminal 2: Ablations (4-5h)
conda activate ace
cd c:\Users\patri\code\ACE
# Run each ablation-seed combination...

# Terminal 3: No-Oracle (5h)
conda activate ace
# Run each seed...
```

---

## Step 3: Monitor Progress

**Check logs:**
```powershell
# Main log
Get-Content results/local_experiments/experiments_*.log -Wait

# Check results incrementally
Get-ChildItem results/*/seed_*/run_*/metrics.csv -Recurse | Measure-Object
```

**Expected timeline:**
- Complex SCM: Start now, finish ~tomorrow afternoon
- Ablations: Can run parallel if you open multiple terminals
- No-Oracle: Can run parallel

---

## Step 4: Verify Results

**After completion:**
```powershell
# Count completed experiments
(Get-ChildItem results/*/seed_*/run_*/metrics.csv -Recurse).Count

# Check summary
Get-Content results/local_experiments/experiment_results.json
```

---

## Emergency Stop

```powershell
# Press Ctrl+C in running terminal
# Or kill Python processes
Get-Process python | Stop-Process
```

---

## Current Status

**Environment creation:** In progress (this takes 5-10 minutes)
**Next:** Activate environment and start experiments
**Estimated total time:** 30 hours sequential, or ~10-12 hours if you run 3 terminals in parallel

**All code is ready** - just waiting for environment setup to complete.
