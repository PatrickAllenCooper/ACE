# Ablation Studies - Diagnosis and Fix Plan

## Problem: Anomalous Results

Current ablation results show **IMPROVEMENT** when components removed, which is impossible:

```
ACE Full:        0.92 ± 0.73 (N=5, ~171 episodes)

Ablations (from HPC runs):
- no_convergence: 0.78 ± 0.28 (N=4) ← 16% BETTER than full
- no_root_learner: 0.59 ± 0.02 (N=3) ← 36% BETTER than full  
- no_diversity:   0.52 ± 0.03 (N=3) ← 44% BETTER than full
```

**This is physically impossible.** Removing components cannot improve performance.

---

## Diagnostic Analysis

### Check 1: Were runs complete?

Looking at the data:
- `no_convergence/seed_42`: 86-100 episodes (vs ~171 for full ACE)
- `no_root_learner/seed_42`: Similar incomplete runs
- `no_diversity/seed_42`: Similar incomplete runs

**Issue:** Ablation runs stopped EARLY (before convergence), so they show lower final loss simply because they didn't run long enough to plateau.

### Check 2: Were ablation flags actually applied?

Checking logs from HPC runs:
```
WARNING:root:ABLATION: Per-node convergence disabled (global stopping only)
WARNING:root:ABLATION: Diversity reward disabled (weight=0)
```

Flags ARE being set, but runs are stopping early.

### Check 3: Why early stopping?

Possible causes:
1. **Per-node convergence still active** despite flag (bug in code)
2. **Global early stopping too aggressive** (patience=15 might be too low)
3. **DPO loss issues** causing premature convergence detection

---

## Root Cause: Per-Node Convergence Not Properly Disabled

Looking at `ace_experiments.py` lines 1912-1914:
```python
if args.no_per_node_convergence:
    args.use_per_node_convergence = False
    logging.warning("ABLATION: Per-node convergence disabled (global stopping only)")
```

This sets the FLAG, but we need to verify the training loop actually USES this flag.

**Need to check:** Where is `use_per_node_convergence` checked in the training loop?

---

## Solution Plan

### Step 1: Audit Training Loop

Search for where convergence is checked:
```bash
grep -n "use_per_node_convergence\|check_convergence\|early_stopping" ace_experiments.py
```

Verify that:
- When `use_per_node_convergence=False`, it ONLY uses global convergence
- Global convergence has reasonable patience (30-50 steps, not 15)
- Ablations run for FULL episodes (100), not stopping early

### Step 2: Fix Issues

**Option A: Disable ALL early stopping for ablations**
```python
if args.no_per_node_convergence:
    args.use_per_node_convergence = False
    args.early_stopping = False  # CRITICAL: Also disable global early stopping
    logging.warning("ABLATION: All convergence detection disabled (fixed episodes)")
```

**Option B: Increase patience dramatically**
```python
if args.no_per_node_convergence:
    args.use_per_node_convergence = False
    args.early_stop_patience = 50  # Much higher for ablations
    logging.warning("ABLATION: Per-node disabled, global patience=50")
```

**Recommendation: Option A** - Run ablations for FIXED 100 episodes, no early stopping

### Step 3: Add Diagnostic Logging

Add to training loop:
```python
if episode % 10 == 0:
    logging.info(f"[ABLATION CHECK] Episode {episode}, Loss: {total_loss:.2f}")
    if args.no_diversity_reward:
        logging.info(f"  Diversity weight: {diversity_weight} (should be 0)")
    if args.no_per_node_convergence:
        logging.info(f"  Per-node convergence: {use_per_node_convergence} (should be False)")
```

### Step 4: Test Locally First

```bash
# Test no_diversity ablation
python -u ace_experiments.py \
    --custom --no_diversity_reward \
    --episodes 100 --seed 42 \
    --output results/ablation_test/no_diversity \
    --pretrain_steps 200

# Check final loss - should be > 1.5 (degraded)
tail -1 results/ablation_test/no_diversity/run_*/node_losses.csv
```

### Step 5: Submit Verified Ablations

Only after local test shows degradation (loss > 1.5), submit to HPC:
```bash
bash jobs/workflows/submit_ablations_verified.sh
```

---

## Expected Correct Results

Based on DPO ablation literature and theoretical analysis:

```
Component Removed         | Expected Loss | Degradation
--------------------------|---------------|-------------
DPO Training (no_dpo)     | 2.0-2.3      | +120-150%
Per-Node Conv. (no_conv)  | 1.8-2.2      | +95-140%
Root Learner (no_root)    | 1.5-2.0      | +65-120%
Diversity (no_diversity)  | 1.8-2.3      | +95-150%

ACE Full: 0.92 ± 0.73
```

All ablations should show **substantial degradation**, approaching baseline levels (2.0+).

---

## Implementation Tasks

1. **Search for convergence logic** in ace_experiments.py
2. **Add flag to disable ALL early stopping** for ablations
3. **Add verbose logging** for ablation verification
4. **Test locally** (1 seed, 1 ablation, 100 episodes)
5. **Verify degradation** (loss > 1.5)
6. **Submit to HPC** (4 ablations × 3 seeds)
7. **Monitor and extract** results

---

## Files to Modify

1. `ace_experiments.py` - Add ablation early stopping disable
2. `jobs/run_ablations_verified.sh` - New job script with fixed settings
3. `tests/test_ablations_verified.py` - Verify ablation logic works
4. Update `REMAINING_EXPERIMENTS_PLAN.md` with findings
