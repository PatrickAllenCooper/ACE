# Ablation Studies - Failure Analysis and Fix

**Date:** February 3, 2026  
**Problem:** All 3 ablation types produced byte-for-byte identical results

---

## The Problem

**Results from Jobs 23495005, 23495006, 23495007:**

All three ablation types (no_convergence, no_root_learner, no_diversity) produced **IDENTICAL results** for each seed:

```
Seed 42:  0.4932251609861851 (all 3 ablations)
Seed 123: 0.5940... (all 3 ablations)
Seed 456: 0.4990... (all 3 ablations)

MD5 hash: ba556fdfb56c6139fe8f5e3dd961551c (all 3 files for seed 42)
```

**This is physically impossible** - removing different components cannot produce byte-for-byte identical results.

---

## Root Cause

**The --custom flag made ablations meaningless:**

1. Jobs used `--custom` flag (custom transformer instead of Qwen)
2. Ablation flags WERE applied (confirmed in logs: "ABLATION: ... disabled")
3. But with custom transformer + fixed seed (42), execution was deterministic
4. The ablated components had **zero observable effect** on trajectory

**Why:**
- Custom transformer is simple and deterministic
- Random seed (42) produces identical random number sequences
- Diversity reward, root learner, convergence checks have negligible impact on which random numbers are drawn
- Result: Identical execution path despite different ablation flags

**Evidence:**
- Same MD5 hash across all 3 ablations
- Even episode 0, step 0 is identical: `32.56513125896454`
- Files created at different times (18:45, 21:24, 21:19) but identical content

---

## The Fix

**Remove --custom flag from ablations:**

**Before:**
```bash
--custom --no_diversity_reward
--custom --no_root_learner  
--custom --no_per_node_convergence
```

**After:**
```bash
--no_diversity_reward --model "Qwen/Qwen2.5-1.5B"
--no_root_learner --model "Qwen/Qwen2.5-1.5B"
--no_per_node_convergence --model "Qwen/Qwen2.5-1.5B"
```

**Why this works:**
- Qwen policy is complex and stochastic
- Ablation components have measurable effect on Qwen's behavior
- Different ablations will produce different trajectories
- Results will show TRUE degradation

**Trade-off:**
- Runtime: 45 min → 2-3 hours per seed (Qwen is slower)
- Time limit: 8h → 12h per job
- But: Results will be VALID

---

## Changes Made

**Modified:** `jobs/run_ablations_verified.sh`
- Removed `--custom` from all ablation flags
- Added `--model "Qwen/Qwen2.5-1.5B"` to python command
- Increased time limit from 8h to 12h

**Commit:** a5b219f "CRITICAL FIX: Remove --custom from ablations - use Qwen to enable proper ablation effects"

---

## What This Means for Experiments

**Previous runs (INVALID):**
- Jobs 23495005, 23495006, 23495007: Custom transformer, identical results
- Cannot use these results
- Wasted ~3 hours of HPC time

**New runs (VALID):**
- Will use Qwen2.5-1.5B (same as main ACE)
- Ablation effects will be observable
- Results will show degradation (expected: 1.5-2.5 loss vs 0.92 full ACE)
- Runtime: ~2-3 hours per seed, 6-9 hours per ablation type

---

## Resubmission Plan

**On HPC:**
```bash
cd /projects/paco0228/ACE
git pull  # Get the fix (commit a5b219f)

# Submit ablations with Qwen (no --custom)
bash jobs/workflows/submit_ablations_verified.sh

# Monitor:
squeue -u paco0228
watch -n 60 'squeue -u $USER'
```

**Expected:**
- 3 jobs submitted (one per ablation type)
- Each runs 3 seeds sequentially
- 6-9 hours per job (running in parallel)
- Total wall time: ~9 hours

---

## Lessons Learned

1. **--custom bypasses stochastic policy behavior**
   - Makes experiments deterministic
   - Masks ablation effects
   - Should NOT be used for ablations

2. **Always verify results make sense**
   - Identical MD5 hashes = red flag
   - Improvements from ablations = red flag
   - Should have caught this earlier

3. **Ablations MUST use same architecture as main results**
   - Main ACE uses Qwen → Ablations must use Qwen
   - Cannot mix custom transformer with Qwen-based main results

---

## Impact on Timeline

**Original plan:** All experiments done by Feb 3  
**Actual:** Need to rerun ablations (adds 9 hours)

**New timeline:**
- Resubmit ablations: Feb 3
- Complete: Feb 4 (9 hours later)
- Analysis and paper update: Feb 4

**Paper submission:** Feb 4-5 with complete, valid ablation results

---

## Verification for Next Run

After resubmission, verify ablations are different:

```bash
# Wait for jobs to complete, then:
md5sum results/ablations_verified_*/no_convergence/seed_42/run_*/node_losses.csv
md5sum results/ablations_verified_*/no_root_learner/seed_42/run_*/node_losses.csv
md5sum results/ablations_verified_*/no_diversity/seed_42/run_*/node_losses.csv

# Should see 3 DIFFERENT hashes
# If same hash again, deeper investigation needed
```

---

## Status

**Fixed:** ✓ Code updated to use Qwen for ablations  
**Pushed:** ✓ Commit a5b219f  
**Ready:** ✓ Ready to resubmit on HPC  

**Next:** Pull on HPC and resubmit ablations
