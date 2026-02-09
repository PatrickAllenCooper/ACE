# Resubmit Ablations - Complete Guide

**Date:** February 3, 2026  
**Status:** Ablations ready to resubmit with fixes

---

## What Was Fixed

### Issue 1: --custom Flag
**Problem:** Custom transformer produced identical results across all ablations  
**Fix:** Now uses `--model "Qwen/Qwen2.5-1.5B"` (same as main ACE)

### Issue 2: Timeouts
**Problem:** Jobs timed out at 12 hours, only completed 1-2 seeds  
**Fix:** Increased to 24 hours

### Issue 3: Data Loss on Timeout
**Problem:** Results only saved at END, timeout = no data  
**Fix:** Now saves node_losses.csv and metrics.csv every 10 episodes

### Issue 4: Slow Storage
**Problem:** Direct writes to projects/ directory (slow)  
**Fix:** Uses scratch storage, copies to projects/ after each seed

---

## New Architecture

**Per Seed Runtime:** ~6-8 hours with Qwen (vs 30-45 min with --custom)  
**Per Job (3 seeds):** 18-24 hours  
**All 3 Jobs (Parallel):** ~24 hours total  

**With incremental saves:**
- Results.csv updated every 10 episodes
- If timeout at 24h: Get partial results (70-80 episodes minimum)
- Still enough for valid ablation analysis

---

## Commands to Resubmit

```bash
# On HPC:
cd /projects/paco0228/ACE

# Pull latest fixes
git pull

# Verify you have the fixes:
git log -1 --oneline
# Should show: "Add incremental CSV saving..."

# Verify --custom removed from ablations:
grep "ABLATION_FLAGS=" jobs/run_ablations_verified.sh
# Should NOT see --custom, should see --model or nothing

# Verify Qwen model specified:
grep "Qwen" jobs/run_ablations_verified.sh
# Should show: --model "Qwen/Qwen2.5-1.5B"

# Submit all 3 ablation types:
bash jobs/workflows/submit_ablations_verified.sh

# Note the 3 job IDs that get printed
```

---

## Monitoring

```bash
# Watch queue (refreshes every 60 seconds)
watch -n 60 'squeue -u $USER'

# When jobs start, verify Qwen is loading (first 2-5 minutes):
tail -f logs/ablations_verified_*.out

# Should see:
# [STARTUP] Loading Qwen2.5-1.5B policy...
# [STARTUP] Model loaded successfully
# [STARTUP] Oracle pretraining (200 steps)...

# Check progress periodically (every hour):
tail -20 logs/ablations_verified_*.err | grep "Episode\|PROGRESS"

# Check if results are being saved incrementally:
find results/ablations_verified_* -name "node_losses.csv" -mmin -30 -ls
```

---

## Expected Timeline

**Model Download (first time):**
- First job: 5-15 minutes to download Qwen (3GB)
- Subsequent jobs: Use cached model (instant)

**Per Seed:**
- Qwen loading: 1-2 minutes (from cache)
- Oracle pretraining: 5-10 minutes (200 steps)
- Training (100 episodes): 5-7 hours
- Total: ~6-8 hours per seed

**Per Job (3 seeds sequential):**
- ~18-24 hours

**All 3 Jobs (parallel):**
- ~24 hours total wall time

**Completion:** ~24 hours from submission

---

## Verification After Completion

```bash
# Check all jobs completed successfully:
sacct -j JOB1,JOB2,JOB3 --format=JobID,JobName,State,ExitCode,Elapsed,End

# Check results exist:
ls -lh results/ablations_verified_*/*/seed_*/run_*/node_losses.csv | tail -10

# CRITICAL: Verify results are DIFFERENT (not identical like before):
tail -1 results/ablations_verified_*/no_convergence/seed_42/run_*/node_losses.csv
tail -1 results/ablations_verified_*/no_root_learner/seed_42/run_*/node_losses.csv
tail -1 results/ablations_verified_*/no_diversity/seed_42/run_*/node_losses.csv

# Should see 3 DIFFERENT final loss values

# Verify with MD5 (should be 3 different hashes):
find results/ablations_verified_* -name "node_losses.csv" -path "*/seed_42/*" -mtime -1 -exec md5sum {} \;
```

---

## Expected Results (This Time)

**With Qwen policy, ablations should show DEGRADATION:**

```
no_convergence:  1.8-2.5 loss (vs 0.92 full ACE) = +95-170% degradation
no_root_learner: 1.5-2.0 loss (vs 0.92 full ACE) = +65-120% degradation
no_diversity:    1.8-2.5 loss (vs 0.92 full ACE) = +95-150% degradation
```

**And most importantly: Results should be DIFFERENT across types**

---

## If Jobs Timeout Again (24h not enough)

**Partial results are still valid:**
- Incremental saves mean you get 70-80 episodes minimum
- Can analyze with whatever episodes completed
- Still shows relative degradation between ablations

**Or split into smaller jobs:**
- Run 1 seed per job (8 hours)
- 9 jobs total (3 ablations × 3 seeds)
- More parallelism, faster completion

---

## After Results Retrieved

I'll analyze with:
```powershell
cd C:\Users\patri\code\ACE
python analyze_final_results.py
```

And update the paper with actual ablation values.

---

## Commit Summary

**Changes pushed (commit 9a3f44e):**
- ✓ Incremental CSV saving every 10 episodes (ace_experiments.py)
- ✓ Scratch storage for faster I/O (run_ablations_verified.sh)
- ✓ Copy to projects/ after each seed completes
- ✓ 24-hour time limit (vs 12h before)

**Ready to resubmit on HPC.**
