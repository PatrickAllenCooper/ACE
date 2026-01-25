# Experimental Results Analysis
**Generated:** January 25, 2026  
**Status:** Waiting for multi-seed runs to complete

---

## Current Results Inventory

### Multi-Seed Validation (CRITICAL - PENDING)
**Directory:** `results/multi_run_20260123_152058/`  
**Status:** EMPTY (jobs still running/queued on HPC)  
**Expected:** 5 seeds × 5 experiments = 25 complete runs

**What's Needed:**
- Mean ± std for all metrics
- 95% confidence intervals
- Statistical significance tests (paired t-tests)
- Complete data for Table 1

**HPC Jobs:**
- Submitted: Jan 23, 15:20
- Status: 1 running, 16 pending (17/33 jobs visible)
- Likely: 16 jobs already completed (check with `sacct`)
- Expected completion: Jan 24-25

---

### Ablation Studies (PENDING)
**Directory:** `results/ablations_*/`  
**Status:** Not found in local results  
**Jobs:** 23193211-23193214 (4 jobs queued on HPC)

**What's Needed:**
- Full ACE vs ablated versions
- Performance degradation metrics
- Table for ablation section

---

### Observational Ratio Ablation (PENDING)
**Directory:** `results/obs_ratio_ablation_20260123_153216/`  
**Status:** Empty  
**Jobs:** 23193219-23193222 (4 jobs queued on HPC)

**What's Needed:**
- Comparison of intervals 2, 3, 4, 5
- Optimal ratio identification

---

## Preliminary Single-Run Results (Available)

### WARNING: These are from OLD CODE
These results were generated BEFORE:
- Intervention masking verification
- Observational ratio tuning
- All critical improvements

**Use only for:** Initial analysis, debugging, sanity checks  
**NOT for:** Final paper, statistical validation

---

### Baseline Results (5 runs available)

**Latest:** `baselines/baselines_20260124_182827/`

**Available data:**
- Random policy results
- Round-Robin results
- Max-Variance results
- PPO results
- Baseline comparison plots
- Intervention distribution plots

**Example metrics (Random, latest run):**
- Final total loss: ~0.02 (episode 99, step 19)
- Episodes: 100
- Convergence: Appears to converge

---

### Duffing Oscillator (5 runs available)

**Latest:** `duffing/duffing_20260124_103527/`

**Available data:**
- Episode-by-episode loss
- Learning curves (plots)
- 100 episodes per run

**Example metrics (latest run):**
- Final loss: ~0.04
- Target: X3 (concentrated interventions)
- Episodes: 100

---

### Complex SCM (6 runs available)

**Latest runs:**
- `complex_scm_random_20260124_191932/`
- `complex_scm_random_20260124_191023/`
- `complex_scm_random_20260124_183203/`
- `complex_scm_smart_random_20260124_115218/`
- `complex_scm_greedy_collider_20260124_130852/`
- `complex_scm_random_20260124_103629/`

**Available data:**
- Strategy comparison (random vs smart vs greedy)
- Results CSV files
- Analysis plots

---

## Paper Support Status

### What's Ready Now (Preliminary)

**Can populate (with caveats):**
- Duffing experiment description (validated approach works)
- Complex SCM experiment structure
- Baseline comparison framework
- Figure examples (learning curves, distributions)

**CANNOT populate yet:**
- Table 1 (needs multi-seed statistics)
- Statistical significance claims
- Ablation results
- Final quantitative claims

---

### What's Missing for Final Paper

**Critical (In Progress):**
1. Multi-seed validation results (5 seeds × 5 experiments)
2. Statistical analysis (mean, std, CI, t-tests)
3. Ablation study results (4 configurations)
4. Observational ratio optimization results

**Status:** All jobs submitted, waiting for HPC completion

**Expected:** Results ready Jan 24-25, 2026

---

## Recommendations

### Immediate Actions

1. **Wait for multi-seed jobs to complete**
   - Check HPC with `sacct --starttime today -u paco0228`
   - Monitor with `squeue -u paco0228`
   
2. **Do NOT use single-run results for final paper**
   - These are from OLD CODE (before improvements)
   - No statistical validation
   - Suitable only for initial analysis

3. **Prepare for data processing**
   - When jobs complete, run:
     ```bash
     ./ace.sh consolidate-multi-seed results/multi_run_20260123_152058
     python scripts/statistical_tests.py results/multi_run_20260123_152058
     python scripts/analyze_ablations.py results/ablations_*
     ```

---

## Timeline Estimate

**Now:** Jan 25, 11:00 AM
- Jobs running/queued on HPC
- Some may have completed already

**By:** Jan 25, evening or Jan 26, morning
- Multi-seed jobs should complete
- Ablation jobs should complete
- All data ready for analysis

**Then:** Jan 26
- Consolidate multi-seed results
- Run statistical tests
- Generate tables and figures
- Populate paper TODOs
- **Paper ready for submission!**

---

## Next Steps

1. Check HPC job status:
   ```bash
   sacct --starttime 2026-01-23 -u paco0228 --format=JobID,JobName,State,Elapsed
   ```

2. When jobs complete, sync results:
   ```bash
   # Option 1: Use sync script
   ./ace.sh sync-hpc
   
   # Option 2: Manual rsync
   rsync -avz paco0228@hpc:~/ACE/results/multi_run_* results/
   rsync -avz paco0228@hpc:~/ACE/results/ablations_* results/
   rsync -avz paco0228@hpc:~/ACE/results/obs_ratio_* results/
   ```

3. Process results:
   ```bash
   ./ace.sh consolidate-multi-seed results/multi_run_20260123_152058
   python scripts/statistical_tests.py results/multi_run_20260123_152058
   python scripts/analyze_ablations.py results/ablations_*
   ```

4. Fill paper with actual data!

---

## Summary

**Available Now:** Preliminary single runs (OLD CODE, informational only)  
**Critical Missing:** Multi-seed validation (NEW CODE, statistical rigor)  
**Status:** Waiting for HPC jobs (17 visible, likely 16 completed)  
**Expected:** Complete data within 12-24 hours  
**Paper:** Ready to populate once multi-seed data arrives

**You're on track for paper completion this weekend!**
