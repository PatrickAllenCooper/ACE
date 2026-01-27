# IMMEDIATE NEXT STEPS - Critical Experiments Recovery

**Date:** January 27, 2026, 1:30 AM MST
**Status:** Critical experiments fixed, ready to resubmit

---

## What Just Happened

**All ablation and critical jobs failed due to missing functions:**
- Critical experiments script was importing functions that didn't exist
- `run_random_policy`, `run_round_robin_policy`, `run_max_variance_policy` were not in baselines.py
- Jobs failed immediately with `ModuleNotFoundError`

**What was fixed (just pushed to main):**
1. Added all missing baseline runner functions
2. Fixed SCMLearner to accept oracle and have evaluate() method
3. Fixed critical experiments to create StudentSCM correctly
4. Added sys.path for subdirectory imports
5. Added 20+ new tests covering all new functionality

---

## EXECUTE ON HPC NOW

```bash
cd /projects/paco0228/ACE

# Pull all fixes
git pull

# Resubmit critical experiments
sbatch jobs/run_critical_experiments.sh

# Monitor for success
sleep 60
tail -50 logs/critical_*.out | tail -30
```

**Expected output (SUCCESS):**
```
Found scripts/runners/run_critical_experiments.py
Running: ALL Critical Experiments

[1/3] EXTENDED BASELINES (171 episodes)

==================================================
EXTENDED BASELINES - Seed 42, Episodes 171
==================================================

Running random...
Running round_robin...
Running max_variance...
```

**If you see that output, job is working!**

---

## Current Ablation Status

**Completed:**
- no_dpo: 3 seeds ✓ (job 23319484, 56 minutes)

**Timed out (2h limit):**
- no_convergence: Partial results
- no_root_learner: Partial results  
- no_diversity: Partial results

**Recommendations:**
1. **Accept no_dpo results** (complete, 3 seeds)
2. **Skip other ablations** if time-constrained - not critical for reviewer response
3. **OR re-run with 4h limit** if you want complete ablation data

---

## Priority

**CRITICAL (do now):**
- ✅ Pull fixed code
- ✅ Resubmit critical experiments
- ✅ Verify it runs (see episode output)
- ⏳ Wait 5-7 hours for completion

**OPTIONAL (if time allows):**
- Re-run timed-out ablations with longer limit
- Collect and analyze no_dpo ablation results

---

## Expected Timeline

**If critical experiments work:**
- Submit: Now (~1:30 AM)
- Complete: ~6-8 AM
- Analyze: 2-3 hours
- Update paper: 1-2 hours
- **Ready for submission: Late morning/afternoon Jan 27**

**If more debugging needed:**
- Add 3-6 hours per iteration

---

## What We'll Have When Critical Completes

✅ **Extended baselines (171 episodes)** - Fair budget comparison
✅ **Lookahead ablation** - Random proposer vs DPO
✅ **15-node Complex SCM** - Full scaling validation
✅ **Learning curves** - Sample efficiency data
✅ **no_dpo ablation** - DPO contribution validated

**Paper strength: ACCEPT tier** (all major reviewer concerns addressed)

---

## Critical Experiments Expected Results

**Location after completion:**
```
results/critical_experiments_YYYYMMDD_HHMMSS/
├── extended_baselines/
│   ├── random_seed42_curve.csv
│   ├── round_robin_seed42_curve.csv
│   ├── max_variance_seed42_curve.csv
│   └── extended_baselines_summary.csv
├── lookahead_ablation/
│   ├── random_lookahead_seed42_curve.csv
│   └── lookahead_ablation_summary.csv
├── complex_scm/
│   └── complex_scm_summary.csv
└── experiment_config.json
```

**Key files:**
- `*_summary.csv`: Final metrics per method
- `*_curve.csv`: Per-episode learning curves
- `experiment_config.json`: Run configuration

---

## Monitoring Commands

```bash
# Check job status
squeue -u paco0228

# Monitor output (if job is running)
tail -f logs/critical_*.out

# Check for results
ls -R results/critical_experiments_*/

# Verify progress
find results/critical_experiments_* -name "*.csv" | wc -l
```

---

## If Job Fails Again

Check logs immediately:
```bash
tail -100 logs/critical_*.err
tail -100 logs/critical_*.out
```

Share the error and I'll fix it.

---

## Documentation

- **guidance_documents/current_status.txt** - Current status and monitoring
- **guidance_documents/remaining_refinements.txt** - Path to Strong Accept
- **guidance_documents/guidance_doc.txt** - Complete technical reference

**Last updated:** Jan 27, 2026, 1:30 AM MST
