# ACE Optimization Checklist for Success
**Date:** January 25, 2026  
**Goal:** Ensure ACE outperforms baselines with fewer episodes

---

## Current Baseline Performance (Target to Beat)

From 5 independent runs (N=5):

| Method | Final Loss | Std | Episodes |
|--------|------------|-----|----------|
| **Max-Variance** | **2.05** | **0.12** | **100** |
| PPO | 2.11 | 0.13 | 100 |
| Round-Robin | 2.15 | 0.08 | 100 |
| Random | 2.18 | 0.06 | 100 |

**ACE Target:** Final loss ≤ 2.05, Episodes < 60 (40% reduction)

---

## Critical Parameters for ACE Success

### 1. Early Stopping (CRITICAL)

**Current settings:**
```python
--early_stopping
--use_per_node_convergence
--early_stop_min_episodes 40
```

**What to verify:**
- ACE should stop at 40-60 episodes, NOT 200
- Per-node convergence should trigger when all nodes < threshold
- Check logs for "Early stopping triggered" message

**If ACE runs to 200 episodes:** Early stopping is broken!

---

### 2. Observational Training (CRITICAL for Root Nodes)

**Current settings:**
```python
--obs_train_interval 3      # Every 3 steps
--obs_train_samples 200     # Samples per injection
--obs_train_epochs 100      # Training epochs
```

**What to verify:**
- Root nodes (X1, X4) should converge
- Without this, roots stay at ~1.0 loss
- Check node_losses.csv: loss_X1 and loss_X4 should decrease

**Optimal:** May need to test intervals 2, 3, 4 (run obs-ablation)

---

### 3. Intervention Masking (VERIFIED)

**Status:** ✓ Tested and working (6 tests passing)

**What it does:**
- Prevents training on intervened nodes
- Preserves causal semantics
- Critical for correctness

**Verification:** Assertions will catch if this breaks

---

### 4. Dedicated Root Learner (CRITICAL)

**Current setting:**
```python
--use_dedicated_root_learner
```

**What to verify:**
- X1 and X4 (roots) should learn their distributions
- Without this: roots never converge
- Check: loss_X1 and loss_X4 in final results should be < 1.0

---

### 5. Diversity Reward (Prevents Collapse)

**Current setting:**
```python
# Enabled by default
--no_diversity_reward  # Don't use this!
```

**What to verify:**
- Interventions spread across multiple nodes
- Not >90% on single node
- Check metrics.csv: target column should vary

---

## Pre-Flight Checklist

Before running full multi-seed:

### Step 1: Quick Test (1 episode, local)
```bash
./test_ace_quick.sh
```

**Expected output:**
- Completes in ~5 minutes
- No errors
- Creates node_losses.csv, metrics.csv
- Shows intervention proposals

**Check:**
- [ ] Script completes without errors
- [ ] CSV files created
- [ ] Node losses decrease episode-to-episode

---

### Step 2: Small Test (10 episodes, local or HPC)
```bash
./run_ace_only.sh --test
```

**Expected output:**
- Completes in ~30-60 minutes
- 10 episodes of training
- Per-node convergence checks
- DPO training metrics

**Check:**
- [ ] Early stopping evaluated (may not trigger at 10 eps)
- [ ] All node losses decreasing
- [ ] Root nodes (X1, X4) learning (should see loss < 1.0)
- [ ] No intervention masking assertion errors

---

### Step 3: Full Multi-Seed (5 seeds, HPC)
```bash
./run_ace_only.sh --seeds 5
```

**Expected output:**
- 5 HPC jobs submitted
- Each runs to convergence or 200 episode max
- Early stopping should trigger at 40-60 episodes
- Results in ace_multi_seed_TIMESTAMP/seed_*/

**Check:**
- [ ] All 5 jobs complete successfully
- [ ] Early stopping triggered (check logs for "Early stopping" message)
- [ ] Episodes < 100 for all seeds
- [ ] Final losses competitive with Max-Variance (2.05)

---

## Success Criteria

ACE is successful if:

1. **Performance:** Final loss ≤ 2.11 (matches PPO)
   - Ideal: ≤ 2.05 (matches Max-Variance)
   
2. **Efficiency:** Mean episodes < 100
   - Ideal: 40-60 episodes (early stopping working)
   - Claimed: 50% reduction = ~50 episodes

3. **Convergence:** All node losses < threshold
   - Mechanisms (X2, X3, X5): < 0.5
   - Roots (X1, X4): < 1.0

4. **Stability:** Std dev reasonable across seeds
   - Loss std: < 0.15
   - Episode std: < 20

---

## Troubleshooting

### If ACE runs to 200 episodes:

**Problem:** Early stopping not triggering

**Fixes:**
1. Check thresholds are appropriate:
   ```python
   --early_stop_patience 10
   --early_stop_min_episodes 40
   ```

2. Verify per-node convergence criteria in logs

3. May need to adjust node-specific thresholds

### If ACE loss > 2.20 (worse than Random):

**Problem:** Something fundamentally wrong

**Fixes:**
1. Check logs for errors
2. Verify intervention masking working (no assertion errors)
3. Verify root learner enabled
4. Check DPO loss is decreasing

### If Root nodes (X1, X4) don't converge:

**Problem:** Dedicated root learner not working

**Fixes:**
1. Verify `--use_dedicated_root_learner` is set
2. Check observational training frequency
3. May need more obs_train_samples or epochs

---

## Expected Timeline

**Quick test (1 ep):** 5 minutes  
**Small test (10 ep):** 30-60 minutes  
**Full multi-seed (5 seeds):** 8-12 hours on HPC

---

## After Multi-Seed Completes

### Analysis:
```bash
# Consolidate results
python scripts/compute_statistics.py results/ace_multi_seed_*/seed_*/ ace

# Compare to baselines
python scripts/statistical_tests.py results/ace_multi_seed_*/ results/baselines/

# Visualize
python visualize.py results/ace_multi_seed_*/seed_*/
```

### Expected Results:
- ACE: X.XX ± Y.YY (ZZ episodes)
- Compare to Max-Variance: 2.05 ± 0.12 (100 eps)
- Calculate improvement percentage
- Test statistical significance

### Populate Paper:
If ACE ≤ 2.11 and episodes < 100:
- Update Table 1 with ACE results
- Add episode reduction percentage
- Include statistical significance
- Uncomment learning curve figures

---

## Commands Reference

### Local Testing
```bash
# Ultra-quick test (1 episode)
./test_ace_quick.sh

# Small test (10 episodes)
./run_ace_only.sh --test
```

### HPC Production
```bash
# Full 5-seed run
./run_ace_only.sh --seeds 5

# Monitor
squeue -u $USER

# Check progress
tail -f logs/ace_seed42_*.out
```

### Analysis
```bash
# After completion
python scripts/compute_statistics.py results/ace_multi_seed_*/ ace
python scripts/statistical_tests.py results/ace_multi_seed_*/ 
```

---

## Key Success Indicators

**During Run:**
- No assertion errors (masking working)
- DPO loss decreasing
- Node losses all converging
- Early stopping triggers at 40-60 eps

**After Run:**
- Final loss ≤ 2.11 (competitive)
- Episodes < 100 (efficient)
- All nodes converged (< thresholds)
- Stable across seeds (std < 0.15)

**If all criteria met:** ACE is working as designed!

---

## Next Steps

1. Run quick test: `./test_ace_quick.sh`
2. Review test output
3. If good, run 10-episode test: `./run_ace_only.sh --test`
4. If good, submit full run: `./run_ace_only.sh --seeds 5`
5. Wait for completion
6. Analyze and populate paper

**Start with the quick test to validate everything works!**
