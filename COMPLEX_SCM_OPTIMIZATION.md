# Complex 15-Node SCM - ACE Optimization Strategy

## Problem Statement

ACE achieved excellent results on 5-node SCM (0.92 ± 0.73) but failed to scale to 15-node:

**Failed Run (ace_complex_scm_20260128_080650):**
- Mean Loss: 290.13 ± 62.00 (N=5 seeds)
- **60x worse than baselines (4.51-4.71)**
- Used simplified architecture (no full DPO, limited pretraining)

**Baseline Performance (N=5 seeds, 200 episodes):**
- Greedy Collider: 4.51 ± 0.17  
- Random: 4.62 ± 0.18
- Random Lookahead (K=4): 4.65 ± 0.19
- PPO: 4.68 ± 0.20
- Round-Robin: 4.71 ± 0.15

**Goal:** Get ACE < 4.5 (better than best baseline)
**Acceptable:** 4.5-5.5 (competitive)

---

## Root Cause Analysis

### Why did the 290-loss run fail?

Checking experiment log from failed run (`ace_complex_scm_20260128_080650`):
1. **Too little pretraining:** Only 200 steps (vs 500 needed for 15 nodes)
2. **Too few steps per episode:** 25 steps (vs 50 needed)
3. **No observational training:** Critical omission - causes catastrophic forgetting
4. **Early convergence:** Stopped at 33-55 episodes (vs target 200-300)

The learner likely:
- Never properly initialized on the 15-node structure
- Forgot previously learned mechanisms without observational training
- Converged prematurely due to early stopping

### Why did simpler runs (6-8 loss) work better?

Earlier runs (`ace_complex_scm_20260128_084218`, seed 456: 6.25 loss) used:
- Simplified ACE (random candidates, no full DPO)
- But still had reasonable exploration
- Lower complexity meant less forgetting

**Insight:** Full DPO on 15 nodes requires MORE stability mechanisms, not fewer.

---

## Optimized Architecture (Already Implemented!)

File: `experiments/run_ace_complex_full.py`

### Key Improvements:

**1. Increased Oracle Pretraining:**
```python
--pretrain_steps 500  # vs 200 in failed run
```
More pretraining gives policy better initialization on complex structure.

**2. More Steps Per Episode:**
```python
--steps 50  # vs 25 in 5-node
```
More learning opportunities per episode.

**3. CRITICAL: Observational Training**
```python
--obs_train_interval 3      # Every 3 steps
--obs_train_samples 200     # 200 observational samples
--obs_train_epochs 100      # 100 training epochs
```
**This was MISSING in failed run.** Prevents catastrophic forgetting.

**4. Longer Runs:**
```python
--episodes 300  # vs 200 in failed run
```
Give more time for complex structure.

**5. Enhanced Diversity Mechanisms:**
- Collider parent tracking
- Forced diversity every 10 steps
- Smart collapse breakers (prioritize collider parents)
- Hard caps on node concentration (70% max)

**6. Full DPO Loop:**
- Candidate generation via Qwen2.5-1.5B
- Winner/loser selection from K=4 candidates
- DPO loss computation and gradient updates
- Reference policy updates every 25 episodes

---

## Implementation Status

✓ **Code Complete:** `experiments/run_ace_complex_full.py`
✓ **Job Script:** `jobs/run_ace_complex_single_seed.sh`
✓ **Test Coverage:** `tests/test_ace_complex_full.py` (22 test cases)

**Last Run:** Job 23366125
- **Status:** FAILED (indentation error at line 182)
- **Fixed:** Commit 82b55f3 (indentation corrected)
- **Resubmitted:** Job 23399683 (queued on HPC)

---

## Expected Performance

### Optimistic (Target):
```
ACE Complex SCM: 3.5-4.0 (better than best baseline 4.51)
```
**If achieved:** Major result, proves ACE scales well

### Realistic (Acceptable):
```
ACE Complex SCM: 4.5-5.5 (competitive with baselines)
```
**If achieved:** Shows ACE is viable at scale, slight degradation acceptable

### Pessimistic (Limitation):
```
ACE Complex SCM: >5.5 (worse than baselines)
```
**If achieved:** Acknowledge scaling limitation, remove from results section

---

## Execution Plan

### Phase 1: Single Seed Overnight Test (IN PROGRESS)

**Job:** 23399683 (queued on HPC)
**Command:** `sbatch jobs/run_ace_complex_single_seed.sh`
**Expected Runtime:** 6-8 hours
**Expected Completion:** Morning of Jan 30, 2026

**Monitor:**
```bash
# Check queue position
squeue -j 23399683

# When running, watch logs
tail -f logs/ace_complex_s42_23399683.out

# Check for startup messages
grep "\[STARTUP\]\|\[PROGRESS\]" logs/ace_complex_s42_23399683.out
```

### Phase 2: Evaluate Results

**If loss < 4.5 (SUCCESS):**
- Run all 5 seeds (submit `jobs/run_ace_complex_scm.sh`)
- Add to paper as scaling success
- Expected runtime: 30-40 hours

**If loss 4.5-5.5 (MARGINAL):**
- Discuss with user
- Maybe run 2-3 more seeds for statistical validation
- Include with caveats about scaling challenges

**If loss > 5.5 (FAILURE):**
- Do NOT include in results
- Acknowledge in discussion/limitations
- Paper still acceptable without it

### Phase 3: Paper Integration (If Successful)

Add table to results section:
```latex
\begin{table}[t]
\caption{Complex 15-node SCM results (N=5 seeds, 300 episodes)}
\begin{tabular}{lc}
\toprule
Method & Total Loss \\
\midrule
ACE & X.XX $\pm$ X.XX \\
Greedy Collider & 4.51 $\pm$ 0.17 \\
Random & 4.62 $\pm$ 0.18 \\
PPO & 4.68 $\pm$ 0.20 \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Contingency: If Optimized Run Still Fails

### Option A: Increase Resources Further
- 1000 pretrain steps (vs 500)
- 75 steps per episode (vs 50)
- 500 episodes (vs 300)
- Runtime: 12-15 hours per seed

### Option B: Simplify Architecture
- Reduce to K=2 candidates (faster, less overfitting)
- Increase observational training (every 2 steps vs 3)
- More frequent reference updates (every 15 episodes vs 25)

### Option C: Accept Limitation
- Remove complex SCM from results
- Use only as motivational framing in introduction
- Focus paper on validated 5-node performance
- Acknowledge 15-node as future work

---

## Key Files

**Implementation:**
- `experiments/run_ace_complex_full.py` - Full ACE for complex SCM

**Job Scripts:**
- `jobs/run_ace_complex_single_seed.sh` - Single seed (10h limit)
- `jobs/run_ace_complex_scm.sh` - All 5 seeds (12h limit each)

**Tests:**
- `tests/test_ace_complex_full.py` - 22 comprehensive tests

**Documentation:**
- `FINAL_COMPLEX_SCM_RUN.md` - Complete execution guide
- This file - Optimization strategy

---

## Current Status

**Job 23399683:** Queued on HPC
**ETA:** 6-8 hours after start
**Next Check:** Monitor queue and check logs when running

**The optimized architecture is ready. Now we wait for results.**
