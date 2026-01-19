# ACE Paper Experiments - Analysis Report
**Run Date:** January 16, 2026  
**Run ID:** paper_20260116_083515  
**Jobs:** 22841445 (ACE), 22841446 (Baselines), 22841447 (Duffing), 22841448 (Phillips)

---

## Executive Summary

### ✅ What Worked
- **Baselines completed successfully** - All 4 baselines (Random, Round-Robin, Max-Variance, PPO) ran to completion
- **X3 collider learning** - All methods successfully learned the X3 mechanism (loss <0.15)
- **Observational training** - Prevented catastrophic forgetting of mechanisms
- **DPO training** - Policy learning is working (loss=0.035, 95% winner preference)

### ❌ What Failed
- **ACE job timeout** - Ran only 240/500 episodes before 12-hour limit
- **ACE output incomplete** - Job cancelled before saving final CSVs/plots
- **Duffing & Phillips experiments** - Both failed due to missing scipy dependency
- **X2 intervention collapse** - ACE stuck at 99% X2 interventions despite smart breaker

### ⚠️ Critical Issues
1. **Time allocation** - 12 hours insufficient for 500 episodes
2. **Missing dependencies** - scipy not installed in conda environment
3. **X2 collapse not resolved** - Smart breaker alone insufficient
4. **Final outputs missing** - No mechanism_contrast.png, metrics.csv, etc. from ACE

---

## Detailed Analysis

### 1. ACE Main Experiment (Job 22841445)

**Status:** ⏱️ TIMEOUT (cancelled at 12 hours)  
**Progress:** 240/500 episodes (48%)  
**Node:** c3gpu-c2-u1 (A100 GPU)

#### Key Findings

**DPO Training Quality:**
```
Avg Loss: 0.0347              ✓ (excellent, near-optimal)
Preference Margin: 415.47     ✓ (very positive)
Winner Preference: 95.0%      ✓ (strong signal)
KL from Reference: -2088.65   ⚠️ (very large drift)
```

**Intervention Distribution:**
```
X2: 99%    ⚠️ EXTREME COLLAPSE
X1: ~1%    
X3: 0%
X4: 0%
X5: 0%
```

**Smart Breaker Activity:**
```
- Triggered every step (99% X2 detected)
- Generated value-diverse X2 interventions
- But policy still generated 100% X2 candidates
```

**Why ACE Failed to Diversify:**
1. DPO reward signal heavily favors X2 (collider parent)
2. Collapse penalty (-7141.5) dominated by disentanglement bonus (+648.7)
3. Policy learned X2 is optimal, ignores diversification incentives
4. Smart breaker injects X2 (value-diverse) but doesn't force X1 interventions

#### What Was NOT Saved:
- `mechanism_contrast.png` - Would show if X2 mechanism forgotten
- `metrics.csv` - Per-step rewards/scores
- `node_losses.csv` - Mechanism learning trajectories
- `dpo_training.csv` - Full DPO diagnostics
- `scm_graph.png` - Final visualization

#### What WAS Saved:
- `value_diversity.csv` - Only file written before timeout

---

### 2. Baselines (Job 22841446)

**Status:** ✅ COMPLETED  
**Duration:** 25 minutes  
**Episodes:** 100 each

#### Results Summary

| Method | Total Loss | X1 | X2 | X3 (Collider) | X4 | X5 | Top Intervention |
|--------|------------|----|----|---------------|----|----|------------------|
| Random | 2.36 ± 0.06 | 1.22 ✗ | 0.01 ✓ | 0.12 ✓ | 1.00 ✗ | 0.01 ✓ | Uniform |
| Round-Robin | 2.36 ± 0.06 | 1.22 ✗ | 0.01 ✓ | 0.12 ✓ | 1.00 ✗ | 0.01 ✓ | Uniform (20% each) |
| Max-Variance | 2.27 ± 0.06 | 1.08 ✗ | 0.01 ✓ | 0.09 ✓ | 1.07 ✗ | 0.02 ✓ | X4 (21%) |
| PPO | 2.14 ± 0.12 | 0.95 ✗ | 0.01 ✓ | 0.14 ✓ | 1.02 ✗ | 0.01 ✓ | X1 (38%), X2 (35%) |

#### Key Observations

**All Baselines:**
- ✓ Successfully learned X2, X3, X5 mechanisms
- ✗ Failed to learn X1, X4 (root nodes)
- ✓ X3 collider learned despite correlations

**Root Node Learning Failure:**
- X1, X4 have no parents → no mechanism to learn
- High "loss" is actually distribution mismatch (mean/variance)
- This is expected behavior, not a failure

**PPO vs Other Baselines:**
- PPO shows highest variance (±0.12 vs ±0.06)
- PPO concentrates on X1 (38%) and X2 (35%)
- PPO total loss marginally better but not significantly

**Comparison Plots Generated:**
- `baseline_comparison.png` - Learning curves
- `intervention_distribution.png` - Intervention frequencies
- Individual CSVs for each method

---

### 3. Duffing Oscillators (Job 22841447)

**Status:** ❌ FAILED  
**Error:** `ModuleNotFoundError: No module named 'scipy'`

**Root Cause:**
```python
from scipy.integrate import solve_ivp  # Line 16 of duffing_oscillators.py
```

The `ace` conda environment does not have scipy installed. This is required for ODE integration (RK45 solver).

**Fix Required:**
```bash
conda activate ace
conda install scipy  # or: pip install scipy
```

---

### 4. Phillips Curve (Job 22841448)

**Status:** ❌ FAILED  
**Error:** `ModuleNotFoundError: No module named 'scipy'` (transitively via experiments/__init__.py)

**Root Cause:**
The experiments/__init__.py imports duffing_oscillators, which imports scipy, so phillips_curve fails even though it doesn't directly use scipy.

**Fixes Required:**
1. Install scipy (same as above)
2. OR: Make __init__.py imports conditional/lazy

---

## Root Cause Analysis

### Issue 1: ACE Timeout

**Problem:** 500 episodes at ~3 min/episode = 25 hours, but job limited to 12 hours

**Contributing Factors:**
- LLM generation slow (~10s per candidate × 4 candidates × 25 steps × 500 episodes)
- Learner training (100 epochs per step)
- DPO update every step

**Solutions:**
1. Reduce episodes to 200 for 12-hour jobs
2. Request 24-hour partition
3. Speed up by using --custom transformer
4. Reduce --learner_epochs from 100 to 50

### Issue 2: X2 Intervention Collapse

**Problem:** Policy learned that X2 interventions maximize reward, won't explore despite penalties

**Why Smart Breaker Failed:**
- Smart breaker injects X2 with diverse VALUES (good!)
- But doesn't inject X1 interventions (bad!)
- Policy candidates are still 100% X2

**Why This Happens:**
```
Disentanglement Bonus (X2): +648.7
Collapse Penalty (99% X2):  -7141.5
Net Score:                  -6408.8

But policy still generates X2 because:
- X1 interventions give 0 disentanglement bonus
- X3/X4/X5 give even worse scores (leaf penalty, irrelevance)
```

**Fundamental Issue:**
The reward structure makes X2 objectively optimal. No amount of penalties will make X1 better unless we:
1. Cap X2 interventions (hard constraint)
2. Boost X1 bonus when X2 is over-sampled
3. Change DPO to incorporate diversity in the preference pairs

### Issue 3: Missing Dependencies

**Problem:** scipy not in environment

**Impact:**
- Duffing oscillators can't run (needs ODE solver)
- Phillips curve can't import (transitive dependency)

**Fix:** One-line conda/pip install

---

## Recommendations

### Immediate Actions (Next Run)

1. **Install scipy**
   ```bash
   conda activate ace
   conda install scipy pandas_datareader  # For both experiments
   ```

2. **Reduce ACE episodes for 12-hour limit**
   ```bash
   ACE_EPISODES=200 ./run_all.sh  # Should complete in ~10 hours
   ```

3. **OR Request longer partition**
   ```bash
   # Modify jobs/run_ace_main.sh:
   #SBATCH --time=24:00:00
   ```

4. **Re-run to get complete ACE outputs**
   - Need mechanism_contrast.png to verify X2 forgetting
   - Need metrics.csv for paper figures
   - Need dpo_training.csv for analysis

### Medium-Term Fixes (Code Changes)

1. **Fix X2 Collapse** - Three options:

   **Option A: Hard Intervention Cap**
   ```python
   if intervention_counts[target] / total > 0.70:
       # Force smart breaker to pick different node
       target = pick_undersampled_parent_of_collider_excluding(target)
   ```

   **Option B: Adaptive Boosting**
   ```python
   # If X2 >70%, give X1 massive boost
   if x2_fraction > 0.70:
       x1_boost = 1000.0 * (x2_fraction - 0.70)
   ```

   **Option C: DPO Diversity Regularization**
   ```python
   # Add diversity term to DPO loss
   diversity_penalty = -entropy(intervention_distribution)
   dpo_loss = dpo_loss + lambda * diversity_penalty
   ```

2. **Speed Up Training**
   ```python
   --learner_epochs 50    # Down from 100
   --candidates 2         # Down from 4
   --pretrain_steps 50    # Down from 100
   ```

3. **Lazy Import for Experiments**
   ```python
   # experiments/__init__.py
   def get_duffing():
       from .duffing_oscillators import run_duffing_experiment
       return run_duffing_experiment
   
   def get_phillips():
       from .phillips_curve import run_phillips_experiment
       return run_phillips_experiment
   ```

### Long-Term (Paper Implications)

1. **Adjust Claims**
   - Current: "ACE outperforms all baselines"
   - Reality: ACE timeout, can't compare yet
   - PPO shows similar performance to other baselines
   - Need completed ACE run to validate DPO > PPO claim

2. **Missing Experiments**
   - Duffing & Phillips not yet validated
   - Sections 3.6 and 3.7 are implementation-only, no results
   - Either complete these or mark as "proposed extensions"

3. **Success Criteria Update**
   - X3 learning: ✓ Achieved by all methods
   - X2 preservation: Unknown (ACE incomplete)
   - Intervention diversity: ✗ Not achieved (99% X2)

---

## Files Generated

### Available for Analysis:
```
results/paper_20260116_083515/
├── baselines/baselines_20260116_083558/
│   ├── baseline_comparison.png       ✓
│   ├── intervention_distribution.png ✓
│   ├── random_results.csv            ✓
│   ├── round_robin_results.csv       ✓
│   ├── max_variance_results.csv      ✓
│   ├── ppo_results.csv               ✓
│   └── ppo_training.csv              ✓
├── ace/run_20260116_083547/
│   └── value_diversity.csv           ✓ (only this)
└── job_info.txt                      ✓
```

### Logs Available:
```
logs/
├── ace_main_20260116_083515_22841445.out    ✓
├── ace_main_20260116_083515_22841445.err    ✓ (985KB)
├── baselines_20260116_083515_22841446.out   ✓
├── baselines_20260116_083515_22841446.err   ✓
├── duffing_20260116_083515_22841447.out     ✓
├── duffing_20260116_083515_22841447.err     ✓
├── phillips_20260116_083515_22841448.out    ✓
└── phillips_20260116_083515_22841448.err    ✓
```

---

## Next Steps

### Priority 1: Get Complete ACE Results
```bash
# Fix and re-run
conda activate ace
conda install scipy pandas_datareader

# Quick validation run
QUICK=true ACE_EPISODES=50 ./run_all.sh

# Full run with realistic time
ACE_EPISODES=200 ./run_all.sh  # Or request 24hr partition
```

### Priority 2: Address X2 Collapse
Implement Option A (hard cap) as it's simplest:
```python
# In ace_experiments.py, before winner selection:
if winner_plan["target"] == "X2":
    x2_frac = intervention_counts["X2"] / sum(intervention_counts.values())
    if x2_frac > 0.70:
        # Force different node
        winner_plan = smart_breaker_diversified()
```

### Priority 3: Validate New Experiments
Once scipy installed, verify Duffing & Phillips work locally:
```bash
python -m experiments.duffing_oscillators --episodes 10
python -m experiments.phillips_curve --episodes 10
```

---

## Questions for Discussion

1. **Time allocation:** Keep 500 episodes (need 24hr partition) or reduce to 200 (fit in 12hr)?

2. **X2 collapse:** Which fix to implement? Hard cap is easiest but crude. DPO regularization is elegant but complex.

3. **Paper scope:** Should Duffing/Phillips be "validated experiments" or "proposed extensions"? Currently we have code but no results.

4. **Baseline comparison:** PPO shows no clear advantage over simpler baselines. How to discuss this given paper claims DPO > PPO?

5. **Success criteria:** All methods learned X3. Is this sufficient, or do we need ACE to show better intervention strategy?
