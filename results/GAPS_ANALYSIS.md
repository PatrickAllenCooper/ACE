# Results We Still Need to Prove
## Gap Analysis - January 21, 2026

**Purpose:** Identify which paper claims lack supporting evidence.

---

## 🚦 Status Legend

- ✅ **PROVEN** - Have solid evidence
- ⚠️ **PARTIAL** - Have some evidence, needs more
- ❌ **MISSING** - No evidence yet
- 🔄 **IN PROGRESS** - Currently running

---

## CRITICAL GAPS (Blockers for Submission)

### ❌ **Gap 1: ACE Final Performance Numbers**

**Paper Claims:**
- Line 439: "ACE achieves [TOTAL_MSE] total MSE"
- Line 439: "ACE achieves [X3_MSE] on collider vs [X3_BASELINE]"
- Table 1 (lines 428-437): All ACE numbers missing

**What We Have:**
- ❌ NO successful ACE run to completion
- Latest run failed (99% zero rewards, zero gradients)

**What We Need:**
```bash
# Run ACE with Jan 21 fixes
sbatch jobs/run_ace_main.sh

# Extract:
# - Final total loss (target: < 2.0 to beat Random)
# - Per-node losses (X1, X2, X3, X4, X5)
# - Number of episodes to convergence
# - Runtime
```

**Target Values (to support claims):**
- Total Loss: < 1.99 (beat Round-Robin, the best baseline)
- X3 Loss: < 0.06 (beat baseline ~0.06-0.08)
- Episodes: 40-80 (support "early stopping" claim)
- Runtime: < 2 hours (support "80% reduction" claim)

**Status:** ❌ **BLOCKING PAPER SUBMISSION**

---

### ❌ **Gap 2: ACE Intervention Distribution**

**Paper Claims:**
- Line 485: "ACE concentrates [X2_PCT]% on X2 and [X1_PCT]% on X1"
- Figure 3: Intervention distribution bar chart

**What We Have:**
- ⚠️ Log shows X2 at 70.2% (but this is from failed run with diversity issues)

**What We Need:**
```python
# From successful ACE run
import pandas as pd
df = pd.read_csv('results/paper_*/ace/metrics.csv')
distribution = df['target'].value_counts(normalize=True) * 100

# Extract:
X1_pct = distribution['X1']  # Expected: 15-25%
X2_pct = distribution['X2']  # Expected: 50-65% (strategic, not collapsed)
X3_pct = distribution['X3']  # Expected: 5-10%
X4_pct = distribution['X4']  # Expected: 5-10%
X5_pct = distribution['X5']  # Expected: 5-10%
```

**Target Values:**
- X2: 50-65% (strategic concentration on collider parent)
- X1: 15-25% (other collider parent)
- Combined X1+X2: > 65% (shows collider focus)
- But X2 < 70% (shows it's strategic, not policy collapse)

**Status:** ❌ **BLOCKING TABLE/FIGURE**

---

### ⚠️ **Gap 3: DPO vs PPO Comparison**

**Paper Claims:**
- Line 734: "DPO consistently outperforms PPO"
- Line 378: PPO baseline for fair comparison
- Discussion section depends on this

**What We Have:**
- ✅ PPO baseline run: 2.1835 total loss
- ⚠️ BUT: PPO has implementation bug (shape mismatch warning)
- ❌ Cannot confidently claim superiority with buggy baseline

**What We Need:**
```bash
# 1. Fix PPO bug in baselines.py:573
# 2. Rerun PPO baseline
python baselines.py --baseline ppo --episodes 100

# Extract:
# - Final total loss
# - Per-node losses
# - Intervention distribution
# - Training stability
```

**Target Values:**
- PPO total loss: > ACE total loss (for DPO superiority claim)
- OR: If PPO performs better, we need to:
  - Acknowledge it in discussion
  - Explain why (non-stationary rewards, etc.)
  - Position DPO as "more stable" not necessarily "better"

**Status:** ⚠️ **CREDIBILITY ISSUE**

---

## IMPORTANT GAPS (Weaken Paper if Missing)

### ❌ **Gap 4: Complex 15-Node SCM Results**

**Paper Claims:**
- Lines 609: "strategic advantage becomes more pronounced at scale"
- Table 2 (lines 600-608): Collider vs non-collider MSE
- Lines 532-609: Entire subsection

**What We Have:**
- ✅ Random policy: Complete
- ✅ Smart Random policy: Complete
- 🔄 Greedy Collider policy: IN PROGRESS

**What We Need:**
```bash
# Wait for completion of greedy_collider
# Then extract from results/paper_*/complex_scm/

# Metrics:
# - Collider loss (average of 5 colliders)
# - Non-collider loss
# - Total loss
# - Compare: random vs smart_random vs greedy_collider
```

**Expected Finding:**
- Greedy collider should have lower collider loss than random
- Should show strategic advantage scales with problem size

**Status:** 🔄 **WAITING FOR COMPLETION**

---

### ⏳ **Gap 5: Clamping Strategy in Duffing** ⚠️ CRITICAL ISSUE

**Paper Claims:**
- Line 661: "ACE discovers a 'clamping' strategy"
- Line 661: "by intervening to hold the middle oscillator fixed (do(X2 = 0))"

**What We Have:**
- ✅ Duffing experiment completed
- ❌ Haven't verified if this ACTUALLY emerged
- 🔴 **ALIGNMENT ISSUE:** Duffing uses RANDOM policy (not ACE)

**Critical Finding:**
The duffing_oscillators.py experiment uses a simple random intervention policy:
```python
target = np.random.choice(oracle.nodes)
value = np.random.uniform(-2, 2)
```
It does NOT use ACE/DPO. The paper claim "ACE discovers" is incorrect.

**What We Need:**
```bash
# Analyze Duffing intervention logs
grep "DO X2 = " results/paper_*/duffing/*/experiment.log | \
  awk -F '=' '{print $2}' > x2_values.txt

# Check:
# 1. Are X2 interventions clustered near 0?
# 2. Mean of X2 interventions (should be near 0)
# 3. Standard deviation (should be low if clamping)
# 4. Did ACE learn this or did we program it?
```

**Options:**
1. **Verify and revise** (Recommended): Run detector, then revise paper
   ```bash
   python clamping_detector.py
   # If no clamping: Revise to "random interventions enable structure discovery"
   ```
2. **Implement ACE on Duffing** (8+ hours): Add LLM policy, actually discover clamping
3. **Soften claim**: "Interventions enable structure identification"

**Status:** 🔴 **CRITICAL - PAPER CLAIM INCORRECT**

---

### ⏳ **Gap 6: Regime Selection in Phillips** ⚠️ ALIGNMENT ISSUE

**Paper Claims:**
- Line 714: "ACE learns to query high-volatility regimes"
- Line 714: "allocates [HIGH_VOL_PCT]% to pre-1985 data"

**What We Have:**
- ✅ Phillips experiment completed
- ❌ Haven't verified actual regime selection behavior
- 🔴 **ALIGNMENT ISSUE:** Phillips uses HARDCODED regime selection (not learned)

**Critical Finding:**
The phillips_curve.py experiment has a hardcoded regime selection strategy:
```python
if step < 5:
    regime = None
elif step < 10:
    regime = "great_inflation"  # HARDCODED
```
It does NOT learn. The paper claim "ACE learns" is incorrect.

**What We Need:**
```python
# Analyze Phillips intervention history
df = pd.read_csv('results/paper_*/phillips/*/results.csv')

# Map episodes to historical periods
# Classify: high-vol (1970s, 2008) vs low-vol (1990s-2007)

# Calculate:
high_vol_pct = (interventions in 1970s + 2008-2009) / total * 100
pre_1985_pct = (interventions before 1985) / total * 100

# Compare to dataset composition:
dataset_pre_1985 = [calculate from date range]
```

**Expected Finding:**
- High-vol queries should be > random baseline
- Should show ACE preferentially selects informative regimes

**Options:**
1. **Revise paper claim** (Recommended, 30 min):
   ```latex
   Systematic querying of high-volatility regimes achieves [OOS_MSE]...
   ```
2. **Implement learning** (8+ hours): Add actual regime selection policy
3. **Verify and report**: Run analyzer, note it's hardcoded in discussion

**Status:** 🔴 **PAPER CLAIM INCORRECT - MUST REVISE**

---

### ❌ **Gap 7: 80% Computational Savings**

**Paper Claims:**
- Line 123: "80% computational savings"
- Line 767: "80% reduction in computational time (40-60 episodes vs. 200)"

**What We Have:**
- ❌ Latest ACE run: 7+ hours for 31 episodes (SLOWER, not faster)
- ✅ Fixes implemented (Jan 21) that should address this

**What We Need:**
```bash
# From successful ACE run after fixes:
# - Actual episodes to convergence
# - Actual runtime
# - Compare to baseline 100-200 episodes

# Calculate:
episode_reduction = (baseline_episodes - ace_episodes) / baseline_episodes * 100
time_reduction = (baseline_time - ace_time) / baseline_time * 100
```

**Reality Check:**
- Baselines run 100 episodes each
- IF ACE converges in 50 episodes: 50% reduction (not 80%)
- IF ACE converges in 40 episodes: 60% reduction (not 80%)
- Original "80%" may have been from different baseline (200 episodes)

**Options:**
1. IF we achieve 40-60 episodes: Update to "50-60% reduction"
2. IF we achieve 80% savings: Keep claim
3. IF worse: Acknowledge and explain (per-node convergence trades speed for completeness)

**Status:** ❌ **LIKELY NEED TO REVISE CLAIM**

---

## MINOR GAPS (Nice to Have)

### ⏳ **Gap 8: Learning Curves Figure**

**Paper Claims:**
- Line 441: Figure 2 showing learning curves
- Line 482: "dashed line indicates ACE's early stopping point"

**What We Have:**
- ✅ Baseline learning curves (from completed runs)
- ❌ ACE learning curve (need successful run)

**What We Need:**
```bash
# Generate figure comparing all methods
python visualize.py results/paper_*/ace/
cp results/paper_*/ace/learning_curves.png results/figures/

# Should show:
# - ACE converging faster than baselines
# - Early stopping point marked
# - All 4 baselines for comparison
```

**Status:** ⏳ **WAITING ON ACE RUN**

---

### ⏳ **Gap 9: Intervention Distribution Figure**

**Paper Claims:**
- Line 487: Figure 3 comparing ACE vs Random distribution

**What We Have:**
- ✅ Random distribution: 20% uniform (by design)
- ❌ ACE distribution (need successful run)

**What We Need:**
```python
# Generate bar chart
import matplotlib.pyplot as plt
import pandas as pd

ace_df = pd.read_csv('results/paper_*/ace/metrics.csv')
ace_dist = ace_df['target'].value_counts(normalize=True) * 100

# Create figure showing ACE concentrated on X1, X2
# Random uniform at 20%
```

**Status:** ⏳ **WAITING ON ACE RUN**

---

### ❌ **Gap 10: Structure Identification F1 Score (Duffing)**

**Paper Claims:**
- Line 661: "[STRUCTURE_F1] F1 score" for ACE
- Line 661: "[BASELINE_F1]" for Max-Variance

**What We Have:**
- ✅ Duffing completed
- ❌ Haven't calculated F1 score for structure identification

**What We Need:**
```python
# Compare learned graph to true graph
true_graph = {(X1, X2), (X2, X3)}  # Chain
learned_graph = extract_from_results()

# Calculate F1:
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * precision * recall / (precision + recall)
```

**Status:** ❌ **NEEDS CALCULATION**

---

## SUMMARY: What We Must Run/Extract

### **CRITICAL (Before Submission):**

1. **✅ RUN: ACE with Jan 21 fixes** (4-6 hours)
   - Fills Table 1
   - Provides intervention distribution
   - Validates early stopping
   - Tests "80% savings" claim

2. **⚠️ FIX & RUN: PPO baseline** (2-3 hours)
   - Validates DPO superiority claim
   - Critical for discussion section

3. **🔄 WAIT: Complex SCM completion** (already running)
   - Fills Table 2
   - Supports scaling claim

### **IMPORTANT (Strengthen Paper):**

4. **⏳ VERIFY: Clamping in Duffing** (30 min analysis)
   - Either keep or remove claim
   - Check if behavior actually emerged

5. **⏳ VERIFY: Regime selection in Phillips** (30 min analysis)
   - Either keep or remove claim
   - Check if behavior actually emerged

6. **⏳ CALCULATE: Structure F1 for Duffing** (1 hour)
   - Fill placeholder in paper
   - Quantify structure learning

### **OPTIONAL (Nice to Have):**

7. **📊 GENERATE: All figures** (1 hour after ACE completes)
   - Learning curves
   - Intervention distributions
   - Publication-ready

8. **📊 STATISTICAL: Multiple runs** (20-40 hours)
   - Confidence intervals
   - Significance tests
   - Acknowledge limitation if not done

---

## Timeline to Fill Gaps

### **Optimistic (3 days):**
```
Day 1: Test fixes (30m) → ACE run (4-6h) → Extract numbers (2h)
Day 2: Fix PPO (30m) → PPO run (2h) → Verify behaviors (2h) → Fill tables (2h)
Day 3: Generate figures (1h) → Final review (2h) → SUBMIT ✅
```

### **Realistic (5 days):**
```
Day 1: Test fixes → debug minor issues → retest → launch ACE
Day 2: ACE completes → extract numbers → find 1-2 issues → rerun subset
Day 3: Fix PPO → PPO run → verify behaviors → find clamping didn't emerge → revise claim
Day 4: Wait for complex_scm → fill all tables → generate figures
Day 5: Final review → address any discrepancies → SUBMIT ✅
```

---

## Go/No-Go Decision Points

### **CAN SUBMIT IF:**
- ✅ ACE total loss < 2.0 (competitive with baselines)
- ✅ X3 collider loss < 0.5 (key mechanism learned)
- ✅ All tables filled (no placeholders)
- ✅ All 4 domains complete
- ⚠️ PPO fixed OR discussion acknowledges limitation

### **NEED MORE WORK IF:**
- ⚠️ ACE total loss > 2.0 (worse than all baselines)
- ⚠️ Claims don't match results (clamping, regime selection)
- ⚠️ Major bugs in baselines

### **CANNOT SUBMIT IF:**
- ❌ No successful ACE run
- ❌ Tables still empty
- ❌ Critical claims have no evidence

---

## Current Status: ⚠️ **CANNOT SUBMIT YET**

**Blockers:**
1. ❌ No successful ACE run
2. ⚠️ PPO has bugs
3. ❌ All result tables empty

**Next Actions:**
1. 🔥 Test Jan 21 fixes (30 min) - **DO THIS NOW**
2. 🔥 Launch full ACE run (4-6 hours) - **IF TEST PASSES**
3. 🔥 Fill tables (2 hours) - **AFTER ACE COMPLETES**

**ETA to Submission-Ready:** 3-5 days (if fixes work)

---

**Last Updated:** January 21, 2026, 08:45 AM  
**Next Review:** After test run completes
