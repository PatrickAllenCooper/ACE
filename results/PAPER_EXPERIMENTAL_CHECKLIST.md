# Paper Section → Experimental Requirements Checklist
## What We Need to Fill Each Section

---

## 📊 RESULTS SECTION REQUIREMENTS

### **Table 1: Synthetic 5-Node Results** (Lines 428-437)

| Column | Source | Status |
|--------|--------|--------|
| **Random** | ✅ `logs copy/baselines_*.out` | Complete |
| **Round-Robin** | ✅ `logs copy/baselines_*.out` | Complete |
| **Max-Variance** | ✅ `logs copy/baselines_*.out` | Complete |
| **PPO** | ⚠️ `logs copy/baselines_*.out` | Bug - needs rerun |
| **ACE (ours)** | ❌ Need fresh ACE run | **MISSING** |

**How to Extract:**
```bash
# Baselines (already have)
grep "Final Total Loss:" logs copy/baselines_20260120_142711_23026272.out

# ACE (need new run)
grep "Final mechanism losses:" results/paper_YYYYMMDD_HHMMSS/ace/experiment.log
```

**What We Need:**
- Per-node losses: X1, X2, X3, X4, X5
- Total MSE
- Number of episodes to convergence

---

### **Table 2: Complex 15-Node Results** (Lines 600-608)

| Metric | Source | Status |
|--------|--------|--------|
| **Random** | `logs copy/complex_scm_*_random_*.out` | ✅ Complete |
| **Smart Random** | `logs copy/complex_scm_*_smart_random_*.out` | ✅ Complete |
| **Greedy Collider** | `logs copy/complex_scm_*_greedy_collider_*.out` | ⏳ Running |

**How to Extract:**
```bash
# Check if complete
ls results/paper_20260120_142711/complex_scm/

# Extract losses
grep "Total Loss\|Collider Loss" results/paper_*/complex_scm/*/results.csv
```

**What We Need:**
- Collider MSE (average across 5 colliders)
- Non-collider MSE
- Total MSE
- Episodes used

---

### **Table 3: Duffing Oscillators** (Lines 652-660)

| Metric | Source | Status |
|--------|--------|--------|
| **Mechanism Recovery** | `logs copy/duffing_*.out` | ✅ Complete |

**How to Extract:**
```bash
grep "Episode.*Loss" logs copy/duffing_20260120_142711_23026274.err

# Check results file
cat results/paper_20260120_142711/duffing/duffing_20260121_064331/results.csv
```

**What We Need:**
- Learned graph structure vs true structure
- Structure identification F1 score
- Coupling parameter accuracy
- Damping parameter accuracy

**Special Validation:**
- **CRITICAL:** Verify if "clamping strategy" actually emerged (paper line 661)
- Check if ACE learned to set DO(X2=0) to decouple oscillators
- This is a specific behavioral claim that needs verification

---

### **Table 4: Phillips Curve** (Lines 705-713)

| Metric | Source | Status |
|--------|--------|--------|
| **Regime Selection** | `logs copy/phillips_*.out` | ✅ Complete |

**How to Extract:**
```bash
grep "Episode.*Eval Loss" logs copy/phillips_20260120_142711_23026275.err

# Check results
cat results/paper_20260120_142711/phillips/phillips_20260121_070621/results.csv
```

**What We Need:**
- In-sample MSE
- Out-of-sample MSE
- Regime coverage statistics
- % queries to pre-1985 data vs total available
- % queries to high-volatility periods

**Special Validation:**
- **CRITICAL:** Verify if ACE actually selected high-volatility regimes (paper line 714)
- Not just claimed behavior, but actual observed behavior

---

### **Table 5: Summary Across Domains** (Lines 718-727)

| Domain | ACE Final Loss | Best Baseline | % Improvement | Episodes |
|--------|---------------|---------------|---------------|----------|
| Synthetic 5-node | ❌ Need | ✅ Have | Calculate | From Table 1 |
| Complex 15-node | ⏳ Running | ✅ Have | Calculate | From Table 2 |
| Duffing | ✅ Have | ✅ Have | Calculate | From Table 3 |
| Phillips | ✅ Have | ✅ Have | Calculate | From Table 4 |

**Calculation:**
```python
improvement = (baseline_loss - ace_loss) / baseline_loss * 100
episode_reduction = (baseline_episodes - ace_episodes) / baseline_episodes * 100
```

---

## 📈 FIGURE REQUIREMENTS

### **Figure 2: Learning Curves** (Line 441)

**Need:**
- ACE learning curve (total loss vs episode)
- All 4 baseline curves on same plot
- Annotate ACE early stopping point

**Source:**
```bash
# ACE
results/paper_*/ace/training_curves.png

# Baselines  
results/paper_*/baselines/baselines_*/learning_curves_*.png
```

**Status:** ❌ Need fresh ACE run

---

### **Figure 3: Intervention Distribution** (Line 487)

**Need:**
- Bar chart: % interventions per node
- ACE vs Random comparison
- Highlight X1, X2 (collider parents)

**How to Generate:**
```python
import pandas as pd
df = pd.read_csv('results/paper_*/ace/metrics.csv')
distribution = df['target'].value_counts(normalize=True) * 100
```

**Status:** ❌ Need fresh ACE run

---

## 🔢 INLINE NUMBERS TO FILL

### Abstract (Lines 112-113)
- [ ] "competitive performance" → Need ACE total loss
- [ ] "superior collider identification" → Need X3 comparison
- [ ] "80% computational savings" → Verify episode count

### Results - Synthetic Benchmark (Line 439)
- [ ] [TOTAL_MSE] = ?
- [ ] [NUM_EPISODES] = ?
- [ ] [BASELINE_MSE] = ? (Max-Variance)
- [ ] [SPEEDUP] = ?
- [ ] [X3_MSE] = ? (ACE)
- [ ] [X3_BASELINE] = ? (best baseline)

### Results - Synthetic Distribution (Line 485)
- [ ] [X2_PCT] = ? (ACE concentration on X2)
- [ ] [X1_PCT] = ? (ACE concentration on X1)
- [ ] [RANDOM_UNIFORM] = 20% (by definition)

### Results - Complex SCM (Line 609)
- [ ] [RANDOM_COLLIDER] = ?
- [ ] [ACE_COLLIDER] = ?
- [ ] [IMPROVEMENT] = ?
- [ ] [ACE_NONCOLLIDER] = ?
- [ ] [RANDOM_NONCOLLIDER] = ?

### Results - Duffing (Line 661)
- [ ] [STRUCTURE_F1] = ? (ACE)
- [ ] [BASELINE_F1] = ? (Max-Variance)

### Results - Phillips (Line 714)
- [ ] [OOS_MSE] = ? (ACE)
- [ ] [BASELINE_OOS] = ? (chronological)
- [ ] [HIGH_VOL_PCT] = ? (% queries to high-vol periods)
- [ ] [PRE1985_PCT] = ? (% of dataset that's pre-1985)

### Results - Summary (Line 728)
- [ ] [AVG_REDUCTION] = ? (average episode reduction across domains)

### Conclusion (Line 767)
- [ ] Verify "40-60 episodes" claim
- [ ] Verify "80% reduction" claim

---

## ✅ VALIDATION CHECKLIST

### **Critical Behavioral Claims to Verify:**

#### 1. Clamping Strategy (Line 661)
- [ ] Parse intervention logs for Duffing experiment
- [ ] Check if X2 interventions cluster near 0
- [ ] Verify this strategy emerged vs was programmed
- [ ] If not emerged, revise claim

#### 2. Regime Selection (Line 714)
- [ ] Extract Phillips curve intervention history
- [ ] Map interventions to historical periods
- [ ] Calculate % high-volatility vs low-volatility
- [ ] Calculate % pre-1985 vs post-1985
- [ ] If not selective, revise claim

#### 3. Superior Collider ID (Line 439)
- [ ] Extract X3 loss: ACE vs all baselines
- [ ] Statistical test (if multiple runs)
- [ ] Verify ACE is significantly better
- [ ] If not, revise to "competitive"

#### 4. Strategic Concentration (Line 485)
- [ ] Extract intervention distribution from ACE
- [ ] Verify X1 + X2 > 50% of interventions
- [ ] Compare to 20% uniform (Random)
- [ ] Check concentration is strategic, not collapse

#### 5. Early Stopping (Line 767)
- [ ] Verify ACE stops at 40-60 episodes
- [ ] Not 31 (too early) or 200 (no stopping)
- [ ] Check per-node convergence triggered
- [ ] If different, update numbers

#### 6. 80% Speedup (Line 767)
- [ ] ACE runtime vs baseline runtime
- [ ] ACE episodes vs baseline episodes
- [ ] Calculate actual % reduction
- [ ] If not 80%, update claim

---

## 🔬 DATA EXTRACTION SCRIPTS

### **Script 1: Extract All Baseline Results**
```bash
#!/bin/bash
# extract_baselines.sh

LOGS_DIR="logs copy"
OUTPUT="baseline_results.txt"

echo "=== BASELINE RESULTS ===" > $OUTPUT

for baseline in random round_robin max_variance ppo; do
    echo "" >> $OUTPUT
    echo "--- $baseline ---" >> $OUTPUT
    grep -A 10 "BASELINE.*$baseline" $LOGS_DIR/baselines_*.out | \
        grep -E "Final Total Loss:|X[1-5]:" >> $OUTPUT
done
```

### **Script 2: Extract ACE Final Results**
```bash
#!/bin/bash
# extract_ace.sh

RESULTS_DIR="results/paper_20260120_142711/ace"
OUTPUT="ace_results.txt"

echo "=== ACE RESULTS ===" > $OUTPUT

# Final losses
echo "Final Losses:" >> $OUTPUT
grep "Final mechanism losses:" $RESULTS_DIR/experiment.log >> $OUTPUT

# Intervention distribution
echo "" >> $OUTPUT
echo "Intervention Distribution:" >> $OUTPUT
python3 << EOF
import pandas as pd
df = pd.read_csv('$RESULTS_DIR/metrics.csv')
dist = df['target'].value_counts(normalize=True) * 100
print(dist)
EOF
```

### **Script 3: Verify Specific Claims**
```bash
#!/bin/bash
# verify_claims.sh

echo "=== CLAIM VERIFICATION ==="

# 1. Clamping in Duffing
echo "1. Checking Duffing clamping..."
grep "DO X2 = " results/paper_*/duffing/*/experiment.log | \
    awk -F '=' '{print $2}' | \
    python3 -c "import sys; vals=[float(x.strip()) for x in sys.stdin]; print(f'X2 mean: {sum(vals)/len(vals):.2f}, std: {(sum((x-sum(vals)/len(vals))**2 for x in vals)/len(vals))**0.5:.2f}')"

# 2. Regime selection in Phillips  
echo "2. Checking Phillips regime selection..."
python3 << EOF
import pandas as pd
df = pd.read_csv('results/paper_*/phillips/*/results.csv')
# Map episodes to historical periods and calculate distribution
# (Need to implement based on actual data structure)
EOF

echo "Done"
```

---

## 📝 PAPER WRITING CHECKLIST

### **Before Submission:**

- [ ] All tables filled (no placeholders)
- [ ] All figures generated and referenced
- [ ] All inline numbers filled (no [BRACKETS])
- [ ] All claims verified against actual data
- [ ] PPO baseline bug fixed and rerun
- [ ] Fresh ACE run completed with Jan 21 fixes
- [ ] Statistical significance computed (if multiple runs)
- [ ] Limitations section acknowledges single-run (line 773)

### **Cross-Checks:**

- [ ] Abstract claims match results section
- [ ] Conclusion claims match experiments
- [ ] Discussion interpretation aligns with data
- [ ] No over-claiming (aspirational vs actual)
- [ ] Error bars / confidence intervals (if applicable)

---

## 🚦 GO/NO-GO DECISION MATRIX

### **Can Submit if:**
- ✅ All tables filled with real numbers
- ✅ ACE outperforms or matches at least 2/4 baselines
- ✅ X3 collider loss < 0.5 (paper threshold)
- ✅ All 4 domains complete
- ✅ Specific claims (clamping, regime selection) verified

### **Need More Work if:**
- ⚠️ ACE worse than all baselines → Need to debug/retrain
- ⚠️ X3 loss > 1.0 → Collider not learned, undermines key claim
- ⚠️ Key claims unverifiable → Need to revise or remove
- ⚠️ PPO significantly better than ACE → Undermines DPO advantage claim

### **Cannot Submit if:**
- ❌ Tables still have placeholders
- ❌ ACE run doesn't complete successfully
- ❌ Major claims contradict data
- ❌ Critical bugs in baselines

---

## 🎯 PRIORITY ORDER

1. **CRITICAL** 🔥 - Must have before submission
   - [ ] Table 1 filled (Synthetic 5-node)
   - [ ] ACE total loss competitive with baselines
   - [ ] X3 collider loss < 0.5

2. **HIGH** ⚠️ - Important for paper quality
   - [ ] All 5 tables filled
   - [ ] PPO baseline fixed
   - [ ] Specific claims verified

3. **MEDIUM** 📊 - Strengthens paper
   - [ ] All figures generated
   - [ ] Statistical validation (multiple runs)
   - [ ] Confidence intervals

4. **LOW** ✨ - Nice to have
   - [ ] Additional ablations
   - [ ] Hyperparameter sensitivity
   - [ ] Scaling analysis

---

**Last Updated:** January 21, 2026  
**Status:** ⏳ Awaiting Jan 21 fix validation  
**Next Action:** Run test, then extract numbers
