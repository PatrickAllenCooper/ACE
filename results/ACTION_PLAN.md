# Action Plan: Proving Remaining Claims
## Step-by-Step Guide to Paper-Ready Results

**Created:** January 21, 2026  
**Status:** ⚠️ Cannot submit - need fresh ACE run

---

## 🎯 Goal: Fill All Evidence Gaps

**Target:** Paper submission-ready in 3-5 days

---

## 🔥 PHASE 1: Critical Path (DO NOW)

### **Action 1.1: Test Jan 21 Fixes** ⏰ 30 minutes

**Why:** Validate fixes work before committing to long runs

```bash
# 1. Run quick test
./test_jan21_fixes.sh

# 2. Check results
tail results/test_jan21_fixes_*/experiment.log

# 3. Look for:
# ✅ Diversity scores > -10 (not -29)
# ✅ Zero rewards < 60% (not 99%)
# ✅ Gradient norms > 0.01 (not 0.0)
# ✅ Episode time < 5 min (not 13.5 min)
```

**Success Criteria:**
- No errors or crashes
- Diversity scores positive
- Some non-zero rewards
- Reasonable episode time

**IF PASS:** → Proceed to Action 1.2  
**IF FAIL:** → Debug issues, fix, retest

**Document in:** `results/RESULTS_LOG.md`

---

### **Action 1.2: Launch Full ACE Run** ⏰ 4-6 hours

**Why:** BLOCKING - Need this to fill Table 1 and all ACE claims

```bash
# 1. Launch on HPC
sbatch jobs/run_ace_main.sh

# 2. Monitor progress
tail -f logs/ace_*.err | grep -E "Episode.*Start|Final Loss|Early stop"

# 3. Track in real-time
# Watch for:
# - Episodes completing every 2-5 minutes (not 13.5 min)
# - Diversity scores healthy
# - Early stopping triggering around episode 40-80
```

**Success Criteria:**
- Completes successfully (doesn't hang/crash)
- Total loss < 2.0 (competitive with baselines)
- X3 loss < 0.5 (collider learned)
- Stops at 40-100 episodes (early stopping works)

**What This Fills:**
- ✅ Table 1 (lines 428-437)
- ✅ Figure 2 (learning curves)
- ✅ Figure 3 (intervention distribution)
- ✅ Lines 439, 485, 767 (inline numbers)
- ✅ Abstract claims about performance

**Document in:** `results/RESULTS_LOG.md` (add entry when complete)

---

### **Action 1.3: Extract ACE Numbers** ⏰ 1 hour

**Why:** Fill empty placeholders in paper

```bash
# 1. Run extraction script
./results/EXTRACTION_SCRIPTS.md # Script 2

# 2. Get key numbers:
# - Final total loss
# - Per-node losses (X1, X2, X3, X4, X5)
# - Episodes to convergence
# - Runtime
# - Intervention distribution

# 3. Update RESULTS_LOG.md
code results/RESULTS_LOG.md
# Add entry with all metrics

# 4. Generate comparison
python compare_methods.py > results/comparison.txt
```

**What to Record:**
- Every metric from the run
- How it compares to baselines
- Whether claims are supported or contradicted
- Any unexpected findings

**Output:** 
- `results/ace_summary.txt`
- `results/comparison.txt`
- Updated entry in `results/RESULTS_LOG.md`

---

### **Action 1.4: Fill Paper Table 1** ⏰ 30 minutes

**Why:** Remove placeholders, make paper submittable

```bash
# 1. Generate LaTeX table
python fill_tables.py

# 2. Copy to paper
code paper/paper.tex
# Replace lines 428-437 with generated table

# 3. Fill inline numbers
# Line 439: [TOTAL_MSE] → actual value
# Line 439: [X3_MSE] → actual value
# Line 485: [X2_PCT]% → actual value
# Line 767: [NUM_EPISODES] → actual value
```

**Success Criteria:**
- No more [PLACEHOLDER] brackets in Table 1
- All numbers match extraction scripts
- Numbers support claims (or claims revised)

**Document:** Note in `results/RESULTS_LOG.md` that Table 1 is filled

---

## ⚠️ PHASE 2: Fix Credibility Issues (PARALLEL)

### **Action 2.1: Fix PPO Baseline Bug** ⏰ 30 minutes

**Why:** Can't claim "DPO > PPO" with buggy PPO

```bash
# 1. Open baselines.py
code baselines.py

# 2. Find line 573 (value_loss calculation)
# Current (broken):
value_loss = nn.functional.mse_loss(new_values, batch_returns)

# Fix (ensure shapes match):
value_loss = nn.functional.mse_loss(new_values.squeeze(), batch_returns)

# 3. Save and commit
git add baselines.py
git commit -m "Fix PPO value loss shape mismatch"
```

**What Changed:**
- Added `.squeeze()` to match tensor shapes
- Should eliminate warning in logs

---

### **Action 2.2: Rerun PPO Baseline** ⏰ 2-3 hours

**Why:** Get clean PPO comparison

```bash
# 1. Run fixed PPO
python baselines.py --baseline ppo --episodes 100 \
    --output results/ppo_fixed_$(date +%Y%m%d_%H%M%S)

# 2. Monitor
tail -f logs/ppo_*.err

# 3. Extract results
grep "Final Total Loss:" results/ppo_fixed_*/experiment.log
```

**Success Criteria:**
- No shape mismatch warnings
- Completes successfully
- Comparable intervention distribution to other baselines

**What to Check:**
- Does PPO still perform poorly (2.18)?
- OR does fixing bug improve it?
- IF PPO better than ACE: Need to revise discussion
- IF PPO still worse: Claim validated

**Document in:** `results/RESULTS_LOG.md`

---

## 🔍 PHASE 3: Verify Behavioral Claims (CRITICAL)

### **Action 3.1: Check Clamping in Duffing** ⏰ 30 minutes

**Why:** Paper line 661 claims specific emergent behavior

```bash
# 1. Run verification script
./verify_claims.sh  # Checks clamping

# 2. Manual analysis
DUFFING_LOG="results/paper_*/duffing/*/experiment.log"
grep "DO X2 = " $DUFFING_LOG | awk -F '=' '{print $2}' > x2_values.txt

# 3. Analyze in Python
python << EOF
import numpy as np
vals = [float(x.strip()) for x in open('x2_values.txt')]
print(f"Mean: {np.mean(vals):.2f}")
print(f"Std: {np.std(vals):.2f}")
print(f"Near zero (<0.5): {sum(1 for v in vals if abs(v) < 0.5)/len(vals)*100:.1f}%")
EOF
```

**Decision Tree:**
- **IF mean ≈ 0 and std < 1.0:** ✅ Clamping detected, keep claim
- **IF mean far from 0 or std > 2.0:** ❌ No clamping, revise claim

**Revisions if needed:**
```latex
% Before (line 661):
ACE discovers a ``clamping'' strategy: by intervening to hold the 
middle oscillator fixed ($\text{do}(X_2 = 0)$), it breaks...

% After (if clamping didn't emerge):
ACE successfully identifies the chain coupling structure, achieving 
[STRUCTURE_F1] F1 score compared to [BASELINE_F1] for Max-Variance.
```

**Document in:** `results/claim_evidence/duffing_clamping.md`

---

### **Action 3.2: Check Regime Selection in Phillips** ⏰ 30 minutes

**Why:** Paper line 714 claims specific selection behavior

```bash
# 1. Analyze Phillips results
python << EOF
import pandas as pd
df = pd.read_csv('results/paper_*/phillips/*/results.csv')

# Map episodes to historical periods
# (Implementation depends on actual data structure)

# Calculate:
# - % queries to 1970s (high volatility)
# - % queries to 2008-2009 (high volatility)  
# - % queries to 1990s-2007 (low volatility)

# Compare to dataset composition
EOF
```

**Decision Tree:**
- **IF high-vol > 40%:** ✅ Selective, keep claim
- **IF high-vol ≈ 30%:** ⚠️ Somewhat selective, hedge claim
- **IF high-vol ≈ random:** ❌ Not selective, revise claim

**Revisions if needed:**
```latex
% Before (line 714):
ACE learns to query high-volatility regimes (1970s stagflation, 
2008 crisis), achieving...

% After (if not selective):
ACE learns from multiple economic regimes, achieving [OOS_MSE] 
out-of-sample MSE compared to [BASELINE_OOS]...
```

**Document in:** `results/claim_evidence/phillips_regimes.md`

---

## 📊 PHASE 4: Complete Remaining Domains

### **Action 4.1: Wait for Complex SCM** ⏰ varies (already running)

**Why:** Fills Table 2, supports scaling claim

```bash
# 1. Check status
ls results/paper_*/complex_scm/

# 2. When complete, extract
python << EOF
import pandas as pd

# Load all three policies
random = pd.read_csv('results/paper_*/complex_scm/complex_scm_random_*/results.csv')
smart = pd.read_csv('results/paper_*/complex_scm/complex_scm_smart_random_*/results.csv')
greedy = pd.read_csv('results/paper_*/complex_scm/complex_scm_greedy_collider_*/results.csv')

# Extract collider vs non-collider losses
# (Implementation depends on data structure)
EOF
```

**What to Extract:**
- Collider MSE (average of 5 colliders)
- Non-collider MSE
- Total MSE
- Compare: random < smart < greedy (expected)

**Fill:** Table 2 (lines 600-608)

**Document in:** `results/RESULTS_LOG.md`

---

### **Action 4.2: Calculate Duffing Structure F1** ⏰ 1 hour

**Why:** Fill placeholder in line 661

```python
# calculate_structure_f1.py

import pandas as pd

# True graph: Chain X1 - X2 - X3
true_edges = {(1, 2), (2, 3)}

# Extract learned graph from results
df = pd.read_csv('results/paper_*/duffing/*/results.csv')
# ... extract learned edges ...

# Calculate F1
TP = len(true_edges & learned_edges)
FP = len(learned_edges - true_edges)
FN = len(true_edges - learned_edges)

precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"Structure F1: {f1:.2f}")
```

**Fill:** Line 661 [STRUCTURE_F1]

**Document in:** `results/RESULTS_LOG.md`

---

## 📈 PHASE 5: Generate Figures (FINAL)

### **Action 5.1: Generate All Figures** ⏰ 1 hour

```bash
# 1. Learning curves (Figure 2)
python visualize.py results/paper_*/ace/
cp results/paper_*/ace/learning_curves.png results/figures/

# 2. Intervention distribution (Figure 3)
python visualize.py results/paper_*/ace/ --intervention-dist
cp results/paper_*/ace/intervention_distribution.png results/figures/

# 3. Ensure publication quality
# - 300 DPI minimum
# - Clear labels
# - Consistent color scheme
# - Proper legends
```

**Checklist:**
- [ ] Figure 2: Learning curves (line 441)
- [ ] Figure 3: Intervention distribution (line 487)
- [ ] Figure 4: Complex SCM structure (line 595)
- [ ] Figure 5: Duffing oscillators (line 647)
- [ ] Figure 6: Phillips curve (line 701)

**Save to:** `results/figures/` with descriptive names

---

## ✅ PHASE 6: Final Validation

### **Action 6.1: Complete RESULTS_LOG** ⏰ 1 hour

```bash
# Update results/RESULTS_LOG.md with:
# - Every completed experiment
# - All extracted metrics
# - Which claims are supported/contradicted
# - Summary dashboard (all ✅)
```

---

### **Action 6.2: Cross-Check Paper** ⏰ 1 hour

```bash
# 1. Search for placeholders
grep -n "\[.*\]" paper/paper.tex

# 2. For each placeholder:
# - Check if we have evidence in RESULTS_LOG.md
# - Fill with actual value OR remove claim

# 3. Verify all tables complete
# Lines: 428-437, 600-608, 652-660, 705-713, 718-727

# 4. Verify all claims match evidence
# Read through paper, cross-reference RESULTS_LOG.md
```

---

### **Action 6.3: Update Guidance Document** ⏰ 15 minutes

```bash
# Update guidance_documents/guidance_doc.txt
# Section: "Current Status"
# - Update "What's Working" with confirmed results
# - Update success criteria checklist
# - Note any claims that changed
```

---

## 📋 SUMMARY CHECKLIST

Before declaring "ready to submit":

### Critical (MUST HAVE):
- [ ] ACE run completed successfully
- [ ] Table 1 filled with real numbers
- [ ] All 4 domains complete (Synthetic, Complex, Duffing, Phillips)
- [ ] PPO baseline fixed and rerun
- [ ] All [PLACEHOLDER] removed from paper
- [ ] `results/RESULTS_LOG.md` has entry for every claim

### Important (SHOULD HAVE):
- [ ] Clamping in Duffing verified (or claim revised)
- [ ] Regime selection in Phillips verified (or claim revised)
- [ ] Structure F1 calculated for Duffing
- [ ] All 5 tables filled
- [ ] All 5 figures generated

### Nice to Have (OPTIONAL):
- [ ] Multiple ACE runs with error bars
- [ ] Statistical significance tests
- [ ] Confidence intervals in tables

---

## 🎯 DECISION POINTS

### After Action 1.2 (ACE Run):

**IF ACE total loss < 2.0:** ✅ Proceed to fill tables  
**IF ACE total loss > 2.0:** ⚠️ Need to:
- Debug why ACE underperforms
- Consider revising claims to "competitive" not "superior"
- Focus on collider identification (even if total loss higher)

### After Action 2.2 (PPO Rerun):

**IF PPO worse than ACE:** ✅ Claim validated  
**IF PPO better than ACE:** ⚠️ Need to:
- Revise "DPO outperforms PPO" claim
- Focus on "DPO more stable" angle
- Discuss non-stationarity issues with PPO

### After Actions 3.1, 3.2 (Behavioral Claims):

**IF both verified:** ✅ Strong paper  
**IF one fails:** ⚠️ Revise that specific claim  
**IF both fail:** ⚠️ Focus on quantitative results, not qualitative behaviors

---

## ⏰ TIMELINE

### Optimistic (3 days):
- **Day 1:** Actions 1.1-1.4, 2.1-2.2 (parallel)
- **Day 2:** Actions 3.1-3.2, 4.1-4.2
- **Day 3:** Actions 5.1, 6.1-6.3, SUBMIT

### Realistic (5 days):
- **Day 1:** Test, debug, launch ACE
- **Day 2:** ACE completes, extract, fill tables
- **Day 3:** Fix PPO, verify behaviors, handle issues
- **Day 4:** Complete domains, generate figures
- **Day 5:** Final validation, SUBMIT

---

## 🚦 CURRENT STATUS

**Can Submit:** ❌ NO  
**Blockers:** ACE run, PPO fix, empty tables  
**Next Action:** Action 1.1 (test fixes) - **DO NOW**  
**ETA:** 3-5 days

---

**Last Updated:** January 21, 2026, 08:45 AM  
**Next Review:** After test run completes (Action 1.1)
