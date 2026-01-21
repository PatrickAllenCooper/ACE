# Experiments Alignment Check
## Do All Experiments Support Paper Claims?

**Date:** January 21, 2026  
**Question:** Do experiments in `experiments/` need pipeline updates?

---

## 📊 Experiments Inventory

### **1. ACE Main** (`ace_experiments.py`)
**Purpose:** Core DPO-based causal discovery framework  
**Paper Sections:** 3.4.1 (Synthetic 5-node), Methods, Discussion  
**Status:** ✅ Updated with Jan 21 fixes  

**What it does:**
- LLM policy generates interventions
- DPO training on preference pairs
- Learns to design experiments through self-play

**Fixes Applied:**
- ✅ Adaptive diversity threshold
- ✅ Value novelty bonus
- ✅ Emergency retraining
- ✅ Speed improvements
- ✅ PPO baseline fix (in baselines.py)

---

### **2. Baselines** (`baselines.py`)
**Purpose:** Comparison methods (Random, Round-Robin, Max-Variance, PPO)  
**Paper Sections:** 3.8 (Baselines), Discussion  
**Status:** ✅ Updated (PPO bug fix)  

**What it does:**
- Random: Uniform sampling
- Round-Robin: Systematic cycling
- Max-Variance: Greedy uncertainty sampling
- PPO: Value-based RL baseline

**Fixes Applied:**
- ✅ PPO shape mismatch bug fixed

---

### **3. Complex 15-Node SCM** (`experiments/complex_scm.py`)
**Purpose:** Hard benchmark to show strategic advantage at scale  
**Paper Sections:** 3.4.2 (Complex SCM)  
**Status:** ⚠️ **NEEDS ALIGNMENT CHECK**  

**What it does:**
- 15 nodes, 5 colliders (vs 5 nodes, 1 collider)
- Tests 3 policies: random, smart_random, greedy_collider
- NO DPO/LLM - uses simple heuristic policies
- Has observational training every 5 steps

**Current Implementation:**
```python
# Line 285: Three simple policies
- random: Uniform sampling
- smart_random: 50% random, 50% collider-focused
- greedy_collider: Always target highest-loss collider parents
```

**Paper Claim (Line 609):**
> "strategic advantage becomes more pronounced at scale"

**ALIGNMENT ISSUES:**
1. ⚠️ These are NOT ACE policies - they're baselines/ablations
2. ⚠️ Paper may imply ACE runs on complex SCM (it doesn't)
3. ✅ Observational training present (line 366)
4. ✅ No DPO issues (no DPO used)
5. ⚠️ No verification of "strategic advantage" claim

**Needs:**
- ⏳ **Clarify in paper** that this tests policy strategies, not ACE itself
- ⏳ **Verify** greedy_collider actually outperforms random (currently running)
- ✅ No code fixes needed (works as designed)

---

### **4. Duffing Oscillators** (`experiments/duffing_oscillators.py`)
**Purpose:** Physics validation (ODE-based continuous dynamics)  
**Paper Sections:** 3.6 (Physics), Line 661 (clamping claim)  
**Status:** ⚠️ **CRITICAL ALIGNMENT ISSUE**  

**What it does:**
- Chain of coupled oscillators (X1 → X2 → X3 → X4)
- Random intervention policy
- Structure discovery via correlation breaking
- Completes in < 1 minute (very fast)

**Current Implementation:**
```python
# Line 172: Simple random policy
for step in range(steps_per_episode):
    target = np.random.choice(oracle.nodes)
    value = np.random.uniform(-2, 2)  # Intervention value
```

**Paper Claim (Line 661):**
> "ACE discovers a 'clamping' strategy: by intervening to hold the middle 
> oscillator fixed (do(X2 = 0))"

**🔴 CRITICAL ALIGNMENT ISSUE:**
1. ❌ **This uses RANDOM policy, not ACE**
2. ❌ **No learning/strategy discovery happening**
3. ❌ **"ACE discovers" is FALSE** - it's just random interventions
4. ❌ **Clamping would need to be verified, not assumed**

**Needs:**
- 🔥 **CRITICAL:** Either:
  - Option A: Run actual ACE on Duffing (add LLM policy support)
  - Option B: Revise paper claim to "random interventions enable structure discovery"
  - Option C: Verify clamping with `clamping_detector.py` and revise if not found
- ⚠️ **Paper misleads** by implying ACE runs on Duffing when it doesn't

---

### **5. Phillips Curve** (`experiments/phillips_curve.py`)
**Purpose:** Economics validation (real FRED data)  
**Paper Sections:** 3.7 (Economics), Line 714 (regime selection)  
**Status:** ⚠️ **ALIGNMENT ISSUE**  

**What it does:**
- Real economic data (UNRATE, FEDFUNDS, MICH → CPILFESL)
- Hardcoded regime selection strategy
- Retrospective learning across historical periods
- Completes in ~30 seconds

**Current Implementation:**
```python
# Line 222-233: HARDCODED regime selection
if step < 5:
    regime = None  # Mixed
elif step < 10:
    regime = "great_inflation"  # HIGH VOL
elif step < 15:
    regime = "great_moderation"  # LOW VOL
else:
    regime = regime_order[step % len(regime_order)]  # Cycle
```

**Paper Claim (Line 714):**
> "ACE learns to query high-volatility regimes"

**🔴 ALIGNMENT ISSUE:**
1. ⚠️ **This is HARDCODED, not learned by ACE**
2. ⚠️ **No actual learning happening** - it's a fixed script
3. ⚠️ **"ACE learns" is MISLEADING** - should say "systematically queries"
4. ✅ But it DOES query high-volatility regimes (by design)

**Needs:**
- ⚠️ **Revise paper claim** to match implementation:
  - Change: "ACE learns to query" 
  - To: "Systematic querying of high-volatility regimes"
- ⏳ **OR: Implement actual learning** (add regime selection policy)
- ✅ No bug fixes needed (works as designed)

---

## 🚨 CRITICAL FINDINGS

### **Paper vs Implementation Mismatch:**

| Experiment | Paper Claims | Actual Implementation | Alignment |
|------------|--------------|----------------------|-----------|
| **ACE Main** | DPO-based learning | ✅ DPO + LLM | ✅ ALIGNED |
| **Baselines** | 4 comparison methods | ✅ 4 methods | ✅ ALIGNED |
| **Complex SCM** | "strategic advantage" | ⚠️ Heuristic policies | ⚠️ CLARIFY |
| **Duffing** | "ACE discovers clamping" | ❌ Random policy | ❌ **MISALIGNED** |
| **Phillips** | "ACE learns regimes" | ❌ Hardcoded | ❌ **MISALIGNED** |

---

## ⚠️ ALIGNMENT ISSUES BY SEVERITY

### **🔴 CRITICAL (Must Fix Before Submission):**

#### **Issue 1: Duffing "ACE discovers" Claim**

**Paper (Line 661):**
```latex
ACE discovers a ``clamping'' strategy: by intervening to hold the 
middle oscillator fixed ($\text{do}(X_2 = 0)$), it breaks the 
correlation chain and reveals the true coupling structure.
```

**Reality:**
- Uses random policy (not ACE)
- No discovery/learning
- Clamping may or may not emerge from random sampling

**Fix Options:**
1. **Option A (Recommended):** Verify with detector, revise based on findings
   ```bash
   python clamping_detector.py
   # If no clamping: Revise to "random interventions enable structure discovery"
   ```

2. **Option B:** Implement actual ACE on Duffing (4-8 hours work)
   - Add LLM policy to duffing_oscillators.py
   - Train with DPO
   - Actually discover clamping strategy

3. **Option C:** Soften claim
   ```latex
   Interventions on the middle oscillator effectively decouple the system,
   enabling structure identification with [STRUCTURE_F1] F1 score.
   ```

---

#### **Issue 2: Phillips "ACE learns" Claim**

**Paper (Line 714):**
```latex
ACE learns to query high-volatility regimes (1970s stagflation, 
2008 crisis), achieving [OOS_MSE] out-of-sample MSE...
```

**Reality:**
- Uses hardcoded regime selection (lines 222-233)
- No learning algorithm
- Does query high-volatility but by design, not learning

**Fix Options:**
1. **Option A (Recommended):** Revise claim to match reality
   ```latex
   Systematic querying of high-volatility regimes (1970s stagflation,
   Great Recession) achieves [OOS_MSE] out-of-sample MSE...
   ```

2. **Option B:** Implement actual learning (8-12 hours work)
   - Add regime selection policy
   - Train to discover which regimes are most informative
   - Actually learn to query high-volatility

---

### **⚠️ MODERATE (Should Clarify):**

#### **Issue 3: Complex SCM "Strategic Advantage"**

**Paper (Line 609):**
```latex
The strategic advantage of ACE becomes more pronounced at scale
```

**Reality:**
- Tests 3 heuristic policies (random, smart_random, greedy_collider)
- NOT actual ACE with DPO/LLM
- Shows heuristics matter, but not that "ACE" matters

**Fix:**
```latex
The advantage of strategic intervention selection becomes more pronounced
at scale: while random sampling achieves [RANDOM_COLLIDER] MSE on 
colliders, greedy collider-focused sampling achieves [GREEDY_COLLIDER]...
```

**OR add footnote:**
```
Note: This experiment tests heuristic policies to establish that
strategic intervention matters. The main ACE framework uses learned
policies via DPO as described in Section 3.
```

---

## 📋 ALIGNMENT CHECKLIST

### **What Works As-Is:**

- ✅ Complex SCM: Tests policy strategies (fine for paper)
- ✅ Duffing: Structure discovery works (just needs claim revision)
- ✅ Phillips: Regime querying works (just needs claim revision)
- ✅ All have observational training where needed
- ✅ All complete successfully
- ✅ All generate useful data

### **What Needs Fixing:**

- ❌ **Paper Line 661:** "ACE discovers" → Should verify or revise
- ❌ **Paper Line 714:** "ACE learns" → Should say "systematic querying"
- ⚠️ **Paper Line 609:** "ACE becomes" → Should say "strategic intervention"

---

## 🔧 RECOMMENDED CHANGES

### **Option 1: Minimal Changes (2 hours) - RECOMMENDED**

**Just revise paper claims to match implementation:**

1. **Line 661 (Duffing):**
   ```latex
   % BEFORE:
   ACE discovers a ``clamping'' strategy:
   
   % AFTER:
   Interventions that decouple the middle oscillator
   ```

2. **Line 714 (Phillips):**
   ```latex
   % BEFORE:
   ACE learns to query high-volatility regimes
   
   % AFTER:
   Systematic querying of high-volatility regimes
   ```

3. **Line 609 (Complex SCM):**
   ```latex
   % BEFORE:
   The strategic advantage of ACE becomes more pronounced
   
   % AFTER:
   The advantage of strategic intervention selection becomes more pronounced
   ```

**Impact:**
- ✅ Paper accurate
- ✅ No code changes needed
- ✅ Can submit immediately (after ACE main run)

---

### **Option 2: Full Implementation (16-24 hours work)**

**Make experiments actually use ACE:**

1. **Add ACE to Duffing** (8 hours)
   - Implement LLM policy for Duffing
   - Train with DPO
   - Actually discover clamping

2. **Add ACE to Phillips** (8 hours)
   - Implement regime selection policy
   - Train to learn informative regimes
   - Actually learn rather than hardcode

3. **Add ACE to Complex SCM** (8 hours)
   - Implement LLM policy for 15-node system
   - Train with DPO
   - Compare learned vs heuristic

**Impact:**
- ✅ Paper claims accurate as written
- ❌ Delays submission by 1-2 weeks
- ❌ High complexity
- ❓ May not improve results (heuristics already work)

---

## 💡 RECOMMENDATION: **Option 1 (Minimal Changes)**

### **Why:**
1. Current experiments serve their purpose (validation across domains)
2. They demonstrate the PROBLEM and that interventions work
3. Main ACE experiment demonstrates the LEARNING/DPO approach
4. Paper just needs accurate descriptions of what each does
5. No bugs or failures - they work correctly

### **What to Change:**

**In Paper:**
- Line 353-354: Clarify "We apply ACE framework principles to physics..."
- Line 356-359: Clarify "We demonstrate regime selection on economics..."
- Line 661: Remove "ACE discovers", use "Decoupling interventions enable..."
- Line 714: Remove "ACE learns", use "Systematic querying of..."
- Line 609: Remove "ACE becomes", use "Strategic intervention becomes..."

**In Code:**
- ✅ Nothing (experiments work as designed)

---

## 🔍 VERIFICATION NEEDED

### **Run These Checks:**

1. **Verify clamping in Duffing:**
   ```bash
   python clamping_detector.py
   ```
   **If clamping detected:** Can soften claim to "interventions enable discovery"  
   **If NO clamping:** Must revise claim

2. **Verify regime selection in Phillips:**
   ```bash
   python regime_analyzer.py
   ```
   **Expected:** High-vol queries > random (it's hardcoded to do this)  
   **But:** Can't claim it was "learned"

3. **Verify strategic advantage in Complex SCM:**
   ```bash
   # After complex_scm completes, check:
   grep "FINAL RESULTS" results/paper_*/complex_scm/*/experiment.log
   
   # Verify: greedy_collider < smart_random < random
   ```

---

## 📊 CURRENT STATUS OF EXPERIMENTS

### **From logs copy/ analysis:**

| Experiment | Status | Runtime | Issues | Needs Work |
|------------|--------|---------|--------|------------|
| ACE Main | ⏳ Need rerun | 7h (31 ep) | Training failures | ✅ Fixed Jan 21 |
| Baselines | ✅ Complete | 22 min | PPO bug | ✅ Fixed Jan 21 |
| Complex SCM | 🔄 Running | ~4h each | None | ⏳ Wait for completion |
| Duffing | ✅ Complete | <1 min | **Claim mismatch** | ⚠️ Revise paper |
| Phillips | ✅ Complete | ~30 sec | **Claim mismatch** | ⚠️ Revise paper |

---

## 🎯 WHAT NEEDS TO ALIGN

### **Code Changes Needed:** ✅ **NONE**

All experiments work correctly. They just don't do exactly what the paper claims.

### **Paper Changes Needed:** ⚠️ **3 Claims to Revise**

1. Line 661: "ACE discovers clamping"
2. Line 714: "ACE learns regimes"
3. Line 609: "ACE becomes more pronounced"

### **Verification Needed:** ✅ **Tools Ready**

- `clamping_detector.py` - Check if clamping actually emerged
- `regime_analyzer.py` - Verify regime selection behavior
- Both ready to run after experiments complete

---

## 📋 ALIGNMENT ACTION PLAN

### **Immediate (Before Next Run):**

1. ✅ **No code changes needed in experiments/**
   - complex_scm.py works as designed
   - duffing_oscillators.py works as designed
   - phillips_curve.py works as designed

2. ⏳ **Wait for complex_scm to complete** (already running)

3. ⏳ **Verify behavioral claims** after runs:
   ```bash
   python clamping_detector.py  # Duffing
   python regime_analyzer.py     # Phillips
   ```

### **After Verification (Update Paper):**

4. ⏳ **Revise paper claims based on verifier output:**
   - If clamping NOT found: Revise line 661
   - If regimes not selective: Revise line 714 (but they should be - hardcoded)
   - Clarify line 609 about strategic policies vs ACE specifically

---

## 💯 FINAL ASSESSMENT

### **Do experiments need pipeline updates?**

**Answer:** ✅ **NO** - They work correctly as validation experiments

**But:** ⚠️ **Paper needs minor revisions** to accurately describe what they do

---

### **Current Experiment Purposes:**

| Experiment | Current Purpose | Paper Position |
|------------|----------------|----------------|
| **ACE Main** | Demonstrates DPO learning | ✅ Correctly described |
| **Baselines** | Comparison methods | ✅ Correctly described |
| **Complex SCM** | Shows strategic policies matter | ⚠️ Needs clarification |
| **Duffing** | Physics domain validation | ❌ Over-claims learning |
| **Phillips** | Economics domain validation | ❌ Over-claims learning |

---

### **Alignment Path Forward:**

**DO THIS:**
1. ✅ No code changes in experiments/ (they work)
2. ✅ Keep pipeline fixes in ACE main and baselines
3. ⏳ Run verifiers after experiments complete
4. ⏳ Revise 3 paper claims based on verification
5. ⏳ Add clarifying notes about what each experiment tests

**DON'T DO THIS:**
- ❌ Re-implement Duffing/Phillips with ACE (too much work, low value)
- ❌ Change complex_scm (it serves its purpose)
- ❌ Add unnecessary complexity

---

## 🚀 NEXT STEPS

### **For Code/Experiments:**
```bash
# 1. Test ACE main fixes
./pipeline_test.sh

# 2. Run ACE main
sbatch jobs/run_ace_main.sh

# 3. Wait for complex_scm to finish (already running)

# No changes needed to experiments/ directory
```

### **For Paper:**
```bash
# After experiments complete:

# 1. Verify claims
python clamping_detector.py
python regime_analyzer.py

# 2. Update paper based on findings
code paper/paper.tex
# Revise lines 609, 661, 714 to match reality
```

---

## ✅ CONCLUSION

**Alignment Status:** ⚠️ **MOSTLY ALIGNED** 

**Code:** ✅ All experiments work correctly  
**Paper:** ⚠️ 3 claims over-state what experiments do  

**Action Required:**
- ✅ No code changes in experiments/
- ⚠️ Minor paper revisions (3 claims)
- ✅ Verification tools ready

**Bottom Line:**
The experiments are fine. The paper just needs to accurately describe them.

---

**Last Updated:** January 21, 2026, 09:45 AM
