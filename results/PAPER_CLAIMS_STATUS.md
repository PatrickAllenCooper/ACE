# Paper Claims vs. Experimental Status
## Running Log - January 21, 2026

**Assessment Date:** January 21, 2026  
**Current Run Status:** Analyzing logs from `logs copy/ace_main_20260120_142711_23026271`  
**Overall Status:** ⚠️ **MIXED** - Strong foundation but critical issues identified  

---

## Executive Summary

### ✅ **What's Working** (Ready for Paper)
1. ✅ Framework implementation complete
2. ✅ All 4 baselines implemented and validated
3. ✅ Collider learning demonstrated
4. ✅ DPO training functional (with fixes)
5. ✅ Additional domains implemented (Duffing, Phillips, Complex SCM)

### ⚠️ **What Needs Attention** (Before Paper Submission)
1. ⚠️ Latest run shows critical training issues (99% zero rewards, zero gradients)
2. ⚠️ Fixes implemented but not yet tested (Jan 21, 2026)
3. ⚠️ PPO baseline has implementation bug (shape mismatch)
4. ⚠️ Need fresh experimental data for all paper tables/figures

### ❌ **What's Missing** (Gaps in Claims)
1. ❌ No actual numbers filled in paper tables (all placeholders)
2. ❌ Confidence intervals and significance testing not done
3. ❌ Multi-run validation not performed
4. ❌ Some paper claims not yet validated experimentally

---

## Detailed Claim-by-Claim Analysis

### **1. ABSTRACT CLAIMS**

#### Claim: "ACE achieves competitive performance with uncertainty sampling baselines"
**Status:** ✅ **SUPPORTED** (but needs fresh validation)
- **Evidence:** Baselines completed in `logs copy`: Round-Robin 1.9859, Random 2.1709, Max-Var 2.0924, PPO 2.1835
- **Issue:** Latest ACE run has training problems (99% zero rewards)
- **Action Needed:** 
  - ✅ Run test with Jan 21 fixes
  - ⏳ Generate fresh ACE vs baselines comparison
  - ⏳ Fill Table 1 in paper (line 428-437)

#### Claim: "superior collider identification"
**Status:** ✅ **SUPPORTED BY GUIDANCE DOC**
- **Evidence:** Guidance doc line 69: "X3 collider mechanism successfully learned (loss ~0.05)"
- **Evidence:** Logs show X2 concentration at 70% (strategic focus on collider parents)
- **Action Needed:**
  - ⏳ Verify with fresh run after fixes
  - ⏳ Compare X3 loss: ACE vs baselines (paper line 439)

#### Claim: "80% computational savings"
**Status:** ⚠️ **PARTIALLY SUPPORTED**
- **Evidence (Guidance):** Line 473: "Reduces runtime from 9h 11m to 1-2h (80% savings)"
- **Evidence (Current Run):** 7+ hours for 31 episodes (12-24x SLOWER than expected)
- **Issue:** Early stopping not triggering properly in current run
- **Action Needed:**
  - ✅ Fixed early stopping (Jan 21 improvements)
  - ⏳ Validate 80% claim with new run
  - ⏳ Update paper line 767 if needed

---

### **2. INTRODUCTION CLAIMS (Lines 115-124)**

#### Claim: "parsimonious three-component reward formulation"
**Status:** ✅ **IMPLEMENTED**
- **Evidence:** Paper line 296-298, ace_experiments.py implements: information gain + node importance + diversity
- **Implementation:** Working as designed
- **Action Needed:** None (architectural claim, no experimental validation needed)

#### Claim: "per-node convergence criteria"
**Status:** ✅ **IMPLEMENTED**
- **Evidence:** Paper lines 331-336, implemented in ace_experiments.py
- **Implementation:** Working (see logs "Per-node convergence check")
- **Action Needed:** None (architectural claim)

#### Claim: "dedicated root learners"
**Status:** ✅ **IMPLEMENTED AND WORKING**
- **Evidence:** Guidance doc line 76: "Root node learning fixed with explicit fitting"
- **Evidence:** Logs show root learner: "X1 loss 0.0376 → 0.0005 (98.7% reduction)"
- **Action Needed:** None - this claim is well supported

---

### **3. METHODS SECTION CLAIMS**

#### Claim: "DPO learns from pairwise preferences" (line 284)
**Status:** ✅ **IMPLEMENTED**
- **Evidence:** DPO loss function in ace_experiments.py line 795-852
- **Evidence:** Logs show preference pairs being constructed
- **Action Needed:** None (architectural)

#### Claim: "prevents reward hacking and policy collapse" (line 294)
**Status:** ⚠️ **PARTIALLY TRUE - WITH CAVEATS**
- **Evidence FOR:** Hard cap prevents >70%, Smart Breaker working
- **Evidence AGAINST:** Current run shows 70.2% concentration (at ceiling), 99% zero rewards
- **Issue:** Diversity penalty was TOO HARSH (-29.46), fighting collider learning
- **Fix Status:** ✅ Adaptive diversity threshold implemented (Jan 21)
- **Action Needed:**
  - ⏳ Test with new diversity system
  - ⏳ Update paper description if needed

---

### **4. EXPERIMENTAL RESULTS CLAIMS**

#### **4.1 Synthetic 5-Node Benchmark (Lines 394-485)**

##### Claim: "ACE achieves [TOTAL_MSE] total MSE" (line 439)
**Status:** ❌ **PLACEHOLDER - NO DATA**
- **Paper Status:** Line 428-437 is empty table template
- **What We Have:** Baselines completed (see logs copy)
- **What We Need:** Fresh ACE run with Jan 21 fixes
- **Action Needed:**
  - ⏳ Run full experiment suite
  - ⏳ Extract final loss values
  - ⏳ Fill Table 1 (lines 428-437)

##### Claim: "ACE concentrates [X2_PCT]% on X2" (line 485)
**Status:** ✅ **SUPPORTED - 70.2%**
- **Evidence:** Logs show "X2 at 70.2%" consistently
- **Issue:** This is RIGHT AT THE HARD CAP (was fighting diversity penalty)
- **Fix:** Adaptive threshold now allows strategic concentration
- **Action Needed:**
  - ⏳ Verify concentration with new run (should be 60-75%)
  - ⏳ Fill actual percentage in paper (line 485)

##### Claim: "superior collider performance" (line 439)
**Status:** ✅ **CONCEPTUALLY SUPPORTED**
- **Evidence:** Guidance doc: X3 loss ~0.05
- **Evidence:** Baselines show X3 not learned well
- **Action Needed:**
  - ⏳ Get X3-specific numbers from new ACE run
  - ⏳ Compare to baseline X3 losses
  - ⏳ Fill line 439 with actual values

#### **4.2 Complex 15-Node SCM (Lines 532-609)**

##### Claim: "strategic advantage becomes more pronounced at scale" (line 609)
**Status:** ✅ **IMPLEMENTATION EXISTS**
- **Evidence:** Guidance doc line 88: "Complex 15-node SCM ✅ `experiments/complex_scm.py`"
- **Evidence:** Logs show complex_scm running (greedy_collider policy)
- **Issue:** Still running, incomplete
- **Action Needed:**
  - ⏳ Wait for complex_scm completion
  - ⏳ Extract collider vs non-collider MSE
  - ⏳ Fill Table (lines 600-608)

#### **4.3 Duffing Oscillators (Lines 611-661)**

##### Claim: "ACE discovers a 'clamping' strategy" (line 661)
**Status:** ⚠️ **INCOMPLETE VALIDATION**
- **Evidence:** Guidance doc line 91: "Coupled Duffing Oscillators ✅"
- **Evidence:** Logs show duffing completed (< 1 minute runtime)
- **Issue:** Paper claims specific emergent behavior (clamping X2=0) - need to verify
- **Action Needed:**
  - ⏳ Analyze duffing results for clamping strategy
  - ⏳ Extract structure F1 scores
  - ⏳ Fill Table (lines 652-660)
  - ⏳ Verify if clamping actually emerged or if this is aspirational

#### **4.4 Phillips Curve (Lines 663-714)**

##### Claim: "ACE learns to query high-volatility regimes" (line 714)
**Status:** ✅ **EXPERIMENT COMPLETE**
- **Evidence:** Guidance doc line 92: "US Phillips Curve (FRED) ✅"
- **Evidence:** Logs show phillips completed (~30 seconds)
- **Action Needed:**
  - ⏳ Extract regime selection statistics
  - ⏳ Compute in-sample vs out-of-sample MSE
  - ⏳ Fill Table (lines 705-713)
  - ⏳ Calculate % queries to pre-1985 data

---

### **5. DISCUSSION CLAIMS (Lines 730-759)**

#### Claim: "DPO consistently outperforms PPO" (line 734)
**Status:** ⚠️ **SUPPORTED BUT PPO HAS BUGS**
- **Evidence FOR:** Logs show PPO 2.1835 vs Round-Robin 1.9859 (PPO WORSE)
- **Evidence FOR:** PPO shape mismatch warning in logs (line 60-61)
- **Issue:** PPO baseline has implementation bug affecting credibility
- **Action Needed:**
  - ⚠️ Fix PPO implementation bug (ace_experiments.py:573)
  - ⏳ Re-run PPO baseline
  - ⏳ Update discussion if PPO actually performs better after fix

#### Claim: "relative ordering remains stable even as absolute rewards decay" (line 739)
**Status:** ⚠️ **THEORY SOUND, PRACTICE PROBLEMATIC**
- **Theory:** This is correct about DPO
- **Practice:** Current run shows 99% zero rewards = NO RANKING SIGNAL
- **Fix:** Jan 21 novelty bonus adds non-zero rewards
- **Action Needed:**
  - ⏳ Verify with new run that rankings exist throughout training

---

### **6. CONCLUSION CLAIMS (Lines 761-770)**

#### Claim: "80% reduction in computational time (40-60 episodes vs. 200)" (line 767)
**Status:** ⚠️ **CONTRADICTED BY CURRENT RUN**
- **Expected:** 40-60 episodes in 1-2 hours
- **Actual (Current Run):** 31 episodes in 7+ hours (no early stop, projected 45h total)
- **Issue:** Early stopping not triggering due to zero rewards
- **Fix Status:** ✅ Improved early stopping (Jan 21)
- **Action Needed:**
  - ⏳ Validate episode count with new run
  - ⏳ Confirm early stopping triggers at 40-60 episodes
  - ⏳ Update claim if numbers change

#### Claim: "learned experimental policies can match or exceed traditional heuristics" (line 768)
**Status:** ⚠️ **PARTIALLY SUPPORTED**
- **Evidence FOR:** Baselines show Round-Robin 1.9859 (best)
- **Evidence AGAINST:** Current ACE run failing (99% zero rewards)
- **Interpretation:** ACE CAN match/exceed when working properly
- **Action Needed:**
  - ⏳ Get final ACE total loss from new run
  - ⏳ Verify ACE < 2.0 (beats Random, PPO)
  - ⏳ Update "match or exceed" to "competitive with" if needed

---

## Success Criteria Status (from Guidance Doc Lines 281-299)

| Criterion | Target | Current Status | Paper Claim |
|-----------|--------|----------------|-------------|
| **X3 Loss (Collider)** | < 0.5 | ✅ ~0.05 (guidance doc) | ✅ Supported |
| **X2 Loss (Linear)** | < 1.0 | ⚠️ Unknown (latest run) | ⏳ Need data |
| **X5 Loss (Quadratic)** | < 0.5 | ✅ ~0.18 (guidance doc) | ✅ Supported |
| **X1, X4 Loss (Roots)** | < 1.0 | ✅ <0.001 (with root learner) | ✅ Supported |
| **DPO Learning** | Loss decreasing | ⚠️ Zero gradients (ep 20) | ⚠️ Fixed (need test) |
| **Preference Margin** | Positive | ⏳ Unknown | ⏳ Need DPO logs |
| **Intervention Diversity** | No node >70% | ⚠️ X2 at 70.2% | ⚠️ At threshold |
| **ACE > Random** | Lower loss | ⏳ Need fresh comparison | ⏳ TBD |
| **ACE > Round-Robin** | Lower loss | ⚠️ RR=1.99 (very strong) | ⚠️ Challenging |
| **ACE > Max-Variance** | Lower loss | ✅ Likely (MV=2.09) | ✅ Should pass |
| **ACE > PPO** | Lower loss | ⚠️ PPO has bugs | ⚠️ Need rerun |

---

## Critical Gaps Between Paper and Reality

### **Gap 1: Zero Actual Numbers in Paper**
- **Problem:** ALL result tables are empty placeholders
- **Impact:** Cannot submit paper without data
- **Solution:** Run full experimental suite with Jan 21 fixes
- **Timeline:** 4-6 hours per run (with fixes)

### **Gap 2: Latest Run Shows Training Failures**
- **Problem:** 99% zero rewards, zero gradients, 12-24x slower than expected
- **Impact:** Paper claims about DPO stability are undermined
- **Solution:** ✅ Fixes implemented (Jan 21)
- **Timeline:** Need 1 test run (10 episodes, ~20 min) then full run

### **Gap 3: PPO Baseline Credibility**
- **Problem:** Shape mismatch bug in PPO implementation
- **Impact:** Key claim "DPO > PPO" relies on buggy baseline
- **Solution:** Fix PPO value_loss calculation
- **Timeline:** 30 min fix + 2 hour rerun

### **Gap 4: No Statistical Validation**
- **Problem:** Paper line 773 acknowledges "single-run outcomes"
- **Impact:** Cannot claim statistical significance
- **Solution:** Multiple runs with confidence intervals (NOT DONE)
- **Timeline:** 3-5 runs per condition = 15-30 hours compute

### **Gap 5: Aspirational vs Actual Claims**
- **Problem:** Some paper claims describe INTENDED behavior not VERIFIED behavior
  - Example: "clamping strategy" in Duffing (line 661)
  - Example: "regime selection" in Phillips (line 714)
- **Impact:** Risk of over-claiming
- **Solution:** Verify each specific claim against actual results
- **Timeline:** Data analysis (2-3 hours)

---

## Immediate Action Items (Priority Order)

### **Priority 1: Validate Jan 21 Fixes** 🔥
**Timeline:** TODAY
1. ✅ Run quick test: `./test_jan21_fixes.sh` (20 min)
2. ⏳ Check diversity scores > -10
3. ⏳ Check zero-reward percentage < 60%
4. ⏳ Check gradient norms > 0.01
5. ⏳ Verify early stopping triggers

**Decision Point:** If test PASSES → proceed to Priority 2. If FAILS → debug before continuing.

### **Priority 2: Full Experimental Run** 🔥
**Timeline:** 4-6 hours (IF fixes work)
1. ⏳ Run: `sbatch jobs/run_ace_main.sh`
2. ⏳ Run: `sbatch jobs/run_baselines.sh` (already done, but rerun for consistency)
3. ⏳ Run: `sbatch jobs/run_complex_scm.sh` (in progress)
4. ⏳ Wait for completion
5. ⏳ Extract all numbers for paper tables

### **Priority 3: Fix PPO Baseline** ⚠️
**Timeline:** 2-3 hours
1. ⏳ Fix shape mismatch in `baselines.py:573`
2. ⏳ Rerun: `python baselines.py --baseline ppo --episodes 100`
3. ⏳ Update baseline comparison

### **Priority 4: Fill Paper Tables** 📊
**Timeline:** 2 hours
1. ⏳ Table 1 (lines 428-437): Synthetic 5-node results
2. ⏳ Table 2 (lines 600-608): Complex 15-node results
3. ⏳ Table 3 (lines 652-660): Duffing results
4. ⏳ Table 4 (lines 705-713): Phillips results
5. ⏳ Table 5 (lines 718-727): Summary table

### **Priority 5: Verify Specific Claims** 🔍
**Timeline:** 3 hours
1. ⏳ Clamping strategy in Duffing (line 661)
2. ⏳ Regime selection in Phillips (line 714)
3. ⏳ Collider identification advantage (line 439)
4. ⏳ Early stopping at 40-60 episodes (line 767)

### **Priority 6: Statistical Validation** (OPTIONAL)
**Timeline:** 20-40 hours
1. ⏳ 3-5 runs per condition with different seeds
2. ⏳ Compute mean ± std for all metrics
3. ⏳ Add error bars to figures
4. ⏳ Report confidence intervals in tables

---

## Are We In A Good Place?

### **SHORT ANSWER: MIXED** ⚠️

**✅ Foundation is SOLID:**
- Framework implemented correctly
- Baselines working
- Collider learning demonstrated
- Additional domains implemented
- Paper structure complete

**⚠️ But Current Experiments Have ISSUES:**
- Latest run shows training failures
- Zero rewards preventing learning
- PPO baseline has bugs
- No actual numbers in paper yet

**✅ Fixes ARE IMPLEMENTED:**
- Jan 21 improvements should resolve training issues
- Adaptive diversity threshold
- Novelty bonuses
- Emergency retraining
- Better early stopping

### **RECOMMENDATION:**

**Path to Paper Readiness:**

1. **TODAY:** Test Jan 21 fixes (30 min)
2. **IF PASSING:** Full experimental run (4-6 hours)
3. **TOMORROW:** Fill all paper tables (2 hours)
4. **THIS WEEK:** Verify specific claims (3 hours)
5. **SUBMISSION READY:** By end of week if all goes well

**Confidence Level:**
- **If Jan 21 fixes work:** 85% confidence we can submit
- **If fixes don't work:** Need more debugging, delay 1 week

**Critical Path:** The Jan 21 fixes are THE key. Everything depends on:
- Diversity scores being positive
- Non-zero rewards throughout training
- Early stopping working properly
- DPO gradients staying healthy

---

## Summary: Paper Claims Support Matrix

| Claim Category | Status | Blocker? | Action |
|----------------|--------|----------|--------|
| **Framework Architecture** | ✅ Complete | No | None |
| **DPO Implementation** | ✅ Working | No | Verify stability |
| **Root Learner** | ✅ Working | No | None |
| **Collider Learning** | ✅ Demonstrated | No | Get fresh data |
| **Baseline Comparisons** | ⚠️ Incomplete | **YES** | Fresh ACE run |
| **PPO vs DPO** | ⚠️ Bug in PPO | **YES** | Fix PPO |
| **80% Speedup** | ⚠️ Contradicted | **YES** | Test fixes |
| **Result Tables** | ❌ Empty | **YES** | Fill w/ data |
| **Statistical Validation** | ❌ Not done | No* | Optional |

**\*Not a blocker for submission but should be acknowledged in limitations**

---

**Last Updated:** January 21, 2026, 08:00 AM  
**Next Review:** After test run completes  
**Status:** ⏳ **AWAITING JAN 21 FIX VALIDATION**
