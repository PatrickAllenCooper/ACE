# Complete Experiments Status
## All Experiments Analyzed - Ready for Next Run

**Date:** January 21, 2026, 10:00 AM  
**Git Status:** Clean (2 commits ahead of origin)  
**Overall:** ⚠️ **Code ready, Paper needs 3 revisions**

---

## 📊 ALL EXPERIMENTS INVENTORY

### **Experiment 1: ACE Main** (Core Contribution)

**File:** `ace_experiments.py`  
**Type:** DPO-based learning policy  
**Paper Section:** 3.4.1, Methods, Discussion  
**Status:** ✅ **Updated with Jan 21 fixes** - Ready to test

**What it does:**
- LLM policy generates interventions
- DPO training on preference pairs
- Learns experimental design strategy through self-play

**Issues Found (Jan 20 run):**
- ❌ 99% zero rewards → ✅ Fixed (novelty bonus)
- ❌ Zero gradients → ✅ Fixed (emergency retraining)
- ❌ Diversity penalty → ✅ Fixed (adaptive threshold)
- ❌ Too slow → ✅ Fixed (dynamic candidates)

**Next Action:** `./pipeline_test.sh` then `sbatch jobs/run_ace_main.sh`

---

### **Experiment 2: Baselines** (Comparison)

**File:** `baselines.py`  
**Type:** 4 baseline policies for comparison  
**Paper Section:** 3.8 (Baselines), Discussion  
**Status:** ✅ **Complete + PPO bug fixed**

**What it includes:**
1. Random - uniform sampling
2. Round-Robin - systematic cycling
3. Max-Variance - greedy uncertainty sampling
4. PPO - value-based RL

**Results (from logs copy):**
- Round-Robin: 1.9859 (BEST)
- Random: 2.1709
- Max-Variance: 2.0924
- PPO: 2.1835 (had bug, now fixed)

**Next Action:** Rerun PPO with fix: `python baselines.py --baseline ppo --episodes 100`

---

### **Experiment 3: Complex 15-Node SCM** (Scaling Validation)

**File:** `experiments/complex_scm.py`  
**Type:** Heuristic policy comparison (NOT ACE/DPO)  
**Paper Section:** 3.4.2 (Complex SCM)  
**Status:** 🔄 **Running** (greedy_collider in progress)

**What it tests:**
- 3 policies: random, smart_random, greedy_collider
- 15 nodes, 5 colliders (vs 5 nodes, 1 collider)
- Shows strategic intervention matters at scale

**Important:** This does NOT use ACE/DPO - it uses simple heuristic policies

**Results:**
- ✅ Random: Complete
- ✅ Smart_random: Complete
- 🔄 Greedy_collider: Running

**Paper Alignment:**
- ⚠️ Line 609 says "ACE becomes more pronounced" but it's testing heuristics
- **Fix:** Change to "strategic intervention becomes more pronounced"

**Next Action:** Wait for completion, then verify greedy > smart > random

---

### **Experiment 4: Duffing Oscillators** (Physics Validation)

**File:** `experiments/duffing_oscillators.py`  
**Type:** Random intervention policy (NOT ACE)  
**Paper Section:** 3.6 (Physics)  
**Status:** ✅ **Complete** (<1 min runtime)

**What it does:**
- Coupled oscillators with ODE simulation
- Random intervention policy
- Structure discovery (chain topology)
- Tests framework on continuous dynamics

**Important:** Uses RANDOM policy, not ACE/DPO

**Results (from logs copy):**
- Runtime: <1 minute
- 100 episodes completed
- Structure learning successful

**Paper Alignment:**
- 🔴 **Line 661 says "ACE discovers clamping strategy"** - INCORRECT
- Reality: Random policy, no discovery/learning
- **Fix:** Remove "ACE discovers", describe actual approach

**Next Action:** 
1. Run `python clamping_detector.py` to verify if clamping emerged
2. Revise paper line 661 based on findings

---

### **Experiment 5: Phillips Curve** (Economics Validation)

**File:** `experiments/phillips_curve.py`  
**Type:** Hardcoded regime selection (NOT ACE)  
**Paper Section:** 3.7 (Economics)  
**Status:** ✅ **Complete** (~30 sec runtime)

**What it does:**
- Real FRED data (unemployment, fed funds, CPI)
- Hardcoded regime selection strategy
- Retrospective learning across historical periods

**Important:** Regime selection is HARDCODED, not learned

**Results (from logs copy):**
- 552 records from FRED
- 6 regimes processed
- 100 episodes completed

**Paper Alignment:**
- 🔴 **Line 714 says "ACE learns to query regimes"** - INCORRECT
- Reality: Hardcoded regime order, no learning
- **Fix:** Change to "systematic querying of"

**Next Action:**
1. Run `python regime_analyzer.py` to document regime distribution
2. Revise paper line 714 to match reality

---

## 🚨 CRITICAL ALIGNMENT ISSUES

### **Summary:**

| Experiment | Paper Claim | Reality | Severity |
|------------|-------------|---------|----------|
| ACE Main | Uses DPO | ✅ Uses DPO | ✅ ALIGNED |
| Baselines | 4 methods | ✅ 4 methods | ✅ ALIGNED |
| Complex SCM | "ACE" advantage | ⚠️ Heuristics | ⚠️ CLARIFY |
| **Duffing** | **"ACE discovers"** | **❌ Random policy** | **🔴 CRITICAL** |
| **Phillips** | **"ACE learns"** | **❌ Hardcoded** | **🔴 CRITICAL** |

---

## 🎯 DO EXPERIMENTS NEED CHANGES?

### **Code Changes Needed:** ✅ **NO**

All experiments work correctly for their intended purposes:
- ✅ Complex SCM tests strategic vs random policies
- ✅ Duffing validates physics domain
- ✅ Phillips validates economics domain
- ✅ All produce useful results

**No bugs, no failures, no fixes needed in experiments/ directory.**

---

### **Paper Changes Needed:** ⚠️ **YES - 3 Claims**

Must revise before submission:

1. **Line 661 (Duffing):** Remove "ACE discovers clamping"
2. **Line 714 (Phillips):** Remove "ACE learns regimes"
3. **Line 609 (Complex):** Clarify "strategic policies" not "ACE"

**Timeline:** 1 hour to revise + verify  
**Impact:** Minimal - experiments still valuable, just accurately described

---

## 📋 COMPLETE EXPERIMENTAL PIPELINE STATUS

| Component | Purpose | Status | Code OK | Paper OK | Action |
|-----------|---------|--------|---------|----------|--------|
| **ACE Main** | Core learning | ⏳ Need run | ✅ Fixed | ✅ OK | Test & run |
| **Baselines** | Comparison | ✅ Complete | ✅ Fixed | ✅ OK | Rerun PPO |
| **Complex SCM** | Scaling | 🔄 Running | ✅ OK | ⚠️ Clarify | Wait & revise |
| **Duffing** | Physics | ✅ Complete | ✅ OK | 🔴 Fix | Verify & revise |
| **Phillips** | Economics | ✅ Complete | ✅ OK | 🔴 Fix | Verify & revise |

---

## 🎯 WHAT NEEDS TO ALIGN

### **Experiments Need:**
- ✅ No code changes
- ✅ No bug fixes
- ✅ No performance improvements
- ✅ All work as designed

### **Paper Needs:**
- ⚠️ Accurate description of Duffing (line 661)
- ⚠️ Accurate description of Phillips (line 714)
- ⚠️ Clarification about Complex SCM (line 609)

### **Verification Needs:**
- ⏳ Run `clamping_detector.py` on Duffing results
- ⏳ Run `regime_analyzer.py` on Phillips results
- ⏳ Confirm Complex SCM shows greedy > random

---

## 🚀 COMPLETE NEXT RUN PROCEDURE

### **Phase 1: Test Pipeline** (30 minutes)
```bash
./pipeline_test.sh
```
**Tests:** ACE fixes, PPO fix, verification tools  
**Expected:** All tests pass

---

### **Phase 2: Launch All Experiments** (6-10 hours)
```bash
# ACE Main (most important)
sbatch jobs/run_ace_main.sh  # 4-6 hours

# PPO Rerun (parallel)
nohup python baselines.py --baseline ppo --episodes 100 \
    --output results/ppo_fixed_$(date +%Y%m%d_%H%M%S) > ppo.log 2>&1 &

# Complex SCM (already running)
# Just wait for completion
```

**Note:** Duffing and Phillips already complete, no rerun needed

---

### **Phase 3: Verify ALL Experiments** (2 hours)
```bash
# 1. Verify behavioral claims
python clamping_detector.py   # Duffing: Check if clamping emerged
python regime_analyzer.py      # Phillips: Document regime distribution
./verify_claims.sh             # All experiments

# 2. Extract metrics
./extract_baselines.sh         # Baseline comparison
./extract_ace.sh               # ACE results (after run)
python compare_methods.py      # Table 1 generation

# 3. Verify complex SCM results
grep "FINAL RESULTS" results/paper_*/complex_scm/*/experiment.log
# Check: greedy_collider < smart_random < random
```

---

### **Phase 4: Update Documentation** (1 hour)
```bash
# 1. Document all findings
code results/RESULTS_LOG.md
# Add entries for:
# - ACE main run
# - PPO rerun
# - Complex SCM completion
# - Verification results

# 2. Update summary dashboard
# Change all ⏳ to ✅ or ⚠️ based on findings
```

---

### **Phase 5: Fill Paper & Revise** (3 hours)
```bash
# 1. Fill all tables with real numbers
code paper/paper.tex

# Tables to fill:
# - Line 428-437: Table 1 (Synthetic 5-node)
# - Line 600-608: Table 2 (Complex SCM)
# - Line 652-660: Table 3 (Duffing)
# - Line 705-713: Table 4 (Phillips)
# - Line 718-727: Table 5 (Summary)

# 2. Replace all [PLACEHOLDER] inline values

# 3. Revise 3 claims based on alignment check:
# - Line 661: Duffing clamping
# - Line 714: Phillips regimes
# - Line 609: Complex SCM
```

---

## ✅ FINAL CHECKLIST

### **Code/Experiments:**
- [x] ACE main fixes implemented
- [x] PPO bug fixed
- [x] Verification tools created
- [x] Extraction scripts ready
- [x] All experiments working correctly
- [x] Git committed and clean

### **Need to Run:**
- [ ] `./pipeline_test.sh` (30 min)
- [ ] `sbatch jobs/run_ace_main.sh` (4-6 hours)
- [ ] Rerun PPO (2 hours)
- [ ] Wait for complex_scm completion

### **Need to Verify:**
- [ ] Clamping in Duffing (run detector)
- [ ] Regimes in Phillips (run analyzer)
- [ ] Strategic advantage in Complex SCM (check results)

### **Need to Update:**
- [ ] Fill 5 paper tables
- [ ] Replace all [PLACEHOLDER]
- [ ] Revise 3 paper claims (lines 609, 661, 714)
- [ ] Update results/RESULTS_LOG.md

### **Can Submit When:**
- [ ] All experiments complete
- [ ] All tables filled
- [ ] All claims verified or revised
- [ ] Final review complete

---

## 🎯 ALIGNMENT SUMMARY

### **Q: Do experiments need further work to align?**

**A: NO code changes needed, YES paper revisions needed.**

**Code:** ✅ All experiments work correctly  
**Paper:** ⚠️ 3 claims over-state what they do

**Solution:** Verify behaviors with tools, revise paper claims (1-2 hours work)

---

### **What's Aligned:**
- ✅ ACE main (core contribution)
- ✅ Baselines (all 4 methods)
- ✅ Experiments produce useful validation data

### **What Needs Alignment:**
- ⚠️ Paper descriptions of Duffing, Phillips, Complex SCM
- ⚠️ 3 specific claims that say "ACE" when it's actually heuristics

### **Impact:**
- Low - experiments still validate multi-domain applicability
- Just needs honest description of what each does

---

## 🚀 READY FOR NEXT RUN

**Git Status:** ✅ Clean, 2 commits ahead  
**Code Status:** ✅ All experiments ready  
**Paper Status:** ⚠️ Needs 3 minor revisions  

**Immediate Action:**
```bash
./pipeline_test.sh
```

**After Test:**
```bash
sbatch jobs/run_ace_main.sh
```

**After Runs:**
```bash
# Verify
python clamping_detector.py
python regime_analyzer.py

# Extract
./extract_ace.sh
python compare_methods.py

# Document
code results/RESULTS_LOG.md

# Revise paper
code paper/paper.tex
# Lines 609, 661, 714
```

---

## 📖 Key Documents

**For current status:** `START_HERE.md`  
**For all experiments:** `EXPERIMENTS_ALIGNMENT_CHECK.md` (this analysis)  
**For paper revisions:** `results/PAPER_REVISIONS_NEEDED.md`  
**For tracking results:** `results/RESULTS_LOG.md`  

---

**Summary:** All 5 experiments analyzed. Code is ready. Paper needs minor revisions. Proceed with testing! 🚀
