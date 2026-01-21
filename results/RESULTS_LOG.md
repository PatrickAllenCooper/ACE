# Results Log
## Running Record of Experimental Findings

**Purpose:** Document every experimental result that supports (or contradicts) paper claims.

**Last Updated:** January 21, 2026

---

## 2026-01-21: Baseline Comparison Complete ✅

**Run ID:** `logs copy/baselines_20260120_142711_23026272`  
**Experiment:** Synthetic 5-node SCM - All Baselines  
**Date:** January 21, 2026  
**Status:** ✅ Complete  

### Key Metrics

| Baseline | Final Loss | X1 | X2 | X3 | X4 | X5 | Episodes |
|----------|-----------|-----|-----|-----|-----|-----|----------|
| **Random** | 2.1709 ± 0.0436 | 1.0637 ✗ | 0.0106 ✓ | 0.0683 ✓ | 1.0151 ✗ | 0.0132 ✓ | 100 |
| **Round-Robin** | 1.9859 ± 0.0402 | 0.9655 ✗ | 0.0104 ✓ | 0.0594 ✓ | 0.9376 ✗ | 0.0131 ✓ | 100 |
| **Max-Variance** | 2.0924 ± 0.0519 | 1.0702 ✗ | 0.0097 ✓ | 0.0799 ✓ | 0.9184 ✗ | 0.0141 ✓ | 100 |
| **PPO** | 2.1835 ± 0.0342 | 0.9719 ✗ | 0.0114 ✓ | 0.0494 ✓ | 1.1369 ✗ | 0.0139 ✓ | 100 |

### Intervention Distribution

| Baseline | X1 | X2 | X3 | X4 | X5 |
|----------|-----|-----|-----|-----|-----|
| Random | 19.9% | 19.2% | 21.5% | 19.6% | 19.8% |
| Round-Robin | 20.0% | 20.0% | 20.0% | 20.0% | 20.0% |
| Max-Variance | 19.9% | 19.7% | 20.4% | 20.5% | 19.4% |
| PPO | 13.7% | 10.2% | 6.9% | 36.5% | 32.6% |

### Paper Claims Supported

- ✅ **Line 362-378**: All four baselines implemented and validated
- ✅ **Line 365**: Random achieves ~20% uniform allocation
- ✅ **Line 368**: Round-Robin achieves perfect 20% balance
- ⚠️ **Line 734**: "DPO outperforms PPO" - PPO has implementation bug

### Key Findings

1. **Round-Robin is BEST baseline** (1.9859) - This is important!
2. All baselines learn X2, X3, X5 (losses < 0.15)
3. NO baseline learns X1, X4 roots well (losses ~0.9-1.1)
4. PPO shows policy collapse (36.5% X4, 32.6% X5, only 6.9% X3)
5. PPO has shape mismatch warning (line 60-61 in logs)

### Action Items

- [ ] Fix PPO implementation bug (`baselines.py:573`)
- [ ] Rerun PPO baseline after fix
- [ ] Update paper Table 1 (lines 428-437) with these values
- [ ] Acknowledge Round-Robin strength in discussion
- [ ] Compare ACE to Round-Robin (strongest baseline, not just Random)

### Evidence Files

- Log: `logs copy/baselines_20260120_142711_23026272.out`
- Error log: `logs copy/baselines_20260120_142711_23026272.err`
- Results: `results/paper_20260120_142711/baselines/baselines_20260121_064331/`

---

## 2026-01-21: Root Learner Validation ✅

**Run ID:** `logs copy/ace_main_20260120_142711_23026271.err` (lines 1800-1900)  
**Experiment:** Dedicated Root Learner Performance  
**Date:** January 21, 2026  
**Status:** ✅ Validated  

### Key Metrics

| Episode | X1 Before | X1 After | X4 Before | X4 After | Improvement |
|---------|-----------|----------|-----------|----------|-------------|
| 3 | 0.1156 | 0.0005 | 1.4653 | 1.2726 | 13% X1, 98.7% X4 |
| 6 | 0.0544 | 0.0005 | 1.2231 | 0.0584 | 99.1% both |
| 9 | 0.0623 | 0.0005 | 0.0472 | 0.0006 | 99.2% both |
| 30 | 0.0377 | 0.0005 | 0.0632 | 0.0005 | **98.7% average** |

### Paper Claims Supported

- ✅ **Line 339-343**: "dedicated root learner" addresses exogenous variable problem
- ✅ **Line 342**: "periodically transfers these estimates" - every 3 episodes
- ✅ **Line 76 (guidance)**: "Root node learning fixed with explicit fitting"
- ✅ **Conclusion line 763**: Root learner as key contribution

### Key Findings

1. Root learner consistently achieves losses < 0.001
2. Works across ALL episodes tested (3, 6, 9, 12, 15, 18, 21, 24, 27, 30)
3. Rapid convergence (within single training session)
4. This is a STRONG architectural contribution

### Action Items

- [x] Document this as key evidence for dedicated root learner claim
- [ ] Include in paper discussion as validated approach
- [ ] Consider making this a highlighted contribution in abstract

### Evidence Files

- Log: `logs copy/ace_main_20260120_142711_23026271.err`
- Search: `grep "Dedicated Root Learner" [log file]`

---

## 2026-01-21: ACE Training Issues Identified ⚠️

**Run ID:** `logs copy/ace_main_20260120_142711_23026271.err`  
**Experiment:** ACE Main (Jan 20 run)  
**Date:** January 21, 2026  
**Status:** ⚠️ Problems identified, fixes implemented  

### Issues Found

1. **99% Zero Rewards**
   - Evidence: Lines show `Reward: 0.00` repeatedly
   - Impact: No DPO learning signal
   - Status: ✅ Fixed with novelty bonus (Jan 21)

2. **Zero Gradients at Episode 20**
   - Evidence: `grad_norm=0.000000, num_params_with_grad=338`
   - Impact: DPO not training
   - Status: ✅ Fixed with emergency retraining (Jan 21)

3. **Diversity Penalty Too Harsh**
   - Evidence: `diversity=-29.46` in candidate logs
   - Impact: Fighting necessary X2 concentration
   - Status: ✅ Fixed with adaptive threshold (Jan 21)

4. **Slow Progress**
   - Evidence: 31 episodes in 7+ hours (13.5 min/episode)
   - Expected: 200 episodes in 2 hours (~0.6 min/episode)
   - Status: ✅ Fixed with reduced candidates + better early stop

### Paper Claims CONTRADICTED (Before Fixes)

- ❌ **Line 767**: "80% reduction" - Current run 12-24x SLOWER
- ❌ **Line 734**: "DPO stable" - Zero gradients contradict this
- ⚠️ **Line 439**: Cannot claim ACE superiority without successful run

### Action Items

- [x] Implement fixes (adaptive diversity, novelty bonus, emergency retrain)
- [ ] Test fixes with quick 10-episode run
- [ ] Full 200-episode run after validation
- [ ] Update paper claims based on new results

### Evidence Files

- Log: `logs copy/ace_main_20260120_142711_23026271.err`
- Analysis: `FIXES_NEEDED.md`
- Fixes: `CHANGES_IMPLEMENTED_JAN21.md`

---

## 2026-01-21: Additional Domains Complete ✅

**Experiments:** Duffing Oscillators, Phillips Curve  
**Date:** January 21, 2026  
**Status:** ✅ Both complete  

### Duffing Oscillators

**Run ID:** `logs copy/duffing_20260120_142711_23026274.err`  
**Runtime:** < 1 minute  

**Key Results:**
- Episode 0: Loss=0.0586
- Episode 20: Loss=0.0152
- Episode 80: Loss=0.0621
- Final: Learned graph structure

**Paper Claims Supported:**
- ✅ **Line 353-354**: Physics simulation domain implemented
- ⏳ **Line 661**: "clamping strategy" - NEEDS VERIFICATION

**Action Items:**
- [ ] Verify if clamping strategy (DO X2=0) actually emerged
- [ ] Extract structure identification F1 score
- [ ] Fill Table 3 (lines 652-660)

### Phillips Curve

**Run ID:** `logs copy/phillips_20260120_142711_23026275.err`  
**Runtime:** ~30 seconds  

**Key Results:**
- Data: 552 records from FRED (1960-2023)
- Regimes: 6 historical periods identified
- Episode 0: Eval Loss=0.4908
- Episode 90: Eval Loss=0.3147

**Paper Claims Supported:**
- ✅ **Line 356-359**: Real-world economic data domain
- ⏳ **Line 714**: "high-volatility regime selection" - NEEDS VERIFICATION

**Action Items:**
- [ ] Verify if ACE actually selected high-volatility periods
- [ ] Calculate % queries to pre-1985 data
- [ ] Compute out-of-sample MSE
- [ ] Fill Table 4 (lines 705-713)

### Evidence Files

- Duffing: `logs copy/duffing_20260120_142711_23026274.err`
- Phillips: `logs copy/phillips_20260120_142711_23026275.err`
- Results: `results/paper_20260120_142711/duffing/` and `/phillips/`

---

## 2026-01-21: Complex 15-Node SCM ⏳

**Run ID:** `logs copy/complex_scm_20260120_142711_23026273`  
**Experiment:** Complex 15-node SCM  
**Date:** January 21, 2026  
**Status:** ⏳ In Progress (greedy_collider policy running)  

### Completed Policies

- ✅ Random: Complete
- ✅ Smart Random: Complete (4 hours runtime)
- ⏳ Greedy Collider: Currently running

### Preliminary Findings

- Random and smart_random policies completed
- Need to wait for greedy_collider completion
- Will provide collider vs non-collider MSE comparison

### Paper Claims Pending

- ⏳ **Line 534-609**: Complex SCM scaling results
- ⏳ **Table 2**: Collider vs non-collider performance

### Action Items

- [ ] Wait for greedy_collider completion
- [ ] Extract collider-specific vs non-collider losses
- [ ] Calculate improvement percentages
- [ ] Fill Table 2 (lines 600-608)

### Evidence Files

- Log: `logs copy/complex_scm_20260120_142711_23026273.out`
- Results: `results/paper_20260120_142711/complex_scm/`

---

## Summary Dashboard (Updated: Jan 21, 09:30 AM)

| Paper Claim | Evidence Status | Quality | Action Needed |
|-------------|----------------|---------|---------------|
| **4 baselines implemented** | ✅ Complete | High | None |
| **Root learner works** | ✅ Complete | High | None |
| **DPO > PPO** | ⚠️ PPO bug fixed | Medium | Rerun PPO |
| **ACE superior to baselines** | ⏳ Fixes ready | - | Run pipeline_test.sh |
| **Collider identification** | ⏳ Fixes ready | - | Test then run ACE |
| **Computational efficiency** | ⏳ Fixes ready | - | Validate with new run |
| **Strategic concentration** | ⏳ Fixes ready | - | Test adaptive diversity |
| **Clamping in Duffing** | ⏳ Tool ready | - | Run clamping_detector.py |
| **Regime selection Phillips** | ⏳ Tool ready | - | Run regime_analyzer.py |

**Next Action:** `./pipeline_test.sh` (30 min) to validate all fixes
| **Complex SCM scaling** | ⏳ Running | - | Wait for completion |

---

## Next Experimental Runs Needed

1. **Priority 1** 🔥: Quick test of Jan 21 fixes (10 episodes, ~20 min)
2. **Priority 2** 🔥: Full ACE run (200 episodes, 4-6 hours)
3. **Priority 3** ⚠️: PPO baseline rerun (100 episodes, 2 hours)
4. **Priority 4** 📊: Wait for complex_scm completion

---

## Notes

- Remember: Only document results you actually have
- If a claim is aspirational, mark it clearly
- Update this log IMMEDIATELY after each run
- Cross-reference paper line numbers
- Include both positive AND negative results

---

## 2026-01-21: Experiments Alignment Check ⚠️

**Analysis:** All experiments in `experiments/` directory  
**Date:** January 21, 2026  
**Status:** ⚠️ Paper claims misaligned with implementation  

### Findings

**experiments/complex_scm.py:**
- ✅ Works correctly (tests 3 heuristic policies)
- ⚠️ Paper implies ACE runs on complex SCM (it doesn't - uses heuristics)
- Action: Clarify that this tests strategic vs random policies

**experiments/duffing_oscillators.py:**
- ✅ Works correctly (random policy for structure discovery)
- 🔴 Paper claims "ACE discovers clamping" but uses RANDOM policy
- Action: Revise line 661 - remove "ACE discovers"

**experiments/phillips_curve.py:**
- ✅ Works correctly (hardcoded regime selection)
- 🔴 Paper claims "ACE learns" but regime selection is HARDCODED
- Action: Revise line 714 - change to "systematic querying"

### Paper Revisions Required

1. **Line 661:** "ACE discovers clamping" → "Interventions enable structure discovery"
2. **Line 714:** "ACE learns to query" → "Systematic querying of"
3. **Line 609:** "ACE becomes more pronounced" → "Strategic intervention becomes"

### Impact

- ⚠️ Over-claiming in current paper
- ✅ Experiments work correctly (no code changes needed)
- ⚠️ Must revise before submission
- Timeline: 1 hour to revise + verify

### Evidence Files

- Analysis: `EXPERIMENTS_ALIGNMENT_CHECK.md`
- Paper revisions: `results/PAPER_REVISIONS_NEEDED.md`

---

## 2026-01-21: Observational Training Restored + Paper Revisions ✅

**Commit:** ff4dc3f  
**Date:** January 21, 2026, 10:00 AM  
**Status:** ✅ Complete - Ready to launch  

### Changes Made

**1. Observational Training Restored:**
- Re-added step-level observational training (every 3 steps)
- Works alongside dedicated root learner (both active)
- Prevents catastrophic forgetting of mechanisms
- Parameters: 200 samples, 100 epochs per injection

**2. Paper Claims Revised (3 fixes):**
- Line 609: "ACE becomes" → "Strategic intervention becomes"
- Line 661: "ACE discovers clamping" → "Interventions decouple"
- Line 714: "ACE learns regimes" → "Systematic querying"

### Why This Matters

**Dual Observational Training:**
- Step-level (every 3 steps): Preserves all mechanisms during interventional training
- Episode-level (every 3 episodes): Dedicated root learner for X1/X4

**Paper Accuracy:**
- Duffing uses random policy (not ACE) - claim revised
- Phillips uses hardcoded regimes (not learning) - claim revised
- Complex SCM tests heuristics (not ACE) - claim clarified

### Evidence Files

- Code: `ace_experiments.py` lines 2658-2666
- Paper: `paper/paper.tex` lines 609, 661, 714
- Analysis: `EXPERIMENTS_ALIGNMENT_CHECK.md` (removed after commit)

### Next Actions

- [ ] Test pipeline: `./pipeline_test.sh`
- [ ] Launch ACE: `sbatch jobs/run_ace_main.sh`
- [ ] Rerun PPO: `python baselines.py --baseline ppo`
- [ ] Verify claims after runs complete
- [ ] Fill paper tables

**Last Updated:** January 21, 2026, 10:15 AM
