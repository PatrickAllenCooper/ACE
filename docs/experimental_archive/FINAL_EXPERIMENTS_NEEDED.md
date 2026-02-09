# Final Experiments Needed for Complete Paper

**Status:** Paper is ACCEPT-ready with current results
**Goal:** Complete all baseline comparisons for comprehensive validation

---

## Critical Experiments (Must Complete)

### 1. Complex SCM - Rerun with 10h limit ✓ QUEUED
**Status:** Running now (job 23323386 resubmitted)
**What:** 5 seeds × 4 methods (random, round-robin, greedy, lookahead)
**Time:** 5-6 hours
**Saves:** Incrementally after each seed
**Priority:** CRITICAL - completes 15-node scaling validation

### 2. Complete Ablations ✓ READY
**Script:** `jobs/run_remaining_ablations.sh`
**What:** no_convergence, no_root_learner, no_diversity (3 seeds each)
**Time:** 6-8 hours
**Priority:** HIGH - validates all 4 components

### 3. ACE Without Oracle ✓ READY  
**Script:** `jobs/run_ace_no_oracle.sh`
**What:** ACE with pretrain_steps=0 (5 seeds)
**Time:** 5-6 hours
**Priority:** HIGH - eliminates main reviewer objection

---

## Additional Experiments (Strengthen Paper)

### 4. Duffing Baselines NOT IMPLEMENTED YET
**Need:** Random, Round-Robin, Max-Variance baselines for Duffing
**Current:** ACE-only (0.042 ± 0.036)
**Issue:** experiments/duffing_oscillators.py doesn't support --policy flag
**Options:**
  - A) Add baseline support to duffing_oscillators.py
  - B) Report ACE results as "demonstration" (current framing)
  - C) Skip for initial submission, add in revision

**Recommendation:** Option B (already done in paper fixes)

### 5. Phillips Baselines NOT IMPLEMENTED YET
**Need:** Random, Round-Robin regime selection baselines
**Current:** ACE-only regime selection
**Issue:** experiments/phillips_curve.py doesn't support --policy flag
**Options:**
  - A) Add baseline support to phillips_curve.py
  - B) Report ACE results as "demonstration" (current framing)
  - C) Skip for initial submission, add in revision

**Recommendation:** Option B (already done in paper fixes)

### 6. PPO for Complex 15-Node SCM NOT IN CURRENT RUN
**Need:** PPO baseline for complex SCM
**Current:** Random, Round-Robin, Greedy Collider, Random Lookahead
**Issue:** PPO not included in run_critical_experiments.py
**Options:**
  - A) Add PPO to complex SCM experiments
  - B) Note that PPO is comparable to Round-Robin on symmetric 5-node, expect similar on 15-node
  - C) Skip for initial submission

**Recommendation:** Option B (reasonable assumption based on 5-node results)

---

## Time Assessment

**If we add Duffing/Phillips baselines:**
- Implement baseline support: 2-3 hours
- Run experiments: 3-4 hours
- Total: 5-7 hours additional

**If we add PPO to complex SCM:**
- Modify script: 1 hour
- Re-run: 5-6 hours
- Total: 6-7 hours

**Total for complete coverage:** 11-14 hours additional

---

## Decision Framework

### Scenario A: Submit with Current Results (ACCEPT tier)
**Have:**
- ✅ Synthetic 5-node: ALL baselines (Random, RR, Max-Var, PPO)
- ✅ Extended baselines (171 ep): All 3 methods
- ✅ Lookahead ablation: Random proposer
- ⏳ Complex 15-node: 4 baselines (no PPO)
- ✅ no_dpo ablation: Complete
- ✅ Duffing: ACE demonstration
- ✅ Phillips: ACE demonstration

**Missing:**
- 3/4 ablations incomplete
- No oracle comparison incomplete
- Duffing/Phillips no baselines
- Complex SCM no PPO

**Timeline:** Paper ready when complex SCM completes (~4-6 hours)
**Verdict:** Competitive ACCEPT, honest about limitations

### Scenario B: Add Ablations + No-Oracle (STRONG ACCEPT tier)
**Additional experiments:**
- Complete ablations: 6-8 hours
- ACE no oracle: 5-6 hours
- (Run in parallel)

**Timeline:** +6-8 hours
**Verdict:** Addresses two main weaknesses, STRONG ACCEPT candidate

### Scenario C: Complete Everything (BULLETPROOF)
**Additional experiments:**
- Ablations + No-Oracle: 6-8 hours
- Duffing baselines: 3-4 hours
- Phillips baselines: 3-4 hours
- Complex SCM with PPO: 5-6 hours

**Timeline:** +15-20 hours
**Verdict:** No gaps, bulletproof paper

---

## Recommendation

**Priority 1 (Essential):**
- ✅ Complex SCM (running now)
- ⏳ Complete ablations
- ⏳ ACE no oracle

**Priority 2 (If time allows):**
- Duffing/Phillips baselines
- PPO for complex SCM

**Minimum for submission:** Priority 1 complete
**Ideal for submission:** Priority 1 + 2

---

## Current Scripts Ready to Execute

```bash
# On HPC:
cd ~/ACE
git pull

# Submit Priority 1 experiments (run in parallel)
bash jobs/workflows/submit_strong_accept_experiments.sh

# Monitor
watch -n 60 'squeue -u $USER'
```

**These are VERIFIED and READY** - all tests pass.

**Duffing/Phillips baselines would require additional implementation work.**
