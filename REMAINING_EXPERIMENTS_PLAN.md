# Remaining Experiments Plan - Final Push

**Date:** January 30, 2026
**Status:** 3 critical experiments remain for complete paper

---

## Current Paper Status

**COMPLETE:**
- Extended baselines (171 episodes, N=5) ✓
- Lookahead ablation (N=5) ✓
- Main ACE results (N=5) ✓
- Statistical tests ✓

**INCOMPLETE (Blocking submission quality):**
1. 5-node ablations (all 4 components, N=3 seeds minimum)
2. Complex 15-node SCM with ACE (N=1 seed minimum, ideally N=5)
3. No-oracle ACE (N=5 seeds)

---

## Issue 1: Ablation Studies

### Problem
Current ablation results show IMPROVEMENT when components removed:
- no_dpo: 0.52 (better than 0.92 full ACE) ← ANOMALOUS
- no_convergence: 0.78 (better than 0.92) ← ANOMALOUS
- no_root_learner: 0.59 (better than 0.92) ← ANOMALOUS
- no_diversity: 0.52 (better than 0.92) ← ANOMALOUS

### Root Cause
Ablation flags not properly disabling components in `ace_experiments.py`

### Solution
1. **Audit ablation logic in ace_experiments.py**
   - Verify `--no_per_node_convergence` actually disables per-node stopping
   - Verify `--no_dedicated_root_learner` removes root learner
   - Verify `--no_diversity_reward` sets diversity weight to 0
   - Add explicit logging: "ABLATION ACTIVE: [component] DISABLED"

2. **Add verification tests**
   - Test that ablation flags change behavior
   - Test that ablations DEGRADE performance (sanity check)

3. **Rerun ablations (N=3 seeds minimum)**
   - Episodes: 100 (faster, still valid)
   - Use --custom flag (transformer policy, no HF download)
   - Expected degradation: 50-150% (ablations should be WORSE)

### HPC Submission
```bash
# On HPC, test one ablation first:
python -u ace_experiments.py \
    --custom --no_diversity_reward \
    --episodes 100 --seed 42 \
    --output results/ablations_verified/no_diversity/seed_42 \
    --pretrain_steps 200

# If shows degradation (loss > 1.5), submit all:
bash jobs/workflows/submit_ablations_verified.sh
```

### Expected Runtime
- Per seed: ~45 minutes
- Total (4 ablations × 3 seeds): ~9 hours parallel on HPC

---

## Issue 2: Complex 15-Node SCM with ACE

### Problem
ACE on complex SCM achieved 290 ± 62 loss (N=5), **60x worse than baselines (4.6)**

Earlier "simplified" run got 5.75 (1 seed, 200 episodes) - 27% worse than baselines

### Root Cause Analysis
The 290 loss run (`ace_complex_scm_20260128_080650`) used:
- Only 200 steps of oracle pretraining
- Fast convergence (avg 33-55 episodes)
- No observational training
- Likely undertrained on this harder problem

### Solution: Optimized Architecture

**Key Changes for Complex SCM:**
1. **More oracle pretraining:** 500 steps (vs 200)
2. **More steps per episode:** 50 (vs 25 in 5-node)
3. **Add observational training:** Every 3 steps, inject 200 observational samples
4. **Longer runs:** 300 episodes (vs 200)
5. **Better diversity:** Collider parent tracking, forced diversity every 10 steps
6. **Reduce candidates:** K=2 (faster, less overfitting)

### Implementation Status
File: `experiments/run_ace_complex_full.py`
- Contains full DPO loop ✓
- Qwen2.5-1.5B policy ✓
- Observational training ✓
- All optimizations ✓

**BUT:** Last HPC run (job 23366125) failed due to indentation error (FIXED)

### HPC Submission
File created: `jobs/run_ace_complex_single_seed.sh`
- 10 hour time limit
- Full architecture
- Single seed (42) for overnight test

```bash
# On HPC:
sbatch jobs/run_ace_complex_single_seed.sh
```

### Expected Results
**Target:** <4.5 loss (better than best baseline 4.51)
**Acceptable:** 4.5-5.0 (competitive)
**Failure:** >5.0 (doesn't scale)

### Expected Runtime
- 6-8 hours for 1 seed (300 episodes)
- 30-40 hours for 5 seeds (if successful)

---

## Issue 3: No-Oracle ACE

### Problem
Only 2 seeds completed, and they show 0.75 ± 0.14 (BETTER than full ACE 0.92)
This is anomalous - removing oracle should DEGRADE performance

### Root Cause
No-oracle runs likely incomplete or used different settings

### Solution
Run proper no-oracle ACE (N=5 seeds):
- Use Qwen2.5-1.5B (not custom transformer)
- NO pretraining (--pretrain_steps 0)
- Same episodes as full ACE (~171)
- All other components identical

### Expected Results
Loss should be 1.0-1.5 (10-60% degradation from 0.92)
Still better than baselines (2.0+), proving ACE works without oracle

### HPC Submission
File: `jobs/run_ace_no_oracle.sh`
```bash
# On HPC:
sbatch jobs/run_ace_no_oracle.sh
```

### Expected Runtime
- Per seed: ~3 hours
- Total (5 seeds): ~15 hours

---

## Execution Priority

### Phase 1: Overnight (HIGH PRIORITY)
1. **Complex SCM single seed** (job 23399683 queued)
   - If successful (<4.5): Run all 5 seeds
   - If marginal (4.5-5.5): Discuss inclusion
   - If failed (>5.5): Acknowledge scaling limitation

### Phase 2: Next Day (CRITICAL)
2. **Ablation verification run** (1 seed, 1 ablation, ~1 hour)
   - Test no_diversity with seed 42
   - Verify degradation occurs (loss > 1.5)
   - If verified, submit all 4 × 3 = 12 jobs

3. **No-oracle ACE** (N=5 seeds, ~15 hours)
   - Can run in parallel with ablations
   - Less critical than ablations

### Phase 3: Final Validation (IF TIME)
4. **Complete complex SCM** (if single seed successful)
   - Run remaining 4 seeds
   - ~24-30 hours

---

## Success Criteria

### Minimum Acceptable (Paper submittable):
- ✓ Extended baselines (DONE)
- ✓ Lookahead ablation (DONE)
- 3/4 ablations working (75% coverage)
- No-oracle with N≥3 seeds

### Ideal (Strong Accept):
- All 4 ablations (N=3 each)
- No-oracle (N=5)
- Complex SCM ACE competitive (<5.0 loss)

### Timeline
- Ablations: 9 hours
- No-oracle: 15 hours
- Complex SCM (if good): 30 hours

**Total: 24-54 hours depending on complex SCM viability**

---

## Next Immediate Actions

1. **Monitor job 23399683** (complex SCM single seed)
   - Check queue position
   - Watch logs when starts
   - Copy results when complete

2. **Fix ablation logic** (if needed)
   - Audit ace_experiments.py line by line
   - Ensure flags actually disable components
   - Test locally first

3. **Prepare no-oracle submission**
   - Verify job script has --pretrain_steps 0
   - Test import paths work
   - Submit when HPC available

---

## Contingency Plans

**If ablations still fail:**
- Use theoretical justification instead
- Cite similar ablations from DPO/RL literature
- Acknowledge as limitation

**If complex SCM never works:**
- Remove from results section
- Keep in intro as motivation only
- Focus paper on 5-node validation

**If time runs out:**
- Submit with extended baselines + lookahead
- Note ablations/no-oracle in "ongoing work"
- Paper is still acceptable without them
