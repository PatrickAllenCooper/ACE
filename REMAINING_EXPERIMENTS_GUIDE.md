# Remaining Experiments Guide

**Status:** Main results complete! Optional experiments to strengthen paper before submission.

---

## Quick Status

### ✓ COMPLETE (Paper-Ready)
- [x] ACE multi-seed (N=5) - **0.61 median, 55-58% improvement**
- [x] Baselines (N=5 each) - Random, Round-Robin, Max-Variance, PPO
- [x] Duffing (N=5) - Physics validation
- [x] Phillips (N=5) - Economics validation
- [x] Complex SCM (N=6) - Strategic comparison

**Paper is 90% ready with these results alone!**

### ⚠ RECOMMENDED (Strengthen for Reviewers)
- [ ] **Ablation studies** (CRITICAL) - Validate component contributions
- [ ] **Statistical tests** (HIGH PRIORITY) - Generate formal p-values
- [ ] **Phillips analysis** (NICE TO HAVE) - Complete economics section

### 🔬 OPTIONAL (Future Work)
- [ ] Large-scale 30-node SCM - Scalability demonstration

---

## Priority 1: Statistical Tests (5 minutes)

**Why:** Confirms p<0.001 claims in paper with formal tests

**How to run:**

```bash
# Can run locally or on HPC
cd ~/ACE  # Or your local path

python scripts/statistical_tests.py \
  --ace results/ace_multi_seed_20260125_115453 \
  --baselines results/baselines/baselines_20260124_182827 \
  --output results/statistical_analysis.txt
```

**Output:**
- Paired t-tests with Bonferroni correction
- Effect sizes (Cohen's d)
- LaTeX table for paper supplement
- Saved to `results/statistical_analysis.txt`

**Value:** HIGH - Reviewers expect this

---

## Priority 2: Ablation Studies (2-4 hours HPC)

**Why:** Quantifies contribution of each ACE component

**Components tested:**
1. **DPO Training** - Is preference learning better than random?
2. **Per-Node Convergence** - Does early stopping save episodes?
3. **Dedicated Root Learner** - Does it improve root variable learning?
4. **Diversity Reward** - Does it prevent policy collapse?

### Option A: Run All Ablations (Recommended)

```bash
# On HPC
cd ~/ACE
source setup_env.sh

# Run all 4 ablations with 3 seeds each (12 total runs)
./run_ablations.sh --seeds 3 --episodes 200

# Monitor (will run sequentially, ~2-4 hours total)
# Progress shown in terminal

# After completion, analyze:
python scripts/analyze_ablations.py results/ablations_20260126_*/
```

### Option B: Test One Ablation First

```bash
# Quick test - just DPO ablation with 1 seed
python ace_experiments.py \
  --episodes 200 \
  --seed 42 \
  --no_dpo_training \
  --output results/ablation_test_no_dpo

# Compare to ACE baseline
# ACE median: 0.61
# Random policy (no DPO) expected: ~2.0+
```

### Analysis

```bash
# After ablations complete:
python scripts/analyze_ablations.py \
  results/ablations_TIMESTAMP \
  --latex \
  --output results/ablation_table.tex

# This generates:
# 1. Degradation per component (e.g., +50% without DPO)
# 2. Statistical significance tests
# 3. LaTeX table for paper
```

**Expected Results:**
- **No DPO:** ~2.0 loss (+227% degradation) - Shows DPO is essential
- **No convergence:** Same loss, but ~199 episodes vs 60-171
- **No root learner:** Root losses worse (~1.5 vs 1.0)
- **No diversity:** Policy may collapse to single node

**Value:** CRITICAL - Reviewers will ask for this

---

## Priority 3: Phillips Curve Analysis (10 minutes)

**Why:** Complete the economics section with detailed metrics

**How to run:**

```bash
# Analyze existing Phillips results
python scripts/analyze_phillips.py \
  results/phillips/phillips_20260124_*/phillips_results.csv \
  --out-of-sample \
  --output results/phillips_analysis.txt
```

**Note:** If `analyze_phillips.py` doesn't exist, I can create it. Let me know the specific analyses you need for the Phillips section.

**Value:** MEDIUM - Nice but not critical

---

## Optional: Large-Scale 30-Node SCM

**Why:** Demonstrate scalability to realistic system sizes

**Status:** Currently marked as TODO in paper (Line 469)

**How to run:**

```bash
# Check if experiment exists
ls experiments/large_scale_scm.py

# If exists, test run:
cd ~/ACE
source setup_env.sh

python -m experiments.large_scale_scm \
  --policy ace \
  --episodes 200 \
  --seed 42 \
  --output results/large_scale_test

# If successful, multi-seed:
# (Would need to create run script)
```

**Value:** LOW - Nice extension but not essential for paper

---

## Recommended Workflow

### Minimal (Before Submission)

```bash
# 1. Statistical tests (5 min)
python scripts/statistical_tests.py \
  --ace results/ace_multi_seed_20260125_115453 \
  --baselines results/baselines/baselines_20260124_182827 \
  --output results/statistical_analysis.txt

# 2. Review paper
# 3. Submit!
```

**Paper readiness:** 90% → 95%

### Recommended (Strengthen Paper)

```bash
# 1. Statistical tests (5 min)
python scripts/statistical_tests.py \
  --ace results/ace_multi_seed_20260125_115453 \
  --baselines results/baselines/baselines_20260124_182827 \
  --output results/statistical_analysis.txt

# 2. Ablation studies (2-4 hrs HPC)
cd ~/ACE
source setup_env.sh
./run_ablations.sh --seeds 3 --episodes 200

# Wait for completion...

# 3. Analyze ablations
python scripts/analyze_ablations.py results/ablations_*/ --latex

# 4. Add ablation results to paper Discussion section
# 5. Submit!
```

**Paper readiness:** 90% → 98%

### Complete (If Time Permits)

```bash
# Do Recommended workflow, plus:

# 3. Phillips analysis
python scripts/analyze_phillips.py results/phillips/*/ --out-of-sample

# 4. (Optional) Large-scale experiment
python -m experiments.large_scale_scm --policy ace --episodes 200
```

**Paper readiness:** 90% → 100%

---

## Time Estimates

| Task | Time | Priority | Value |
|------|------|----------|-------|
| Statistical tests | 5 min | HIGH | HIGH |
| Ablations (all) | 2-4 hrs | CRITICAL | CRITICAL |
| Phillips analysis | 10 min | MEDIUM | MEDIUM |
| Large-scale SCM | 4-6 hrs | LOW | LOW |

**Recommended minimum:** Statistical tests + Ablations = 2-4 hours total

---

## What Reviewers Will Ask For

Based on typical ML conference reviews:

### Very Likely (>90% chance)
1. **Ablation studies** - "How do you know each component helps?"
   - Solution: Run `./run_ablations.sh`
   
2. **Statistical significance** - "Are improvements significant?"
   - Solution: Run `scripts/statistical_tests.py`

### Likely (50-70% chance)
3. **Error bars on learning curves** - "What's the variance across seeds?"
   - Solution: Generate multi-seed learning curve plots
   
4. **Hyperparameter sensitivity** - "Did you tune hyperparameters?"
   - Response: Document default choices, no tuning on test set

### Possible (20-40% chance)
5. **Larger scale experiments** - "Does it scale?"
   - Solution: Run 30-node SCM (if time permits)
   
6. **More baseline comparisons** - "What about method X?"
   - Response: We compare against 4 diverse baselines including RL

---

## Current Script Inventory

### Ready to Use
- `run_ablations.sh` - Run all ablation studies (NEW - I just created this)
- `scripts/statistical_tests.py` - Formal significance tests (NEW)
- `scripts/analyze_ablations.py` - Analyze ablation results (NEW)
- `scripts/compute_statistics.py` - Aggregate multi-seed stats
- `run_ace_only.sh` - Run ACE multi-seed (already used)

### Existing Scripts
- `ace_experiments.py` - Main ACE implementation
- `baselines.py` - All baseline methods
- `experiments/complex_scm.py` - 15-node benchmark
- `experiments/duffing_oscillators.py` - Physics domain
- `experiments/phillips_curve.py` - Economics domain

### May Need to Create
- `scripts/analyze_phillips.py` - Phillips detailed analysis
- `jobs/run_large_scale.sh` - 30-node SCM job script
- Multi-seed learning curve visualization

---

## Questions?

**Q: Are ablations absolutely necessary?**
A: Not to submit, but highly likely reviewers will ask. Better to have them ready.

**Q: Can I skip statistical tests?**
A: The paper claims "p<0.001" - you should verify this formally.

**Q: What if ablations show a component doesn't help?**
A: Honest reporting! If diversity reward doesn't help, that's a valid finding to discuss.

**Q: How long until paper is submittable?**
A: With current results: NOW (90% ready)
   With stats + ablations: 2-4 hours (98% ready)

---

## Bottom Line

**Minimal path to submission:**
```bash
# 5 minutes:
python scripts/statistical_tests.py \
  --ace results/ace_multi_seed_20260125_115453 \
  --baselines results/baselines/baselines_20260124_182827 \
  --output results/statistical_analysis.txt

# Add results to paper supplement
# SUBMIT!
```

**Recommended path (bulletproof for reviewers):**
```bash
# 5 minutes:
python scripts/statistical_tests.py [...args...]

# 2-4 hours on HPC:
./run_ablations.sh --seeds 3 --episodes 200

# 5 minutes:
python scripts/analyze_ablations.py results/ablations_*/ --latex

# Add ablation table to paper
# SUBMIT!
```

Your main results are excellent (55-58% improvement, 99.8% concentration).
The remaining experiments just add polish and preempt reviewer questions!
