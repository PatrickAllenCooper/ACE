# Plan to Reach STRONG ACCEPT

**Current Status:** Paper is ACCEPT-competitive after complex SCM completes
**Goal:** Address remaining weaknesses for STRONG ACCEPT

---

## What We Have (After Complex SCM)

✅ **Core Results:**
- ACE: 0.92 ± 0.73 (median 0.61, N=5 seeds)
- Extended baselines (171 ep): 2.03-2.10
- 70-71% improvement at matched budget
- Statistical significance: p<0.001, d≈2

✅ **Major Concerns Addressed:**
- Unfair budget: Extended baselines show improvement persists
- Lookahead confound: Random lookahead (2.10) shows DPO is key
- Statistical rigor: Paired t-tests, Bonferroni correction

✅ **Partial Ablations:**
- no_dpo: 2.12 ± 0.10 (130% degradation) - validates DPO contribution

---

## Remaining Weaknesses

❌ **Incomplete ablations:** Only 1/4 complete (no_dpo done, 3 pending)
❌ **Oracle pretraining:** 200 interventions in main results

---

## Experiments to Run

### Job 1: Complete Ablations (8h limit)

**What:** no_convergence, no_root_learner, no_diversity
**Seeds:** 3 each (42, 123, 456)
**Expected runtime:** ~6 hours
**Submit:** `sbatch jobs/run_remaining_ablations.sh`

**Validates:**
- Per-node convergence prevents premature termination
- Dedicated root learner enables stable root distribution learning
- Diversity reward prevents policy collapse

### Job 2: ACE Without Oracle (8h limit)

**What:** ACE with pretrain_steps=0
**Seeds:** 5 (42, 123, 456, 789, 1011)
**Expected runtime:** ~5-6 hours
**Submit:** `sbatch jobs/run_ace_no_oracle.sh`

**Eliminates:**
- Main reviewer objection (oracle pretraining)
- Use this as headline result if comparable to oracle version
- Move oracle-pretrained to supplementary

---

## Submit Both in Parallel

```bash
cd ~/ACE
git pull

# Submit both jobs (run in parallel)
bash jobs/workflows/submit_strong_accept_experiments.sh

# Monitor
watch -n 60 'squeue -u $USER'
```

**Expected completion:** 6-8 hours (parallel execution)

---

## Timeline

**If submitted now (assume ~1 PM MST):**
- Submit: 1 PM
- Complete: 7-9 PM (both jobs in parallel)
- Analyze & update paper: 2-3 hours
- **Paper ready:** Late evening/tomorrow morning

---

## Expected Results (if positive)

**With complete ablations:**
- Table shows all 4 components contribute
- Validates entire architecture
- No gaps in experimental validation

**With no-oracle results:**
- Headline becomes: "ACE without oracle: X.XX ± Y.YY"
- No reviewer objection about privileged info
- Strengthens "learned from scratch" narrative

**If both succeed:** Paper moves from ACCEPT to STRONG ACCEPT tier

---

## Risk Assessment

**Ablations:** Low risk
- no_dpo worked (2.12 ± 0.10)
- Other 3 should work similarly
- All have intermediate saves now

**No Oracle:** Medium risk
- Might perform worse without oracle warm-start
- If comparable to oracle version (0.92 ± 0.73): GREAT, use as main result
- If substantially worse: Still publishable, shows oracle contribution

---

## Fallback Plan

**If no-oracle performs poorly:**
- Keep oracle version as main result
- Report no-oracle in ablations
- Acknowledge limitation honestly
- Still have complete ablations → validates architecture

**If either job fails:**
- Use partial results (intermediate saves)
- Report: "N=2 seeds due to computational constraints"
- Still strengthens paper vs current state

---

## Commands

**Submit both experiments:**
```bash
cd ~/ACE
git pull
bash jobs/workflows/submit_strong_accept_experiments.sh
```

**Monitor progress:**
```bash
watch -n 60 'squeue -u $USER'
tail -f logs/ablations_final_*.out
tail -f logs/ace_no_oracle_*.out
```

**Check intermediate results:**
```bash
# Ablations
ls -lh results/ablations_complete/*/seed_*/run_*/metrics.csv

# No oracle
ls -lh results/ace_no_oracle/seed_*/run_*/node_losses.csv
```

---

## After Completion

1. Copy all results to local
2. Analyze no-oracle performance
3. Update paper:
   - Complete ablation table
   - Use no-oracle as main result (if comparable)
   - Update abstract/conclusion accordingly
4. Final proofread
5. Submit

**This completes the experimental work for STRONG ACCEPT.**
