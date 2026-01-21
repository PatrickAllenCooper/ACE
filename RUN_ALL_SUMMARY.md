# run_all.sh Summary
## Complete Paper Experiments with All Latest Fixes

---

## ✅ What `./run_all.sh` Does

**Submits 5 SLURM jobs in parallel:**

1. **ACE Main** → `jobs/run_ace_main.sh`
   - Uses: All Jan 21 fixes + observational training
   - Runs: 200 episodes (early stop expected at 40-80)
   - Time: 4-6 hours
   - Output: `results/paper_TIMESTAMP/ace/`

2. **Baselines** → `jobs/run_baselines.sh`
   - Includes: Random, Round-Robin, Max-Variance, PPO
   - Uses: PPO bug fix + observational training
   - Runs: 100 episodes each
   - Time: 2-3 hours
   - Output: `results/paper_TIMESTAMP/baselines/`

3. **Complex 15-Node SCM** → `jobs/run_complex_scm.sh`
   - Policies: random, smart_random, greedy_collider
   - Runs: 200 episodes each (3 policies)
   - Time: 8-12 hours total
   - Output: `results/paper_TIMESTAMP/complex_scm/`

4. **Duffing Oscillators** → `jobs/run_duffing.sh`
   - Random intervention policy
   - Runs: 100 episodes
   - Time: <1 minute
   - Output: `results/paper_TIMESTAMP/duffing/`

5. **Phillips Curve** → `jobs/run_phillips.sh`
   - Hardcoded regime selection
   - Runs: 100 episodes
   - Time: ~30 seconds
   - Output: `results/paper_TIMESTAMP/phillips/`

---

## 🎯 All Latest Fixes Included

### **ACE Main Gets:**
- ✅ Adaptive diversity threshold
- ✅ Value novelty bonus
- ✅ Emergency retraining
- ✅ Dynamic candidate reduction
- ✅ Improved early stopping
- ✅ **Observational training** (every 3 steps)
- ✅ Dedicated root learner (every 3 episodes)

### **Baselines Get:**
- ✅ PPO bug fix
- ✅ Observational training (every 3 steps)
- ✅ All 4 methods run

### **Other Experiments Get:**
- ✅ All work as designed
- ✅ No changes needed

---

## 🚀 How to Use

### **Full Paper Experiments (Recommended):**
```bash
./run_all.sh
```
**Time:** 8-12 hours total (jobs run in parallel)  
**Output:** All results in `results/paper_TIMESTAMP/`

---

### **Quick Validation (10 episodes):**
```bash
QUICK=true ./run_all.sh
```
**Time:** ~30 minutes  
**Use:** Verify everything works before full run

---

### **Monitor Progress:**
```bash
# Check job queue
squeue -u $USER

# Watch ACE training
tail -f logs/ace_main_*.out

# Check all logs
ls -lth logs/*.out | head -10
```

---

### **After Completion:**
```bash
# Verify claims
./verify_claims.sh

# Extract metrics
./extract_ace.sh
python compare_methods.py

# Document
code results/RESULTS_LOG.md

# Fill paper tables
code paper/paper.tex
```

---

## 📊 Expected Output Structure

```
results/paper_20260121_HHMMSS/
├── ace/
│   ├── experiment.log
│   ├── metrics.csv
│   ├── node_losses.csv
│   ├── learning_curves.png
│   └── ...
├── baselines/
│   └── baselines_TIMESTAMP/
│       ├── results.csv
│       ├── learning_curves.png
│       └── ...
├── complex_scm/
│   ├── complex_scm_random_TIMESTAMP/
│   ├── complex_scm_smart_random_TIMESTAMP/
│   └── complex_scm_greedy_collider_TIMESTAMP/
├── duffing/
│   └── duffing_TIMESTAMP/
├── phillips/
│   └── phillips_TIMESTAMP/
└── job_info.txt
```

---

## ⏰ Timeline

**Parallel execution (all jobs run simultaneously):**
- Duffing: <1 min
- Phillips: ~30 sec
- Baselines: 2-3 hours
- ACE Main: 4-6 hours
- Complex SCM: 8-12 hours

**Total wall time:** ~12 hours (longest job)  
**With early stopping:** ~6-8 hours likely

---

## ✅ What's Included

| Job | Script | Latest Fixes | Status |
|-----|--------|--------------|--------|
| ACE | run_ace_main.sh | ✅ All 8 fixes | Ready |
| Baselines | run_baselines.sh | ✅ PPO fix + obs | Ready |
| Complex | run_complex_scm.sh | ✅ Works as-is | Ready |
| Duffing | run_duffing.sh | ✅ Works as-is | Ready |
| Phillips | run_phillips.sh | ✅ Works as-is | Ready |

---

## 🎯 Recommendation

**Use `./run_all.sh` to:**
- Get complete paper results in one command
- Run all experiments with latest fixes
- Parallel execution (saves time)
- Organized output structure

**Alternative (if testing first):**
```bash
# Test ACE fixes first
./pipeline_test.sh

# Then launch all
./run_all.sh
```

---

**Bottom line:** `./run_all.sh` is ready to go with all latest improvements! 🚀
