# Quick Start - Improved ACE (January 20, 2026)

## 🚀 Ready-to-Run Commands

### Quick Test (30 minutes)
```bash
cd /Users/patrickcooper/code/ACE

python ace_experiments.py \
  --episodes 10 \
  --steps 25 \
  --early_stopping \
  --root_fitting \
  --output results/quick_test_jan20
```

**Expected output:**
- Runs in ~20-30 minutes
- Shows "TRAINING IMPROVEMENTS ENABLED" in logs
- May trigger early stopping if converges fast

---

### Recommended Full Run (1-2 hours)
```bash
cd /Users/patrickcooper/code/ACE

python ace_experiments.py \
  --episodes 200 \
  --steps 25 \
  --early_stopping \
  --early_stop_patience 20 \
  --obs_train_interval 3 \
  --obs_train_samples 200 \
  --obs_train_epochs 100 \
  --root_fitting \
  --root_fit_interval 5 \
  --undersampled_bonus 200.0 \
  --diversity_reward_weight 0.3 \
  --max_concentration 0.5 \
  --update_reference_interval 25 \
  --pretrain_steps 200 \
  --pretrain_interval 25 \
  --smart_breaker \
  --output results/ace_improved_jan20
```

**Expected output:**
- Runtime: 1-2 hours (vs 9h baseline)
- Early stopping may trigger before 200 episodes
- All nodes should learn (loss < 0.5)

---

### Minimal (Backwards Compatible)
```bash
python ace_experiments.py --episodes 200
```
Note: Uses improved defaults but no early stopping

---

## 📊 What to Check After Run

### 1. Check if early stopping triggered:
```bash
tail -100 results/ace_improved_jan20/run_*/experiment.log | grep "Early stopping"
```

### 2. Check final losses:
```bash
tail -1 results/ace_improved_jan20/run_*/node_losses.csv
```
**Target:** X1, X4 < 0.5

### 3. Check intervention distribution:
```bash
tail -20 results/ace_improved_jan20/run_*/experiment.log | grep "Final Target Distribution"
```
**Target:** No node > 60%

### 4. Visualize:
```bash
python visualize.py results/ace_improved_jan20/run_*/
```

---

## 🔍 Monitoring During Run

### Watch logs in real-time:
```bash
tail -f results/ace_improved_jan20/run_*/experiment.log
```

### Look for:
- ✓ "TRAINING IMPROVEMENTS ENABLED" at start
- ✓ "[Root Fitting]" messages every 5 episodes
- ✓ "[Ref Update]" messages every 25 episodes  
- ✓ "[Early Stop Monitor]" zero-reward percentage
- ✓ "Early stopping triggered" when converged

---

## ⚡ What Changed

### Fixes Implemented:
1. **Early Stopping** - Saves 80% compute time
2. **3x Observational Training** - Fixes X1/X4 learning
3. **Root Fitting** - Explicit root distribution learning
4. **Diversity Penalties** - Prevents 99% X2 collapse
5. **Reference Updates** - Stabilizes KL divergence

### Expected Improvements:
| Metric | Before | After |
|--------|--------|-------|
| Runtime | 9h | 1-2h |
| X1 Loss | 0.88 | <0.3 |
| X4 Loss | 0.94 | <0.3 |
| X2 Concentration | 69% | <50% |

---

## 📝 Files to Review After Run

### Results Directory Structure:
```
results/ace_improved_jan20/
└── run_YYYYMMDD_HHMMSS/
    ├── experiment.log           # Main log - CHECK THIS FIRST
    ├── node_losses.csv          # Per-node losses over time
    ├── metrics.csv              # All interventions
    ├── dpo_training.csv         # DPO training details
    ├── mechanism_contrast.png   # Learned vs true mechanisms
    ├── training_curves.png      # Loss/reward over time
    ├── strategy_analysis.png    # Intervention distribution
    └── scm_graph.png            # Causal graph with losses
```

### Key Files to Check:
1. **experiment.log** - Did early stopping work? Were improvements active?
2. **node_losses.csv** - Did X1/X4 improve over time?
3. **metrics.csv** - Is intervention distribution balanced?
4. **mechanism_contrast.png** - Do learned mechanisms match ground truth?

---

## ❓ Troubleshooting

### Early stopping triggered too soon?
Add: `--early_stop_patience 30`

### Root nodes still not learning?
Add: `--obs_train_interval 2 --root_fit_interval 3`

### Policy still collapsing to X2?
Add: `--undersampled_bonus 300.0 --max_concentration 0.4`

### Want to see more logs?
Add: `--debug_parsing` (verbose logging)

---

## 📚 Documentation

- **`IMPLEMENTATION_SUMMARY.md`** - Executive summary of changes
- **`CODE_IMPROVEMENTS_IMPLEMENTED.md`** - Technical details
- **`COMPREHENSIVE_TRAINING_ANALYSIS_Jan19_2026.md`** - Why these changes were needed
- **`guidance_documents/guidance_doc.txt`** - Updated with new parameters

---

**Status:** ✅ Ready to run  
**Last Updated:** January 20, 2026  
**Next Step:** Run quick test above
