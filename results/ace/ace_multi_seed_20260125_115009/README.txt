ACE Multi-Seed Run
==================
Started: Sun Jan 25 11:50:10 MST 2026
Seeds: 42 123 456 789 1011
Episodes: 200
Output: results/ace_multi_seed_20260125_115009

Configuration:
--------------
- Early stopping: ENABLED
- Per-node convergence: ENABLED  
- Dedicated root learner: ENABLED
- Obs train interval: 3 (every 3 steps)
- Obs train samples: 200
- Obs train epochs: 100

Expected:
---------
- ACE should stop at 40-60 episodes (early stopping)
- Final loss should be competitive with Max-Variance (2.05)
- Per-node losses should all converge

To analyze:
-----------
python scripts/compute_statistics.py results/ace_multi_seed_20260125_115009 ace
python scripts/statistical_tests.py results/ace_multi_seed_20260125_115009 baselines

Compare against:
----------------
Baseline results (N=5, 100 episodes):
  Max-Variance: 2.05 ± 0.12 (BEST)
  PPO: 2.11 ± 0.13
  Round-Robin: 2.15 ± 0.08
  Random: 2.18 ± 0.06
