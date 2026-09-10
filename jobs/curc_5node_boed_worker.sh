#!/bin/bash
# 5-node Bayesian-OED worker (Table 1 row) -- OUT via env
# Mirrors jobs/curc_cpu_suite.sh's --bayesian-baseline call; since the Sept 2026
# audit run_reviewer_experiments.py records both evaluators in
# bayesian_oed_summary.csv (total_loss = observational, ace_total_loss = ACE's).
source /projects/paco0228/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate ace 2>/dev/null || true
export MPLCONFIGDIR="/projects/paco0228/cache/matplotlib"; mkdir -p "$MPLCONFIGDIR"
cd /projects/paco0228/ACE
echo "5-node Bayesian OED started at $(date)"
python -u scripts/runners/run_reviewer_experiments.py \
    --bayesian-baseline \
    --seeds 42 123 456 789 1011 314 271 577 618 141 \
    --episodes 171 \
    --output "$OUT/suite"
echo "5-node Bayesian OED finished at $(date)"
