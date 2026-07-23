#!/bin/bash
# =============================================================================
# Bayesian OED baseline at 30 nodes (5 seeds).
#
# Fixes the contradiction reviewer F4Cb caught: the NeurIPS 2026 submission
# claimed Bayesian OED "does not scale to 30 nodes within our budget" while
# its own appendix reported OED at 4-6 min/step on CPU vs ACE at 22 min/step
# on an A100. scripts/runners/run_30node_baseline_seed.py already implements
# BayesianOEDFast for this scale (reduced candidate/MC-sample search); this
# script just submits it with the same 5-seed set used for the other 30-node
# baselines so the comparison in the paper is no longer selectively omitted.
#
# Usage (from /projects/paco0228/ACE):
#   cd /projects/paco0228/ACE
#   git pull
#   bash jobs/curc_submit_30node_bayesian_oed.sh
#
# Output: results/curc_30node_baselines/bayesian_oed/seed_{seed}/
#   (co-located with random/round_robin/max_variance so
#    aggregate_30node_baselines.py picks it up directly)
#
# Each run also writes query_budget.json with the candidate_probe/executed
# breakdown -- BayesianOEDFast queries the oracle n_candidates x n_mc_samples
# times per step to score candidates (the same lookahead/execution asymmetry
# raised against ACE), so this also produces the OED-side budget numbers
# needed for the paper's accounting section.
#
# SLURM resources per job (CPU only, no GPU needed):
#   partition : amilan
#   time      : 10:00:00 (OED's own candidate-probe cost makes each step
#               slower than the other CPU baselines; budget generously)
# =============================================================================

set -euo pipefail

cd /projects/paco0228/ACE

source /projects/paco0228/miniconda3/etc/profile.d/conda.sh
conda activate ace

OUT="/scratch/alpine1/paco0228/ACE/results/curc_30node_baselines"
mkdir -p "$OUT/logs"

echo "================================================================"
echo " Bayesian OED @ 30 nodes -- CURC SLURM (5 jobs)"
echo "================================================================"
echo " Output : $OUT"
echo " Started: $(date)"
echo "================================================================"

SEEDS="42 123 456 789 1011"

for SEED in $SEEDS; do
    JOB=$(sbatch --parsable \
        --job-name="oed30_s${SEED}" \
        --partition=amilan \
        --qos=normal \
        --nodes=1 --ntasks=1 \
        --cpus-per-task=4 \
        --mem=8G \
        --time=10:00:00 \
        --output="$OUT/logs/bayesian_oed_seed${SEED}_%j.out" \
        --error="$OUT/logs/bayesian_oed_seed${SEED}_%j.err" \
        --export=ALL,METHOD=bayesian_oed,SEED=$SEED,OUT=$OUT \
        jobs/curc_30node_baseline_seed.sh)
    echo "  Submitted: Bayesian-OED seed=$SEED -> Job $JOB"
done

echo ""
echo "5 jobs submitted."
echo "Monitor with:  squeue -u \$USER"
echo "Logs in:       $OUT/logs/"
echo ""
echo "When complete, pull results locally with:"
echo "  scp -r paco0228@login.rc.colorado.edu:$OUT ./results/"
echo ""
echo "Then re-run: python scripts/runners/aggregate_30node_baselines.py"
