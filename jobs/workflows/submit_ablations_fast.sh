#!/bin/bash
# Submit all fast ablation jobs in parallel
# Each job runs one ablation type across 3 seeds
# 4 jobs total, expected 1-2 hours each

echo "=============================================="
echo "Submitting Fast Ablation Jobs (Parallel)"
echo "=============================================="

cd "$(dirname "$0")/../.."

# Submit each ablation as separate job
ABLATIONS=("no_dpo" "no_convergence" "no_root_learner" "no_diversity")

for abl in "${ABLATIONS[@]}"; do
    echo "Submitting: $abl"
    ABLATION=$abl sbatch jobs/run_ablations_fast.sh
done

echo ""
echo "=============================================="
echo "Submitted 4 ablation jobs"
echo "Monitor with: squeue -u \$USER"
echo "=============================================="
