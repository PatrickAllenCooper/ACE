#!/bin/bash

# Submit verified ablations (3 types × 3 seeds each)
# Each type runs as separate job to maximize parallelism

echo "=============================================="
echo "Submitting Verified Ablation Jobs (Parallel)"
echo "=============================================="

# Submit each ablation type as separate job
for ABLATION in no_convergence no_root_learner no_diversity; do
    echo "Submitting: $ABLATION"
    sbatch --export=ALL,ABLATION=$ABLATION jobs/run_ablations_verified.sh
done

echo ""
echo "=============================================="
echo "Submitted 3 ablation jobs (3 seeds each)"
echo "Monitor with: squeue -u \$USER"
echo "Expected runtime: ~3 hours per job"
echo "=============================================="
