#!/bin/bash
# Submit ablation jobs using simple direct execution (no Python wrapper)

echo "=============================================="
echo "Submitting Simple Ablation Jobs (Parallel)"
echo "=============================================="

cd "$(dirname "$0")/../.."

# Cancel old jobs if still queued
OLD_JOBS=(23319191 23319192 23319193 23319194)
for job in "${OLD_JOBS[@]}"; do
    scancel $job 2>/dev/null
done

echo "Cancelled old wrapper-based jobs"
echo ""

# Submit new simplified jobs
for abl in no_dpo no_convergence no_root_learner no_diversity; do
    echo "Submitting: $abl"
    ABLATION=$abl sbatch jobs/run_single_ablation.sh
done

echo ""
echo "=============================================="
echo "Submitted 4 ablation jobs (direct execution)"
echo "Monitor with: squeue -u \$USER"
echo "=============================================="
