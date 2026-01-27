#!/bin/bash
# Submit experiments to reach STRONG ACCEPT tier
# Run these in parallel for faster completion

echo "=============================================="
echo "Submitting STRONG ACCEPT Experiments"
echo "=============================================="

cd "$(dirname "$0")/../.."

echo "Job 1: Remaining ablations (3 types × 3 seeds)"
JOB1=$(sbatch jobs/run_remaining_ablations.sh | awk '{print $4}')
echo "  Submitted: $JOB1"

echo "Job 2: ACE without oracle (N=5 seeds)"
JOB2=$(sbatch jobs/run_ace_no_oracle.sh | awk '{print $4}')
echo "  Submitted: $JOB2"

echo ""
echo "=============================================="
echo "Submitted 2 jobs (will run in parallel)"
echo "=============================================="
echo "Ablations: $JOB1 (8h limit, ~6h expected)"
echo "No Oracle: $JOB2 (8h limit, ~5-6h expected)"
echo ""
echo "Monitor:"
echo "  squeue -j $JOB1,$JOB2"
echo "  watch -n 60 'squeue -u \$USER'"
echo ""
echo "Expected completion: 6-8 hours"
echo "=============================================="
