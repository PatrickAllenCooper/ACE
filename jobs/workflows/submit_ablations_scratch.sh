#!/bin/bash
#
# Submit ablation studies using SCRATCH storage
# Much more space available on scratch (9TB vs 250GB on projects)
#
# Usage:
#   bash submit_ablations_scratch.sh --seeds 3
#

set -e

# Default parameters
SEEDS=3
EPISODES=200
STEPS=25
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --seeds)
            SEEDS="$2"
            shift 2
            ;;
        --episodes)
            EPISODES="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=================================================="
echo "Submitting ACE Ablation Studies (SCRATCH mode)"
echo "=================================================="
echo "Timestamp: $TIMESTAMP"
echo "Seeds: $SEEDS"
echo "Episodes: $EPISODES"
echo ""
echo "NOTE: Jobs will use /scratch for intermediate files"
echo "      Only final results copied to /projects"
echo "      This saves ~80% disk space"
echo ""

# Seed array
SEED_ARRAY=(42 123 456 789 1011)

# Ablations
ABLATIONS=(no_dpo no_convergence no_root_learner no_diversity)
ABLATION_NAMES=(
    "No DPO (Custom Policy)"
    "No Per-Node Convergence"
    "No Dedicated Root Learner"
    "No Diversity Reward"
)

# Submit all jobs
JOB_IDS=()

for ablation_idx in "${!ABLATIONS[@]}"; do
    ablation="${ABLATIONS[$ablation_idx]}"
    ablation_name="${ABLATION_NAMES[$ablation_idx]}"
    
    echo ""
    echo "Submitting: $ablation_name"
    
    for i in $(seq 0 $((SEEDS-1))); do
        seed=${SEED_ARRAY[$i]}
        
        job_id=$(sbatch \
            --export=ALL,ABLATION="$ablation",SEED="$seed",EPISODES="$EPISODES",STEPS="$STEPS",TIMESTAMP="$TIMESTAMP" \
            jobs/run_ablations_scratch.sh | awk '{print $NF}')
        
        JOB_IDS+=($job_id)
        echo "  Seed $seed: Job $job_id"
    done
done

echo ""
echo "=================================================="
echo "SUBMISSION COMPLETE"
echo "=================================================="
echo "Total jobs: ${#JOB_IDS[@]}"
echo "Job IDs: ${JOB_IDS[@]}"
echo ""
echo "Results will be in: results/ablations_${TIMESTAMP}/"
echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  tail -f logs/ace_ablation_*.out"
echo ""
echo "After completion:"
echo "  python scripts/analyze_ablations.py results/ablations_${TIMESTAMP}/ --latex"
echo ""
echo "Disk usage: Minimal (~2-3GB total in /projects)"
echo "             All temp files on /scratch (auto-cleaned)"
echo ""
