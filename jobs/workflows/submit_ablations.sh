#!/bin/bash
#
# Submit all ablation studies as separate SLURM jobs
#
# Usage:
#   ./submit_ablations.sh --seeds 3
#   ./submit_ablations.sh --seeds 5 --episodes 200
#

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
        --steps)
            STEPS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --seeds N --episodes M --steps K"
            exit 1
            ;;
    esac
done

echo "=================================================="
echo "Submitting ACE Ablation Studies"
echo "=================================================="
echo "Timestamp: $TIMESTAMP"
echo "Seeds: $SEEDS"
echo "Episodes: $EPISODES"
echo "Steps per episode: $STEPS"
echo ""

# Array of seed values
SEED_ARRAY=(42 123 456 789 1011)

# Array of ablations
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
    echo "=================================================="
    echo "Submitting: $ablation_name"
    echo "=================================================="
    
    for i in $(seq 0 $((SEEDS-1))); do
        seed=${SEED_ARRAY[$i]}
        
        # Submit job
        job_id=$(sbatch \
            --output="logs/ablation_${ablation}_${seed}_${TIMESTAMP}_%j.out" \
            --error="logs/ablation_${ablation}_${seed}_${TIMESTAMP}_%j.err" \
            --export=ALL,ABLATION="$ablation",SEED="$seed",EPISODES="$EPISODES",STEPS="$STEPS",OUTPUT_DIR="results/ablations_${TIMESTAMP}/${ablation}/seed_${seed}" \
            jobs/run_ablations.sh | awk '{print $NF}')
        
        JOB_IDS+=($job_id)
        
        echo "  Seed $seed: Job $job_id"
    done
done

echo ""
echo "=================================================="
echo "SUBMISSION COMPLETE"
echo "=================================================="
echo "Total jobs submitted: ${#JOB_IDS[@]}"
echo "Job IDs: ${JOB_IDS[@]}"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f logs/ablation_*_${TIMESTAMP}_*.out"
echo ""
echo "After completion, analyze with:"
echo "  python scripts/analyze_ablations.py results/ablations_${TIMESTAMP}/ --latex"
echo ""
echo "Expected completion: 2-4 hours"
echo ""
