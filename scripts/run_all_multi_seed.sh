#!/bin/bash

# ============================================================================
# Multi-Seed Experimental Run for Statistical Validation
# ============================================================================
# Runs all experiments 5 times with different random seeds to enable
# statistical analysis (mean, std, confidence intervals, significance tests).
#
# Usage:
#   ./scripts/run_all_multi_seed.sh
#
# Or with custom seeds:
#   SEEDS="42 100 200 300 400" ./scripts/run_all_multi_seed.sh
# ============================================================================

set -e

# Configuration
SEEDS=${SEEDS:-"42 123 456 789 1011"}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MULTI_RUN_DIR="results/multi_run_${TIMESTAMP}"
mkdir -p "$MULTI_RUN_DIR"

echo "========================================"
echo "ACE Multi-Seed Experimental Runs"
echo "========================================"
echo "Base directory: $MULTI_RUN_DIR"
echo "Seeds: $SEEDS"
echo "Started: $(date)"
echo ""

# Track job IDs for all runs
declare -a ALL_JOB_IDS

# Submit jobs for each seed
SEED_NUM=1
for SEED in $SEEDS; do
    echo "Submitting run $SEED_NUM/5 (seed=$SEED)..."
    
    # Set output directory for this seed
    SEED_OUTPUT="$MULTI_RUN_DIR/seed_${SEED}"
    mkdir -p "$SEED_OUTPUT"
    
    # Submit all 5 jobs with this seed
    # Modified to pass RANDOM_SEED to job scripts
    
    # Job 1: ACE Main
    JOB1=$(sbatch --parsable \
        --output=logs/ace_main_seed${SEED}_${TIMESTAMP}_%j.out \
        --error=logs/ace_main_seed${SEED}_${TIMESTAMP}_%j.err \
        --export=ALL,EPISODES=200,OUTPUT_DIR=$SEED_OUTPUT/ace,RANDOM_SEED=$SEED \
        jobs/run_ace_main.sh)
    ALL_JOB_IDS+=($JOB1)
    echo "  ACE Main: $JOB1"
    
    # Job 2: Baselines
    JOB2=$(sbatch --parsable \
        --output=logs/baselines_seed${SEED}_${TIMESTAMP}_%j.out \
        --error=logs/baselines_seed${SEED}_${TIMESTAMP}_%j.err \
        --export=ALL,EPISODES=100,OUTPUT_DIR=$SEED_OUTPUT/baselines,RANDOM_SEED=$SEED \
        jobs/run_baselines.sh)
    ALL_JOB_IDS+=($JOB2)
    echo "  Baselines: $JOB2"
    
    # Job 3: Complex SCM
    JOB3=$(sbatch --parsable \
        --output=logs/complex_seed${SEED}_${TIMESTAMP}_%j.out \
        --error=logs/complex_seed${SEED}_${TIMESTAMP}_%j.err \
        --export=ALL,EPISODES=100,OUTPUT_DIR=$SEED_OUTPUT/complex_scm,RANDOM_SEED=$SEED \
        jobs/run_complex_scm.sh)
    ALL_JOB_IDS+=($JOB3)
    echo "  Complex SCM: $JOB3"
    
    # Job 4: Duffing
    JOB4=$(sbatch --parsable \
        --output=logs/duffing_seed${SEED}_${TIMESTAMP}_%j.out \
        --error=logs/duffing_seed${SEED}_${TIMESTAMP}_%j.err \
        --export=ALL,EPISODES=100,OUTPUT_DIR=$SEED_OUTPUT/duffing,RANDOM_SEED=$SEED \
        jobs/run_duffing.sh)
    ALL_JOB_IDS+=($JOB4)
    echo "  Duffing: $JOB4"
    
    # Job 5: Phillips
    JOB5=$(sbatch --parsable \
        --output=logs/phillips_seed${SEED}_${TIMESTAMP}_%j.out \
        --error=logs/phillips_seed${SEED}_${TIMESTAMP}_%j.err \
        --export=ALL,EPISODES=50,OUTPUT_DIR=$SEED_OUTPUT/phillips,RANDOM_SEED=$SEED \
        jobs/run_phillips.sh)
    ALL_JOB_IDS+=($JOB5)
    echo "  Phillips: $JOB5"
    
    echo ""
    ((SEED_NUM++))
done

echo "========================================"
echo "All Multi-Seed Jobs Submitted!"
echo "========================================"
echo "Total jobs submitted: ${#ALL_JOB_IDS[@]}"
echo "Multi-run directory: $MULTI_RUN_DIR"
echo ""
echo "Monitor all jobs:"
echo "  squeue -j $(IFS=,; echo "${ALL_JOB_IDS[*]}")"
echo ""
echo "After all complete, consolidate results:"
echo "  ./scripts/consolidate_multi_runs.sh $MULTI_RUN_DIR"
echo ""

# Save run info
cat > "$MULTI_RUN_DIR/multi_run_info.txt" <<EOF
Multi-Seed Experimental Run
===========================
Submitted: $(date)
Base Directory: $MULTI_RUN_DIR
Seeds: $SEEDS
Total Jobs: ${#ALL_JOB_IDS[@]}

Job IDs:
$(for id in "${ALL_JOB_IDS[@]}"; do echo "  $id"; done)

Monitor: squeue -j $(IFS=,; echo "${ALL_JOB_IDS[*]}")

Consolidate After Completion:
  ./scripts/consolidate_multi_runs.sh $MULTI_RUN_DIR
EOF

echo "Run info saved to: $MULTI_RUN_DIR/multi_run_info.txt"
