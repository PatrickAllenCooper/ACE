#!/bin/bash

# ============================================================================
# ACE-Only Multi-Seed Experimental Runs
# ============================================================================
# Runs ONLY ACE experiments (not baselines) with multiple seeds
# for statistical validation.
#
# Baselines already complete from previous runs.
# This script focuses on ACE to compare against existing baseline data.
#
# Usage:
#   ./run_ace_only.sh [--test] [--seeds N]
#
# Options:
#   --test       Quick test mode (10 episodes, 1 seed)
#   --seeds N    Number of seeds (default: 5)
# ============================================================================

set -e

# Parse arguments
TEST_MODE=false
N_SEEDS=5

for arg in "$@"; do
    case $arg in
        --test)
            TEST_MODE=true
            ;;
        --seeds)
            shift
            N_SEEDS=$1
            ;;
    esac
done

# Configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

if [ "$TEST_MODE" = true ]; then
    EPISODES=10
    RUN_DIR="results/ace_test_${TIMESTAMP}"
    SEEDS=("42")
    echo "=========================================="
    echo "  ACE TEST MODE"
    echo "=========================================="
    echo "Episodes: $EPISODES (quick test)"
    echo "Seeds: 1 (seed 42 only)"
    echo "Output: $RUN_DIR"
else
    EPISODES=200
    RUN_DIR="results/ace_multi_seed_${TIMESTAMP}"
    
    case $N_SEEDS in
        3) SEEDS=("42" "123" "456") ;;
        5) SEEDS=("42" "123" "456" "789" "1011") ;;
        10) SEEDS=("42" "123" "456" "789" "1011" "314" "271" "577" "618" "141") ;;
        *) echo "Error: Supported seeds: 3, 5, 10"; exit 1 ;;
    esac
    
    echo "=========================================="
    echo "  ACE MULTI-SEED VALIDATION"
    echo "=========================================="
    echo "Episodes: $EPISODES"
    echo "Seeds: ${#SEEDS[@]} (${SEEDS[*]})"
    echo "Output: $RUN_DIR"
fi

mkdir -p "$RUN_DIR"
echo "Started: $(date)" >> "$RUN_DIR/run_info.txt"

# ============================================================================
# Submit ACE Jobs
# ============================================================================

declare -a JOB_IDS

for SEED in "${SEEDS[@]}"; do
    SEED_DIR="$RUN_DIR/seed_${SEED}"
    mkdir -p "$SEED_DIR"
    
    echo ""
    echo "Submitting ACE with seed $SEED..."
    
    if [ "$TEST_MODE" = true ]; then
        # Test mode: still submit to SLURM (needs GPU) but with shorter time limit
        echo "  Submitting test job (shorter time limit)..."
        JOB=$(sbatch --parsable \
            --nodes=1 --partition=aa100 --qos=normal \
            --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=2:00:00 \
            --job-name=ace_test_s${SEED} \
            --output=logs/ace_test_seed${SEED}_${TIMESTAMP}_%j.out \
            --error=logs/ace_test_seed${SEED}_${TIMESTAMP}_%j.err \
            --wrap="python ace_experiments.py \
                --episodes $EPISODES \
                --seed $SEED \
                --early_stopping \
                --use_per_node_convergence \
                --use_dedicated_root_learner \
                --obs_train_interval 3 \
                --obs_train_samples 200 \
                --obs_train_epochs 100 \
                --output $SEED_DIR")
        
        JOB_IDS+=($JOB)
        echo "  Job ID: $JOB"
        echo "  Monitor: tail -f logs/ace_test_seed${SEED}_${TIMESTAMP}_${JOB}.out"
        
    else
        # Production mode: submit to HPC
        JOB=$(sbatch --parsable \
            --nodes=1 --partition=aa100 --qos=normal \
            --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=10:00:00 \
            --job-name=ace_s${SEED} \
            --output=logs/ace_seed${SEED}_${TIMESTAMP}_%j.out \
            --error=logs/ace_seed${TIMESTAMP}_%j.err \
            --wrap="python ace_experiments.py \
                --episodes $EPISODES \
                --seed $SEED \
                --early_stopping \
                --use_per_node_convergence \
                --use_dedicated_root_learner \
                --obs_train_interval 3 \
                --obs_train_samples 200 \
                --obs_train_epochs 100 \
                --output $SEED_DIR")
        
        JOB_IDS+=($JOB)
        echo "  Job ID: $JOB"
        
        # Save job info
        echo "Seed $SEED: Job $JOB" >> "$RUN_DIR/run_info.txt"
    fi
done

echo ""
echo "=========================================="

if [ "$TEST_MODE" = true ]; then
    echo "  TEST COMPLETE"
    echo "=========================================="
    echo ""
    echo "Review test output:"
    echo "  cat $RUN_DIR/seed_42/test_output.log"
    echo ""
    echo "Check results:"
    echo "  head -20 $RUN_DIR/seed_42/node_losses.csv"
    echo "  tail -5 $RUN_DIR/seed_42/node_losses.csv"
    echo ""
    echo "If test looks good, run full experiment:"
    echo "  ./run_ace_only.sh --seeds 5"
    
else
    echo "  ACE MULTI-SEED SUBMITTED"
    echo "=========================================="
    echo ""
    echo "Jobs: ${JOB_IDS[*]}"
    echo "Seeds: ${SEEDS[*]}"
    echo "Results: $RUN_DIR"
    echo ""
    echo "Monitor: squeue -j $(IFS=,; echo "${JOB_IDS[*]}")"
    echo ""
    echo "After completion, analyze with:"
    echo "  python scripts/compute_statistics.py $RUN_DIR ace"
    echo "  python scripts/statistical_tests.py $RUN_DIR"
fi

echo ""
cat > "$RUN_DIR/README.txt" <<EOF
ACE Multi-Seed Run
==================
Started: $(date)
Seeds: ${SEEDS[*]}
Episodes: $EPISODES
Output: $RUN_DIR

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
python scripts/compute_statistics.py $RUN_DIR ace
python scripts/statistical_tests.py $RUN_DIR baselines

Compare against:
----------------
Baseline results (N=5, 100 episodes):
  Max-Variance: 2.05 ± 0.12 (BEST)
  PPO: 2.11 ± 0.13
  Round-Robin: 2.15 ± 0.08
  Random: 2.18 ± 0.06
EOF

echo "Run info saved to: $RUN_DIR/README.txt"
