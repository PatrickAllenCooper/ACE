#!/bin/bash

# ============================================================================
# ACE Paper Experiments - Job Orchestrator
# ============================================================================
# This script submits all paper experiments as separate SLURM jobs.
# DO NOT run this with 'sbatch' - just execute it directly: ./run_all.sh
#
# Usage:
#   ./run_all.sh                    # Submit all jobs
#   QUICK=true ./run_all.sh         # Quick validation run
#   ACE_EPISODES=200 ./run_all.sh   # Custom episode counts
# ============================================================================

set -e

# --- Configuration ---
EPISODES=${ACE_EPISODES:-500}
BASELINE_EPISODES=${BASELINE_EPISODES:-100}
QUICK=${QUICK:-false}

if [ "$QUICK" = "true" ]; then
    EPISODES=10
    BASELINE_EPISODES=10
fi

# --- Setup ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT="results/paper_${TIMESTAMP}"
mkdir -p logs "$OUT"

echo "========================================"
echo "ACE Paper Experiments - Job Submission"
echo "========================================"
echo "Timestamp: $TIMESTAMP"
echo "Output: $OUT"
echo "ACE Episodes: $EPISODES"
echo "Baseline Episodes: $BASELINE_EPISODES"
echo ""

# Check if we're on a login node (good) or compute node (bad)
if [ -n "$SLURM_JOB_ID" ]; then
    echo "ERROR: This script should NOT be run via sbatch!"
    echo "Run it directly on the login node: ./run_all.sh"
    exit 1
fi

# Check if sbatch is available
if ! command -v sbatch &> /dev/null; then
    echo "ERROR: sbatch command not found. Are you on an HPC cluster?"
    exit 1
fi

# Check if job scripts exist
if [ ! -f "jobs/run_ace_main.sh" ]; then
    echo "ERROR: Job scripts not found in jobs/ directory"
    exit 1
fi

echo "Submitting jobs..."
echo ""

# Array to track job IDs
declare -a JOB_IDS
declare -a JOB_NAMES

# --- Job 1: ACE Main Experiment ---
JOB1=$(sbatch --parsable \
    --output=logs/ace_main_${TIMESTAMP}_%j.out \
    --error=logs/ace_main_${TIMESTAMP}_%j.err \
    --export=ALL,EPISODES=$EPISODES,OUTPUT_DIR=$OUT/ace \
    jobs/run_ace_main.sh)

JOB_IDS+=($JOB1)
JOB_NAMES+=("ACE_Main")
echo "[1/4] ACE Main Experiment"
echo "      Job ID: $JOB1"
echo "      Script: jobs/run_ace_main.sh"
echo "      Output: logs/ace_main_${TIMESTAMP}_${JOB1}.out"
echo ""

# --- Job 2: Baselines ---
JOB2=$(sbatch --parsable \
    --output=logs/baselines_${TIMESTAMP}_%j.out \
    --error=logs/baselines_${TIMESTAMP}_%j.err \
    --export=ALL,EPISODES=$BASELINE_EPISODES,OUTPUT_DIR=$OUT/baselines \
    jobs/run_baselines.sh)

JOB_IDS+=($JOB2)
JOB_NAMES+=("Baselines")
echo "[2/4] Baselines (Random, Round-Robin, Max-Variance, PPO)"
echo "      Job ID: $JOB2"
echo "      Script: jobs/run_baselines.sh"
echo "      Output: logs/baselines_${TIMESTAMP}_${JOB2}.out"
echo ""

# --- Job 3: Duffing Oscillators ---
JOB3=$(sbatch --parsable \
    --output=logs/duffing_${TIMESTAMP}_%j.out \
    --error=logs/duffing_${TIMESTAMP}_%j.err \
    --export=ALL,EPISODES=$BASELINE_EPISODES,OUTPUT_DIR=$OUT/duffing \
    jobs/run_duffing.sh)

JOB_IDS+=($JOB3)
JOB_NAMES+=("Duffing")
echo "[3/4] Duffing Oscillators (Physics)"
echo "      Job ID: $JOB3"
echo "      Script: jobs/run_duffing.sh"
echo "      Output: logs/duffing_${TIMESTAMP}_${JOB3}.out"
echo ""

# --- Job 4: Phillips Curve ---
JOB4=$(sbatch --parsable \
    --output=logs/phillips_${TIMESTAMP}_%j.out \
    --error=logs/phillips_${TIMESTAMP}_%j.err \
    --export=ALL,EPISODES=$BASELINE_EPISODES,OUTPUT_DIR=$OUT/phillips \
    jobs/run_phillips.sh)

JOB_IDS+=($JOB4)
JOB_NAMES+=("Phillips")
echo "[4/4] Phillips Curve (Economics)"
echo "      Job ID: $JOB4"
echo "      Script: jobs/run_phillips.sh"
echo "      Output: logs/phillips_${TIMESTAMP}_${JOB4}.out"
echo ""

# --- Summary ---
echo "========================================"
echo "All jobs submitted successfully!"
echo "========================================"
echo "Results directory: $OUT"
echo ""
echo "Job Summary:"
for i in "${!JOB_IDS[@]}"; do
    echo "  ${JOB_NAMES[$i]}: ${JOB_IDS[$i]}"
done

echo ""
echo "Monitor jobs:"
echo "  squeue -u \$USER"
echo "  squeue -j ${JOB_IDS[*]}"
echo ""
echo "View logs (live):"
echo "  tail -f logs/ace_main_${TIMESTAMP}_${JOB1}.out"
echo "  tail -f logs/baselines_${TIMESTAMP}_${JOB2}.out"
echo ""
echo "Cancel all jobs:"
echo "  scancel ${JOB_IDS[*]}"
echo "========================================"

# Save job info for later reference
cat > "$OUT/job_info.txt" <<EOF
ACE Paper Experiments
Submitted: $(date)
Timestamp: $TIMESTAMP
Output: $OUT

Job IDs and Scripts:
  ACE Main:    $JOB1  (jobs/run_ace_main.sh)
  Baselines:   $JOB2  (jobs/run_baselines.sh)
  Duffing:     $JOB3  (jobs/run_duffing.sh)
  Phillips:    $JOB4  (jobs/run_phillips.sh)

Monitor: squeue -j ${JOB_IDS[*]}
Cancel:  scancel ${JOB_IDS[*]}

Logs:
  logs/ace_main_${TIMESTAMP}_${JOB1}.out
  logs/baselines_${TIMESTAMP}_${JOB2}.out
  logs/duffing_${TIMESTAMP}_${JOB3}.out
  logs/phillips_${TIMESTAMP}_${JOB4}.out
EOF

echo ""
echo "Job information saved to: $OUT/job_info.txt"
