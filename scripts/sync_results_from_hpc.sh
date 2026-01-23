#!/bin/bash

# ============================================================================
# Sync Results from HPC to Local Machine
# ============================================================================
# Downloads results from HPC server after run_all.sh completes.
#
# Usage:
#   ./scripts/sync_results_from_hpc.sh
#
# Or specify custom local directory:
#   ./scripts/sync_results_from_hpc.sh /path/to/local/results
#
# This script:
# 1. Finds latest results directory on HPC
# 2. Downloads to local machine
# 3. Runs post-processing locally
# 4. Generates all tables and figures
# ============================================================================

set -e

# Configuration
HPC_HOST="paco0228@login.rc.colorado.edu"
HPC_PROJECT_DIR="/projects/paco0228/ACE"
LOCAL_DEST=${1:-"./results_from_hpc"}

echo "========================================"
echo "Syncing Results from HPC"
echo "========================================"
echo "HPC: $HPC_HOST:$HPC_PROJECT_DIR"
echo "Local: $LOCAL_DEST"
echo "Started: $(date)"
echo ""

# Create local destination
mkdir -p "$LOCAL_DEST"

# ============================================================================
# Step 1: Find Latest Results on HPC
# ============================================================================
echo "[1/4] Finding latest results on HPC..."

LATEST_RESULTS=$(ssh $HPC_HOST "cd $HPC_PROJECT_DIR && ls -td results/paper_* 2>/dev/null | head -1" 2>/dev/null || echo "")

if [ -z "$LATEST_RESULTS" ]; then
    echo "ERROR: No results found on HPC"
    echo "Have you run ./run_all.sh on the HPC?"
    exit 1
fi

echo "  Latest results: $LATEST_RESULTS"
RESULTS_NAME=$(basename "$LATEST_RESULTS")
echo "  Basename: $RESULTS_NAME"
echo ""

# ============================================================================
# Step 2: Check if Jobs Are Complete
# ============================================================================
echo "[2/4] Checking job completion status..."

# Check if job_info.txt exists and all jobs are done
JOB_STATUS=$(ssh $HPC_HOST "cd $HPC_PROJECT_DIR && cat $LATEST_RESULTS/job_info.txt 2>/dev/null | grep 'Job IDs'" || echo "")

if [ -n "$JOB_STATUS" ]; then
    echo "  Job info found:"
    echo "  $JOB_STATUS"
    
    # Try to check if jobs are still running
    RUNNING=$(ssh $HPC_HOST "squeue -u paco0228 2>/dev/null | wc -l" || echo "0")
    
    if [ "$RUNNING" -gt 1 ]; then
        echo ""
        echo "⚠️  WARNING: $((RUNNING - 1)) jobs still running on HPC"
        echo "  You can still download partial results, but some experiments may be incomplete"
        echo ""
        read -p "Continue with download? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Cancelled."
            exit 0
        fi
    else
        echo "  ✓ No jobs currently running - safe to download"
    fi
else
    echo "  ⚠️  job_info.txt not found - cannot verify completion status"
fi

echo ""

# ============================================================================
# Step 3: Download Results
# ============================================================================
echo "[3/4] Downloading results from HPC..."
echo "  Source: $HPC_HOST:$HPC_PROJECT_DIR/$LATEST_RESULTS"
echo "  Dest: $LOCAL_DEST/$RESULTS_NAME"
echo ""

# Use rsync for efficient transfer (with progress)
rsync -avz --progress \
    $HPC_HOST:$HPC_PROJECT_DIR/$LATEST_RESULTS/ \
    $LOCAL_DEST/$RESULTS_NAME/

if [ $? -eq 0 ]; then
    echo ""
    echo "  ✓ Download complete"
else
    echo ""
    echo "  ✗ Download failed"
    exit 1
fi

# Also download logs
echo ""
echo "  Downloading logs..."
TIMESTAMP=$(echo $RESULTS_NAME | sed 's/paper_//')

rsync -avz --progress \
    $HPC_HOST:$HPC_PROJECT_DIR/logs/*${TIMESTAMP}* \
    $LOCAL_DEST/logs/ 2>/dev/null || echo "  (some logs may not exist yet)"

echo ""

# ============================================================================
# Step 4: Run Post-Processing Locally
# ============================================================================
echo "[4/4] Running post-processing locally..."
echo ""

if [ -f "./scripts/process_all_results.sh" ]; then
    echo "  Processing results..."
    ./scripts/process_all_results.sh "$LOCAL_DEST/$RESULTS_NAME"
    
    echo ""
    echo "  ✓ Post-processing complete"
    echo "  Results in: $LOCAL_DEST/$RESULTS_NAME/processed/"
else
    echo "  ⚠️  process_all_results.sh not found"
    echo "  Skipping post-processing (run manually later)"
fi

echo ""
echo "========================================"
echo "Sync Complete!"
echo "========================================"
echo "Finished: $(date)"
echo ""
echo "Results location: $LOCAL_DEST/$RESULTS_NAME"
echo ""
echo "Quick checks:"
echo "  ls -lh $LOCAL_DEST/$RESULTS_NAME/*/"
echo "  cat $LOCAL_DEST/$RESULTS_NAME/processed/PROCESSING_SUMMARY.txt"
echo ""
echo "View processed outputs:"
echo "  cat $LOCAL_DEST/$RESULTS_NAME/processed/tables/table1.txt"
echo "  open $LOCAL_DEST/$RESULTS_NAME/processed/figures/"
echo ""
echo "Next steps:"
echo "  1. Review verification reports in processed/verification/"
echo "  2. Copy Table 1 data to paper"
echo "  3. Copy figures to paper directory"
echo "  4. Fill TODO markers in paper.tex"
echo ""
