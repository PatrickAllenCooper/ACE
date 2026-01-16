#!/bin/bash

# ============================================================================
# ACE Workspace Cleanup Script
# ============================================================================
# Removes experiment outputs to prepare for new runs while preserving code.
#
# Usage:
#   ./cleanup.sh               # Interactive mode (asks for confirmation)
#   ./cleanup.sh --dry-run     # Show what would be deleted
#   ./cleanup.sh --full        # Also clean caches and temp files
#   ./cleanup.sh --force       # Skip confirmation prompts
#   ./cleanup.sh --keep-latest # Keep the most recent result directory
# ============================================================================

set -e  # Exit on error

# --- Configuration ---
DRY_RUN=false
FULL_CLEAN=false
FORCE=false
KEEP_LATEST=false

# --- Parse Arguments ---
for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --full)
            FULL_CLEAN=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --keep-latest)
            KEEP_LATEST=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dry-run      Show what would be deleted without deleting"
            echo "  --full         Also clean cache directories and temp files"
            echo "  --force        Skip confirmation prompts"
            echo "  --keep-latest  Keep the most recent results directory"
            echo "  --help         Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# --- Helper Functions ---
remove_item() {
    local item=$1
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] Would remove: $item"
    else
        if [ -e "$item" ]; then
            echo "Removing: $item"
            rm -rf "$item"
        fi
    fi
}

count_files() {
    local pattern=$1
    find . -name "$pattern" 2>/dev/null | wc -l | tr -d ' '
}

# --- Display Header ---
echo "========================================"
echo "ACE Workspace Cleanup"
echo "========================================"
echo "Current directory: $(pwd)"
echo "Dry run: $DRY_RUN"
echo "Full clean: $FULL_CLEAN"
echo "Keep latest: $KEEP_LATEST"
echo ""

# --- Check what will be cleaned ---
echo "Analyzing workspace..."
echo ""

# Count items
RESULTS_COUNT=$(find results -maxdepth 1 -type d -name "paper_*" 2>/dev/null | wc -l | tr -d ' ') || RESULTS_COUNT=0
RUN_COUNT=$(find results -maxdepth 1 -type d -name "run_*" 2>/dev/null | wc -l | tr -d ' ') || RUN_COUNT=0
LOG_COUNT=$(count_files "*.log")
OUT_COUNT=$(find logs -name "*.out" 2>/dev/null | wc -l | tr -d ' ') || OUT_COUNT=0
ERR_COUNT=$(find logs -name "*.err" 2>/dev/null | wc -l | tr -d ' ') || ERR_COUNT=0

echo "Items to clean:"
echo "  - Result directories (paper_*): $RESULTS_COUNT"
echo "  - Result directories (run_*): $RUN_COUNT"
echo "  - Log files (*.log): $LOG_COUNT"
echo "  - SLURM output files (*.out): $OUT_COUNT"
echo "  - SLURM error files (*.err): $ERR_COUNT"

if [ "$FULL_CLEAN" = true ]; then
    echo "  - Cache directories (HF_HOME, MPLCONFIGDIR)"
    echo "  - Python cache files (__pycache__, *.pyc)"
fi

echo ""

# --- Confirmation ---
if [ "$FORCE" = false ] && [ "$DRY_RUN" = false ]; then
    echo "This will permanently delete the above items."
    read -p "Continue? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cleanup cancelled."
        exit 0
    fi
fi

# --- Clean Results Directories ---
echo ""
echo "Cleaning results directories..."

if [ -d "results" ]; then
    if [ "$KEEP_LATEST" = true ]; then
        # Find the most recent directory
        LATEST=$(find results -maxdepth 1 -type d \( -name "paper_*" -o -name "run_*" \) -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
        
        if [ -n "$LATEST" ]; then
            echo "Keeping latest: $LATEST"
        fi
        
        # Remove all except latest
        for dir in results/paper_* results/run_*; do
            if [ -d "$dir" ] && [ "$dir" != "$LATEST" ]; then
                remove_item "$dir"
            fi
        done
    else
        # Remove all result directories
        for dir in results/paper_* results/run_*; do
            if [ -d "$dir" ]; then
                remove_item "$dir"
            fi
        done
    fi
fi

# --- Clean Log Files ---
echo ""
echo "Cleaning log files..."

# Remove standalone .log files
for log in results/*.log; do
    [ -e "$log" ] && remove_item "$log"
done

# --- Clean SLURM Logs ---
echo ""
echo "Cleaning SLURM logs..."

if [ -d "logs" ]; then
    for log in logs/*.out logs/*.err; do
        [ -e "$log" ] && remove_item "$log"
    done
fi

# --- Full Clean (Caches and Temp Files) ---
if [ "$FULL_CLEAN" = true ]; then
    echo ""
    echo "Performing full clean..."
    
    # Python cache
    echo "Removing Python cache files..."
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    find . -type f -name "*.pyo" -delete 2>/dev/null || true
    
    # Jupyter checkpoints
    if [ "$DRY_RUN" = false ]; then
        find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
    else
        echo "[DRY RUN] Would remove .ipynb_checkpoints directories"
    fi
    
    # Cache directories (be careful with these!)
    if [ -n "$HF_HOME" ] && [ -d "$HF_HOME" ]; then
        echo "Note: HF_HOME is set to: $HF_HOME"
        echo "      (Not deleting to preserve downloaded models)"
    fi
    
    if [ -n "$MPLCONFIGDIR" ] && [ -d "$MPLCONFIGDIR" ]; then
        echo "Note: MPLCONFIGDIR is set to: $MPLCONFIGDIR"
        echo "      (Not deleting to preserve matplotlib config)"
    fi
fi

# --- Summary ---
echo ""
echo "========================================"
if [ "$DRY_RUN" = true ]; then
    echo "Dry run complete. No files were deleted."
    echo "Run without --dry-run to perform cleanup."
else
    echo "Cleanup complete!"
    echo ""
    echo "Remaining structure:"
    du -sh results 2>/dev/null || echo "  results/: (empty or removed)"
    du -sh logs 2>/dev/null || echo "  logs/: (empty or removed)"
fi
echo "========================================"
