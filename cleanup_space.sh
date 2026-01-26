#!/bin/bash
#
# Clean up disk space in ACE repository
# Removes old results, checkpoints, and cached files
#
# Usage:
#   ./cleanup_space.sh           # Interactive prompts
#   ./cleanup_space.sh --yes     # Auto-confirm all
#   ./cleanup_space.sh --dry-run # Show what would be deleted
#

set -e

DRY_RUN=false
AUTO_YES=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --yes|-y)
            AUTO_YES=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dry-run] [--yes]"
            exit 1
            ;;
    esac
done

echo "=================================================="
echo "ACE Disk Space Cleanup"
echo "=================================================="
echo ""

# Show current usage
if command -v curc-quota &> /dev/null; then
    echo "Current disk usage:"
    curc-quota
    echo ""
fi

# Function to remove directory with confirmation
remove_dir() {
    local dir=$1
    local description=$2
    
    if [ ! -d "$dir" ]; then
        return
    fi
    
    local size=$(du -sh "$dir" 2>/dev/null | cut -f1)
    
    if $DRY_RUN; then
        echo "[DRY RUN] Would remove: $dir ($size) - $description"
        return
    fi
    
    if $AUTO_YES; then
        echo "Removing: $dir ($size) - $description"
        rm -rf "$dir"
    else
        echo "Remove: $dir ($size) - $description"
        read -p "Continue? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$dir"
            echo "  -> Removed"
        else
            echo "  -> Skipped"
        fi
    fi
}

# Function to remove files matching pattern
remove_files() {
    local pattern=$1
    local description=$2
    
    local files=$(find . -name "$pattern" 2>/dev/null)
    if [ -z "$files" ]; then
        return
    fi
    
    local count=$(echo "$files" | wc -l)
    local size=$(du -sh $files 2>/dev/null | tail -1 | cut -f1)
    
    if $DRY_RUN; then
        echo "[DRY RUN] Would remove: $count files matching $pattern ($size) - $description"
        return
    fi
    
    echo "Found: $count files matching $pattern ($size) - $description"
    if $AUTO_YES; then
        echo "$files" | xargs rm -f
        echo "  -> Removed"
    else
        read -p "Remove these files? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "$files" | xargs rm -f
            echo "  -> Removed"
        else
            echo "  -> Skipped"
        fi
    fi
}

echo "=================================================="
echo "1. OLD PAPER RESULT DIRECTORIES"
echo "=================================================="
echo ""

# Keep only most recent paper_* directory, remove rest
PAPER_DIRS=$(ls -td results/paper_* 2>/dev/null || true)
if [ -n "$PAPER_DIRS" ]; then
    LATEST=$(echo "$PAPER_DIRS" | head -1)
    echo "Latest paper results: $LATEST (KEEPING)"
    echo ""
    
    # Remove all except latest
    for dir in $PAPER_DIRS; do
        if [ "$dir" != "$LATEST" ]; then
            remove_dir "$dir" "Old paper results"
        fi
    done
else
    echo "No paper_* directories found"
fi

echo ""
echo "=================================================="
echo "2. OLD ACE MULTI-SEED RUNS"
echo "=================================================="
echo ""

# Keep only most recent ace_multi_seed, remove rest
ACE_DIRS=$(ls -td results/ace_multi_seed_* 2>/dev/null || true)
if [ -n "$ACE_DIRS" ]; then
    LATEST_ACE=$(echo "$ACE_DIRS" | head -1)
    echo "Latest ACE run: $LATEST_ACE (KEEPING)"
    echo ""
    
    for dir in $ACE_DIRS; do
        if [ "$dir" != "$LATEST_ACE" ]; then
            remove_dir "$dir" "Old ACE multi-seed run"
        fi
    done
else
    echo "No ace_multi_seed_* directories found"
fi

echo ""
echo "=================================================="
echo "3. TEST RUNS"
echo "=================================================="
echo ""

# Remove test runs (ace_quick_test, ace_test)
for dir in results/ace_quick_test_* results/ace_test_*; do
    if [ -d "$dir" ]; then
        remove_dir "$dir" "Test run (not needed for paper)"
    fi
done

echo ""
echo "=================================================="
echo "4. PYTORCH CHECKPOINTS"
echo "=================================================="
echo ""

# Remove .pt checkpoint files (these can be regenerated)
remove_files "*.pt" "PyTorch checkpoints"

echo ""
echo "=================================================="
echo "5. HUGGINGFACE CACHE"
echo "=================================================="
echo ""

# Clean HuggingFace cache (can be large)
if [ -d "$HF_HOME" ]; then
    HF_SIZE=$(du -sh "$HF_HOME" 2>/dev/null | cut -f1)
    echo "HuggingFace cache: $HF_HOME ($HF_SIZE)"
    
    if $DRY_RUN; then
        echo "[DRY RUN] Would clean HF cache"
    else
        read -p "Clean HuggingFace cache? (Models will re-download if needed) (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$HF_HOME"/*
            echo "  -> Cleaned"
        else
            echo "  -> Skipped"
        fi
    fi
fi

echo ""
echo "=================================================="
echo "6. OLD LOG FILES"
echo "=================================================="
echo ""

# Remove logs older than 7 days
OLD_LOGS=$(find logs/ -type f -mtime +7 2>/dev/null || true)
if [ -n "$OLD_LOGS" ]; then
    LOG_COUNT=$(echo "$OLD_LOGS" | wc -l)
    echo "Found $LOG_COUNT log files older than 7 days"
    
    if $DRY_RUN; then
        echo "[DRY RUN] Would remove old logs"
    else
        read -p "Remove old logs? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "$OLD_LOGS" | xargs rm -f
            echo "  -> Removed"
        else
            echo "  -> Skipped"
        fi
    fi
else
    echo "No old logs to remove"
fi

echo ""
echo "=================================================="
echo "CLEANUP COMPLETE"
echo "=================================================="
echo ""

# Show updated usage
if command -v curc-quota &> /dev/null; then
    echo "Updated disk usage:"
    curc-quota
fi

echo ""
echo "Recommended to keep:"
echo "  - results/ace_multi_seed_20260125_115453/ (latest ACE results)"
echo "  - results/baselines/baselines_20260124_182827/ (baseline comparison)"
echo "  - results/duffing/, results/phillips/, results/complex_scm/ (domain experiments)"
echo ""
echo "Safe to remove:"
echo "  - Old paper_* directories (already copied out)"
echo "  - Test runs (ace_quick_test, ace_test)"
echo "  - PyTorch checkpoints (.pt files)"
echo "  - Old logs (>7 days)"
echo ""
