#!/bin/bash

# ============================================================================
# HPC Repository Cleanup Script
# ============================================================================
# Removes old results, checkpoints, and logs to free up disk space.
# IMPORTANT: Only run this AFTER copying results locally!
#
# Usage:
#   ./cleanup_hpc.sh [--dry-run] [--aggressive]
#
# Options:
#   --dry-run      Show what would be deleted without deleting
#   --aggressive   Remove everything except latest run
#   --keep-latest  Keep only the most recent results (default)
#   --help         Show this help
# ============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Parse arguments
DRY_RUN=false
AGGRESSIVE=false
KEEP_LATEST=true

for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN=true
            ;;
        --aggressive)
            AGGRESSIVE=true
            ;;
        --keep-latest)
            KEEP_LATEST=true
            ;;
        --help)
            grep "^#" $0 | grep -v "^#!/" | sed 's/^# //'
            exit 0
            ;;
        *)
            log_error "Unknown option: $arg"
            echo "Use --help for usage"
            exit 1
            ;;
    esac
done

if [ "$DRY_RUN" = true ]; then
    log_warning "DRY RUN MODE - Nothing will be deleted"
fi

# ============================================================================
# Disk Usage Before
# ============================================================================

log_info "Checking current disk usage..."
echo ""
du -sh . 2>/dev/null || echo "Total: Unknown"
echo ""
echo "Breakdown by directory:"
du -sh results/ checkpoints/ logs/ 2>/dev/null || true
echo ""

BEFORE_SIZE=$(du -s . 2>/dev/null | awk '{print $1}')

# ============================================================================
# Cleanup Functions
# ============================================================================

cleanup_old_results() {
    log_info "Cleaning old results directories..."
    
    if [ "$AGGRESSIVE" = true ]; then
        # Keep only the absolute latest
        LATEST=$(ls -td results/multi_run_* results/paper_* results/ablations_* results/obs_ratio_* 2>/dev/null | head -1)
        
        if [ -n "$LATEST" ]; then
            log_warning "Keeping only: $LATEST"
            
            for dir in results/*/; do
                if [ "$dir" != "$LATEST/" ] && [ "$dir" != "results/claim_evidence/" ]; then
                    if [ "$DRY_RUN" = true ]; then
                        echo "  Would remove: $dir"
                    else
                        rm -rf "$dir" && log_success "Removed: $dir"
                    fi
                fi
            done
        fi
    else
        # Keep latest multi_run, ablations, obs_ratio, and most recent paper_
        KEEP_MULTI=$(ls -td results/multi_run_* 2>/dev/null | head -1)
        KEEP_ABLATION=$(ls -td results/ablations_* 2>/dev/null | head -1)
        KEEP_OBS=$(ls -td results/obs_ratio_* 2>/dev/null | head -1)
        KEEP_PAPER=$(ls -td results/paper_* 2>/dev/null | head -1)
        
        log_info "Keeping latest of each type:"
        [ -n "$KEEP_MULTI" ] && echo "  Multi-run: $KEEP_MULTI"
        [ -n "$KEEP_ABLATION" ] && echo "  Ablations: $KEEP_ABLATION"
        [ -n "$KEEP_OBS" ] && echo "  Obs-ratio: $KEEP_OBS"
        [ -n "$KEEP_PAPER" ] && echo "  Paper: $KEEP_PAPER"
        echo ""
        
        # Remove old results
        for dir in results/paper_*/; do
            if [ "$dir" != "$KEEP_PAPER/" ]; then
                if [ "$DRY_RUN" = true ]; then
                    echo "  Would remove: $dir"
                else
                    rm -rf "$dir" && echo "  Removed: $dir"
                fi
            fi
        done
        
        # Remove old multi_run (keep latest)
        for dir in results/multi_run_*/; do
            if [ "$dir" != "$KEEP_MULTI/" ]; then
                if [ "$DRY_RUN" = true ]; then
                    echo "  Would remove: $dir"
                else
                    rm -rf "$dir" && echo "  Removed: $dir"
                fi
            fi
        done
        
        # Remove old ablations (keep latest)
        for dir in results/ablations_*/; do
            if [ "$dir" != "$KEEP_ABLATION/" ]; then
                if [ "$DRY_RUN" = true ]; then
                    echo "  Would remove: $dir"
                else
                    rm -rf "$dir" && echo "  Removed: $dir"
                fi
            fi
        done
        
        # Remove old obs_ratio (keep latest)
        for dir in results/obs_ratio_*/; do
            if [ "$dir" != "$KEEP_OBS/" ]; then
                if [ "$DRY_RUN" = true ]; then
                    echo "  Would remove: $dir"
                else
                    rm -rf "$dir" && echo "  Removed: $dir"
                fi
            fi
        done
    fi
}

cleanup_checkpoints() {
    log_info "Cleaning checkpoints..."
    
    if [ -d "checkpoints" ]; then
        if [ "$DRY_RUN" = true ]; then
            echo "  Would remove: checkpoints/ ($(du -sh checkpoints/ 2>/dev/null | awk '{print $1}'))"
        else
            SIZE=$(du -sh checkpoints/ 2>/dev/null | awk '{print $1}')
            rm -rf checkpoints/*
            log_success "Removed checkpoints/ ($SIZE freed)"
        fi
    fi
}

cleanup_old_logs() {
    log_info "Cleaning old log files (keeping latest 10)..."
    
    # Keep only 10 most recent .out and .err files
    if [ "$DRY_RUN" = true ]; then
        echo "  Would keep 10 newest .out/.err files"
        ls -t logs/*.out logs/*.err 2>/dev/null | tail -n +21 | wc -l | xargs echo "  Would remove:"
    else
        REMOVED=0
        for ext in out err; do
            ls -t logs/*.$ext 2>/dev/null | tail -n +11 | while read file; do
                rm -f "$file" && ((REMOVED++)) || true
            done
        done
        log_success "Removed old log files"
    fi
}

cleanup_python_cache() {
    log_info "Cleaning Python cache..."
    
    if [ "$DRY_RUN" = true ]; then
        find . -type d -name "__pycache__" 2>/dev/null | wc -l | xargs echo "  Would remove __pycache__ dirs:"
        find . -name "*.pyc" 2>/dev/null | wc -l | xargs echo "  Would remove .pyc files:"
    else
        find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
        find . -name "*.pyc" -delete 2>/dev/null || true
        rm -rf .pytest_cache 2>/dev/null || true
        rm -f .coverage 2>/dev/null || true
        log_success "Removed Python cache"
    fi
}

cleanup_misplaced_slurm_logs() {
    log_info "Cleaning misplaced SLURM logs in base directory..."
    
    # Find slurm-*.out files in base directory (should be in logs/)
    SLURM_FILES=$(ls slurm-*.out 2>/dev/null | wc -l)
    
    if [ "$SLURM_FILES" -gt 0 ]; then
        if [ "$DRY_RUN" = true ]; then
            echo "  Would remove $SLURM_FILES slurm-*.out files from base directory"
            ls slurm-*.out 2>/dev/null | head -5
            [ "$SLURM_FILES" -gt 5 ] && echo "  ... and $((SLURM_FILES - 5)) more"
        else
            rm -f slurm-*.out 2>/dev/null || true
            log_success "Removed $SLURM_FILES slurm-*.out files from base directory"
        fi
    else
        log_info "  No misplaced SLURM logs found"
    fi
}

# ============================================================================
# Main Cleanup
# ============================================================================

log_warning "=========================================="
log_warning "  HPC REPOSITORY CLEANUP"
log_warning "=========================================="
echo ""

if [ "$DRY_RUN" = false ]; then
    log_warning "This will DELETE old results and checkpoints!"
    log_warning "Make sure you've copied important data locally first."
    echo ""
    read -p "Continue? (yes/no): " -r
    if [[ ! $REPLY =~ ^yes$ ]]; then
        log_info "Cleanup cancelled"
        exit 0
    fi
    echo ""
fi

# Run cleanup operations
cleanup_misplaced_slurm_logs
echo ""
cleanup_old_results
echo ""
cleanup_checkpoints
echo ""
cleanup_old_logs
echo ""
cleanup_python_cache
echo ""

# ============================================================================
# Disk Usage After
# ============================================================================

if [ "$DRY_RUN" = false ]; then
    log_info "Checking disk usage after cleanup..."
    echo ""
    du -sh . 2>/dev/null || echo "Total: Unknown"
    echo ""
    
    AFTER_SIZE=$(du -s . 2>/dev/null | awk '{print $1}')
    SAVED=$((BEFORE_SIZE - AFTER_SIZE))
    SAVED_MB=$((SAVED / 1024))
    
    log_success "=========================================="
    log_success "  CLEANUP COMPLETE"
    log_success "=========================================="
    echo ""
    echo "Space freed: ${SAVED_MB} MB"
    echo ""
    
    log_info "Remaining files:"
    ls -lh results/ 2>/dev/null | grep "^d" || echo "  (no results directories)"
    
else
    log_info "=========================================="
    log_info "  DRY RUN COMPLETE"
    log_info "=========================================="
    echo ""
    log_info "Run without --dry-run to actually delete files"
fi
