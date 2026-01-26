#!/bin/bash
#
# AGGRESSIVE cleanup for HPC
# Removes EVERYTHING except what's needed for paper
#
# IMPORTANT: Only run this AFTER copying results locally!
#

set -e

echo "=================================================="
echo "AGGRESSIVE HPC CLEANUP"
echo "=================================================="
echo ""
echo "WARNING: This will remove MOST data from ACE directory!"
echo "Only keeping essential results for paper."
echo ""
echo "Will KEEP:"
echo "  - results/ace_multi_seed_20260125_115453/ (latest ACE)"
echo "  - results/baselines/baselines_20260124_182827/ (baselines)"
echo "  - results/duffing/duffing_20260124_*/ (5 runs)"
echo "  - results/phillips/phillips_20260124_*/ (5 runs)"
echo "  - results/complex_scm/complex_scm_*/ (6 runs)"
echo "  - Source code (.py files)"
echo "  - Job scripts"
echo ""
echo "Will REMOVE:"
echo "  - ALL paper_* directories"
echo "  - ALL old ace_multi_seed runs"
echo "  - ALL test runs"
echo "  - ALL checkpoints"
echo "  - ALL .pt files"
echo "  - ALL old logs"
echo "  - Python cache"
echo ""

read -p "Are you SURE results are backed up locally? (type 'yes'): " -r
if [[ ! $REPLY == "yes" ]]; then
    echo "Cleanup cancelled"
    exit 0
fi

# Work from current directory (don't assume ~/ACE)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo ""
echo "Working directory: $(pwd)"
echo "Starting cleanup..."
echo ""

# 1. Remove ALL paper_* directories
echo "1. Removing paper_* directories..."
rm -rf results/paper_*
echo "   Done"

# 2. Remove old ACE multi-seed (keep only 20260125_115453)
echo "2. Removing old ACE multi-seed runs..."
cd results
for dir in ace_multi_seed_*; do
    if [ "$dir" != "ace_multi_seed_20260125_115453" ] && [ -d "$dir" ]; then
        rm -rf "$dir"
    fi
done
cd "$SCRIPT_DIR"
echo "   Done"

# 3. Remove test runs
echo "3. Removing test runs..."
rm -rf results/ace_quick_test_* results/ace_test_*
echo "   Done"

# 4. Remove ALL checkpoints
echo "4. Removing checkpoints..."
rm -rf checkpoints 2>/dev/null || true
find . -name "*.pt" -delete 2>/dev/null || true
echo "   Done"

# 5. Clean logs (keep only 5 most recent)
echo "5. Cleaning logs..."
cd logs
ls -t *.out 2>/dev/null | tail -n +6 | xargs rm -f 2>/dev/null || true
ls -t *.err 2>/dev/null | tail -n +6 | xargs rm -f 2>/dev/null || true
cd "$SCRIPT_DIR"
echo "   Done"

# 6. Remove Python cache
echo "6. Removing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
rm -rf .pytest_cache 2>/dev/null || true
echo "   Done"

# 7. Remove any slurm files in base directory
echo "7. Cleaning misplaced SLURM logs..."
rm -f slurm-*.out 2>/dev/null || true
echo "   Done"

# 8. Remove old baseline runs (keep only 20260124_182827)
echo "8. Cleaning old baseline runs..."
cd results/baselines
for dir in baselines_*; do
    if [ "$dir" != "baselines_20260124_182827" ] && [ -d "$dir" ]; then
        rm -rf "$dir"
    fi
done
cd "$SCRIPT_DIR"
echo "   Done"

echo ""
echo "=================================================="
echo "CLEANUP COMPLETE"
echo "=================================================="
echo ""

# Show results
echo "Remaining size:"
du -sh .
echo ""

echo "Results kept:"
du -sh results/ace_multi_seed_20260125_115453 2>/dev/null || echo "  ACE: Not found!"
du -sh results/baselines/baselines_20260124_182827 2>/dev/null || echo "  Baselines: Not found!"
du -sh results/duffing 2>/dev/null || echo "  Duffing: Not found!"
du -sh results/phillips 2>/dev/null || echo "  Phillips: Not found!"
du -sh results/complex_scm 2>/dev/null || echo "  Complex SCM: Not found!"
echo ""

# Show quota
if command -v curc-quota &> /dev/null; then
    echo "Updated disk usage:"
    curc-quota
fi

echo ""
echo "=================================================="
echo "WHAT'S LEFT"
echo "=================================================="
ls -lh results/
echo ""

echo "You should now have 100-200GB free!"
echo ""
