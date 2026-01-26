#!/bin/bash
#
# Find what's using disk space in ACE directory
#

# Work from current directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=================================================="
echo "DISK SPACE ANALYSIS"
echo "=================================================="
echo ""

echo "Total ACE directory:"
du -sh .
echo ""

echo "Top-level subdirectories:"
du -sh * | sort -rh | head -20
echo ""

echo "=================================================="
echo "RESULTS DIRECTORY BREAKDOWN"
echo "=================================================="
du -sh results/* 2>/dev/null | sort -rh | head -20
echo ""

echo "=================================================="
echo "LARGEST INDIVIDUAL FILES (Top 20)"
echo "=================================================="
find . -type f -exec du -h {} + 2>/dev/null | sort -rh | head -20
echo ""

echo "=================================================="
echo "FILE TYPE SUMMARY"
echo "=================================================="
echo "CSV files:"
find . -name "*.csv" -exec du -ch {} + 2>/dev/null | tail -1
echo ""
echo "PNG files:"
find . -name "*.png" -exec du -ch {} + 2>/dev/null | tail -1
echo ""
echo "PyTorch checkpoints (.pt):"
find . -name "*.pt" -exec du -ch {} + 2>/dev/null | tail -1
echo ""
echo "Log files:"
du -sh logs/ 2>/dev/null
echo ""

echo "=================================================="
echo "CHECKPOINT DIRECTORIES"
echo "=================================================="
find . -type d -name "checkpoints" -exec du -sh {} \; 2>/dev/null
echo ""

echo "=================================================="
echo "CACHE DIRECTORIES"
echo "=================================================="
echo "HuggingFace cache:"
du -sh ${HF_HOME:-/projects/$USER/cache/huggingface} 2>/dev/null || echo "Not found"
echo ""
echo "Matplotlib cache:"
du -sh ${MPLCONFIGDIR:-/projects/$USER/cache/matplotlib} 2>/dev/null || echo "Not found"
echo ""
