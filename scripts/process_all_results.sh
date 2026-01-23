#!/bin/bash

# ============================================================================
# Post-Processing Script for ACE Paper Experiments
# ============================================================================
# This script processes results from run_all.sh to generate:
# - Extracted metrics for all experiments
# - Verification of all paper claims
# - Comparison tables (Table 1)
# - All paper figures
# - Summary report
#
# Usage:
#   ./scripts/process_all_results.sh results/paper_YYYYMMDD_HHMMSS
#
# Or automatically find latest:
#   ./scripts/process_all_results.sh $(ls -td results/paper_* | head -1)
# ============================================================================

set -e

# Check argument
if [ $# -eq 0 ]; then
    echo "Usage: $0 <results_directory>"
    echo ""
    echo "Example:"
    echo "  $0 results/paper_20260122_120000"
    echo ""
    echo "Or find latest automatically:"
    echo "  $0 \$(ls -td results/paper_* | head -1)"
    exit 1
fi

RESULTS_DIR=$1

# Validate directory exists
if [ ! -d "$RESULTS_DIR" ]; then
    echo "ERROR: Directory not found: $RESULTS_DIR"
    exit 1
fi

echo "========================================"
echo "ACE Results Post-Processing"
echo "========================================"
echo "Results directory: $RESULTS_DIR"
echo "Started: $(date)"
echo ""

# Create output directories
PROCESSED_DIR="$RESULTS_DIR/processed"
mkdir -p "$PROCESSED_DIR/tables"
mkdir -p "$PROCESSED_DIR/figures"
mkdir -p "$PROCESSED_DIR/verification"

# ============================================================================
# Step 1: Extract Metrics
# ============================================================================
echo "[1/5] Extracting metrics..."
echo ""

if [ -f "./scripts/extract_ace.sh" ]; then
    echo "  Extracting ACE metrics..."
    ./scripts/extract_ace.sh "$RESULTS_DIR" > "$PROCESSED_DIR/ace_metrics.txt" 2>&1
    echo "  ✓ ACE metrics saved to $PROCESSED_DIR/ace_metrics.txt"
else
    echo "  ⚠️  extract_ace.sh not found"
fi

if [ -f "./scripts/extract_baselines.sh" ]; then
    echo "  Extracting baseline metrics..."
    ./scripts/extract_baselines.sh "$RESULTS_DIR" > "$PROCESSED_DIR/baseline_metrics.txt" 2>&1
    echo "  ✓ Baseline metrics saved to $PROCESSED_DIR/baseline_metrics.txt"
else
    echo "  ⚠️  extract_baselines.sh not found"
fi

echo ""

# ============================================================================
# Step 2: Verify Paper Claims
# ============================================================================
echo "[2/5] Verifying paper claims..."
echo ""

if [ -f "./scripts/verify_claims.sh" ]; then
    echo "  Running claim verification..."
    ./scripts/verify_claims.sh "$RESULTS_DIR" > "$PROCESSED_DIR/verification/claim_verification.txt" 2>&1
    echo "  ✓ Claim verification saved to $PROCESSED_DIR/verification/claim_verification.txt"
else
    echo "  ⚠️  verify_claims.sh not found"
fi

# Verify clamping strategy (Line 661)
if [ -d "$RESULTS_DIR/duffing" ] && [ -f "clamping_detector.py" ]; then
    echo "  Verifying clamping strategy (Line 661)..."
    python clamping_detector.py "$RESULTS_DIR/duffing" > "$PROCESSED_DIR/verification/clamping_verification.txt" 2>&1
    echo "  ✓ Clamping verification saved"
else
    echo "  ⚠️  Duffing results not found or clamping_detector.py missing"
fi

# Verify regime selection (Line 714)
if [ -d "$RESULTS_DIR/phillips" ] && [ -f "regime_analyzer.py" ]; then
    echo "  Verifying regime selection (Line 714)..."
    python regime_analyzer.py "$RESULTS_DIR/phillips" > "$PROCESSED_DIR/verification/regime_verification.txt" 2>&1
    echo "  ✓ Regime verification saved"
else
    echo "  ⚠️  Phillips results not found or regime_analyzer.py missing"
fi

echo ""

# ============================================================================
# Step 3: Generate Comparison Tables
# ============================================================================
echo "[3/5] Generating comparison tables..."
echo ""

if [ -f "compare_methods.py" ]; then
    echo "  Generating Table 1 (method comparison)..."
    python compare_methods.py "$RESULTS_DIR" > "$PROCESSED_DIR/tables/table1.txt" 2>&1
    echo "  ✓ Table 1 saved to $PROCESSED_DIR/tables/table1.txt"
    
    # TODO: Add --latex flag when implemented
    # python compare_methods.py "$RESULTS_DIR" --latex > "$PROCESSED_DIR/tables/table1.tex" 2>&1
else
    echo "  ⚠️  compare_methods.py not found"
fi

echo ""

# ============================================================================
# Step 4: Generate Figures
# ============================================================================
echo "[4/5] Generating figures..."
echo ""

if [ -f "visualize.py" ]; then
    # Generate visualizations for each experiment
    for exp_dir in "$RESULTS_DIR"/*/ ; do
        if [ -d "$exp_dir" ]; then
            exp_name=$(basename "$exp_dir")
            echo "  Visualizing $exp_name..."
            python visualize.py "$exp_dir" > /dev/null 2>&1 || echo "    (visualization may have failed, check manually)"
        fi
    done
    
    # Copy generated figures to processed directory
    find "$RESULTS_DIR" -name "*.png" -exec cp {} "$PROCESSED_DIR/figures/" \; 2>/dev/null || true
    
    num_figures=$(ls -1 "$PROCESSED_DIR/figures/"*.png 2>/dev/null | wc -l)
    echo "  ✓ $num_figures figures copied to $PROCESSED_DIR/figures/"
else
    echo "  ⚠️  visualize.py not found"
fi

echo ""

# ============================================================================
# Step 5: Generate Summary Report
# ============================================================================
echo "[5/5] Generating summary report..."
echo ""

cat > "$PROCESSED_DIR/PROCESSING_SUMMARY.txt" <<EOF
ACE Experiment Results Processing Summary
==========================================

Processed: $(date)
Results Directory: $RESULTS_DIR
Processed Output: $PROCESSED_DIR

Experiments Run:
================

EOF

# Check which experiments completed
for exp in ace baselines complex_scm duffing phillips; do
    if [ -d "$RESULTS_DIR/$exp" ]; then
        echo "✓ $exp" >> "$PROCESSED_DIR/PROCESSING_SUMMARY.txt"
    else
        echo "✗ $exp (not found)" >> "$PROCESSED_DIR/PROCESSING_SUMMARY.txt"
    fi
done

cat >> "$PROCESSED_DIR/PROCESSING_SUMMARY.txt" <<EOF

Metrics Extracted:
==================

EOF

if [ -f "$PROCESSED_DIR/ace_metrics.txt" ]; then
    echo "✓ ACE metrics extracted" >> "$PROCESSED_DIR/PROCESSING_SUMMARY.txt"
else
    echo "✗ ACE metrics missing" >> "$PROCESSED_DIR/PROCESSING_SUMMARY.txt"
fi

if [ -f "$PROCESSED_DIR/baseline_metrics.txt" ]; then
    echo "✓ Baseline metrics extracted" >> "$PROCESSED_DIR/PROCESSING_SUMMARY.txt"
else
    echo "✗ Baseline metrics missing" >> "$PROCESSED_DIR/PROCESSING_SUMMARY.txt"
fi

cat >> "$PROCESSED_DIR/PROCESSING_SUMMARY.txt" <<EOF

Paper Claims Verified:
======================

EOF

if [ -f "$PROCESSED_DIR/verification/claim_verification.txt" ]; then
    echo "✓ General claims verified" >> "$PROCESSED_DIR/PROCESSING_SUMMARY.txt"
else
    echo "✗ General verification missing" >> "$PROCESSED_DIR/PROCESSING_SUMMARY.txt"
fi

if [ -f "$PROCESSED_DIR/verification/clamping_verification.txt" ]; then
    echo "✓ Clamping strategy (Line 661)" >> "$PROCESSED_DIR/PROCESSING_SUMMARY.txt"
else
    echo "✗ Clamping verification missing" >> "$PROCESSED_DIR/PROCESSING_SUMMARY.txt"
fi

if [ -f "$PROCESSED_DIR/verification/regime_verification.txt" ]; then
    echo "✓ Regime selection (Line 714)" >> "$PROCESSED_DIR/PROCESSING_SUMMARY.txt"
else
    echo "✗ Regime verification missing" >> "$PROCESSED_DIR/PROCESSING_SUMMARY.txt"
fi

cat >> "$PROCESSED_DIR/PROCESSING_SUMMARY.txt" <<EOF

Tables Generated:
=================

EOF

if [ -f "$PROCESSED_DIR/tables/table1.txt" ]; then
    echo "✓ Table 1 (method comparison)" >> "$PROCESSED_DIR/PROCESSING_SUMMARY.txt"
else
    echo "✗ Table 1 missing" >> "$PROCESSED_DIR/PROCESSING_SUMMARY.txt"
fi

cat >> "$PROCESSED_DIR/PROCESSING_SUMMARY.txt" <<EOF

Figures Generated:
==================

$(ls -1 "$PROCESSED_DIR/figures/"*.png 2>/dev/null | wc -l) figures saved

Output Files:
=============

Metrics:
  - $PROCESSED_DIR/ace_metrics.txt
  - $PROCESSED_DIR/baseline_metrics.txt

Verification:
  - $PROCESSED_DIR/verification/claim_verification.txt
  - $PROCESSED_DIR/verification/clamping_verification.txt
  - $PROCESSED_DIR/verification/regime_verification.txt

Tables:
  - $PROCESSED_DIR/tables/table1.txt

Figures:
  - $PROCESSED_DIR/figures/ (all PNG files)

Next Steps:
===========

1. Review verification reports for claim support
2. Copy Table 1 data to paper/paper.tex
3. Copy figures to paper/ directory
4. Update paper with actual numerical results

Commands:
---------

# View claim verification
cat "$PROCESSED_DIR/verification/claim_verification.txt"

# View Table 1
cat "$PROCESSED_DIR/tables/table1.txt"

# View figures
open "$PROCESSED_DIR/figures/"

EOF

echo "✓ Summary report generated"
cat "$PROCESSED_DIR/PROCESSING_SUMMARY.txt"

echo ""
echo "========================================"
echo "Post-Processing Complete!"
echo "========================================"
echo "Finished: $(date)"
echo ""
echo "Summary report: $PROCESSED_DIR/PROCESSING_SUMMARY.txt"
echo ""
echo "Next steps:"
echo "  1. Review: cat $PROCESSED_DIR/PROCESSING_SUMMARY.txt"
echo "  2. Verify claims: cat $PROCESSED_DIR/verification/*.txt"
echo "  3. View Table 1: cat $PROCESSED_DIR/tables/table1.txt"
echo "  4. View figures: open $PROCESSED_DIR/figures/"
echo ""
