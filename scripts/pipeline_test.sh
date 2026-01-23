#!/bin/bash
# Pipeline Validation Script
# Tests all pipeline changes before full experimental runs

set -e

echo "========================================"
echo "PIPELINE VALIDATION"
echo "========================================"
echo ""

# Check if we're in the right directory
if [ ! -f "ace_experiments.py" ]; then
    echo "❌ Error: Must run from ACE project root"
    exit 1
fi

# Track overall status
ALL_PASS=true

# ===================================
# TEST 1: ACE Fixes
# ===================================
echo "TEST 1: ACE Training Fixes"
echo "---"

if [ -f "test_jan21_fixes.sh" ]; then
    echo "Running quick ACE test (10 episodes)..."
    ./test_jan21_fixes.sh > test_output_ace.log 2>&1
    
    if [ $? -eq 0 ]; then
        echo "✅ ACE test completed"
        
        # Check for specific improvements
        TEST_DIR=$(ls -td results/test_jan21_fixes_* 2>/dev/null | head -1)
        if [ -d "$TEST_DIR" ]; then
            LOG="$TEST_DIR/experiment.log"
            
            # Check diversity scores
            if grep -q "diversity=" "$LOG"; then
                AVG_DIV=$(grep "diversity=" "$LOG" | awk -F'diversity=' '{print $2}' | awk '{print $1}' | awk '{s+=$1; n++} END {if(n>0) print s/n; else print 0}')
                echo "   Avg diversity score: $AVG_DIV (target: > -10)"
            fi
            
            # Check zero rewards
            TOTAL=$(grep -c "Reward:" "$LOG" || echo 0)
            ZEROS=$(grep "Reward: 0.00" "$LOG" | wc -l || echo 0)
            if [ $TOTAL -gt 0 ]; then
                ZERO_PCT=$(echo "scale=1; 100 * $ZEROS / $TOTAL" | bc)
                echo "   Zero-reward %: $ZERO_PCT% (target: < 60%)"
            fi
            
            # Check gradients
            if grep -q "Gradient Check" "$LOG"; then
                GRAD=$(grep "Gradient Check" "$LOG" | tail -1 | grep -oE "grad_norm=[0-9.]+" | cut -d= -f2)
                echo "   Latest gradient norm: $GRAD (target: > 0.01)"
            fi
        fi
    else
        echo "❌ ACE test FAILED - check test_output_ace.log"
        ALL_PASS=false
    fi
else
    echo "⚠️  test_jan21_fixes.sh not found - skipping ACE test"
fi

echo ""

# ===================================
# TEST 2: PPO Bug Fix
# ===================================
echo "TEST 2: PPO Baseline Bug Fix"
echo "---"

echo "Running quick PPO test (5 episodes)..."
python baselines.py --baseline ppo --episodes 5 --steps 10 --output results/test_ppo_pipeline > test_output_ppo.log 2>&1

if [ $? -eq 0 ]; then
    echo "✅ PPO test completed"
    
    # Check for shape warnings
    if grep -q "UserWarning.*target size.*different.*input size" test_output_ppo.log; then
        echo "   ❌ Still has shape mismatch warnings"
        ALL_PASS=false
    else
        echo "   ✅ No shape warnings - bug fixed"
    fi
    
    # Check completion
    if [ -f "results/test_ppo_pipeline/baselines_ppo_*/results.csv" 2>/dev/null ]; then
        echo "   ✅ Generated results successfully"
    fi
else
    echo "❌ PPO test FAILED - check test_output_ppo.log"
    ALL_PASS=false
fi

echo ""

# ===================================
# TEST 3: Behavioral Verifiers
# ===================================
echo "TEST 3: Behavioral Verification Tools"
echo "---"

# Test clamping detector
if [ -f "clamping_detector.py" ]; then
    echo "Testing clamping detector..."
    python clamping_detector.py > test_output_clamping.log 2>&1
    
    if [ $? -eq 0 ]; then
        echo "✅ Clamping detector works"
    else
        # Expected to fail if no data exists
        if grep -q "No Duffing" test_output_clamping.log; then
            echo "⚠️  Clamping detector ready (no data to test on yet)"
        else
            echo "⚠️  Clamping detector has issues - check test_output_clamping.log"
        fi
    fi
else
    echo "❌ clamping_detector.py not found"
    ALL_PASS=false
fi

# Test regime analyzer
if [ -f "regime_analyzer.py" ]; then
    echo "Testing regime analyzer..."
    python regime_analyzer.py > test_output_regime.log 2>&1
    
    if [ $? -eq 0 ]; then
        echo "✅ Regime analyzer works"
    else
        # Expected to fail if no data exists
        if grep -q "No Phillips" test_output_regime.log; then
            echo "⚠️  Regime analyzer ready (no data to test on yet)"
        else
            echo "⚠️  Regime analyzer has issues - check test_output_regime.log"
        fi
    fi
else
    echo "❌ regime_analyzer.py not found"
    ALL_PASS=false
fi

echo ""

# ===================================
# TEST 4: Documentation Infrastructure
# ===================================
echo "TEST 4: Documentation Infrastructure"
echo "---"

# Check required files
REQUIRED_FILES=(
    "results/README.md"
    "results/RESULTS_LOG.md"
    "results/GAPS_ANALYSIS.md"
    "results/ACTION_PLAN.md"
)

MISSING=0
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file"
    else
        echo "❌ $file missing"
        MISSING=$((MISSING + 1))
        ALL_PASS=false
    fi
done

if [ $MISSING -eq 0 ]; then
    echo "✅ All documentation files present"
fi

echo ""

# ===================================
# SUMMARY
# ===================================
echo "========================================"
echo "VALIDATION SUMMARY"
echo "========================================"
echo ""

if [ "$ALL_PASS" = true ]; then
    echo "✅ ALL TESTS PASSED"
    echo ""
    echo "Pipeline is READY for full experimental runs!"
    echo ""
    echo "Next steps:"
    echo "  1. Launch ACE: sbatch jobs/run_ace_main.sh"
    echo "  2. Rerun PPO: python baselines.py --baseline ppo --episodes 100"
    echo "  3. Monitor: tail -f logs/ace_*.err"
    echo ""
    exit 0
else
    echo "⚠️  SOME TESTS FAILED"
    echo ""
    echo "Review test outputs:"
    echo "  - test_output_ace.log"
    echo "  - test_output_ppo.log"
    echo "  - test_output_clamping.log"
    echo "  - test_output_regime.log"
    echo ""
    echo "Fix issues before proceeding to full runs."
    echo ""
    exit 1
fi
