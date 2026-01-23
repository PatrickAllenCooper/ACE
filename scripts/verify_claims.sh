#!/bin/bash
# Verify specific behavioral claims from paper

echo "========================================"
echo "VERIFYING PAPER CLAIMS"
echo "========================================"
echo ""

# ===================================
# Claim 1: Clamping Strategy (Line 661)
# ===================================
echo "1. CLAMPING STRATEGY (Paper Line 661)"
echo "   Claim: 'ACE discovers a clamping strategy (do(X2 = 0))'"
echo ""

if [ -f "clamping_detector.py" ]; then
    python clamping_detector.py
else
    echo "   ❌ clamping_detector.py not found"
fi

echo ""
echo "---"
echo ""

# ===================================
# Claim 2: Regime Selection (Line 714)
# ===================================
echo "2. REGIME SELECTION (Paper Line 714)"
echo "   Claim: 'ACE learns to query high-volatility regimes'"
echo ""

if [ -f "regime_analyzer.py" ]; then
    python regime_analyzer.py
else
    echo "   ❌ regime_analyzer.py not found"
fi

echo ""
echo "---"
echo ""

# ===================================
# Claim 3: Early Stopping 40-60 Episodes (Line 767)
# ===================================
echo "3. EARLY STOPPING (Paper Line 767)"
echo "   Claim: '40-60 episodes vs. 200'"
echo ""

ACE_LOG=$(ls -t results/paper_*/ace/experiment.log 2>/dev/null | head -1)

if [ -f "$ACE_LOG" ]; then
    echo "   Checking ACE episodes..."
    
    # Look for early stopping or final episode count
    STOP_LINE=$(grep -E "Early stopping|saturation detected|episodes trained:" "$ACE_LOG" | tail -1)
    
    if [ ! -z "$STOP_LINE" ]; then
        echo "   $STOP_LINE"
        
        # Extract episode number
        NUM=$(echo "$STOP_LINE" | grep -oE '[0-9]+' | head -1)
        
        if [ ! -z "$NUM" ]; then
            if [ $NUM -ge 40 ] && [ $NUM -le 60 ]; then
                echo "   ✅ CLAIM SUPPORTED: $NUM episodes (within 40-60 range)"
            elif [ $NUM -ge 60 ] && [ $NUM -le 80 ]; then
                echo "   ⚠️  CLOSE: $NUM episodes (slightly above 40-60, still good)"
            elif [ $NUM -lt 40 ]; then
                echo "   ⚠️  TOO FEW: $NUM episodes (below 40, may have stopped too early)"
            else
                echo "   ⚠️  TOO MANY: $NUM episodes (above 80, early stopping may not have worked)"
            fi
            
            # Calculate vs baselines
            REDUCTION=$(echo "scale=1; 100 * (100 - $NUM) / 100" | bc)
            echo "   Episode reduction: ${REDUCTION}% (claimed 80%)"
            
            if [ $(echo "$REDUCTION >= 70" | bc) -eq 1 ]; then
                echo "   ✅ Close to 80% claim"
            elif [ $(echo "$REDUCTION >= 40" | bc) -eq 1 ]; then
                echo "   ⚠️  Significant but not 80% - consider revising claim"
            else
                echo "   ⚠️  Minimal reduction - revise claim needed"
            fi
        fi
    else
        echo "   ⚠️  No early stopping information found"
    fi
else
    echo "   ❌ No ACE log found"
    echo "   Expected: results/paper_*/ace/experiment.log"
fi

echo ""
echo "---"
echo ""

# ===================================
# Claim 4: Strategic Concentration (Line 485)
# ===================================
echo "4. STRATEGIC CONCENTRATION (Paper Line 485)"
echo "   Claim: 'ACE concentrates on X1 and X2 (collider parents)'"
echo ""

if [ -f "$ACE_LOG" ]; then
    METRICS=$(dirname "$ACE_LOG")/metrics.csv
    
    if [ -f "$METRICS" ]; then
        python3 << 'EOF'
import pandas as pd
import glob

# Find most recent metrics
metrics_files = glob.glob("results/paper_*/ace/metrics.csv")
if metrics_files:
    df = pd.read_csv(sorted(metrics_files)[-1])
    dist = df['target'].value_counts(normalize=True) * 100
    
    x1_pct = dist.get('X1', 0)
    x2_pct = dist.get('X2', 0)
    combined = x1_pct + x2_pct
    
    print(f"   X1: {x1_pct:.1f}%")
    print(f"   X2: {x2_pct:.1f}%")
    print(f"   Combined (X1+X2): {combined:.1f}%")
    print()
    
    if combined > 60:
        print(f"   ✅ STRATEGIC CONCENTRATION: {combined:.1f}% on collider parents")
    elif combined > 40:
        print(f"   ⚠️  MODERATE: {combined:.1f}% on collider parents (expected >60%)")
    else:
        print(f"   ❌ NO CONCENTRATION: Only {combined:.1f}% on collider parents")
    
    # Check for collapse vs strategy
    if x2_pct > 75:
        print(f"   ⚠️  X2 at {x2_pct:.1f}% may indicate policy collapse (check diversity scores)")
    elif x2_pct > 55:
        print(f"   ✅ X2 at {x2_pct:.1f}% appears strategic (not collapsed)")
else:
    print("   ❌ No metrics.csv found")
EOF
    else
        echo "   ❌ No metrics.csv found"
    fi
else
    echo "   ❌ No ACE results found"
fi

echo ""
echo "========================================"
echo "CLAIM VERIFICATION COMPLETE"
echo "========================================"
echo ""
echo "Summary will be saved to: results/claim_verification_summary.txt"
