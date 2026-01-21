#!/bin/bash
# Launch DPO Training with All Improvements
# January 21, 2026

set -e

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║              LAUNCHING DPO TRAINING WITH OBSERVATIONAL DATA                  ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Check we're in the right directory
if [ ! -f "ace_experiments.py" ]; then
    echo "❌ Error: Must run from ACE project root"
    exit 1
fi

echo "📋 Pre-Flight Checklist:"
echo "  ✅ Observational training: ACTIVE (every 3 steps)"
echo "  ✅ Dedicated root learner: ACTIVE (every 3 episodes)"
echo "  ✅ Adaptive diversity: ENABLED"
echo "  ✅ Novelty bonus: ENABLED"
echo "  ✅ Emergency retraining: ENABLED"
echo "  ✅ Speed improvements: ENABLED"
echo "  ✅ Early stopping: ENABLED"
echo ""

# Option 1: Test first (recommended)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "RECOMMENDED: Test pipeline first (30 minutes)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Would you like to:"
echo "  1) Test pipeline first (./pipeline_test.sh)"
echo "  2) Skip test and launch full training (sbatch jobs/run_ace_main.sh)"
echo "  3) Cancel"
echo ""
read -p "Enter choice (1-3): " choice

case $choice in
    1)
        echo ""
        echo "🧪 Running pipeline test..."
        ./pipeline_test.sh
        
        if [ $? -eq 0 ]; then
            echo ""
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "✅ TESTS PASSED - Ready for full training"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo ""
            read -p "Launch full training? (y/n): " launch
            
            if [ "$launch" = "y" ] || [ "$launch" = "Y" ]; then
                echo ""
                echo "🚀 Launching full DPO training..."
                sbatch jobs/run_ace_main.sh
                echo ""
                echo "✅ Job submitted!"
                echo ""
                echo "Monitor with:"
                echo "  tail -f logs/ace_*.err | grep -E 'Episode.*Start|Obs Training|diversity='"
            else
                echo "Training not launched."
            fi
        else
            echo ""
            echo "❌ Tests failed - review output above"
            echo "Fix issues before launching full training"
            exit 1
        fi
        ;;
    
    2)
        echo ""
        echo "⚠️  Skipping tests (not recommended)"
        echo ""
        read -p "Are you sure? (y/n): " confirm
        
        if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
            echo ""
            echo "🚀 Launching full DPO training..."
            sbatch jobs/run_ace_main.sh
            echo ""
            echo "✅ Job submitted!"
            echo ""
            echo "Monitor with:"
            echo "  tail -f logs/ace_*.err | grep -E 'Episode.*Start|Obs Training'"
        else
            echo "Cancelled."
        fi
        ;;
    
    3)
        echo "Cancelled."
        exit 0
        ;;
    
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Next steps after training completes:"
echo "  1. ./verify_claims.sh"
echo "  2. ./extract_ace.sh"
echo "  3. python compare_methods.py"
echo "  4. code results/RESULTS_LOG.md  (document findings)"
echo "  5. code paper/paper.tex          (fill tables)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
