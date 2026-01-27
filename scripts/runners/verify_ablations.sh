#!/bin/bash
# Verification script for ablation setup
# Tests everything that can be tested without running actual experiments

echo "=============================================="
echo "Ablation System Verification"
echo "=============================================="

ERRORS=0

# Test 1: Python runner script exists and has help
echo -n "Test 1: Runner script help... "
if python scripts/runners/run_ablations_fast.py --help > /dev/null 2>&1; then
    echo "✓ PASS"
else
    echo "✗ FAIL"
    ((ERRORS++))
fi

# Test 2: Runner accepts all ablation types
echo -n "Test 2: Valid ablation types... "
ALL_VALID=true
for abl in no_dpo no_convergence no_root_learner no_diversity; do
    if ! python scripts/runners/run_ablations_fast.py --help 2>&1 | grep -q "$abl"; then
        ALL_VALID=false
    fi
done

if $ALL_VALID; then
    echo "✓ PASS"
else
    echo "✗ FAIL"
    ((ERRORS++))
fi

# Test 3: ace_experiments.py has ablation flags in help
echo -n "Test 3: Main script ablation flags... "
if python ace_experiments.py --help 2>&1 | grep -q "no_per_node_convergence" &&
   python ace_experiments.py --help 2>&1 | grep -q "no_dedicated_root_learner" &&
   python ace_experiments.py --help 2>&1 | grep -q "no_diversity_reward" &&
   python ace_experiments.py --help 2>&1 | grep -q "custom"; then
    echo "✓ PASS"
else
    echo "✗ FAIL - Ablation flags not in help"
    ((ERRORS++))
fi

# Test 4: Job script has QoS
echo -n "Test 4: Job script QoS... "
if grep -q "#SBATCH --qos=normal" jobs/run_ablations_fast.sh; then
    echo "✓ PASS"
else
    echo "✗ FAIL - Missing QoS"
    ((ERRORS++))
fi

# Test 5: Job script calls correct Python file
echo -n "Test 5: Job script path... "
if grep -q "scripts/runners/run_ablations_fast.py" jobs/run_ablations_fast.sh; then
    echo "✓ PASS"
else
    echo "✗ FAIL - Wrong Python path"
    ((ERRORS++))
fi

# Test 6: Scratch script stays in submit dir
echo -n "Test 6: Scratch script import fix... "
if grep -A5 "Run Ablation" jobs/run_ablations_scratch.sh | grep -q "cd.*SLURM_SUBMIT_DIR"; then
    echo "✓ PASS"
else
    echo "✗ FAIL - Scratch script doesn't cd to SUBMIT_DIR"
    ((ERRORS++))
fi

# Test 7: ace_experiments.py has ablation logic
echo -n "Test 7: Ablation logic implemented... "
if grep -q "if args.no_diversity_reward:" ace_experiments.py &&
   grep -q "if args.no_dedicated_root_learner:" ace_experiments.py &&
   grep -q "if args.no_per_node_convergence:" ace_experiments.py; then
    echo "✓ PASS"
else
    echo "✗ FAIL - Ablation logic not implemented"
    ((ERRORS++))
fi

# Summary
echo "=============================================="
if [ $ERRORS -eq 0 ]; then
    echo "✓ All verification tests passed!"
    echo "System ready for HPC execution"
else
    echo "✗ $ERRORS test(s) failed"
    echo "Fix errors before HPC submission"
fi
echo "=============================================="

exit $ERRORS
