#!/bin/bash
# Comprehensive verification of STRONG ACCEPT experiment scripts
# Tests everything that can be tested without GPU/execution

echo "=============================================="
echo "STRONG ACCEPT Scripts Verification"
echo "=============================================="

ERRORS=0

# Test 1: Scripts exist
echo -n "Test 1: Scripts exist... "
if [ -f "jobs/run_remaining_ablations.sh" ] && \
   [ -f "jobs/run_ace_no_oracle.sh" ] && \
   [ -f "jobs/workflows/submit_strong_accept_experiments.sh" ]; then
    echo "[PASS]"
else
    echo "[FAIL]"
    ((ERRORS++))
fi

# Test 2: Scripts have QoS
echo -n "Test 2: QoS specified... "
if grep -q "#SBATCH --qos=normal" jobs/run_remaining_ablations.sh && \
   grep -q "#SBATCH --qos=normal" jobs/run_ace_no_oracle.sh; then
    echo "[PASS]"
else
    echo "[FAIL]"
    ((ERRORS++))
fi

# Test 3: Ablation flags are implemented in ace_experiments.py
echo -n "Test 3: Ablation logic exists... "
if grep -q "if args.no_per_node_convergence:" ace_experiments.py && \
   grep -q "if args.no_dedicated_root_learner:" ace_experiments.py && \
   grep -q "if args.no_diversity_reward:" ace_experiments.py; then
    echo "[PASS]"
else
    echo "[FAIL]"
    ((ERRORS++))
fi

# Test 4: Pretrain_steps flag exists
echo -n "Test 4: Pretrain_steps flag... "
if grep -q '"--pretrain_steps"' ace_experiments.py; then
    echo "[PASS]"
else
    echo "[FAIL]"
    ((ERRORS++))
fi

# Test 5: Scripts use python -u (unbuffered)
echo -n "Test 5: Unbuffered output... "
if grep -q "python -u ace_experiments.py" jobs/run_remaining_ablations.sh && \
   grep -q "python -u ace_experiments.py" jobs/run_ace_no_oracle.sh; then
    echo "[PASS]"
else
    echo "[FAIL]"
    ((ERRORS++))
fi

# Test 6: Scripts run in SLURM_SUBMIT_DIR (not scratch)
echo -n "Test 6: Correct working directory... "
if grep -q "cd \$SLURM_SUBMIT_DIR" jobs/run_remaining_ablations.sh && \
   grep -q "cd \$SLURM_SUBMIT_DIR" jobs/run_ace_no_oracle.sh; then
    echo "[PASS]"
else
    echo "[FAIL]"
    ((ERRORS++))
fi

# Test 7: Time limits are adequate
echo -n "Test 7: Time limits... "
if grep -q "#SBATCH --time=08:00:00" jobs/run_remaining_ablations.sh && \
   grep -q "#SBATCH --time=08:00:00" jobs/run_ace_no_oracle.sh; then
    echo "[PASS]"
else
    echo "[FAIL]"
    ((ERRORS++))
fi

# Test 8: Output directories specified
echo -n "Test 8: Output directories... "
if grep -q "BASE_OUTPUT=" jobs/run_remaining_ablations.sh && \
   grep -q "BASE_OUTPUT=" jobs/run_ace_no_oracle.sh; then
    echo "[PASS]"
else
    echo "[FAIL]"
    ((ERRORS++))
fi

# Test 9: Syntax check on bash scripts
echo -n "Test 9: Bash syntax... "
if bash -n jobs/run_remaining_ablations.sh 2>/dev/null && \
   bash -n jobs/run_ace_no_oracle.sh 2>/dev/null; then
    echo "[PASS]"
else
    echo "[FAIL]"
    ((ERRORS++))
fi

# Test 10: Python syntax check
echo -n "Test 10: Python syntax... "
if python -m py_compile ace_experiments.py 2>/dev/null; then
    echo "[PASS]"
else
    echo "[FAIL]"
    ((ERRORS++))
fi

echo "=============================================="
if [ $ERRORS -eq 0 ]; then
    echo "[PASS] All 10 verification tests passed"
    echo "Scripts are ready for HPC execution"
else
    echo "[FAIL] $ERRORS test(s) failed"
    echo "Fix errors before submission"
fi
echo "=============================================="

exit $ERRORS
