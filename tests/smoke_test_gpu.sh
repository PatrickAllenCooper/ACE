#!/bin/bash
# ============================================================================
# GPU Smoke Test: verify all three fixed experiments run end-to-end
# ============================================================================
# Runs each experiment for just 3-5 episodes to catch runtime errors
# before deploying to Azure. Takes ~15-30 minutes on a 3080.
#
# Usage (from repo root):
#   bash tests/smoke_test_gpu.sh
# ============================================================================

set -uo pipefail
cd "$(dirname "$0")/.."

PASS=0
FAIL=0
OUT="results/smoke_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

echo "================================================================"
echo " GPU Smoke Test -- 3 Fixed Experiments"
echo "================================================================"
echo " Output: $OUT"
echo ""

# ---------------------------------------------------------------------------
# Test 1: Graph misspecification (5 episodes, 1 seed, missing_edge)
# ---------------------------------------------------------------------------
echo "--- Test 1/5: Graph misspec (missing_edge) ---"
python -u ace_experiments.py \
    --episodes 5 \
    --seed 42 \
    --graph_misspec missing_edge \
    --early_stopping \
    --early_stop_min_episodes 3 \
    --pretrain_steps 10 \
    --pretrain_interval 50 \
    --obs_train_interval 3 \
    --obs_train_samples 50 \
    --obs_train_epochs 10 \
    --use_dedicated_root_learner \
    --root_fitting \
    --root_fit_samples 50 \
    --root_fit_epochs 10 \
    --smart_breaker \
    --output "$OUT/misspec_missing" \
    2>&1 | tail -5

if ls "$OUT"/misspec_missing/run_*/node_losses.csv 1>/dev/null 2>&1; then
    echo "  PASS: node_losses.csv written"
    PASS=$((PASS+1))
else
    echo "  FAIL: no node_losses.csv"
    FAIL=$((FAIL+1))
fi

# ---------------------------------------------------------------------------
# Test 2: Graph misspec (reversed_edge -- the hardest case)
# ---------------------------------------------------------------------------
echo ""
echo "--- Test 2/5: Graph misspec (reversed_edge) ---"
python -u ace_experiments.py \
    --episodes 5 \
    --seed 42 \
    --graph_misspec reversed_edge \
    --early_stopping \
    --early_stop_min_episodes 3 \
    --pretrain_steps 10 \
    --pretrain_interval 50 \
    --obs_train_interval 3 \
    --obs_train_samples 50 \
    --obs_train_epochs 10 \
    --use_dedicated_root_learner \
    --root_fitting \
    --root_fit_samples 50 \
    --root_fit_epochs 10 \
    --smart_breaker \
    --output "$OUT/misspec_reversed" \
    2>&1 | tail -5

if ls "$OUT"/misspec_reversed/run_*/node_losses.csv 1>/dev/null 2>&1; then
    echo "  PASS: node_losses.csv written"
    PASS=$((PASS+1))
else
    echo "  FAIL: no node_losses.csv"
    FAIL=$((FAIL+1))
fi

# ---------------------------------------------------------------------------
# Test 3: Large-scale 30-node SCM (3 episodes)
# ---------------------------------------------------------------------------
echo ""
echo "--- Test 3/5: Large-scale 30-node SCM ---"
python -u ace_experiments.py \
    --large_scale 30 \
    --episodes 3 \
    --seed 42 \
    --early_stopping \
    --early_stop_min_episodes 2 \
    --pretrain_steps 10 \
    --pretrain_interval 50 \
    --obs_train_interval 3 \
    --obs_train_samples 50 \
    --obs_train_epochs 10 \
    --use_dedicated_root_learner \
    --root_fitting \
    --root_fit_samples 50 \
    --root_fit_epochs 10 \
    --smart_breaker \
    --output "$OUT/large_scale_30" \
    2>&1 | tail -5

if ls "$OUT"/large_scale_30/run_*/node_losses.csv 1>/dev/null 2>&1; then
    echo "  PASS: node_losses.csv written"
    PASS=$((PASS+1))
else
    echo "  FAIL: no node_losses.csv"
    FAIL=$((FAIL+1))
fi

# ---------------------------------------------------------------------------
# Test 4: Hyperparameter variation (single cell: cov_bonus=30, diversity=0.1)
# ---------------------------------------------------------------------------
echo ""
echo "--- Test 4/5: Hyperparameter variation (alpha=0.05, gamma=0.1) ---"
python -u ace_experiments.py \
    --episodes 5 \
    --seed 42 \
    --cov_bonus 30.0 \
    --diversity_reward_weight 0.1 \
    --early_stopping \
    --early_stop_min_episodes 3 \
    --pretrain_steps 10 \
    --pretrain_interval 50 \
    --obs_train_interval 3 \
    --obs_train_samples 50 \
    --obs_train_epochs 10 \
    --use_dedicated_root_learner \
    --root_fitting \
    --root_fit_samples 50 \
    --root_fit_epochs 10 \
    --smart_breaker \
    --output "$OUT/hyperparam_test" \
    2>&1 | tail -5

if ls "$OUT"/hyperparam_test/run_*/node_losses.csv 1>/dev/null 2>&1; then
    echo "  PASS: node_losses.csv written"
    PASS=$((PASS+1))
else
    echo "  FAIL: no node_losses.csv"
    FAIL=$((FAIL+1))
fi

# ---------------------------------------------------------------------------
# Test 5: Default ACE (baseline sanity check, 5 episodes)
# ---------------------------------------------------------------------------
echo ""
echo "--- Test 5/5: Default ACE (sanity check) ---"
python -u ace_experiments.py \
    --episodes 5 \
    --seed 42 \
    --early_stopping \
    --early_stop_min_episodes 3 \
    --pretrain_steps 10 \
    --pretrain_interval 50 \
    --obs_train_interval 3 \
    --obs_train_samples 50 \
    --obs_train_epochs 10 \
    --use_dedicated_root_learner \
    --root_fitting \
    --root_fit_samples 50 \
    --root_fit_epochs 10 \
    --smart_breaker \
    --output "$OUT/default_ace" \
    2>&1 | tail -5

if ls "$OUT"/default_ace/run_*/node_losses.csv 1>/dev/null 2>&1; then
    echo "  PASS: node_losses.csv written"
    PASS=$((PASS+1))
else
    echo "  FAIL: no node_losses.csv"
    FAIL=$((FAIL+1))
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo " SMOKE TEST RESULTS: $PASS passed, $FAIL failed"
echo "================================================================"
if [ $FAIL -eq 0 ]; then
    echo " All tests passed -- safe to deploy to Azure."
else
    echo " FAILURES DETECTED -- fix before deploying."
    echo " Check logs in $OUT/"
fi
