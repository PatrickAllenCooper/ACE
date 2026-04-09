#!/bin/bash
# ============================================================================
# Azure Dual-A100 -- Fixed Reviewer Experiments
# ============================================================================
#
# Runs the three experiments that were broken in the Lambda run:
#   1. Hyperparameter sensitivity grid (ACE, not random policy)
#   2. Graph misspecification ablation (ACE, not random policy)
#   3. 30-node large-scale SCM (ACE, not just random baseline)
#
# Parallelizes across 2 GPUs:
#   GPU 0: Hyperparameter grid (32 ACE runs: 4x4 grid x 2 seeds)
#   GPU 1: Graph misspecification (15 ACE runs: 5 types x 3 seeds)
#   Then:  30-node ACE (3 seeds, either GPU)
#
# Usage:
#   ssh into Azure VM, clone repo, then:
#     cd ACE && bash jobs/run_azure_dual_a100.sh
#
# Estimated runtime: 18-28 hours on dual A100
# ============================================================================

set -uo pipefail

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
echo "================================================================"
echo " ACE -- Fixed Reviewer Experiments (Azure Dual A100)"
echo "================================================================"
echo " Host      : $(hostname)"
echo " GPUs      : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | tr '\n' ', ')"
echo " Started   : $(date)"
echo "================================================================"

# Install dependencies if needed
pip install --quiet transformers accelerate datasets scipy pandas matplotlib seaborn networkx tqdm 2>/dev/null || true
pip install --quiet --upgrade Pillow 2>/dev/null || true

# Verify GPUs
python -c "
import torch
n = torch.cuda.device_count()
print(f'PyTorch {torch.__version__}, CUDA GPUs: {n}')
for i in range(n):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
assert n >= 1, 'No GPUs found'
"

TS=$(date +%Y%m%d_%H%M%S)
OUT="results/reviewer_azure_${TS}"
mkdir -p "$OUT" "$OUT/logs"

# Ensure /scratch exists (checkpoint writes)
sudo mkdir -p /scratch 2>/dev/null && sudo chmod 777 /scratch 2>/dev/null || mkdir -p /tmp/ace_scratch

echo ""
echo "Output: $OUT"
echo ""

# ---------------------------------------------------------------------------
# GPU 0: Hyperparameter Grid (background)
# ---------------------------------------------------------------------------
echo ">>> Starting: Hyperparameter grid on GPU 0 (background) <<<"
CUDA_VISIBLE_DEVICES=0 python -u scripts/runners/run_reviewer_experiments.py \
    --hyperparam-grid \
    --seeds 42 123 \
    --episodes 100 \
    --output "$OUT/hyperparam" \
    2>&1 | tee "$OUT/logs/hyperparam.log" &
PID_HYPERPARAM=$!
echo "  PID: $PID_HYPERPARAM"

# ---------------------------------------------------------------------------
# GPU 1: Graph Misspecification Ablation (background)
# ---------------------------------------------------------------------------
echo ">>> Starting: Graph misspecification on GPU 1 (background) <<<"
CUDA_VISIBLE_DEVICES=1 python -u scripts/runners/run_reviewer_experiments.py \
    --graph-misspec \
    --seeds 42 123 456 \
    --episodes 171 \
    --output "$OUT/misspec" \
    2>&1 | tee "$OUT/logs/misspec.log" &
PID_MISSPEC=$!
echo "  PID: $PID_MISSPEC"

# ---------------------------------------------------------------------------
# Wait for both to finish
# ---------------------------------------------------------------------------
echo ""
echo ">>> Both experiments running in parallel. Waiting... <<<"
echo "    Monitor: tail -f $OUT/logs/hyperparam.log"
echo "    Monitor: tail -f $OUT/logs/misspec.log"
echo ""

wait $PID_HYPERPARAM
HYPERPARAM_EXIT=$?
echo ">>> Hyperparameter grid finished (exit=$HYPERPARAM_EXIT) <<<"

wait $PID_MISSPEC
MISSPEC_EXIT=$?
echo ">>> Graph misspecification finished (exit=$MISSPEC_EXIT) <<<"

# ---------------------------------------------------------------------------
# 30-Node Large-Scale ACE (sequential, uses GPU 0)
# ---------------------------------------------------------------------------
echo ""
echo ">>> Starting: 30-node ACE (3 seeds) on GPU 0 <<<"

for SEED in 42 123 456; do
    echo "  30-node ACE seed $SEED ($(date))"
    CUDA_VISIBLE_DEVICES=0 python -u ace_experiments.py \
        --large_scale 30 \
        --episodes 300 \
        --seed "$SEED" \
        --early_stopping \
        --early_stop_patience 20 \
        --use_dedicated_root_learner \
        --dedicated_root_interval 3 \
        --obs_train_interval 3 \
        --obs_train_samples 200 \
        --obs_train_epochs 100 \
        --root_fitting \
        --root_fit_interval 5 \
        --root_fit_samples 500 \
        --root_fit_epochs 100 \
        --undersampled_bonus 200.0 \
        --diversity_reward_weight 0.3 \
        --max_concentration 0.7 \
        --concentration_penalty 150.0 \
        --update_reference_interval 25 \
        --pretrain_steps 200 \
        --pretrain_interval 25 \
        --smart_breaker \
        --output "$OUT/large_scale/seed_${SEED}" \
        2>&1 | tee "$OUT/logs/large_scale_seed_${SEED}.log" \
        || echo "  WARN: 30-node seed $SEED had errors"
    echo "  30-node ACE seed $SEED finished ($(date))"
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo " ALL FIXED EXPERIMENTS COMPLETE"
echo "================================================================"
echo " Finished : $(date)"
echo " Results  : $OUT"
echo ""
echo " Files to check:"
echo "   $OUT/hyperparam/*/hyperparam_grid_summary.csv"
echo "   $OUT/misspec/*/graph_misspec_summary.csv"
echo "   $OUT/large_scale/seed_*/run_*/node_losses.csv"
echo ""
echo " Copy results:"
echo "   scp -r $(whoami)@$(hostname):$(pwd)/$OUT ."
echo ""
echo " Then DEALLOCATE the Azure VM to stop billing!"
echo "================================================================"
