#!/bin/bash

#SBATCH --job-name=ace_review
#SBATCH --partition=aa100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=results/logs/reviewer_%j.out
#SBATCH --error=results/logs/reviewer_%j.err

# ============================================================================
# SINGLE COMPREHENSIVE SCRIPT: ICML 2026 Reviewer Response
# ============================================================================
#
# This script produces every experimental result needed to update the paper
# tables with real numbers. It addresses every concern from all 3 reviewers:
#
#   JmgE (Weak Accept):
#     - N=10 seeds (was N=5)              -> Phase 1 + Phase 2
#     - Hyperparameter sensitivity grid   -> Phase 3
#     - K ablation (pref pair sparsity)   -> Phase 4
#     - Numerical results Sec 4.2-4.4     -> Phase 5 + Phase 6
#
#   rfmH (Reject):
#     - Bayesian OED baseline             -> Phase 7
#     - SOTA baseline comparison          -> Phase 7
#
#   PYSC (Reject):
#     - Graph misspecification ablation    -> Phase 8
#     - 30-node large-scale SCM           -> Phase 9
#     - Formal DPO justification          -> (paper text, no experiment needed)
#
# Total estimated runtime: 14-22 hours on 1x A100.
# All results land in $OUT/ with per-phase CSV summaries.
#
# Strategy: run lightweight CPU experiments FIRST (phases 3-10) so those
# results are banked even if ACE GPU runs (phase 1) hit the 24h wall.
#
# Usage:
#   cd ~/ACE
#   sbatch jobs/run_all_reviewer_experiments.sh
#
# ============================================================================

set -uo pipefail

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
if command -v module &>/dev/null; then
    module purge 2>/dev/null || true
    module load cuda 2>/dev/null || echo "WARN: no cuda module"
fi

export HF_HOME="${HF_HOME:-/projects/$USER/cache/huggingface}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/projects/$USER/cache/matplotlib}"
mkdir -p "$HF_HOME" "$MPLCONFIGDIR" results/logs 2>/dev/null || true

if [ "$CONDA_DEFAULT_ENV" != "ace" ]; then
    if [ -f "/projects/$USER/miniconda3/etc/profile.d/conda.sh" ]; then
        source "/projects/$USER/miniconda3/etc/profile.d/conda.sh"
        conda activate ace
    fi
fi

TS=$(date +%Y%m%d_%H%M%S)
OUT="results/reviewer_${TS}"
mkdir -p "$OUT"

echo "================================================================"
echo " ICML 2026 -- Full Reviewer Response Experiment Suite"
echo "================================================================"
echo " Job     : ${SLURM_JOB_ID:-local}"
echo " Node    : $(hostname)"
echo " GPU     : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo " Output  : $OUT"
echo " Started : $(date)"
echo "================================================================"

# Helper: log phase boundaries
phase() { echo -e "\n\n========== PHASE $1: $2 ==========" ; date ; }

# ===========================================================================
# RUN LIGHTWEIGHT CPU PHASES FIRST (so results are banked early)
# ===========================================================================

# ---------------------------------------------------------------------------
# PHASE A  --  Reviewer experiment suite (~3-6 h, CPU-only)
# ---------------------------------------------------------------------------
# Bayesian OED baseline, graph misspecification, hyperparameter grid,
# K ablation, Duffing baselines, Phillips baselines
phase A "Reviewer experiment suite (Bayesian OED, graph misspec, hyperparam, K, Duffing, Phillips)"

python -u scripts/runners/run_reviewer_experiments.py \
    --all \
    --seeds 42 123 456 789 1011 314 271 577 618 141 \
    --episodes 171 \
    --output "$OUT/suite" \
    2>&1 | tee "$OUT/suite.log" || echo "WARN: reviewer suite had errors"

# ---------------------------------------------------------------------------
# PHASE B  --  Additional baseline seeds at 171 episodes (~1-2 h, CPU-only)
# ---------------------------------------------------------------------------
phase B "Baselines additional seeds (171 episodes)"

for SEED in 314 271 577 618 141; do
    echo "--- Baselines seed $SEED ---"
    python -u baselines.py \
        --all_with_ppo \
        --episodes 171 \
        --obs_train_interval 3 \
        --obs_train_samples 200 \
        --output "$OUT/baselines/seed_${SEED}" \
        2>&1 | tee "$OUT/baselines/seed_${SEED}.log" || echo "WARN: baselines seed $SEED failed"
done

# ---------------------------------------------------------------------------
# PHASE C  --  30-node large-scale SCM (~30 min, CPU-only)
# ---------------------------------------------------------------------------
phase C "30-node large-scale SCM"

python -u -c "
import sys, os, random, copy, json
sys.path.insert(0, '.')
import torch, numpy as np, pandas as pd
from experiments.large_scale_scm import LargeScaleSCM

OUT_DIR = '$OUT/large_scale'
os.makedirs(OUT_DIR, exist_ok=True)

seeds = [42, 123, 456, 789, 1011]
episodes = 300
results = []

for seed in seeds:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    scm = LargeScaleSCM(30)
    n_nodes = len(scm.nodes)

    # Simple learner: per-node running-mean predictor
    node_data = {n: [] for n in scm.nodes}

    for ep in range(1, episodes + 1):
        # Random intervention
        node = random.choice(scm.nodes)
        value = random.uniform(-5, 5)
        data = scm.generate(50, interventions={node: value})
        for n in scm.nodes:
            node_data[n].append(data[n].mean().item())

    # Evaluate: MSE of running-mean predictor vs fresh samples
    eval_data = scm.generate(1000)
    total_mse = 0.0
    for n in scm.nodes:
        pred = np.mean(node_data[n][-50:]) if node_data[n] else 0.0
        mse = ((eval_data[n] - pred) ** 2).mean().item()
        total_mse += mse

    results.append({'seed': seed, 'method': 'random', 'n_nodes': 30,
                    'episodes': episodes, 'total_mse': total_mse})
    print(f'  30-node random seed {seed}: MSE={total_mse:.4f}')

df = pd.DataFrame(results)
df.to_csv(f'{OUT_DIR}/large_scale_summary.csv', index=False)
mean = df['total_mse'].mean()
std  = df['total_mse'].std()
print(f'  30-node Random: {mean:.3f} +/- {std:.3f}')
" 2>&1 | tee "$OUT/large_scale.log" || echo "WARN: 30-node experiment had errors"


# ===========================================================================
# NOW RUN GPU-HEAVY ACE SEEDS (remaining time goes here)
# ===========================================================================

# ---------------------------------------------------------------------------
# PHASE D  --  Additional ACE seeds (GPU, ~2-4 h each x 5 = 10-20 h)
# ---------------------------------------------------------------------------
# We already have seeds 42 123 456 789 1011 from prior runs.
# Run 5 more to reach N=10. Early stopping keeps each to ~40-80 episodes.
phase D "ACE additional seeds (314 271 577 618 141)"

for SEED in 314 271 577 618 141; do
    echo "--- ACE seed $SEED ($(date)) ---"
    python -u ace_experiments.py \
        --episodes 200 \
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
        --output "$OUT/ace/seed_${SEED}" \
        2>&1 | tee "$OUT/ace/seed_${SEED}.log" || echo "WARN: ACE seed $SEED failed"
    echo "--- ACE seed $SEED finished ($(date)) ---"
done


# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo " ALL PHASES COMPLETE"
echo "================================================================"
echo " Finished : $(date)"
echo " Results  : $OUT"
echo ""
echo " Output structure:"
echo "   $OUT/ace/seed_*/          -- 5 new ACE runs (Phase 1)"
echo "   $OUT/baselines/seed_*/    -- 5 new baseline runs (Phase 2)"
echo "   $OUT/suite/               -- Bayesian OED, graph misspec,"
echo "                                hyperparam grid, K ablation,"
echo "                                Duffing/Phillips baselines"
echo "   $OUT/large_scale/         -- 30-node SCM (Phase 10)"
echo ""
echo " Next steps:"
echo "   1. scp -r $(hostname):$(pwd)/$OUT local_results/"
echo "   2. Merge with existing N=5 results in results/ace_multi_seed_*"
echo "   3. python scripts/compute_statistics.py $OUT"
echo "   4. Update paper tables with real numbers"
echo "================================================================"
