#!/bin/bash
# ============================================================================
# Lambda Cloud Setup & Run Script for ACE Reviewer Experiments
# ============================================================================
#
# Run this ONCE after SSH-ing into a fresh Lambda Cloud A100 instance.
# It installs everything and launches the full experiment suite.
#
# Prerequisites:
#   1. Lambda Cloud account with SSH key configured
#   2. Launch a 1x A100 instance from https://cloud.lambda.ai
#   3. SSH in:  ssh ubuntu@<instance-ip>
#   4. Run:     bash lambda_cloud_setup.sh
#
# Estimated cost: ~$25-30 (1x A100 @ $1.29-1.48/hr for ~20 hours)
# ============================================================================

set -euo pipefail

echo "================================================================"
echo " ACE Reviewer Experiments -- Lambda Cloud Setup"
echo "================================================================"
echo " Instance : $(hostname)"
echo " GPU      : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'checking...')"
echo " Started  : $(date)"
echo "================================================================"

# ---------------------------------------------------------------------------
# 1. Clone the repo
# ---------------------------------------------------------------------------
echo ""
echo ">>> Step 1: Cloning ACE repository <<<"

cd ~
if [ -d "ACE" ]; then
    echo "  ACE directory exists, pulling latest..."
    cd ACE
    git pull origin main
else
    git clone https://github.com/PatrickAllenCooper/ACE.git
    cd ACE
fi

# ---------------------------------------------------------------------------
# 2. Install dependencies
# ---------------------------------------------------------------------------
echo ""
echo ">>> Step 2: Installing Python dependencies <<<"

# Lambda instances come with PyTorch + CUDA pre-installed.
# Just install the additional packages ACE needs.
pip install --quiet \
    transformers \
    accelerate \
    datasets \
    scipy \
    pandas \
    matplotlib \
    seaborn \
    networkx \
    tqdm

# Verify GPU is visible
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"

# ---------------------------------------------------------------------------
# 3. Create output directory
# ---------------------------------------------------------------------------
TS=$(date +%Y%m%d_%H%M%S)
OUT="results/reviewer_lambda_${TS}"
mkdir -p "$OUT"

echo ""
echo ">>> Output directory: $OUT <<<"

# ---------------------------------------------------------------------------
# 4. Run all experiments
# ---------------------------------------------------------------------------
echo ""
echo ">>> Step 3: Running all reviewer experiments <<<"
echo ">>> Estimated runtime: 15-22 hours on 1x A100 <<<"
echo ""

# Phase A: CPU-light reviewer suite (Bayesian OED, graph misspec, etc.)
echo "=== Phase A: Reviewer experiment suite ==="
date
python -u scripts/runners/run_reviewer_experiments.py \
    --all \
    --seeds 42 123 456 789 1011 314 271 577 618 141 \
    --episodes 171 \
    --output "$OUT/suite" \
    2>&1 | tee "$OUT/suite.log" || echo "WARN: suite had errors"

# Phase B: Baselines at 171 episodes for the 5 new seeds
echo ""
echo "=== Phase B: Baseline additional seeds (171 episodes) ==="
date
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

# Phase C: 30-node large-scale SCM
echo ""
echo "=== Phase C: 30-node large-scale SCM ==="
date
python -u -c "
import sys, os, random
sys.path.insert(0, '.')
import torch, numpy as np, pandas as pd
from experiments.large_scale_scm import LargeScaleSCM

OUT_DIR = '$OUT/large_scale'
os.makedirs(OUT_DIR, exist_ok=True)
seeds = [42, 123, 456, 789, 1011]
episodes = 300
results = []
for seed in seeds:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    scm = LargeScaleSCM(30)
    node_data = {n: [] for n in scm.nodes}
    for ep in range(1, episodes + 1):
        node = random.choice(scm.nodes)
        value = random.uniform(-5, 5)
        data = scm.generate(50, interventions={node: value})
        for n in scm.nodes:
            node_data[n].append(data[n].mean().item())
    eval_data = scm.generate(1000)
    total_mse = sum(((eval_data[n] - np.mean(node_data[n][-50:])) ** 2).mean().item() for n in scm.nodes)
    results.append({'seed': seed, 'method': 'random', 'n_nodes': 30, 'episodes': episodes, 'total_mse': total_mse})
    print(f'  30-node random seed {seed}: MSE={total_mse:.4f}')
df = pd.DataFrame(results)
df.to_csv(f'{OUT_DIR}/large_scale_summary.csv', index=False)
print(f'  30-node Random: {df[\"total_mse\"].mean():.3f} +/- {df[\"total_mse\"].std():.3f}')
" 2>&1 | tee "$OUT/large_scale.log" || echo "WARN: 30-node had errors"

# Phase D: ACE additional seeds (GPU-heavy, ~2-4h each)
echo ""
echo "=== Phase D: ACE additional seeds (314 271 577 618 141) ==="
date
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
# 5. Summary
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo " ALL EXPERIMENTS COMPLETE"
echo "================================================================"
echo " Finished : $(date)"
echo " Results  : $OUT"
echo ""
echo " Copy results to your local machine:"
echo "   scp -r ubuntu@$(curl -s ifconfig.me 2>/dev/null || echo '<instance-ip>'):~/ACE/$OUT ."
echo ""
echo " IMPORTANT: Terminate the Lambda instance to stop billing!"
echo "================================================================"
