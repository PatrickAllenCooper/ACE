#!/bin/bash

# ============================================================================
# Ablation Study Experiments
# ============================================================================
# Tests each component by removing it to validate design choices.
#
# Runs 4 ablation configurations:
# 1. No per-node convergence (use global early stopping)
# 2. No dedicated root learner (roots trained with interventional data)
# 3. No diversity reward (diversity_weight = 0)
# 4. Information gain only (no importance, no diversity)
#
# Usage:
#   ./scripts/run_ablations.sh
# ============================================================================

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ABLATION_DIR="results/ablations_${TIMESTAMP}"
mkdir -p "$ABLATION_DIR"

echo "========================================"
echo "ACE Ablation Studies"
echo "========================================"
echo "Output: $ABLATION_DIR"
echo "Started: $(date)"
echo ""

declare -a JOB_IDS

# ============================================================================
# Ablation 1: No Per-Node Convergence
# ============================================================================
echo "[1/4] No Per-Node Convergence"
echo "  Testing: Global early stopping instead of per-node"
echo "  Expected: Premature termination, undertrained slow learners"

JOB1=$(sbatch --parsable \
    --job-name=ablation_no_per_node \
    --output=logs/ablation_no_per_node_${TIMESTAMP}_%j.out \
    --error=logs/ablation_no_per_node_${TIMESTAMP}_%j.err \
    --partition=aa100 --qos=normal --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=8:00:00 \
    --wrap="python ace_experiments.py \
        --episodes 200 \
        --early_stopping \
        --no_per_node_convergence \
        --use_dedicated_root_learner \
        --output $ABLATION_DIR/no_per_node_convergence")

JOB_IDS+=($JOB1)
echo "  Job ID: $JOB1"
echo ""

# ============================================================================
# Ablation 2: No Dedicated Root Learner
# ============================================================================
echo "[2/4] No Dedicated Root Learner"
echo "  Testing: Roots trained with interventional data"
echo "  Expected: Poor root distribution learning (X1, X4 high loss)"

JOB2=$(sbatch --parsable \
    --job-name=ablation_no_root \
    --output=logs/ablation_no_root_${TIMESTAMP}_%j.out \
    --error=logs/ablation_no_root_${TIMESTAMP}_%j.err \
    --partition=aa100 --qos=normal --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=8:00:00 \
    --wrap="python ace_experiments.py \
        --episodes 200 \
        --early_stopping \
        --use_per_node_convergence \
        --no_dedicated_root_learner \
        --output $ABLATION_DIR/no_dedicated_root_learner")

JOB_IDS+=($JOB2)
echo "  Job ID: $JOB2"
echo ""

# ============================================================================
# Ablation 3: No Diversity Reward
# ============================================================================
echo "[3/4] No Diversity Reward"
echo "  Testing: diversity_reward_weight = 0"
echo "  Expected: Policy collapse to single target"

JOB3=$(sbatch --parsable \
    --job-name=ablation_no_diversity \
    --output=logs/ablation_no_diversity_${TIMESTAMP}_%j.out \
    --error=logs/ablation_no_diversity_${TIMESTAMP}_%j.err \
    --partition=aa100 --qos=normal --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=8:00:00 \
    --wrap="python ace_experiments.py \
        --episodes 200 \
        --early_stopping \
        --use_per_node_convergence \
        --use_dedicated_root_learner \
        --no_diversity_reward \
        --output $ABLATION_DIR/no_diversity_reward")

JOB_IDS+=($JOB3)
echo "  Job ID: $JOB3"
echo ""

# ============================================================================
# Ablation 4: Information Gain Only (Bonus)
# ============================================================================
echo "[4/4] Information Gain Only"
echo "  Testing: No node importance, no diversity"
echo "  Expected: Poor exploration, suboptimal performance"

JOB4=$(sbatch --parsable \
    --job-name=ablation_ig_only \
    --output=logs/ablation_ig_only_${TIMESTAMP}_%j.out \
    --error=logs/ablation_ig_only_${TIMESTAMP}_%j.err \
    --partition=aa100 --qos=normal --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=8:00:00 \
    --wrap="python ace_experiments.py \
        --episodes 200 \
        --early_stopping \
        --use_per_node_convergence \
        --use_dedicated_root_learner \
        --no_diversity_reward \
        --undersampled_bonus 0.0 \
        --output $ABLATION_DIR/ig_only")

JOB_IDS+=($JOB4)
echo "  Job ID: $JOB4"
echo ""

echo "========================================"
echo "All Ablation Jobs Submitted!"
echo "========================================"
echo "Jobs: ${JOB_IDS[*]}"
echo "Monitor: squeue -j $(IFS=,; echo "${JOB_IDS[*]}")"
echo ""
echo "After completion, compare results:"
echo "  python scripts/analyze_ablations.py $ABLATION_DIR"
echo ""

# Save ablation info
cat > "$ABLATION_DIR/ablation_info.txt" <<EOF
Ablation Study Run
==================
Submitted: $(date)
Directory: $ABLATION_DIR

Jobs:
  No Per-Node Conv:    $JOB1
  No Root Learner:     $JOB2
  No Diversity Reward: $JOB3
  IG Only:             $JOB4

Expected Results:
  1. No per-node: Earlier termination, undertrained slow mechanisms
  2. No root learner: X1, X4 (roots) have 2-3× higher loss
  3. No diversity: 90%+ interventions on single node, poor total loss
  4. IG only: Similar to no diversity, validates 3-component reward

Monitor: squeue -j $(IFS=,; echo "${JOB_IDS[*]}")

Analyze After Completion:
  python scripts/analyze_ablations.py $ABLATION_DIR
EOF

echo "Ablation info saved to: $ABLATION_DIR/ablation_info.txt"
