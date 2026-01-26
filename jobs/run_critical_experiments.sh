#!/bin/bash
#SBATCH --job-name=critical_expts
#SBATCH --partition=aa100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/critical_%j.out
#SBATCH --error=logs/critical_%j.err

# ============================================================================
# CRITICAL EXPERIMENTS FOR ICML REVIEWER RESPONSE
# ============================================================================
#
# This script runs all experiments needed to address reviewer concerns:
#   1. Extended baselines (171 episodes) - fair comparison
#   2. Lookahead ablation (random proposer) - isolate DPO contribution
#   3. Complex 15-node SCM - validate scaling claims
#
# Expected runtime: 4-6 hours
#
# Usage:
#   sbatch jobs/run_critical_experiments.sh           # Run all
#   sbatch jobs/run_critical_experiments.sh extended  # Just extended baselines
#   sbatch jobs/run_critical_experiments.sh ablation  # Just lookahead ablation
#   sbatch jobs/run_critical_experiments.sh complex   # Just 15-node SCM
# ============================================================================

echo "=============================================="
echo "Critical Experiments for ICML Response"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "=============================================="

# Setup environment
cd $SLURM_SUBMIT_DIR
source ~/.bashrc
conda activate ace_env

# Create logs directory
mkdir -p logs

# Parse argument for selective execution
EXPERIMENT=${1:-all}

case $EXPERIMENT in
    extended)
        echo "Running: Extended Baselines Only"
        python run_critical_experiments.py --extended-baselines --seeds 42 123 456 789 1011
        ;;
    ablation)
        echo "Running: Lookahead Ablation Only"
        python run_critical_experiments.py --lookahead-ablation --seeds 42 123 456 789 1011
        ;;
    complex)
        echo "Running: Complex 15-Node SCM Only"
        python run_critical_experiments.py --complex-scm --seeds 42 123 456 789 1011
        ;;
    all|*)
        echo "Running: ALL Critical Experiments"
        python run_critical_experiments.py --all --seeds 42 123 456 789 1011
        ;;
esac

echo ""
echo "=============================================="
echo "Completed: $(date)"
echo "=============================================="
