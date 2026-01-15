#!/bin/bash
#
# Run All ACE Paper Experiments on HPC (SLURM)
#
# This script submits separate SLURM jobs for:
#   1. ACE (DPO) experiment
#   2. Baselines (Random, Round-Robin, Max-Variance, PPO)
#   3. Comparison analysis (runs after 1 & 2 complete)
#
# Usage:
#   ./run_all_hpc.sh              # Full paper experiments
#   ./run_all_hpc.sh --quick      # Quick validation
#   ./run_all_hpc.sh --skip-ppo   # Skip PPO baseline (faster)
#
# Output:
#   results/paper_run_YYYYMMDD_HHMMSS/
#   logs/ace_*.out, logs/baselines_*.out, logs/comparison_*.out
#

set -e  # Exit on error

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default settings (full paper experiments)
ACE_EPISODES=500
BASELINE_EPISODES=100
ACE_STEPS=25
SKIP_PPO=false
USE_CUSTOM=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            ACE_EPISODES=10
            BASELINE_EPISODES=10
            USE_CUSTOM=true
            shift
            ;;
        --skip-ppo)
            SKIP_PPO=true
            shift
            ;;
        --ace-episodes)
            ACE_EPISODES="$2"
            shift 2
            ;;
        --baseline-episodes)
            BASELINE_EPISODES="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create timestamped run directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="results/paper_run_${TIMESTAMP}"
ACE_OUTPUT="${RUN_DIR}/ace"
BASELINES_OUTPUT="${RUN_DIR}/baselines"
COMPARISON_OUTPUT="${RUN_DIR}/comparison"

mkdir -p "$RUN_DIR"
mkdir -p "$ACE_OUTPUT"
mkdir -p "$BASELINES_OUTPUT"
mkdir -p "$COMPARISON_OUTPUT"
mkdir -p logs

echo "=============================================="
echo "ACE Paper Experiments - HPC Submission"
echo "=============================================="
echo "Timestamp: $TIMESTAMP"
echo "Run Directory: $RUN_DIR"
echo "ACE Episodes: $ACE_EPISODES"
echo "Baseline Episodes: $BASELINE_EPISODES"
echo "Skip PPO: $SKIP_PPO"
echo "Use Custom Transformer: $USE_CUSTOM"
echo "=============================================="

# =============================================================================
# JOB 1: ACE (DPO) Experiment
# =============================================================================

ACE_CUSTOM_FLAG=""
if [ "$USE_CUSTOM" = true ]; then
    ACE_CUSTOM_FLAG="--custom"
fi

ACE_JOB=$(sbatch --parsable << EOF
#!/bin/bash
#SBATCH --job-name=ace_dpo_${TIMESTAMP}
#SBATCH --partition=aa100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/ace_${TIMESTAMP}_%j.out
#SBATCH --error=logs/ace_${TIMESTAMP}_%j.err

# Environment setup
module purge
module load cuda

export HF_HOME="/projects/\$USER/cache/huggingface"
export MPLCONFIGDIR="/projects/\$USER/cache/matplotlib"
mkdir -p \$HF_HOME \$MPLCONFIGDIR

source /projects/\$USER/miniconda3/etc/profile.d/conda.sh
conda activate ace

echo "=== ACE (DPO) Experiment ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Node: \$(hostname)"
echo "Started: \$(date)"
nvidia-smi

cd \$SLURM_SUBMIT_DIR

python ace_experiments.py \\
    --episodes $ACE_EPISODES \\
    --steps $ACE_STEPS \\
    --output "$ACE_OUTPUT" \\
    $ACE_CUSTOM_FLAG

echo "Finished: \$(date)"
EOF
)

echo "Submitted ACE job: $ACE_JOB"

# =============================================================================
# JOB 2: Baselines
# =============================================================================

BASELINE_FLAG="--all"
if [ "$SKIP_PPO" = false ]; then
    BASELINE_FLAG="--all_with_ppo"
fi

BASELINES_JOB=$(sbatch --parsable << EOF
#!/bin/bash
#SBATCH --job-name=baselines_${TIMESTAMP}
#SBATCH --partition=aa100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/baselines_${TIMESTAMP}_%j.out
#SBATCH --error=logs/baselines_${TIMESTAMP}_%j.err

# Environment setup
module purge
module load cuda

export HF_HOME="/projects/\$USER/cache/huggingface"
export MPLCONFIGDIR="/projects/\$USER/cache/matplotlib"
mkdir -p \$HF_HOME \$MPLCONFIGDIR

source /projects/\$USER/miniconda3/etc/profile.d/conda.sh
conda activate ace

echo "=== Baseline Experiments ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Node: \$(hostname)"
echo "Started: \$(date)"
nvidia-smi

cd \$SLURM_SUBMIT_DIR

python baselines.py \\
    $BASELINE_FLAG \\
    --episodes $BASELINE_EPISODES \\
    --steps $ACE_STEPS \\
    --output "$BASELINES_OUTPUT"

echo "Finished: \$(date)"
EOF
)

echo "Submitted Baselines job: $BASELINES_JOB"

# =============================================================================
# JOB 3: Comparison Analysis (depends on Jobs 1 & 2)
# =============================================================================

COMPARISON_JOB=$(sbatch --parsable --dependency=afterok:${ACE_JOB}:${BASELINES_JOB} << EOF
#!/bin/bash
#SBATCH --job-name=comparison_${TIMESTAMP}
#SBATCH --partition=short
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/comparison_${TIMESTAMP}_%j.out
#SBATCH --error=logs/comparison_${TIMESTAMP}_%j.err

# Environment setup
module purge

source /projects/\$USER/miniconda3/etc/profile.d/conda.sh
conda activate ace

echo "=== Comparison Analysis ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Node: \$(hostname)"
echo "Started: \$(date)"

cd \$SLURM_SUBMIT_DIR

# Find the actual run directories (they have timestamps)
ACE_RUN=\$(ls -td ${ACE_OUTPUT}/run_* 2>/dev/null | head -1)
BASELINES_RUN=\$(ls -td ${BASELINES_OUTPUT}/baselines_* 2>/dev/null | head -1)

echo "ACE results: \$ACE_RUN"
echo "Baselines results: \$BASELINES_RUN"

# Generate comparison using Python
python << 'PYTHON_SCRIPT'
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

run_dir = "${RUN_DIR}"
ace_output = "${ACE_OUTPUT}"
baselines_output = "${BASELINES_OUTPUT}"
comparison_output = "${COMPARISON_OUTPUT}"

# Find actual run directories
ace_runs = sorted(glob(f"{ace_output}/run_*"), reverse=True)
baseline_runs = sorted(glob(f"{baselines_output}/baselines_*"), reverse=True)

ace_run = ace_runs[0] if ace_runs else None
baselines_run = baseline_runs[0] if baseline_runs else None

print(f"ACE run: {ace_run}")
print(f"Baselines run: {baselines_run}")

# Load and compare results
results = {}

# Load baseline results
if baselines_run:
    for name in ["random", "round_robin", "max_variance", "ppo"]:
        path = os.path.join(baselines_run, f"{name}_results.csv")
        if os.path.exists(path):
            results[name] = pd.read_csv(path)
            print(f"Loaded {name}: {len(results[name])} records")

# Load ACE results
if ace_run:
    ace_path = os.path.join(ace_run, "node_losses.csv")
    if os.path.exists(ace_path):
        results["ace"] = pd.read_csv(ace_path)
        print(f"Loaded ACE: {len(results['ace'])} records")

if not results:
    print("No results found!")
    sys.exit(1)

# Generate comparison figure
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("ACE Paper: Method Comparison", fontsize=14)

colors = {
    "random": "red", "round_robin": "blue", 
    "max_variance": "green", "ppo": "orange", "ace": "purple"
}
display_names = {
    "random": "Random", "round_robin": "Round-Robin",
    "max_variance": "Max-Variance", "ppo": "PPO", "ace": "ACE (DPO)"
}

# Plot convergence curves
nodes = ["X1", "X2", "X3", "X4", "X5"]

ax = axes[0, 0]
for name, df in results.items():
    if "total_loss" in df.columns:
        mean_loss = df.groupby("step")["total_loss"].mean()
        ax.plot(mean_loss.index, mean_loss.values, 
                label=display_names.get(name, name), color=colors.get(name, "gray"))
ax.set_xlabel("Step")
ax.set_ylabel("Total MSE")
ax.set_title("Total Loss Convergence")
ax.legend(fontsize=8)
ax.set_yscale("log")

for idx, node in enumerate(nodes):
    ax = axes[(idx + 1) // 3, (idx + 1) % 3]
    for name, df in results.items():
        col = f"loss_{node}"
        if col in df.columns:
            mean_loss = df.groupby("step")[col].mean()
            ax.plot(mean_loss.index, mean_loss.values,
                    label=display_names.get(name, name), color=colors.get(name, "gray"))
    ax.set_xlabel("Step")
    ax.set_ylabel("MSE")
    ax.set_title(f"{node} Mechanism")
    ax.legend(fontsize=7)
    ax.set_yscale("log")
    ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(comparison_output, "convergence_curves.png"), dpi=150)
plt.close()
print(f"Saved convergence_curves.png")

# Generate summary table
summary_rows = []
for name, df in results.items():
    final_step = df["step"].max()
    final_df = df[df["step"] == final_step]
    row = {"Method": display_names.get(name, name)}
    if "total_loss" in final_df.columns:
        row["Total Loss"] = f"{final_df['total_loss'].mean():.3f}"
    for node in nodes:
        col = f"loss_{node}"
        if col in final_df.columns:
            row[f"{node}"] = f"{final_df[col].mean():.3f}"
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(comparison_output, "summary_table.csv"), index=False)
print(f"Saved summary_table.csv")
print("\n" + summary_df.to_string(index=False))
PYTHON_SCRIPT

# Generate summary file
cat > "${RUN_DIR}/paper_summary.txt" << SUMMARY
======================================================================
ACE PAPER EXPERIMENT SUMMARY (HPC)
======================================================================
Run Directory: ${RUN_DIR}
Timestamp: ${TIMESTAMP}

Jobs:
  ACE (DPO):    $ACE_JOB
  Baselines:   $BASELINES_JOB
  Comparison:  \$SLURM_JOB_ID

Configuration:
  ACE Episodes: $ACE_EPISODES
  Baseline Episodes: $BASELINE_EPISODES
  Steps per Episode: $ACE_STEPS
  Skip PPO: $SKIP_PPO

Output Files:
  ${COMPARISON_OUTPUT}/convergence_curves.png
  ${COMPARISON_OUTPUT}/summary_table.csv
  ${RUN_DIR}/paper_summary.txt

======================================================================
SUMMARY

echo "Summary written to ${RUN_DIR}/paper_summary.txt"
echo "Finished: \$(date)"
EOF
)

echo "Submitted Comparison job: $COMPARISON_JOB (depends on $ACE_JOB, $BASELINES_JOB)"

# =============================================================================
# SUMMARY
# =============================================================================

echo ""
echo "=============================================="
echo "All jobs submitted!"
echo "=============================================="
echo ""
echo "Job IDs:"
echo "  ACE (DPO):   $ACE_JOB"
echo "  Baselines:   $BASELINES_JOB"
echo "  Comparison:  $COMPARISON_JOB (runs after ACE & Baselines complete)"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  watch -n 30 'squeue -u \$USER'"
echo ""
echo "View logs:"
echo "  tail -f logs/ace_${TIMESTAMP}_*.out"
echo "  tail -f logs/baselines_${TIMESTAMP}_*.out"
echo ""
echo "Results will be in:"
echo "  $RUN_DIR"
echo "=============================================="

# Save job info for later reference
cat > "${RUN_DIR}/job_info.txt" << JOBINFO
Timestamp: $TIMESTAMP
ACE Job: $ACE_JOB
Baselines Job: $BASELINES_JOB
Comparison Job: $COMPARISON_JOB

Configuration:
  ACE Episodes: $ACE_EPISODES
  Baseline Episodes: $BASELINE_EPISODES
  Steps: $ACE_STEPS
  Skip PPO: $SKIP_PPO
  Use Custom: $USE_CUSTOM
JOBINFO

echo "Job info saved to ${RUN_DIR}/job_info.txt"
