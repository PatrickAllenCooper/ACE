#!/bin/bash
#SBATCH --job-name=phillips_baselines
#SBATCH --partition=aa100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/phillips_baselines_%j.out
#SBATCH --error=logs/phillips_baselines_%j.err

# Phillips Curve Baseline Comparison
# Run Random, Round-Robin regime selection for N=5 seeds

echo "=============================================="
echo "Phillips Baselines (2 methods × 5 seeds)"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "=============================================="

# --- Environment Setup ---
if command -v module &> /dev/null; then
    module purge || true
    module load cuda || echo "Warning: Could not load cuda module."
fi

export HF_HOME="${HF_HOME:-/projects/$USER/cache/huggingface}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/projects/$USER/cache/matplotlib}"
mkdir -p "$HF_HOME" "$MPLCONFIGDIR" 2>/dev/null || true

if [ "$CONDA_DEFAULT_ENV" != "ace" ]; then
    if [ -f "/projects/$USER/miniconda3/etc/profile.d/conda.sh" ]; then
        source /projects/$USER/miniconda3/etc/profile.d/conda.sh
        conda activate ace
    fi
fi

cd $SLURM_SUBMIT_DIR
mkdir -p logs

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="results/phillips_baselines_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

# Run baselines for each seed
for SEED in 42 123 456 789 1011; do
    for METHOD in random round_robin; do
        echo ""
        echo "Running Phillips $METHOD seed $SEED..."
        
        python -u experiments/phillips_curve.py \
            --policy $METHOD \
            --episodes 50 \
            --seed $SEED \
            --output "$OUTPUT_DIR/${METHOD}_seed${SEED}"
        
        echo "  ✓ $METHOD seed $SEED complete"
        
        # CRITICAL: Save cumulative summary after each method
        python -c "
import pandas as pd
import glob
import os

csvs = glob.glob('$OUTPUT_DIR/*/phillips_results.csv')
if csvs:
    results = []
    for f in csvs:
        parts = os.path.basename(os.path.dirname(f)).split('_')
        method = '_'.join(parts[:-1]) if len(parts) > 1 else parts[0]
        seed = parts[-1].replace('seed', '') if 'seed' in parts[-1] else '42'
        df = pd.read_csv(f)
        final_loss = df['loss'].iloc[-1] if 'loss' in df.columns else 0
        results.append({'method': method, 'seed': int(seed), 'final_loss': final_loss})
    
    summary = pd.DataFrame(results)
    summary.to_csv('$OUTPUT_DIR/phillips_baselines_summary.csv', index=False)
    print(f'  Saved summary: {len(results)} runs')
"
    done
done

echo ""
echo "=============================================="
echo "Phillips baselines complete"
echo "Completed: $(date)"
echo "Results: $OUTPUT_DIR/phillips_baselines_summary.csv"
echo "=============================================="
