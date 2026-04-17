#!/bin/bash
# ============================================================================
# CURC Resubmit on al40 (less busy partition)
# ============================================================================
#
# Cancels pending aa100 jobs and resubmits to al40 with shorter wall time.
# al40 has near-immediate scheduling vs aa100 which is severely backlogged.
#
# L40 is ~70% the speed of A100 but jobs actually run.
# 8h wall time fits ACE runs (which take ~6h on L40 with early stopping).
#
# Usage:
#   cd /projects/paco0228/ACE
#   bash jobs/curc_resubmit_al40.sh
# ============================================================================

set -euo pipefail

cd /projects/paco0228/ACE
source /projects/paco0228/miniconda3/etc/profile.d/conda.sh
conda activate ace

# Reuse existing output directory (where baselines & Bayesian OED already landed)
OUT="/projects/paco0228/ACE/results/curc_20260415_152624"
mkdir -p "$OUT/logs"

echo "================================================================"
echo " CURC al40 Resubmit (shorter wall time, less busy partition)"
echo "================================================================"
echo " Output : $OUT"
echo ""

# ----------------------------------------------------------------
# Step 1: Cancel old pending aa100 GPU jobs
# ----------------------------------------------------------------
echo ">>> Cancelling old pending aa100 GPU jobs <<<"
PENDING=$(squeue -u paco0228 -p aa100 -t PENDING --noheader -o "%i")
if [ -n "$PENDING" ]; then
    echo "$PENDING" | xargs scancel
    echo "  Cancelled $(echo "$PENDING" | wc -l) pending jobs"
else
    echo "  No pending aa100 jobs to cancel"
fi
sleep 2

# ----------------------------------------------------------------
# Helper: submit GPU job to al40
# ----------------------------------------------------------------
submit_al40() {
    local name="$1"; local script="$2"; shift 2
    sbatch --parsable \
        --job-name="$name" \
        --partition=al40 --qos=normal \
        --nodes=1 --ntasks=1 --gres=gpu:1 \
        --cpus-per-task=8 --mem=32G \
        --time=08:00:00 \
        --output="$OUT/logs/${name}_%j.out" \
        --error="$OUT/logs/${name}_%j.err" \
        --export=ALL,OUT=$OUT,"$@" \
        "$script"
}

# ----------------------------------------------------------------
# Phase A: ACE additional seeds (5 seeds)
# ----------------------------------------------------------------
echo ""
echo ">>> Phase A: ACE seeds on al40 <<<"
for SEED in 314 271 577 618 141; do
    JOB=$(submit_al40 "ace_s${SEED}" jobs/curc_ace_seed.sh "SEED=$SEED")
    echo "  ACE seed $SEED -> Job $JOB"
done

# ----------------------------------------------------------------
# Phase G: Graph misspecification (5 types x 3 seeds)
# ----------------------------------------------------------------
echo ""
echo ">>> Phase G: Graph misspecification on al40 <<<"
for MISSPEC in none missing_edge extra_edge reversed_edge missing_and_extra; do
    for SEED in 42 123 456; do
        JOB=$(submit_al40 "mis_${MISSPEC:0:3}_s${SEED}" jobs/curc_misspec_seed.sh "MISSPEC=$MISSPEC,SEED=$SEED")
        echo "  Misspec $MISSPEC seed $SEED -> Job $JOB"
    done
done

# ----------------------------------------------------------------
# Phase H: Hyperparameter grid (4x4 x 2 seeds = 32 jobs)
# ----------------------------------------------------------------
echo ""
echo ">>> Phase H: Hyperparameter grid on al40 <<<"
for ALPHA_KEY in 0.01 0.05 0.1 0.2; do
    for GAMMA in 0.01 0.05 0.1 0.2; do
        for SEED in 42 123; do
            case $ALPHA_KEY in
                0.01) COV=6.0 ;;
                0.05) COV=30.0 ;;
                0.1)  COV=60.0 ;;
                0.2)  COV=120.0 ;;
            esac
            LABEL="a${ALPHA_KEY}_g${GAMMA}_s${SEED}"
            JOB=$(submit_al40 "hp_${LABEL}" jobs/curc_hyperparam_cell.sh "COV=$COV,GAMMA=$GAMMA,SEED=$SEED,LABEL=$LABEL")
            echo "  Hyperparam $LABEL -> Job $JOB"
        done
    done
done

# ----------------------------------------------------------------
# Phase I: 30-node large-scale ACE (3 seeds, 12h since 30-node is bigger)
# ----------------------------------------------------------------
echo ""
echo ">>> Phase I: 30-node ACE on al40 (12h wall) <<<"
for SEED in 42 123 456; do
    JOB=$(sbatch --parsable \
        --job-name="ls30_s${SEED}" \
        --partition=al40 --qos=normal \
        --nodes=1 --ntasks=1 --gres=gpu:1 \
        --cpus-per-task=8 --mem=32G \
        --time=12:00:00 \
        --output="$OUT/logs/large_scale_seed${SEED}_%j.out" \
        --error="$OUT/logs/large_scale_seed${SEED}_%j.err" \
        --export=ALL,SEED=$SEED,OUT=$OUT \
        jobs/curc_large_scale_seed.sh)
    echo "  30-node seed $SEED -> Job $JOB"
done

echo ""
echo "================================================================"
echo " RESUBMITTED ON al40"
echo "================================================================"
echo " Monitor: squeue -u paco0228"
echo "================================================================"
