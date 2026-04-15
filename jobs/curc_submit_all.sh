#!/bin/bash
# ============================================================================
# CURC Complete Experiment Pipeline for NeurIPS 2026
# ============================================================================
#
# Submits ALL experiments as separate SLURM jobs so they run in parallel.
# Each job fits within the 24h aa100 normal QoS wall limit.
#
# Experiments:
#   A. ACE additional seeds (5 new seeds: 314, 271, 577, 618, 141)
#   B. Baselines at 171 episodes (5 new seeds)
#   C. Bayesian OED baseline (10 seeds)
#   D. Graph misspecification ablation (5 types x 3 seeds = 15 jobs)
#   E. Hyperparameter sensitivity grid (4x4 grid x 2 seeds = 32 jobs)
#   F. K ablation + Duffing + Phillips baselines (CPU, 1 job)
#   G. 30-node large-scale ACE (3 seeds)
#
# Usage (from /projects/paco0228/ACE):
#   cd /projects/paco0228/ACE
#   source setup_env.sh
#   bash jobs/curc_submit_all.sh
#
# Results land in: results/curc_YYYYMMDD_HHMMSS/
# ============================================================================

set -euo pipefail

cd /projects/paco0228/ACE

# Activate conda directly (bypass setup_env.sh which fails on login node)
source /projects/paco0228/miniconda3/etc/profile.d/conda.sh
conda activate ace

TS=$(date +%Y%m%d_%H%M%S)
OUT="/projects/paco0228/ACE/results/curc_${TS}"
mkdir -p "$OUT/logs"

echo "================================================================"
echo " NeurIPS 2026 -- Full Experiment Pipeline (CURC SLURM)"
echo "================================================================"
echo " Output   : $OUT"
echo " Submitted: $(date)"
echo "================================================================"
echo ""

# ----------------------------------------------------------------
# Helper: submit a GPU job
# ----------------------------------------------------------------
submit_gpu() {
    local name="$1"; local time="$2"; local script="$3"
    sbatch --parsable \
        --job-name="$name" \
        --partition=aa100 --qos=normal \
        --nodes=1 --ntasks=1 --gres=gpu:1 \
        --cpus-per-task=8 --mem=32G \
        --time="$time" \
        --output="$OUT/logs/${name}_%j.out" \
        --error="$OUT/logs/${name}_%j.err" \
        "$script"
}

# Helper: submit a CPU job
submit_cpu() {
    local name="$1"; local time="$2"; local script="$3"
    sbatch --parsable \
        --job-name="$name" \
        --partition=amilan --qos=normal \
        --nodes=1 --ntasks=1 \
        --cpus-per-task=8 --mem=16G \
        --time="$time" \
        --output="$OUT/logs/${name}_%j.out" \
        --error="$OUT/logs/${name}_%j.err" \
        "$script"
}

# ----------------------------------------------------------------
# Phase A: ACE additional seeds (5 seeds, one job per seed)
# ----------------------------------------------------------------
echo ">>> Phase A: ACE additional seeds <<<"
for SEED in 314 271 577 618 141; do
    JOB=$(sbatch --parsable \
        --job-name="ace_s${SEED}" \
        --partition=aa100 --qos=normal \
        --nodes=1 --ntasks=1 --gres=gpu:1 \
        --cpus-per-task=8 --mem=32G \
        --time=24:00:00 \
        --output="$OUT/logs/ace_seed${SEED}_%j.out" \
        --error="$OUT/logs/ace_seed${SEED}_%j.err" \
        --export=ALL,SEED=$SEED,OUT=$OUT \
        jobs/curc_ace_seed.sh)
    echo "  ACE seed $SEED -> Job $JOB"
done

# ----------------------------------------------------------------
# Phase B: Baselines at 171 episodes (5 new seeds, CPU)
# ----------------------------------------------------------------
echo ""
echo ">>> Phase B: Baselines 171 episodes <<<"
for SEED in 314 271 577 618 141; do
    JOB=$(sbatch --parsable \
        --job-name="base_s${SEED}" \
        --partition=amilan --qos=normal \
        --nodes=1 --ntasks=1 \
        --cpus-per-task=8 --mem=16G \
        --time=04:00:00 \
        --output="$OUT/logs/baselines_seed${SEED}_%j.out" \
        --error="$OUT/logs/baselines_seed${SEED}_%j.err" \
        --export=ALL,SEED=$SEED,OUT=$OUT \
        jobs/curc_baselines_seed.sh)
    echo "  Baselines seed $SEED -> Job $JOB"
done

# ----------------------------------------------------------------
# Phase C-F: CPU experiments (Bayesian OED, misspec, hyperparam,
#            K ablation, Duffing, Phillips) -- one combined job
# ----------------------------------------------------------------
echo ""
echo ">>> Phase C-F: CPU experiments (Bayesian OED, graph misspec, hyperparam, K, Duffing, Phillips) <<<"
JOB=$(sbatch --parsable \
    --job-name="cpu_suite" \
    --partition=amilan --qos=normal \
    --nodes=1 --ntasks=1 \
    --cpus-per-task=16 --mem=32G \
    --time=24:00:00 \
    --output="$OUT/logs/cpu_suite_%j.out" \
    --error="$OUT/logs/cpu_suite_%j.err" \
    --export=ALL,OUT=$OUT \
    jobs/curc_cpu_suite.sh)
echo "  CPU suite -> Job $JOB"

# ----------------------------------------------------------------
# Phase G: Graph misspecification ablation (ACE, 5 types x 3 seeds)
# ----------------------------------------------------------------
echo ""
echo ">>> Phase G: Graph misspecification ablation (ACE) <<<"
for MISSPEC in none missing_edge extra_edge reversed_edge missing_and_extra; do
    for SEED in 42 123 456; do
        JOB=$(sbatch --parsable \
            --job-name="mis_${MISSPEC:0:3}_s${SEED}" \
            --partition=aa100 --qos=normal \
            --nodes=1 --ntasks=1 --gres=gpu:1 \
            --cpus-per-task=8 --mem=32G \
            --time=24:00:00 \
            --output="$OUT/logs/misspec_${MISSPEC}_seed${SEED}_%j.out" \
            --error="$OUT/logs/misspec_${MISSPEC}_seed${SEED}_%j.err" \
            --export=ALL,MISSPEC=$MISSPEC,SEED=$SEED,OUT=$OUT \
            jobs/curc_misspec_seed.sh)
        echo "  Misspec $MISSPEC seed $SEED -> Job $JOB"
    done
done

# ----------------------------------------------------------------
# Phase H: Hyperparameter grid (4 alpha x 4 gamma x 2 seeds = 32 jobs)
# ----------------------------------------------------------------
echo ""
echo ">>> Phase H: Hyperparameter sensitivity grid <<<"
for ALPHA_KEY in 0.01 0.05 0.1 0.2; do
    for GAMMA in 0.01 0.05 0.1 0.2; do
        for SEED in 42 123; do
            # Map paper alpha to cov_bonus
            case $ALPHA_KEY in
                0.01) COV=6.0 ;;
                0.05) COV=30.0 ;;
                0.1)  COV=60.0 ;;
                0.2)  COV=120.0 ;;
            esac
            LABEL="a${ALPHA_KEY}_g${GAMMA}_s${SEED}"
            JOB=$(sbatch --parsable \
                --job-name="hp_${LABEL}" \
                --partition=aa100 --qos=normal \
                --nodes=1 --ntasks=1 --gres=gpu:1 \
                --cpus-per-task=8 --mem=32G \
                --time=24:00:00 \
                --output="$OUT/logs/hp_${LABEL}_%j.out" \
                --error="$OUT/logs/hp_${LABEL}_%j.err" \
                --export=ALL,COV=$COV,GAMMA=$GAMMA,SEED=$SEED,LABEL=$LABEL,OUT=$OUT \
                jobs/curc_hyperparam_cell.sh)
            echo "  Hyperparam $LABEL -> Job $JOB"
        done
    done
done

# ----------------------------------------------------------------
# Phase I: 30-node large-scale ACE (3 seeds)
# ----------------------------------------------------------------
echo ""
echo ">>> Phase I: 30-node large-scale ACE <<<"
for SEED in 42 123 456; do
    JOB=$(sbatch --parsable \
        --job-name="ls30_s${SEED}" \
        --partition=aa100 --qos=normal \
        --nodes=1 --ntasks=1 --gres=gpu:1 \
        --cpus-per-task=8 --mem=32G \
        --time=24:00:00 \
        --output="$OUT/logs/large_scale_seed${SEED}_%j.out" \
        --error="$OUT/logs/large_scale_seed${SEED}_%j.err" \
        --export=ALL,SEED=$SEED,OUT=$OUT \
        jobs/curc_large_scale_seed.sh)
    echo "  30-node seed $SEED -> Job $JOB"
done

echo ""
echo "================================================================"
echo " ALL JOBS SUBMITTED"
echo "================================================================"
echo " Monitor: squeue -u paco0228"
echo " Logs   : $OUT/logs/"
echo " Results: $OUT/"
echo ""
echo " When complete, copy results locally:"
echo "   scp -r paco0228@login.rc.colorado.edu:$OUT C:\\Users\\patri\\code\\ACE\\results\\"
echo "================================================================"
