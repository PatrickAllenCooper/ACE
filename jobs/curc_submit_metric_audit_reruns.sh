#!/bin/bash
# =============================================================================
# Re-run every baseline cell after the 10 Sept 2026 metric audit
# (docs/development/guidance/metric_audit_2026-09-10.md).
#
# Why: the baselines were scored by a different evaluator than ACE (roots x1.0
# vs x0.2; observational vs broad-range validation), and at >=15 nodes they ran
# on a LargeScaleSCM whose mechanism coefficients were redrawn every batch.
# All runners now (a) freeze the coefficients to ACE's exact per-seed draw and
# (b) log both evaluators per step. Nothing here touches ACE runs.
#
# Every suite writes under a FRESH root so the old (invalid) results stay on
# disk for the record:
#   $ROOT/t1_5node/<method>/seed_<s>/        Table 1 baselines, 171 ep
#   $ROOT/t2_30node/<method>/seed_<s>/       Table 2 baselines, 150 ep (seeds 42..1011)
#                                            + 300 ep on 2022..2026 (expansion controls)
#   $ROOT/scaling/nodes<N>/random/seed_<s>/  scaling-sweep Random arm
#   $ROOT/bf5_baselines/<method>/seed_<s>/   5-node budget-fairness (query-matched)
#   $ROOT/bf30_baselines/<method>/seed_<s>/  30-node budget-fairness (query-matched)
#   $ROOT/boed5/seed_<s>/suite/bayesian_baseline/  Table 1 Bayesian-OED row, one job per seed
#
# Usage (from /projects/paco0228/ACE, after `git pull`):
#   bash jobs/curc_submit_metric_audit_reruns.sh                 # all suites
#   SUITES="t1_5node t2_30node" bash jobs/curc_submit_metric_audit_reruns.sh
# SKIP_COMPLETED=1 (default) skips cells that already have a summary.csv
# (the runners write it only on completion). All CPU; no GPU jobs.
# =============================================================================
set -euo pipefail
cd /projects/paco0228/ACE

ROOT="${ROOT:-/scratch/alpine/paco0228/ACE/results/audit_reruns}"
SUITES="${SUITES:-t1_5node t2_30node scaling_random bf5_baselines bf30_baselines boed5}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
CPU_PART="${CPU_PART:-acpu}"
CPU_QOS="${CPU_QOS:-cpu-normal}"
T1_SEEDS="${T1_SEEDS:-141 271 314 577 618 42 123 456 789 1011}"
T2_SEEDS="${T2_SEEDS:-42 123 456 789 1011}"
T2_EXPANSION_SEEDS="${T2_EXPANSION_SEEDS:-2022 2023 2024 2025 2026}"
SCALING_SEEDS="${SCALING_SEEDS:-42 123 456 789 1011}"
SCALING_EPISODES="${SCALING_EPISODES:-150}"    # what the original Random arms ran (aggregate --max-episode 40 for the plateau-budget view)
BF5_BUDGET="${BF5_BUDGET:-3204460}"            # mean total samples, 5-node ace_env
BF30_BUDGET="${BF30_BUDGET:-483460}"           # mean total samples, 30-node ace_env

mkdir -p "$ROOT/logs"
N=0
done_cell() { [ "$SKIP_COMPLETED" = "1" ] && [ -f "$1/summary.csv" ]; }

# submit_cpu NAME HOURS MEM EXPORTS WORKER
submit_cpu() {
    local name=$1 hours=$2 mem=$3 exports=$4 worker=$5
    local job
    job=$(sbatch --parsable --job-name="$name" \
        --partition="$CPU_PART" --qos="$CPU_QOS" --nodes=1 --ntasks=1 \
        --cpus-per-task=4 --mem="$mem" --time="${hours}:00:00" \
        --output="$ROOT/logs/${name}_%j.out" --error="$ROOT/logs/${name}_%j.err" \
        --export="ALL,$exports" "$worker")
    echo "  Submitted: $name -> Job $job"; N=$((N+1))
}

for suite in $SUITES; do
  echo "== suite: $suite =="
  case $suite in
    t1_5node)
      for M in random round_robin max_variance ppo; do for S in $T1_SEEDS; do
        D="$ROOT/t1_5node/$M/seed_$S"
        if done_cell "$D"; then echo "  SKIP (done): t1 $M s$S"; continue; fi
        submit_cpu "aud5_${M}_s${S}" 3 8G "METHOD=$M,SEED=$S,OUT=$ROOT/t1_5node,EPISODES=171" jobs/curc_5node_baseline_seed.sh
      done; done ;;
    t2_30node)
      for M in random round_robin max_variance bayesian_oed; do
        for S in $T2_SEEDS; do
          D="$ROOT/t2_30node/$M/seed_$S"
          if done_cell "$D"; then echo "  SKIP (done): t2 $M s$S"; continue; fi
          H=8; [ "$M" = bayesian_oed ] && H=12
          submit_cpu "aud30_${M}_s${S}" $H 16G "METHOD=$M,SEED=$S,OUT=$ROOT/t2_30node,EPISODES=150" jobs/curc_30node_baseline_seed.sh
        done
        for S in $T2_EXPANSION_SEEDS; do
          D="$ROOT/t2_30node/$M/seed_$S"
          if done_cell "$D"; then echo "  SKIP (done): t2 $M s$S"; continue; fi
          H=12; [ "$M" = bayesian_oed ] && H=23
          submit_cpu "aud30x_${M}_s${S}" $H 16G "METHOD=$M,SEED=$S,OUT=$ROOT/t2_30node,EPISODES=300" jobs/curc_30node_baseline_seed.sh
        done
      done ;;
    scaling_random)
      for SC in 15 30 50; do for S in $SCALING_SEEDS; do
        D="$ROOT/scaling/nodes$SC/random/seed_$S"
        if done_cell "$D"; then echo "  SKIP (done): scaling N=$SC s$S"; continue; fi
        submit_cpu "audsc${SC}_random_s${S}" 4 16G "SCALE=$SC,METHOD=random,SEED=$S,OUT=$ROOT/scaling,EPISODES=$SCALING_EPISODES" jobs/curc_scaling_seed.sh
      done; done ;;
    boed5)
      # Table 1's Bayesian-OED row via run_reviewer_experiments.py. One seed is
      # ~6h on acpu and the QoS caps jobs at 24h, so one job per seed, each
      # writing its own $ROOT/boed5/seed_<s>/suite/bayesian_baseline/ summary.
      for S in $T1_SEEDS; do
        if [ "$SKIP_COMPLETED" = "1" ] && [ -f "$ROOT/boed5/seed_$S/suite/bayesian_baseline/bayesian_oed_summary.csv" ]; then
          echo "  SKIP (done): boed5 s$S"; continue; fi
        submit_cpu "aud5_boed_s${S}" 12 8G "OUT=$ROOT/boed5/seed_$S,SEEDS=$S" jobs/curc_5node_boed_worker.sh
      done ;;
    bf5_baselines)
      OUT="$ROOT/bf5_baselines" SKIP_COMPLETED="$SKIP_COMPLETED" \
        bash jobs/curc_submit_5node_budget_fairness_baselines.sh "$BF5_BUDGET" ;;
    bf30_baselines)
      OUT="$ROOT/bf30_baselines" \
        bash jobs/curc_submit_30node_budget_fairness_baselines.sh "$BF30_BUDGET" ;;
    *) echo "unknown suite: $suite"; exit 1 ;;
  esac
done
echo
echo "$N job(s) submitted by this script (budget-fairness suites report their own counts)."
echo "Aggregate when done:"
echo "  python scripts/analysis/aggregate_metric_audit.py --reruns $ROOT"
