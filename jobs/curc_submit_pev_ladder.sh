#!/bin/bash
# =============================================================================
# PEV ladder -- the post-audit experiment set (17 Sept 2026)
#
# Everything here is CPU and runs inside the corrected baseline framework
# (frozen coefficients, both evaluators logged), so every arm is on exactly
# the footing of results/audit_reruns/. Arms:
#
#   pev              ensemble student (K=5, bootstrap) + propagated-epistemic-
#                    variance acquisition (baselines.PropagatedVariancePolicy)
#   random_ens       same ensemble student, uniform random policy  <- isolates
#                    the acquisition from the student
#   round_robin_ens  same ensemble student, round-robin
#   random / round_robin (single-student) on the hetero family only -- the
#                    audit already has them on LargeScaleSCM and at 5 nodes
#
# Suites (default ROOT/results/pev_ladder/<suite>/<method>/seed_<s>/):
#   t1_5node    5-node, 171 ep, seeds 141..618 + 42..1011        (3 methods x 10)
#   t2_30node   LargeScaleSCM 30, 150 ep, seeds 42..1011          (3 x 5)
#   hetero30    HeterogeneousSCM 30, 150 ep, seeds 42..1011       (5 x 5)
#   scaling     LargeScaleSCM N=15/50, 40 ep (pev, random_ens)   (2 x 2 x 5)
#   hetero_sc   HeterogeneousSCM N=15/50, 40 ep (pev, random_ens, random) (3 x 2 x 5)
#
# Usage (from /projects/paco0228/ACE, after git pull):
#   bash jobs/curc_submit_pev_ladder.sh
#   SUITES="t1_5node hetero30" bash jobs/curc_submit_pev_ladder.sh
# SKIP_COMPLETED=1 (default) skips cells that already have summary.csv.
# =============================================================================
set -euo pipefail
cd /projects/paco0228/ACE

ROOT="${ROOT:-/scratch/alpine/paco0228/ACE/results/pev_ladder}"
SUITES="${SUITES:-t1_5node t2_30node hetero30 scaling hetero_sc}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
CPU_PART="${CPU_PART:-acpu}"
CPU_QOS="${CPU_QOS:-cpu-normal}"
T1_SEEDS="${T1_SEEDS:-141 271 314 577 618 42 123 456 789 1011}"
T2_SEEDS="${T2_SEEDS:-42 123 456 789 1011}"
K="${K:-5}"

mkdir -p "$ROOT/logs"
N=0
done_cell() { [ "$SKIP_COMPLETED" = "1" ] && [ -f "$1/summary.csv" ]; }
submit_cpu() {  # NAME HOURS MEM EXPORTS WORKER
    local name=$1 hours=$2 mem=$3 exports=$4 worker=$5 job
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
      for M in pev random_ens round_robin_ens; do for S in $T1_SEEDS; do
        D="$ROOT/t1_5node/$M/seed_$S"; done_cell "$D" && { echo "  SKIP (done): $M s$S"; continue; }
        submit_cpu "pev5_${M}_s${S}" 6 8G "METHOD=$M,SEED=$S,OUT=$ROOT/t1_5node,EPISODES=171,EXTRA_ARGS=--ensemble_size $K" jobs/curc_5node_baseline_seed.sh
      done; done ;;
    t2_30node)
      for M in pev random_ens round_robin_ens; do for S in $T2_SEEDS; do
        D="$ROOT/t2_30node/$M/seed_$S"; done_cell "$D" && { echo "  SKIP (done): $M s$S"; continue; }
        submit_cpu "pev30_${M}_s${S}" 20 16G "METHOD=$M,SEED=$S,OUT=$ROOT/t2_30node,EPISODES=150,EXTRA_ARGS=--ensemble_size $K" jobs/curc_30node_baseline_seed.sh
      done; done ;;
    hetero30)
      for M in random round_robin pev random_ens round_robin_ens; do for S in $T2_SEEDS; do
        D="$ROOT/hetero30/$M/seed_$S"; done_cell "$D" && { echo "  SKIP (done): $M s$S"; continue; }
        submit_cpu "het30_${M}_s${S}" 20 16G "METHOD=$M,SEED=$S,OUT=$ROOT/hetero30,EPISODES=150,EXTRA_ARGS=--family hetero --ensemble_size $K" jobs/curc_30node_baseline_seed.sh
      done; done ;;
    scaling)
      for SC in 15 50; do for M in pev random_ens; do for S in $T2_SEEDS; do
        D="$ROOT/scaling/nodes$SC/$M/seed_$S"; done_cell "$D" && { echo "  SKIP (done): N=$SC $M s$S"; continue; }
        submit_cpu "pevsc${SC}_${M}_s${S}" 12 16G "METHOD=$M,SEED=$S,OUT=$ROOT/scaling/nodes$SC,EPISODES=40,N_NODES=$SC,EXTRA_ARGS=--ensemble_size $K" jobs/curc_30node_baseline_seed.sh
      done; done; done ;;
    hetero_sc)
      for SC in 15 50; do for M in pev random_ens random; do for S in $T2_SEEDS; do
        D="$ROOT/hetero_sc/nodes$SC/$M/seed_$S"; done_cell "$D" && { echo "  SKIP (done): N=$SC $M s$S"; continue; }
        submit_cpu "hetsc${SC}_${M}_s${S}" 12 16G "METHOD=$M,SEED=$S,OUT=$ROOT/hetero_sc/nodes$SC,EPISODES=40,N_NODES=$SC,EXTRA_ARGS=--family hetero --ensemble_size $K" jobs/curc_30node_baseline_seed.sh
      done; done; done ;;
    *) echo "unknown suite: $suite"; exit 1 ;;
  esac
done
echo; echo "$N job(s) submitted. Aggregate with:"
echo "  python scripts/analysis/aggregate_metric_audit.py --reruns $ROOT   # plus --arm for the audit baselines / ACE"
