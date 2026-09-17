#!/bin/bash
# =============================================================================
# PEV ladder -- the post-audit experiment set (17 Sept 2026), MATCHED STUDENT
#
# The audit's baseline re-runs used the pre-audit student, a (16,) ReLU MLP
# trained 50 epochs/step; ACE's student is (64, 64) trained 100 epochs/step.
# Every arm here runs the ACE student (runner default --student_arch ace,
# --train_epochs 100) inside the corrected framework (frozen coefficients,
# both evaluators logged), so all arms -- and ACE's own runs -- share one
# learner for the first time. Arms:
#
#   random, round_robin        single student (the passive bar, re-run on the matched student)
#   random_ens, round_robin_ens ensemble student (K=5, bootstrap), passive policies
#   pev                        ensemble student + expected-variance-reduction acquisition
#   pev_var                    ensemble student + naive visited-variance acquisition (ablation)
#
# Suites  (ROOT/<suite>/<arm>/seed_<s>/):
#   t1_5node    5-node, 171 ep, seeds 141..618 + 42..1011            6 arms x 10
#   bf5_matched 5-node, total-query-matched to ACE (3.2M samples)     4 arms x 5
#   t2_30node   LargeScaleSCM-30, 150 ep, seeds 42..1011              6 arms x 5
#   hetero30    HeterogeneousSCM-30, 150 ep, seeds 42..1011           6 arms x 5
#   scaling     LargeScaleSCM N=15/50, 40 ep: random, random_ens, pev 3 x 2 x 5
#   hetero_sc   HeterogeneousSCM N=15/50, 40 ep: same                 3 x 2 x 5
#
# Usage (from /projects/paco0228/ACE, after git pull):
#   bash jobs/curc_submit_pev_ladder.sh
#   SUITES="t1_5node hetero30" bash jobs/curc_submit_pev_ladder.sh
# SKIP_COMPLETED=1 (default) skips cells that already have summary.csv.
# =============================================================================
set -euo pipefail
cd /projects/paco0228/ACE

ROOT="${ROOT:-/scratch/alpine/paco0228/ACE/results/pev_ladder}"
SUITES="${SUITES:-t1_5node bf5_matched t2_30node hetero30 scaling hetero_sc}"
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
# arm -> (method, extra runner args)
arm_method() { case $1 in pev_var) echo pev ;; *) echo "$1" ;; esac; }
arm_extra()  { case $1 in pev_var) echo "--pev_scoring var --tag pev_var" ;; pev) echo "--pev_scoring ivr" ;; *) echo "" ;; esac; }
# hours: single-student arms are ~5x cheaper than K=5 ensembles
arm_hours()  { case $1 in random|round_robin) echo "$2" ;; *) echo "$3" ;; esac; }

for suite in $SUITES; do
  echo "== suite: $suite =="
  case $suite in
    t1_5node)
      for A in random round_robin random_ens round_robin_ens pev pev_var; do for S in $T1_SEEDS; do
        D="$ROOT/t1_5node/$A/seed_$S"; done_cell "$D" && { echo "  SKIP (done): $A s$S"; continue; }
        submit_cpu "pev5_${A}_s${S}" $(arm_hours $A 3 8) 8G "METHOD=$(arm_method $A),SEED=$S,OUT=$ROOT/t1_5node,EPISODES=171,EXTRA_ARGS=--ensemble_size $K $(arm_extra $A)" jobs/curc_5node_baseline_seed.sh
      done; done ;;
    bf5_matched)
      # total-query-matched at ACE's 5-node budget (mean ace_env total samples),
      # matched student: the query-fair version of t1_5node
      for A in random round_robin random_ens pev; do for S in $T2_SEEDS; do
        D="$ROOT/bf5_matched/$A/seed_$S"; done_cell "$D" && { echo "  SKIP (done): $A s$S"; continue; }
        submit_cpu "bf5m_${A}_s${S}" $(arm_hours $A 12 23) 8G "METHOD=$(arm_method $A),SEED=$S,OUT=$ROOT/bf5_matched,EPISODES=2000,EXTRA_ARGS=--query_budget 3204460 --ensemble_size $K $(arm_extra $A)" jobs/curc_5node_baseline_seed.sh
      done; done ;;
    t2_30node|hetero30)
      FAM="large_scale"; [ "$suite" = hetero30 ] && FAM="hetero"
      for A in random round_robin random_ens round_robin_ens pev pev_var; do for S in $T2_SEEDS; do
        D="$ROOT/$suite/$A/seed_$S"; done_cell "$D" && { echo "  SKIP (done): $A s$S"; continue; }
        submit_cpu "${suite}_${A}_s${S}" $(arm_hours $A 10 23) 16G "METHOD=$(arm_method $A),SEED=$S,OUT=$ROOT/$suite,EPISODES=150,EXTRA_ARGS=--family $FAM --ensemble_size $K $(arm_extra $A)" jobs/curc_30node_baseline_seed.sh
      done; done ;;
    scaling|hetero_sc)
      FAM="large_scale"; [ "$suite" = hetero_sc ] && FAM="hetero"
      for SC in 15 50; do for A in random random_ens pev; do for S in $T2_SEEDS; do
        D="$ROOT/$suite/nodes$SC/$A/seed_$S"; done_cell "$D" && { echo "  SKIP (done): N=$SC $A s$S"; continue; }
        submit_cpu "${suite}${SC}_${A}_s${S}" $(arm_hours $A 6 16) 16G "METHOD=$(arm_method $A),SEED=$S,OUT=$ROOT/$suite/nodes$SC,EPISODES=40,N_NODES=$SC,EXTRA_ARGS=--family $FAM --ensemble_size $K $(arm_extra $A)" jobs/curc_30node_baseline_seed.sh
      done; done; done ;;
    *) echo "unknown suite: $suite"; exit 1 ;;
  esac
done
echo; echo "$N job(s) submitted. Aggregate with:"
echo "  python scripts/analysis/aggregate_metric_audit.py --reruns $ROOT   # plus --arm for ACE's own runs"
