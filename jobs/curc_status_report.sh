#!/bin/bash
# =============================================================================
# CURC experiment status dashboard for the ICLR 2027 resubmission.
#
# Run this ON CURC (login node), from /projects/paco0228/ACE:
#   git pull
#   bash jobs/curc_status_report.sh
#
# This sandbox (the Claude Code session that wrote this script) has no CURC
# access -- it cannot run squeue/sacct/find itself. This script is what to
# run by hand (or paste the output back) to get an answer to "what's in,
# what's still in flight."
#
# Reports, for every suite tracked in
# docs/development/guidance/current_status.txt:
#   1. Live queue (squeue)
#   2. Recent job history incl. FAILED/OOM/TIMEOUT (sacct, last 14 days)
#   3. Per-suite cell completion, using each suite's OWN "done" test -- the
#      same cell_done()/SKIP_COMPLETED convention its curc_submit_*.sh
#      script already uses, so "done" here means SKIP_COMPLETED=1 would
#      also skip that cell on resubmission.
#   4. /scratch/alpine quota
# =============================================================================
set -uo pipefail
cd /projects/paco0228/ACE || { echo "Run this on CURC, from /projects/paco0228/ACE"; exit 1; }
source /projects/paco0228/miniconda3/etc/profile.d/conda.sh
conda activate ace

SCRATCH="/scratch/alpine/paco0228/ACE/results"

echo "================================================================"
echo " ACE / ICLR 2027 resubmission -- CURC status report"
echo " $(date)"
echo "================================================================"

echo
echo "---- 1. Live queue (squeue -u \$USER) ----"
squeue -u "$USER" -o "%.10i %.30j %.8T %.10M %.6D %R" 2>/dev/null || echo "(squeue unavailable)"

echo
echo "---- 1b. Queue entries NOT matching any known ACE/ICLR job-name prefix ----"
echo "     (bf5_/bf30_/bf5bl_/bf30bl_/oed30_/ace30_/dpoalt_/nodeimp_/msA_/msB_/fr100_/ksK)"
echo "     If anything prints below, it's either a different project sharing this account,"
echo "     or the ACE scripts' naming has drifted from what's checked into this repo -- do"
echo "     not assume it's ACE-related without checking the job's script/command."
squeue -u "$USER" -o "%.10i %.30j %.8T %R" 2>/dev/null | \
    grep -Ev '^\s*JOBID' | \
    grep -Ev '\b(bf5_|bf30_|bf5bl_|bf30bl_|oed30_|ace30_|dpoalt_|nodeimp_|msA_|msB_|fr100_|ksK)' || \
    echo "  (none -- every queued job matches a known ACE prefix)"

echo
echo "---- 2. Job history, last 14 days (sacct) ----"
echo "     Look for FAILED / OUT_OF_ME+ / TIMEOUT / CANCELLED in the State column."
START=$(date -d '14 days ago' +%Y-%m-%d 2>/dev/null || date -v-14d +%Y-%m-%d)
sacct -u "$USER" -S "$START" \
      -o JobID,JobName%20,State,ExitCode,Elapsed,Start \
      --parsable2 2>/dev/null | grep -Ev '\.(batch|extern)$' || echo "(sacct unavailable)"

echo
echo "================================================================"
echo " 3. Per-suite cell completion"
echo "================================================================"

# episode_ge <node_losses.csv> <threshold> -- exit 0 iff max(episode) >= threshold
episode_ge() {
    python3 -c "
import pandas as pd, sys
try:
    df = pd.read_csv(sys.argv[1])
    sys.exit(0 if 'episode' in df.columns and int(df['episode'].max()) >= int(sys.argv[2]) else 1)
except Exception:
    sys.exit(1)
" "$1" "$2" 2>/dev/null
}

has_file() { find "$1" -name "$2" 2>/dev/null | grep -q .; }

suite_header() { echo; echo "--- $1 ---"; }

# -----------------------------------------------------------------------
# BLOCKING: 5-node budget-fairness Phase 1 (ACE env/student), the decision
# gate this scale is still waiting on (see current_status.txt Aug 12 entry:
# resume-crash bug fixed, resubmit with SKIP_COMPLETED=1).
# -----------------------------------------------------------------------
suite_header "BLOCKING: 5-node budget-fairness Phase 1 (ACE env/student) -- need node_losses.csv episode>=199"
OUT="$SCRATCH/curc_5node_budget_fairness"
done=0; total=0
for MODE in env student; do
    for SEED in 42 123 456 789 1011; do
        total=$((total+1))
        nl=$(find "$OUT/ace_${MODE}/seed_${SEED}" -name node_losses.csv 2>/dev/null | head -1)
        if [[ -n "$nl" ]] && episode_ge "$nl" 199; then
            done=$((done+1))
        else
            maxep="none"
            [[ -n "$nl" ]] && maxep=$(python3 -c "import pandas as pd,sys; print(int(pd.read_csv(sys.argv[1])['episode'].max()))" "$nl" 2>/dev/null || echo "unreadable")
            echo "  MISSING: ace_${MODE} seed=${SEED}  (max episode so far: ${maxep})"
        fi
    done
done
echo "  -> $done / $total cells done"

# -----------------------------------------------------------------------
# BLOCKING: 5-node budget-fairness Phase 2 (baselines at matched query
# budget). Cannot even be submitted until Phase 1 above is 100% done --
# the shared budget is derived from ace_env's query_budget.json.
# -----------------------------------------------------------------------
suite_header "BLOCKING: 5-node budget-fairness Phase 2 (baselines) -- need summary.csv per method/seed"
OUT="$SCRATCH/curc_5node_budget_fairness/baselines"
done=0; total=0
for METHOD in random round_robin max_variance ppo; do
    for SEED in 42 123 456 789 1011; do
        total=$((total+1))
        if [[ -f "$OUT/${METHOD}/seed_${SEED}/summary.csv" ]]; then
            done=$((done+1))
        else
            echo "  MISSING: ${METHOD} seed=${SEED}"
        fi
    done
done
echo "  -> $done / $total cells done"
if [[ $done -eq 0 ]]; then
    echo "  (expected -- per current_status.txt this phase has not been submitted yet;"
    echo "   it needs Phase 1's shared query budget first, see script header of"
    echo "   jobs/curc_submit_5node_budget_fairness.sh)"
fi

# -----------------------------------------------------------------------
# DONE per current_status.txt (Aug 10): 30-node budget-fairness, both
# phases. Verify that's still true rather than trusting the doc blindly.
# -----------------------------------------------------------------------
suite_header "30-node budget-fairness Phase 1 (ACE env/student) -- should be DONE"
OUT="$SCRATCH/curc_30node_budget_fairness"
done=0; total=0
for MODE in env student; do
    for SEED in 42 123 456 789 1011; do
        total=$((total+1))
        nl=$(find "$OUT/ace_${MODE}/seed_${SEED}" -name node_losses.csv 2>/dev/null | head -1)
        [[ -n "$nl" ]] && [[ -f "$(dirname "$nl")/query_budget.json" ]] && done=$((done+1)) || echo "  MISSING: ace_${MODE} seed=${SEED}"
    done
done
echo "  -> $done / $total cells done"

suite_header "30-node budget-fairness Phase 2 (baselines) -- should be DONE"
OUT="$SCRATCH/curc_30node_budget_fairness/baselines"
done=0; total=0
for METHOD in random round_robin max_variance bayesian_oed; do
    for SEED in 42 123 456 789 1011; do
        total=$((total+1))
        [[ -f "$OUT/${METHOD}/seed_${SEED}/summary.csv" ]] && done=$((done+1)) || echo "  MISSING: ${METHOD} seed=${SEED}"
    done
done
echo "  -> $done / $total cells done"

# -----------------------------------------------------------------------
# Non-blocking / secondary suites
# -----------------------------------------------------------------------
suite_header "30-node Bayesian OED (executed-only accounting) -- reported COMPLETED Aug 12"
OUT="$SCRATCH/curc_30node_baselines"
done=0; total=0
for SEED in 42 123 456 789 1011; do
    total=$((total+1))
    [[ -f "$OUT/bayesian_oed/seed_${SEED}/summary.csv" ]] && done=$((done+1)) || echo "  MISSING: seed=${SEED}"
done
echo "  -> $done / $total cells done"

suite_header "30-node ACE seed expansion (new seeds 2022-2026) -- reported 5/5 COMPLETED Aug 9"
OUT="$SCRATCH/curc_30node_baselines/ace"
done=0; total=0
for SEED in 2022 2023 2024 2025 2026; do
    total=$((total+1))
    has_file "$OUT/seed_${SEED}" "node_losses.csv" && done=$((done+1)) || echo "  MISSING: seed=${SEED}"
done
echo "  -> $done / $total cells done"

suite_header "DPO-alternatives comparison (dpo/sft_best/ranking) -- not yet submitted as of last status update"
OUT="$SCRATCH/curc_dpo_alternatives"
done=0; total=0
for MODE in dpo sft_best ranking; do
    for SEED in 42 123 456 789 1011; do
        total=$((total+1))
        has_file "$OUT/${MODE}/seed_${SEED}" "node_losses.csv" && done=$((done+1))
    done
done
echo "  -> $done / $total cells done  (0 is expected until 'bash jobs/curc_submit_dpo_alternatives.sh' runs)"

suite_header "Node-importance ablation (no_node_importance vs baseline) -- not yet submitted as of last status update"
OUT="$SCRATCH/curc_node_importance_ablation"
done=0; total=0
for SEED in 42 123 456; do
    for CONFIG in with_importance no_importance; do
        total=$((total+1))
        has_file "$OUT/${CONFIG}/seed_${SEED}" "node_losses.csv" && done=$((done+1))
    done
done
echo "  -> $done / $total cells found (config names are a guess -- inspect $OUT if this looks wrong;"
echo "     check jobs/curc_node_importance_ablation_seed.sh for the exact directory convention)"

suite_header "Model-scale sweep (secondary; Qwen2.5 0.5B-32B) -- mixed as of Aug 9-10"
OUT="$SCRATCH/curc_model_scale_sweep"
if [[ -d "$OUT" ]]; then
    n=$(find "$OUT" -name node_losses.csv 2>/dev/null | wc -l | tr -d ' ')
    echo "  $n node_losses.csv files found under $OUT (no fixed target -- exploratory sweep, see"
    echo "  docs/development/guidance/current_status.txt Aug 6/9 entries for the full cell matrix)"
else
    echo "  MISSING: $OUT does not exist yet"
fi

suite_header "100-node frontier (secondary) -- status unclear, verify"
OUT="$SCRATCH/curc_100node_frontier"
if [[ -d "$OUT" ]]; then
    n=$(find "$OUT/nodes100" \( -name node_losses.csv -o -name summary.csv \) 2>/dev/null | wc -l | tr -d ' ')
    echo "  $n result files found under $OUT/nodes100"
else
    echo "  MISSING: $OUT does not exist yet"
fi

suite_header "K-scaling under student lookahead (secondary), K in {4,8,16,32}, N=30, 3 seeds"
done=0; total=0
for K in 4 8 16 32; do
    for SEED in 42 123 456; do
        total=$((total+1))
        OUT="$SCRATCH/curc_k_scaling_student/K${K}"
        has_file "$OUT/nodes30/ace/seed_${SEED}" "node_losses.csv" && done=$((done+1))
    done
done
echo "  -> $done / $total cells done  (verify \$SCRATCH/curc_k_scaling_student is the right root -- inspect"
echo "     jobs/curc_submit_k_scaling_student.sh if this looks wrong)"

echo
echo "================================================================"
echo " 4. Scratch quota"
echo "================================================================"
curc-quota 2>/dev/null || echo "(curc-quota command not found / not on a CURC login node)"

echo
echo "================================================================"
echo " Done. Cross-reference against docs/development/guidance/current_status.txt"
echo " before trusting any suite this script could not find a directory for --"
echo " some of the secondary-suite paths above are inferred from the submit"
echo " scripts and were not each independently verified against a real run."
echo "================================================================"
