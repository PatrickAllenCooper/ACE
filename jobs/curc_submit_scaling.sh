#!/bin/bash
# =============================================================================
# Scaling sweep: "ACE scales to larger N without architectural change."
#
# Consistent hierarchical family (LargeScaleSCM) at N in {15, 30, 50}, three
# methods (ACE, ACE-w/o-DPO [LM+lookahead, --no_dpo], Random), reported as
# PER-NODE best MSE so larger graphs are not penalised by the mechanical growth
# of the summed total. N=5 is the paper's bespoke diagnostic SCM and is shown
# as a separately-sourced anchor in the figure, not run here.
#
# Budget rationale (the binding cost is per-episode wallclock, which grows with
# prompt length; best-MSE plateaus early so ACE is capped at ~40 episodes):
#   ACE 15/30  : full prompt, 40 ep, checkpoint-resume across windows
#   ACE 50     : COMPACT prompt (scaling enabler), 40 ep, resume
#   ZSL (no_dpo): 40 ep, fixed policy converges fast
#   Random     : 150 ep, MLP learner only (cheap)
#
# Checkpoint-resume: the worker writes to a STABLE dir (no job-id suffix), so
# re-running this script resubmits and continues 50-node ACE from its last
# checkpoint. Re-run as many windows as needed.
#
# Usage (from /projects/paco0228/ACE):
#   git pull
#   bash jobs/curc_submit_scaling.sh            # main sweep
#   SUBMIT_PARITY=1 bash jobs/curc_submit_scaling.sh   # +30-node compact parity
#
# Output: results/scaling/nodes{N}/{method}/seed_{seed}/run_*/node_losses.csv
# =============================================================================

set -euo pipefail

cd /projects/paco0228/ACE
source /projects/paco0228/miniconda3/etc/profile.d/conda.sh
conda activate ace

OUT="/projects/paco0228/ACE/results/scaling"
mkdir -p "$OUT/logs"

SEEDS="${SEEDS:-42 123 456}"
WORKER="jobs/curc_scaling_seed.sh"

echo "================================================================"
echo " Scaling sweep -- CURC SLURM"
echo " Output : $OUT   Seeds: $SEEDS   Started: $(date)"
echo "================================================================"

submit() {
    # submit <scale> <method> <seed> <mem> <walltime> <extra-export>
    local scale=$1 method=$2 seed=$3 mem=$4 wall=$5 extra=$6
    local name="sc${scale}_${method}_s${seed}"
    local JOB
    JOB=$(sbatch --parsable \
        --job-name="$name" \
        --partition=aa100 --qos=normal \
        --nodes=1 --ntasks=1 --gres=gpu:1 \
        --cpus-per-task=8 --mem="$mem" \
        --time="$wall" \
        --output="$OUT/logs/${name}_%j.out" \
        --error="$OUT/logs/${name}_%j.err" \
        --export=ALL,SCALE=$scale,METHOD=$method,SEED=$seed,OUT=$OUT,$extra \
        "$WORKER")
    echo "  Submitted: $name -> Job $JOB"
}

for SEED in $SEEDS; do
    # ---- N=15 (consistent-family small anchor) -----------------------------
    submit 15 ace          "$SEED" 48G 10:00:00 "PROMPT_STRATEGY=full,EPISODES=40"
    submit 15 zero_shot_lm "$SEED" 48G 06:00:00 "PROMPT_STRATEGY=full,EPISODES=40"
    submit 15 random       "$SEED" 32G 06:00:00 "EPISODES=150"

    # ---- N=30 (paper anchor) ----------------------------------------------
    submit 30 ace          "$SEED" 64G 22:00:00 "PROMPT_STRATEGY=full,EPISODES=40"
    submit 30 zero_shot_lm "$SEED" 64G 10:00:00 "PROMPT_STRATEGY=full,EPISODES=40"
    submit 30 random       "$SEED" 48G 08:00:00 "EPISODES=150"

    # ---- N=50 (compact prompt is the scaling enabler) ---------------------
    submit 50 ace          "$SEED" 78G 24:00:00 "PROMPT_STRATEGY=compact,PROMPT_TOP_M=8,EPISODES=40"
    submit 50 zero_shot_lm "$SEED" 78G 12:00:00 "PROMPT_STRATEGY=compact,PROMPT_TOP_M=8,EPISODES=40"
    submit 50 random       "$SEED" 64G 10:00:00 "EPISODES=150"
done

# ---- Optional: 30-node compact-vs-full parity (Phase 2 validation) --------
# Confirms compact encoding does not regress best-MSE before we trust it at 50.
if [ "${SUBMIT_PARITY:-0}" = "1" ]; then
    echo ""
    echo ">>> 30-node compact parity (compact ACE, same seeds) <<<"
    for SEED in $SEEDS; do
        # Stored under a sibling root so it does not collide with the full-prompt
        # 30-node ACE cell above.
        JOB=$(sbatch --parsable \
            --job-name="sc30_ace_compact_s${SEED}" \
            --partition=aa100 --qos=normal \
            --nodes=1 --ntasks=1 --gres=gpu:1 \
            --cpus-per-task=8 --mem=64G \
            --time=22:00:00 \
            --output="$OUT/logs/sc30_ace_compact_s${SEED}_%j.out" \
            --error="$OUT/logs/sc30_ace_compact_s${SEED}_%j.err" \
            --export=ALL,SCALE=30,METHOD=ace,SEED=$SEED,OUT=$OUT/parity_compact,PROMPT_STRATEGY=compact,PROMPT_TOP_M=8,EPISODES=40 \
            "$WORKER")
        echo "  Submitted: sc30_ace_compact seed=$SEED -> Job $JOB"
    done
fi

echo ""
echo "Sweep submitted. Monitor: squeue -u \$USER"
echo "Resume 50-node ACE by re-running this script (stable dirs continue from checkpoint)."
echo "Logs: $OUT/logs/"
