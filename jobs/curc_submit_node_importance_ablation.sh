#!/bin/bash
# =============================================================================
# Node-importance ablation on the 5-node benchmark: full (cov_bonus=60, paper
# default) vs no_node_importance (cov_bonus=0), isolating w(V_i, {L_j}) from
# the information-gain and diversity reward terms. Complements the existing
# no-diversity row in the component-ablation table (Table 3) and addresses
# reviewer wZrW's request for a node-importance ablation.
#
# 2 configs x 3 seeds = 6 jobs (matches the N=3-seed pilot used for the other
# component-ablation rows).
#
# Usage (from /projects/paco0228/ACE):
#   cd /projects/paco0228/ACE
#   git pull
#   bash jobs/curc_submit_node_importance_ablation.sh
#
# Output: written to /scratch/alpine/paco0228 (NOT /projects, which is at
# 229G/250G quota) -- see curc-quota before adding further large result trees.
#   /scratch/alpine/paco0228/ACE/results/curc_node_importance_ablation/{config}/seed_{seed}/
#
# SLURM resources per job:
#   partition : aa100 (A100 GPU)
#   time      : 08:00:00 (same config as the main 5-node ACE runs)
# =============================================================================

set -euo pipefail

# GPU targeting. Default to RTX Pro 6000 (Aug 2026 Alpine expansion): the
# original aa100/a100-40gb + 32G config OOM'd on every no_node_importance
# cell (jobs 30942627-30942629, Aug 7) for the same host-RAM reason bf5's
# budget-fairness suite did -- see
# jobs/curc_submit_5node_budget_fairness.sh. The "full" cells that
# "completed" in 12-13s on that same run predate the HF Hub 429 fix
# (resolve_local_hf_snapshot(), committed ~Aug 9-10) -- that's a 429 on
# tokenizer load, not a real run. Override if needed:
#   GPU_PARTITION=<partition> GPU_QOS=<qos> GPU_GRES=<gres> bash jobs/<this script>
GPU_PARTITION="${GPU_PARTITION:-artxpro6000}"
GPU_QOS="${GPU_QOS:-gpu-normal}"
GPU_GRES="${GPU_GRES:-gpu:rtx_pro_6000:1}"

cd /projects/paco0228/ACE

source /projects/paco0228/miniconda3/etc/profile.d/conda.sh
conda activate ace

OUT="/scratch/alpine/paco0228/ACE/results/curc_node_importance_ablation"
mkdir -p "$OUT/logs"

echo "================================================================"
echo " Node-importance ablation -- CURC SLURM (6 jobs)"
echo "================================================================"
echo " Output : $OUT"
echo " Started: $(date)"
echo "================================================================"

CONFIGS="full no_node_importance"
SEEDS="42 123 456"

# Set SKIP_COMPLETED=1 to skip any (config, seed) cell whose node_losses.csv
# already reached episode 199 (the 200-episode cap). Presence of the file is
# NOT sufficient -- a wall-time TIMEOUT leaves a partial one behind. Same
# convention as jobs/curc_submit_5node_budget_fairness.sh.
#
# NOTE: completed-output check only; it does NOT detect a duplicate still
# PENDING/RUNNING. Check `squeue -u $USER` before resubmitting.
SKIP_COMPLETED="${SKIP_COMPLETED:-0}"

cell_done() {
    local config=$1 seed=$2
    local nl
    nl=$(find "$OUT/${config}/seed_${seed}" -name node_losses.csv 2>/dev/null | head -1)
    [[ -n "$nl" ]] || return 1
    python -c "
import pandas as pd, sys
df = pd.read_csv(sys.argv[1])
sys.exit(0 if 'episode' in df.columns and int(df['episode'].max()) >= 199 else 1)
" "$nl" 2>/dev/null
}

for CONFIG in $CONFIGS; do
    for SEED in $SEEDS; do
        if [ "$SKIP_COMPLETED" = "1" ] && cell_done "$CONFIG" "$SEED"; then
            echo "  SKIP (done): nodeimp_${CONFIG} seed=$SEED"
            continue
        fi
        JOB=$(sbatch --parsable \
            --job-name="nodeimp_${CONFIG:0:6}_s${SEED}" \
            --partition=$GPU_PARTITION --qos=$GPU_QOS \
            --nodes=1 --ntasks=1 --gres=$GPU_GRES \
            --cpus-per-task=8 --mem=128G \
            --time=24:00:00 \
            --output="$OUT/logs/${CONFIG}_seed${SEED}_%j.out" \
            --error="$OUT/logs/${CONFIG}_seed${SEED}_%j.err" \
            --export=ALL,CONFIG=$CONFIG,SEED=$SEED,OUT=$OUT \
            jobs/curc_node_importance_ablation_seed.sh)
        echo "  Submitted: config=$CONFIG seed=$SEED -> Job $JOB"
    done
done

echo ""
echo "6 jobs submitted."
echo "Monitor with:  squeue -u \$USER"
echo "Logs in:       $OUT/logs/"
echo ""
echo "When complete, pull results locally with:"
echo "  scp -r paco0228@login.rc.colorado.edu:$OUT ./results/"
