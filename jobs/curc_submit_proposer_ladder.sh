#!/bin/bash
# =============================================================================
# LM-proposer ladder at 5 nodes (17 Sept 2026): does the LM add anything on
# top of ACE's learner, lookahead scoring, scaffolding and stopping rule?
#
#   proposer=random     uniform (node, value) candidates, --no_dpo
#   proposer=heuristic  loss-guided direct-child-impact candidates, --no_dpo
#   (proposer=lm --no_dpo is the calibration suite's 'none' row:
#    jobs/curc_submit_dpo_alternatives.sh MODES=none)
#
# Identical to the DPO-alternatives worker otherwise (200 ep, dedicated root
# learner, obs refresh 3/200/100). Output: $OUT/<proposer>/seed_<s>/
# Usage: GPU_PARTITION=ah200 GPU_QOS=gpu-long GPU_GRES=gpu:h200:1 WALL_TIME=48:00:00 \
#        bash jobs/curc_submit_proposer_ladder.sh
# =============================================================================
set -euo pipefail
cd /projects/paco0228/ACE
source /projects/paco0228/miniconda3/etc/profile.d/conda.sh; conda activate ace
OUT="${OUT:-/scratch/alpine/paco0228/ACE/results/proposer_ladder}"
GPU_PARTITION="${GPU_PARTITION:-artxpro6000}"; GPU_QOS="${GPU_QOS:-gpu-normal}"; GPU_GRES="${GPU_GRES:-gpu:rtx_pro_6000:1}"
WALL_TIME="${WALL_TIME:-24:00:00}"; SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
PROPOSERS="${PROPOSERS:-random heuristic}"; SEEDS="${SEEDS:-42 123 456 789 1011}"
mkdir -p "$OUT/logs"; N=0
cell_done() { local nl; nl=$(find "$1" -name node_losses.csv 2>/dev/null | head -1); [[ -n "$nl" ]] || return 1
    python -c "import pandas as pd,sys; d=pd.read_csv(sys.argv[1]); sys.exit(0 if int(d['episode'].max())>=199 else 1)" "$nl" 2>/dev/null; }
for P in $PROPOSERS; do for S in $SEEDS; do
    if [ "$SKIP_COMPLETED" = "1" ] && cell_done "$OUT/$P/seed_$S"; then echo "  SKIP (done): $P s$S"; continue; fi
    JOB=$(sbatch --parsable --job-name="prop_${P}_s${S}" --partition=$GPU_PARTITION --qos=$GPU_QOS --gres=$GPU_GRES \
        --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=128G --time=$WALL_TIME \
        --output="$OUT/logs/${P}_s${S}_%j.out" --error="$OUT/logs/${P}_s${S}_%j.err" \
        --export=ALL,SEED=$S,POLICY_UPDATE=none,PROPOSER=$P,OUT="$OUT" jobs/curc_dpo_alternative_seed.sh)
    echo "  Submitted: proposer=$P seed=$S -> Job $JOB"; N=$((N+1))
done; done
echo "$N job(s) submitted -> $OUT"
