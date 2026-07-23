#!/bin/bash
# =============================================================================
# DPO-alternative comparison on the 5-node benchmark: dpo (paper default) vs
# sft_best (imitate the best lookahead candidate only) vs ranking (pairwise
# Bradley-Terry loss with no reference-policy term).
#
# 3 policy_update modes x 5 seeds = 15 jobs.
#
# Note: per the ICLR resubmission framing, the contribution is LM prior +
# lookahead selection, with the preference-learning rule as one calibration
# component -- a tie between dpo and sft_best/ranking here is an acceptable,
# even informative, result (it says the reference-policy KL anchor or the
# preference-comparison structure isn't where DPO's value comes from), not
# grounds to abandon the method.
#
# Usage (from /projects/paco0228/ACE):
#   cd /projects/paco0228/ACE
#   git pull   # ensure latest --policy_update flag
#   bash jobs/curc_submit_dpo_alternatives.sh
#
# Output: results/curc_dpo_alternatives/{policy_update}/seed_{seed}/
#   Each run writes metrics.csv (with a "policy_update" column) and
#   query_budget.json, same schema as the main ACE runs.
#
# SLURM resources per job:
#   partition : aa100 (A100 GPU)
#   time      : 08:00:00 (same config as the main 5-node ACE runs)
# =============================================================================

set -euo pipefail

cd /projects/paco0228/ACE

source /projects/paco0228/miniconda3/etc/profile.d/conda.sh
conda activate ace

OUT="/scratch/alpine1/paco0228/ACE/results/curc_dpo_alternatives"
mkdir -p "$OUT/logs"

echo "================================================================"
echo " DPO-alternative comparison -- CURC SLURM (15 jobs)"
echo "================================================================"
echo " Output : $OUT"
echo " Started: $(date)"
echo "================================================================"

MODES="dpo sft_best ranking"
SEEDS="42 123 456 789 1011"

for MODE in $MODES; do
    for SEED in $SEEDS; do
        JOB=$(sbatch --parsable \
            --job-name="dpoalt_${MODE:0:4}_s${SEED}" \
            --partition=aa100 --qos=normal \
            --nodes=1 --ntasks=1 --gres=gpu:1 \
            --cpus-per-task=8 --mem=32G \
            --time=08:00:00 \
            --output="$OUT/logs/${MODE}_seed${SEED}_%j.out" \
            --error="$OUT/logs/${MODE}_seed${SEED}_%j.err" \
            --export=ALL,POLICY_UPDATE=$MODE,SEED=$SEED,OUT=$OUT \
            jobs/curc_dpo_alternative_seed.sh)
        echo "  Submitted: policy_update=$MODE seed=$SEED -> Job $JOB"
    done
done

echo ""
echo "15 jobs submitted."
echo "Monitor with:  squeue -u \$USER"
echo "Logs in:       $OUT/logs/"
echo ""
echo "When complete, pull results locally with:"
echo "  scp -r paco0228@login.rc.colorado.edu:$OUT ./results/"
