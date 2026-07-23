#!/bin/bash
# =============================================================================
# Expand ACE 30-node seed count from N=3 (NeurIPS 2026 submission) to N=10,
# and report FINAL MSE as the primary statistic (not just best-so-far).
#
# Reviewers d6tT and TnpG both flagged the 30-node result as under-seeded
# (N=3) relative to the passive baselines (N=5), and objected to reporting
# best-so-far MSE for a non-monotone method as the headline number. This
# submits the 7 additional seeds needed to reach N=10 -- generously above
# the baselines' N=5 -- so seed 42's known training instability (reported in
# the submission's Appendix E.2) is diluted by more independent seeds rather
# than argued away.
#
# Existing seeds (from results/curc_30node_baselines/ace/ and
# results/curc_30node_ace_extra/): 42, 123, 456, 789, 1011
# New seeds to reach N=10: 2022, 2023, 2024, 2025, 2026
#
# Usage (from /projects/paco0228/ACE):
#   cd /projects/paco0228/ACE
#   git pull
#   bash jobs/curc_submit_ace_30node_seed_expansion.sh
#
# Output: results/curc_30node_baselines/ace/seed_{seed}/
#   (co-located with the baselines directory for direct aggregation)
#
# After all 10 seeds are in, report BOTH statistics in the paper:
#   - final MSE (last episode)   <- new primary metric
#   - best MSE (min over training) <- secondary, as in the submission
#
# SLURM resources per job:
#   partition : aa100 (A100 GPU; 30-node ACE needs ~22 min/step per the
#               submission's own Appendix -- budget the full 8h window)
# =============================================================================

set -euo pipefail

cd /projects/paco0228/ACE

source /projects/paco0228/miniconda3/etc/profile.d/conda.sh
conda activate ace

OUT="/scratch/alpine1/paco0228/ACE/results/curc_30node_baselines"
mkdir -p "$OUT/logs"

echo "================================================================"
echo " ACE 30-node seed expansion (N=3 -> N=10) -- CURC SLURM (5 jobs)"
echo "================================================================"
echo " Output : $OUT/ace/"
echo " Started: $(date)"
echo "================================================================"

NEW_SEEDS="2022 2023 2024 2025 2026"

for SEED in $NEW_SEEDS; do
    JOB=$(sbatch --parsable \
        --job-name="ace30_s${SEED}" \
        --partition=aa100 \
        --qos=normal \
        --nodes=1 --ntasks=1 --gres=gpu:1 \
        --cpus-per-task=8 \
        --mem=64G \
        --time=08:00:00 \
        --output="$OUT/logs/ace_seed${SEED}_%j.out" \
        --error="$OUT/logs/ace_seed${SEED}_%j.err" \
        --export=ALL,SEED=$SEED,OUT="$OUT/ace" \
        jobs/curc_large_scale_seed.sh)
    echo "  Submitted: ACE 30-node seed=$SEED -> Job $JOB"
done

echo ""
echo "5 jobs submitted (seeds already run: 42 123 456 789 1011)."
echo "Monitor with:  squeue -u \$USER"
echo "Logs in:       $OUT/logs/"
echo ""
echo "When complete, pull results locally with:"
echo "  scp -r paco0228@login.rc.colorado.edu:$OUT ./results/"
echo ""
echo "Then compute both final-MSE and best-MSE across all 10 seeds for the"
echo "ICLR revision's Table (final MSE as primary statistic)."
