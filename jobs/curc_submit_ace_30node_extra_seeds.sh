#!/bin/bash
# =============================================================================
# Submit 2 more ACE 30-node seeds to reach N=5.
#
# Existing results: seeds 42, 123, 456 (best-loss 2.79, 1.80, 1.27)
# New seeds: 789, 1011 to get N=5 total matching the baseline set
#
# Usage (from /projects/paco0228/ACE):
#   cd /projects/paco0228/ACE
#   bash jobs/curc_submit_ace_30node_extra_seeds.sh
#
# Results land in: results/curc_30node_baselines/ace/seed_{seed}/
# (co-located with baselines so aggregate_30node_baselines.py can pick them up)
#
# Resources per job:
#   partition : aa100 (A100 80GB)
#   QoS       : normal
#   time      : 08:00:00
#   GPUs      : 1
#   CPUs      : 8
#   MEM       : 32G
# =============================================================================

set -euo pipefail

cd /projects/paco0228/ACE

source /projects/paco0228/miniconda3/etc/profile.d/conda.sh
conda activate ace

OUT="/projects/paco0228/ACE/results/curc_30node_ace_extra"
mkdir -p "$OUT/logs"

echo "================================================================"
echo " ACE 30-node extra seeds (to reach N=5)"
echo "================================================================"
echo " Output : $OUT"
echo " Started: $(date)"
echo "================================================================"

# SLURM places us on a random A100 in the aa100 partition, which contains
# both 40GB and 80GB cards. Previous runs that succeeded landed on 80GB by
# luck. The 40GB cards run out of memory because Qwen2.5-1.5B active +
# reference + cloned learners + KV cache + grads exceed 40GB.
#
# Mitigations applied:
#   1. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True in the worker script
#      (reduces fragmentation, may give us enough headroom on 40GB cards).
#   2. Submit each seed twice. Both jobs do the same work, but whichever
#      lands on an 80GB card will complete in ~6h while the 40GB attempt
#      may still OOM. With 2 attempts per seed, probability that at least
#      one attempt per seed gets an 80GB card is high.
#
# After completion, cancel any duplicate that is still running and pick the
# completed run for each seed.

for SEED in 789 1011; do
    for ATTEMPT in 1 2; do
        JOB=$(sbatch --parsable \
            --job-name="ace30_s${SEED}a${ATTEMPT}" \
            --partition=aa100 \
            --qos=normal \
            --nodes=1 --ntasks=1 --gres=gpu:1 \
            --cpus-per-task=8 \
            --mem=64G \
            --time=08:00:00 \
            --output="$OUT/logs/ace30_seed${SEED}_attempt${ATTEMPT}_%j.out" \
            --error="$OUT/logs/ace30_seed${SEED}_attempt${ATTEMPT}_%j.err" \
            --export=ALL,SEED=$SEED,OUT=$OUT \
            jobs/curc_large_scale_seed.sh)
        echo "  Submitted: ACE seed=$SEED attempt=$ATTEMPT -> Job $JOB"
    done
done

echo ""
echo "2 jobs submitted."
echo "Monitor with:  squeue -u \$USER"
echo "Logs in:       $OUT/logs/"
echo ""
echo "When complete, pull locally:"
echo "  scp -r paco0228@login.rc.colorado.edu:$OUT ./results/"
