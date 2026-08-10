#!/bin/bash
# =============================================================================
# Prefetch all Qwen2.5 sizes used by the model-scale sweep into HF_HOME.
#
# Required before flooding the queue with ~60 ACE jobs: without this, every
# job's AutoTokenizer.from_pretrained hits huggingface.co/api/models/... and
# CURC's shared campus IP gets 429 rate-limited (see Aug 9 failure logs).
#
# Usage (from /projects/paco0228/ACE):
#   export HF_TOKEN=hf_...          # https://huggingface.co/settings/tokens
#   bash jobs/curc_prefetch_hf_models.sh
#
# Or run interactively on a login node (fine for small models; 32B is ~60GB):
#   conda activate ace
#   export HF_HOME=/projects/paco0228/cache/huggingface
#   export HF_TOKEN=hf_...
#   python scripts/runners/prefetch_qwen_models.py
# =============================================================================

set -euo pipefail

cd /projects/paco0228/ACE

if [ -z "${HF_TOKEN:-}" ] && [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
    echo "ERROR: export HF_TOKEN=hf_... before running this script."
    echo "  Create a free token at https://huggingface.co/settings/tokens"
    exit 2
fi

source /projects/paco0228/miniconda3/etc/profile.d/conda.sh
conda activate ace

OUT="/scratch/alpine/paco0228/ACE/results/curc_prefetch_hf"
mkdir -p "$OUT"

JOB=$(sbatch --parsable \
    --job-name="ace_hf_prefetch" \
    --partition=acpu --qos=cpu-normal \
    --nodes=1 --ntasks=1 \
    --cpus-per-task=4 --mem=16G \
    --time=06:00:00 \
    --output="$OUT/prefetch_%j.out" \
    --error="$OUT/prefetch_%j.err" \
    --export=ALL,HF_HOME=/projects/paco0228/cache/huggingface,HF_TOKEN \
    --wrap='source /projects/paco0228/miniconda3/etc/profile.d/conda.sh && conda activate ace && cd /projects/paco0228/ACE && python -u scripts/runners/prefetch_qwen_models.py')

echo "Submitted prefetch job $JOB"
echo "Monitor: squeue -j $JOB ; tail -f $OUT/prefetch_${JOB}.out"
echo ""
echo "When it COMPLETED, resubmit the ACE gaps:"
echo "  SKIP_COMPLETED=1 bash jobs/curc_submit_model_scale_sweep.sh"
echo "  SKIP_COMPLETED=1 GPU_ONLY=1 bash jobs/curc_submit_100node_frontier.sh"
echo "  SKIP_COMPLETED=1 bash jobs/curc_submit_k_scaling_student.sh"
