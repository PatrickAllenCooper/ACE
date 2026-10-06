#!/bin/bash
# CPU-only delivery attribution. Runtime and source are explicitly frozen.
#SBATCH --job-name=ace_delivery_attribution
#SBATCH --account=ucb736_asc1
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=8G
#SBATCH --time=08:00:00
set -euo pipefail
: "${ACE_STAGE_ROOT:?}" "${ACE_STAGE_OUT:?}" "${ACE_CODE_REVISION:?}"
cd "$ACE_STAGE_ROOT/code"
test "$(cat source_revision.txt)" = "$ACE_CODE_REVISION"
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=""
exec /projects/paco0228/miniconda3/envs/mono_s2s/bin/python -u scripts/research/delivery_attribution_batch.py run --out "$ACE_STAGE_OUT"
