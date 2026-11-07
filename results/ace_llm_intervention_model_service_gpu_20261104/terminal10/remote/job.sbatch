#!/bin/bash
#SBATCH --job-name=ace_llm_service10
#SBATCH --account=ucb736_asc1
#SBATCH --partition=aa100
#SBATCH --qos=gpu-testing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_model_service_gpu_20261104_10/logs/%j.out
#SBATCH --error=/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_model_service_gpu_20261104_10/logs/%j.err
set -euo pipefail
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
exec /projects/paco0228/miniconda3/envs/ace/bin/python -B -c 'import hashlib, runpy, sys
p = sys.argv[1]
expected = sys.argv[2]
if hashlib.sha256(open(p, "rb").read()).hexdigest() != expected:
    raise ValueError("driver source digest differs")
sys.argv = [p] + sys.argv[3:]
runpy.run_path(p, run_name="__main__")' /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_model_service_gpu_20261104_10/input/qualify_llm_intervention_model_service.py fb3e95013cb46cd53f3b44ee72a141d75616c39b762b8e62ea90d89f3a9de717 --packet /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_model_service_gpu_20261104_10/input/packet.json --packet-sha256 e1a911fcf4d087742e1e0cc2ed87b351ea13ee966113c9b40245704a1afe5c68 --output /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_model_service_gpu_20261104_10/integration10
