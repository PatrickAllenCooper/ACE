#!/bin/bash
#SBATCH --job-name=ace_llm_servicecpu03
#SBATCH --account=ucb736_asc2
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_service_cpu_20261107_03/logs/%j.out
#SBATCH --error=/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_service_cpu_20261107_03/logs/%j.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=
exec /projects/paco0228/miniconda3/envs/ace/bin/python -I -B -c 'import hashlib,sys
from pathlib import Path
path=Path(sys.argv[1]); expected=sys.argv[2]
with path.open('"'"'rb'"'"') as f: data=f.read(1024**2+1)
if len(data)>1024**2 or hashlib.sha256(data).hexdigest()!=expected:
    raise ValueError('"'"'independent bootstrap source pin differs'"'"')
sys.argv=[str(path),*sys.argv[3:]]
exec(compile(data,str(path),'"'"'exec'"'"'),{'"'"'__name__'"'"':'"'"'__main__'"'"','"'"'__file__'"'"':str(path)})
' /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_service_cpu_20261107_03/qualify_llm_intervention_service_cpu.py d658bedd192d5c3e92f8fc22a052c67d6524df296623ef74e2c5f38bf02fbeb5 --packet /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_service_cpu_20261107_03/preflight.json --packet-sha256 3a6ffe2d172943477158dd1a8c4de5fd196fca69e86d887b20cdda8f0966fb29 --output /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_service_cpu_20261107_03/qualification
