#!/bin/bash
#SBATCH --job-name=ace_llm_traincpu01
#SBATCH --account=ucb736_asc2
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_training_cpu_20261108_01/logs/%j.out
#SBATCH --error=/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_training_cpu_20261108_01/logs/%j.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 MKL_THREADING_LAYER=GNU
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=
unset PYTHONHOME PYTHONPATH LD_PRELOAD MKL_SERVICE_FORCE_INTEL
exec /projects/paco0228/miniconda3/envs/ace/bin/python -I -B -c 'import hashlib,sys
from pathlib import Path
path=Path(sys.argv[1]); expected=sys.argv[2]
with path.open('"'"'rb'"'"') as f: data=f.read(1024**2+1)
if len(data)>1024**2 or hashlib.sha256(data).hexdigest()!=expected:
    raise ValueError('"'"'independent bootstrap source pin differs'"'"')
sys.argv=[str(path),*sys.argv[3:]]
exec(compile(data,str(path),'"'"'exec'"'"'),{'"'"'__name__'"'"':'"'"'__main__'"'"','"'"'__file__'"'"':str(path)})
' /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_training_cpu_20261108_01/prepare_llm_intervention_training_cpu.py b2b1425bcd010d5ddc5deeda0dc3bd89817c9fe4cae6c4cc6465ec9f7f998f43 --packet /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_training_cpu_20261108_01/preparation.json --packet-sha256 b4394a7b0160b2d8f23df709334dbbc6cace2bac3fb1cd053602c7260d569d73 --output /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_training_cpu_20261108_01/qualification
