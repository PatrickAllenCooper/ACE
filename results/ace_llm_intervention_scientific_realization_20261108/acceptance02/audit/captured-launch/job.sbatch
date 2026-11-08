#!/bin/bash
#SBATCH --job-name=ace_llm_scientific_realize02
#SBATCH --account=ucb736_asc2
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=7680M
#SBATCH --time=00:05:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_realization_20261108_02/logs/%j.out
#SBATCH --error=/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_realization_20261108_02/logs/%j.err
set -euo pipefail
umask 077
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MKL_THREADING_LAYER=GNU
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=''
unset PYTHONHOME PYTHONPATH LD_PRELOAD MKL_SERVICE_FORCE_INTEL
exec /projects/paco0228/miniconda3/envs/ace/bin/python -I -B -c 'import hashlib,sys
from pathlib import Path
path=Path(sys.argv[1]); expected=sys.argv[2]
with path.open('"'"'rb'"'"') as f: data=f.read(1024**2+1)
if len(data)>1024**2 or hashlib.sha256(data).hexdigest()!=expected:
    raise ValueError('"'"'independent bootstrap source pin differs'"'"')
sys.argv=[str(path),*sys.argv[3:]]
exec(compile(data,str(path),'"'"'exec'"'"'),{'"'"'__name__'"'"':'"'"'__main__'"'"','"'"'__file__'"'"':str(path)})
' /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_realization_20261108_02/inputs/supervise_llm_intervention_scientific_cpu.py e5235f16dd17c72e0310004e7bd8df636008c789db8f793212d528fabfd587f8 --packet /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_realization_20261108_02/inputs/packet.json --packet-sha256 887d6d879a279b1d6291bad3b30425e3a0eae135f2279e1a0794b219769e929a --source-commit a9346022f145cc29439325b6f291597b2e6ec037 --admission /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_realization_20261108_02/inputs/admission.json --admission-sha256 5c4cb07b82c6558904fd4a40b093eba0a76ea6c72ed22e80d8082a8fd42aea09 --output /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_realization_20261108_02/realization
