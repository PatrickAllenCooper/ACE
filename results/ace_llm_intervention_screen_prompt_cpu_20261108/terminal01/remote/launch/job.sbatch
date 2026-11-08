#!/bin/bash
#SBATCH --job-name=ace_llm_screen_prompt_cpu01
#SBATCH --account=ucb736_asc2
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=7680M
#SBATCH --time=00:05:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_screen_prompt_cpu_20261108_01/logs/%j.out
#SBATCH --error=/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_screen_prompt_cpu_20261108_01/logs/%j.err
set -euo pipefail
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MKL_THREADING_LAYER=GNU
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
unset PYTHONHOME PYTHONPATH LD_PRELOAD MKL_SERVICE_FORCE_INTEL
exec /projects/paco0228/miniconda3/envs/ace/bin/python -I -B -c 'import hashlib,sys
from pathlib import Path
path=Path(sys.argv[1]); expected=sys.argv[2]
with path.open('"'"'rb'"'"') as f: data=f.read(1024**2+1)
if len(data)>1024**2 or hashlib.sha256(data).hexdigest()!=expected:
    raise ValueError('"'"'independent bootstrap source pin differs'"'"')
sys.argv=[str(path),*sys.argv[3:]]
exec(compile(data,str(path),'"'"'exec'"'"'),{'"'"'__name__'"'"':'"'"'__main__'"'"','"'"'__file__'"'"':str(path)})
' /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_screen_prompt_cpu_20261108_01/inputs/qualify_llm_intervention_screen_prompt_cpu.py c85747e3958165582fa8dcce6a60ea91c4c050acdad5f61fcccad3aec43c2533 --packet /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_screen_prompt_cpu_20261108_01/inputs/packet.json --packet-sha256 0979cb607643b4b5d01fbf56c6e701a3e474d4d2bbda1e1c9f4e78e91251ccfa --source-commit 21b187887ecb62bdc4b3515f3d27b79b22769d2b --output /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_screen_prompt_cpu_20261108_01/cpu01
