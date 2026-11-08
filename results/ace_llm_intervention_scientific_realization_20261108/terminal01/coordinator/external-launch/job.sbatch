#!/bin/bash
#SBATCH --job-name=ace_llm_scientific_realize01
#SBATCH --account=ucb736_asc2
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=7680M
#SBATCH --time=00:05:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_realization_20261108_01/logs/%j.out
#SBATCH --error=/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_realization_20261108_01/logs/%j.err
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
' /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_realization_20261108_01/inputs/supervise_llm_intervention_scientific_cpu.py 794fd63af304ad7093c943f57dece7668ddd6bd5102a8c5db55cea19ee5054f9 --packet /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_realization_20261108_01/inputs/packet.json --packet-sha256 dc9a181d45a630b83b97027fa03307143faff533de0200dfaa7ee8aa783aee21 --source-commit f97cb08e28532a6773aa5045819fa64d45e49632 --admission /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_realization_20261108_01/inputs/admission.json --admission-sha256 21caf84c6b0eac9a07faab4f206ef157fdf3c0129a66dbf298cf6613e353a5e4 --output /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_realization_20261108_01/realization
