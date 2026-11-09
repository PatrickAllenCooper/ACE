#!/bin/bash
#SBATCH --job-name=ace_llm_scientific_campaign_cpu01
#SBATCH --account=ucb736_asc2
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=7680M
#SBATCH --time=01:40:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_campaign_cpu_20261108_01/logs/%j.out
#SBATCH --error=/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_campaign_cpu_20261108_01/logs/%j.err
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
' /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_campaign_cpu_20261108_01/inputs/prepare_llm_intervention_scientific_campaign_cpu.py ba439d819f49f6832c97181e4ff727c4813f29031e7b67f8085670f845b0f893 --packet /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_campaign_cpu_20261108_01/inputs/packet.json --packet-sha256 ad046a726e3a3a7744e54a860f8cf1a30947a198e8be3bc25ae06ec7b2909344 --source-commit a9346022f145cc29439325b6f291597b2e6ec037 --admission /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_campaign_cpu_20261108_01/inputs/admission.json --admission-sha256 c3ea6ab0f0434a4df3f6ef118e3fa89886d8b06a0e288cdfbd75cec3ba1abda6
