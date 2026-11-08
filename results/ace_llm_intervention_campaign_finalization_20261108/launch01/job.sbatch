#!/bin/bash
#SBATCH --job-name=ace_llm_campaign_final01
#SBATCH --account=ucb736_asc2
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=7680M
#SBATCH --time=00:05:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_campaign_finalization_20261108_01/logs/%j.out
#SBATCH --error=/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_campaign_finalization_20261108_01/logs/%j.err
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
' /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_campaign_finalization_20261108_01/inputs/finalize_llm_intervention_campaign_cpu.py 88102baf56395e7ef06fc7e96aae3ed2d7b8eef4ecb8b71f77acfc7e4b8e813f --packet /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_campaign_finalization_20261108_01/inputs/packet.json --packet-sha256 417bd889465198c0276fcc2a2ebf55168e21efad278c7978a34ed9360761bf6c --output /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_campaign_finalization_20261108_01/finalization
