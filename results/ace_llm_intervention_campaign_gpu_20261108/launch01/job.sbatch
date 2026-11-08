#!/bin/bash
#SBATCH --job-name=ace_llm_campaign_gpu01
#SBATCH --account=ucb736_asc2
#SBATCH --partition=aa100
#SBATCH --qos=gpu-testing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=15360M
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_campaign_gpu_20261108_01/logs/%j.out
#SBATCH --error=/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_campaign_gpu_20261108_01/logs/%j.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MKL_THREADING_LAYER=GNU
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
unset PYTHONHOME PYTHONPATH LD_PRELOAD MKL_SERVICE_FORCE_INTEL
# Preserve scheduler-provided GPU visibility.
exec /projects/paco0228/miniconda3/envs/ace/bin/python -I -B -c 'import hashlib,sys
from pathlib import Path
path=Path(sys.argv[1]); expected=sys.argv[2]
with path.open('"'"'rb'"'"') as f: data=f.read(1024**2+1)
if len(data)>1024**2 or hashlib.sha256(data).hexdigest()!=expected:
    raise ValueError('"'"'independent bootstrap source pin differs'"'"')
sys.argv=[str(path),*sys.argv[3:]]
exec(compile(data,str(path),'"'"'exec'"'"'),{'"'"'__name__'"'"':'"'"'__main__'"'"','"'"'__file__'"'"':str(path)})
' /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_campaign_gpu_20261108_01/inputs/qualify_llm_intervention_campaign_gpu.py 58f64d3eea63ae56cc7d14d9188c5d1f9e6e4126fc5b4512abc9758006a803bc --packet /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_campaign_gpu_20261108_01/inputs/packet.json --packet-sha256 970dc22bdeab1d8cdeded84dfe4f42c58c186cfd77fc9763a9273fbc4a0a3976 --output /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_campaign_artificial_20261108_03/gpu03
