#!/bin/bash
#SBATCH --job-name=ace_llm_scientific_campaign_gpu02
#SBATCH --account=ucb736_asc2
#SBATCH --partition=aa100
#SBATCH --qos=gpu-testing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=15360M
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --time=00:48:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_campaign_gpu_20261109_02/logs/%j.out
#SBATCH --error=/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_campaign_gpu_20261109_02/logs/%j.err
set -euo pipefail
umask 077
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MKL_THREADING_LAYER=GNU
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
unset PYTHONHOME PYTHONPATH LD_PRELOAD MKL_SERVICE_FORCE_INTEL
exec /projects/paco0228/miniconda3/envs/ace/bin/python -I -B -c 'import hashlib,sys
from pathlib import Path
path=Path(sys.argv[1]); expected=sys.argv[2]
with path.open('"'"'rb'"'"') as f: data=f.read(1024**2+1)
if len(data)>1024**2 or hashlib.sha256(data).hexdigest()!=expected:
    raise ValueError('"'"'independent bootstrap source pin differs'"'"')
sys.argv=[str(path),*sys.argv[3:]]
exec(compile(data,str(path),'"'"'exec'"'"'),{'"'"'__name__'"'"':'"'"'__main__'"'"','"'"'__file__'"'"':str(path)})
' /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_campaign_gpu_20261109_02/inputs/qualify_llm_intervention_scientific_campaign_gpu.py 696be95ff571d584f4600b5a87fe6a442eaa2da863d07c0d464ac22d4e1de1fa --packet /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_campaign_gpu_20261109_02/inputs/packet.json --packet-sha256 7d37d1ed4aca9b6471ec2e9a54984ada29c0bc7b36a59e03f10aea51b9998205 --admission /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_campaign_gpu_20261109_02/inputs/admission.json --admission-sha256 945dec5bd002c8ecd860b9cc701bc2dbecfe181a86bbe295a89671ab707c13a6 --output /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_campaign_gpu_20261108_01/gpu
