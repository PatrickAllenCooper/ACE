#!/bin/bash
#SBATCH --job-name=ace_llm_service11
#SBATCH --account=ucb736_asc2
#SBATCH --partition=aa100
#SBATCH --qos=gpu-testing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_model_service_gpu_20261107_11/logs/%j.out
#SBATCH --error=/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_model_service_gpu_20261107_11/logs/%j.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
exec /projects/paco0228/miniconda3/envs/ace/bin/python -I -B -c 'import hashlib,sys
from pathlib import Path
path=Path(sys.argv[1]); expected=sys.argv[2]
with path.open('"'"'rb'"'"') as f: data=f.read(1024**2+1)
if len(data)>1024**2 or hashlib.sha256(data).hexdigest()!=expected:
    raise ValueError('"'"'independent bootstrap source pin differs'"'"')
sys.argv=[str(path),*sys.argv[3:]]
exec(compile(data,str(path),'"'"'exec'"'"'),{'"'"'__name__'"'"':'"'"'__main__'"'"','"'"'__file__'"'"':str(path)})
' /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_model_service_gpu_20261107_11/qualify_llm_intervention_model_service.py ee75aead983f30393774cd239af53f9ef1caf7876bdef23089b9c0ae3f1c1174 --packet /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_model_service_gpu_20261107_11/integration.json --packet-sha256 c49d3cef5064e86aa6b9b660696eb4b73716995ae8e9004ddc29bdce4f41e9b7 --output /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_model_service_gpu_20261107_11/qualification
