#!/bin/bash
#SBATCH --job-name=ace_llm_scientific_train01
#SBATCH --account=ucb736_asc2
#SBATCH --partition=aa100
#SBATCH --qos=gpu-testing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=15360M
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --time=00:45:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_training_gpu_20261108_01/logs/%j.out
#SBATCH --error=/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_training_gpu_20261108_01/logs/%j.err
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
' /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_training_gpu_20261108_01/inputs/supervise_llm_intervention_scientific_training_gpu.py 55be5ba74fa16f3e05d5632a1d0ff4ac1b0a3187507849e82d6594810f3545fd --packet /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_training_gpu_20261108_01/inputs/training.json --packet-sha256 b12fd2d7c0879618b48232cdd46bada5a3c1c16a9773680d3769a24a99e7aaf0 --admission /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_training_gpu_20261108_01/inputs/admission.json --admission-sha256 4f098af6a35c629ce93d7cfc63bdd8400b65e25c8e52b92974b076718c9693ac --output /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_training_gpu_20261108_01/training
