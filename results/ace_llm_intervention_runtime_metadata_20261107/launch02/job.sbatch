#!/bin/bash
#SBATCH --job-name=ace_llm_meta02
#SBATCH --account=ucb736_asc2
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_runtime_metadata_20261107_02/logs/%j.out
#SBATCH --error=/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_runtime_metadata_20261107_02/logs/%j.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=
exec /projects/paco0228/miniconda3/envs/ace/bin/python -I -B -c 'import hashlib,sys
p=sys.argv[1]; raw=open(p,"rb").read(); expected=sys.argv[2]
if hashlib.sha256(raw).hexdigest()!=expected: raise ValueError("diagnostic source pin differs")
sys.argv=[p]+sys.argv[3:];exec(compile(raw,p,"exec"),{"__name__":"__main__","__file__":p})' /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_runtime_metadata_20261107_02/qualify_llm_intervention_runtime_metadata.py 31bb109e2ce9603e72fe97e1cb3fcfdff2b1a72b3f7eaf74431ca92f1cd7c329 --packet /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_runtime_metadata_20261107_02/diagnostic.json --packet-sha256 80107304e27dca7a1ff328487d3189210cd5509bdf2051e26641ab7c28dff151 --self-sha256 31bb109e2ce9603e72fe97e1cb3fcfdff2b1a72b3f7eaf74431ca92f1cd7c329 --helper-sha256 7664772074cc84bdc9d668f5a99aae475a9e54aafc0c85486e23bf9b88ba747e --probe-sha256 8d6f9e2761922802712a8bf2e77df6bdec3708ce1e08272864bebbf76f0d1951
