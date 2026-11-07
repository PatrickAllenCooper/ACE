#!/bin/bash
#SBATCH --job-name=ace_llm_stage01
#SBATCH --account=ucb736_asc2
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_staged_runtime_20261107_01/logs/%j.out
#SBATCH --error=/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_staged_runtime_20261107_01/logs/%j.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=
exec /projects/paco0228/miniconda3/envs/ace/bin/python -B -c 'import hashlib,sys; p=sys.argv[1]; raw=open(p,"rb").read(); expected=sys.argv[2]
if hashlib.sha256(raw).hexdigest()!=expected: raise ValueError("staged qualification source pin differs")
sys.argv=[p]+sys.argv[3:];exec(compile(raw,p,"exec"),{"__name__":"__main__","__file__":p})' /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_staged_runtime_20261107_01/qualify_llm_intervention_staged_import.py 9e63e644aba5c65a5c8b61d24018a8dad2282538b2834132b36da5d7516e72c7 --helper /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_staged_runtime_20261107_01/llm_intervention_runtime_archive.py --helper-sha256 7664772074cc84bdc9d668f5a99aae475a9e54aafc0c85486e23bf9b88ba747e --probe /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_staged_runtime_20261107_01/probe.py --probe-sha256 8d6f9e2761922802712a8bf2e77df6bdec3708ce1e08272864bebbf76f0d1951 --service /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_model_service_gpu_20261104_10/input/llm_intervention_model_service.py --service-sha256 908496e344f97f718e0a5061728bf206cb67df291003ffd2ac85ba4c9cd650a1 --freeze /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_model_service_gpu_20261104_10/input/frozen.json --freeze-sha256 d33bc0f4a23b4a82f1ee8f433688e706a893e9127ef41b8abe970aab46878d37 --output /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_staged_runtime_20261107_01/qualification
