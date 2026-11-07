#!/bin/bash
#SBATCH --job-name=ace_llm_import01
#SBATCH --account=ucb736_asc1
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:08:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_import_diagnostic_20261107_01/logs/%j.out
#SBATCH --error=/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_import_diagnostic_20261107_01/logs/%j.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=
exec /projects/paco0228/miniconda3/envs/ace/bin/python -B -c 'import hashlib,sys; p=sys.argv[1]; raw=open(p,"rb").read(); expected=sys.argv[2]
if hashlib.sha256(raw).hexdigest()!=expected: raise ValueError("diagnostic source pin differs")
sys.argv=[p]+sys.argv[3:];exec(compile(raw,p,"exec"),{"__name__":"__main__","__file__":p})' /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_import_diagnostic_20261107_01/probe.py ed42935c49a842e417dd083a2f5c4ad6d1d9f87b802d2a820b03f859516aee98 --service /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_model_service_gpu_20261104_10/input/llm_intervention_model_service.py --service-sha256 908496e344f97f718e0a5061728bf206cb67df291003ffd2ac85ba4c9cd650a1 --freeze /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_model_service_gpu_20261104_10/input/frozen.json --freeze-sha256 d33bc0f4a23b4a82f1ee8f433688e706a893e9127ef41b8abe970aab46878d37 --output /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_import_diagnostic_20261107_01/diagnostic
