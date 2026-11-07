#!/bin/bash
#SBATCH --job-name=ace_llm_meta01
#SBATCH --account=ucb736_asc2
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_runtime_metadata_20261107_01/logs/%j.out
#SBATCH --error=/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_runtime_metadata_20261107_01/logs/%j.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=
exec /projects/paco0228/miniconda3/envs/ace/bin/python -I -B -c 'import hashlib,sys
p=sys.argv[1]; raw=open(p,"rb").read(); expected=sys.argv[2]
if hashlib.sha256(raw).hexdigest()!=expected: raise ValueError("diagnostic source pin differs")
sys.argv=[p]+sys.argv[3:];exec(compile(raw,p,"exec"),{"__name__":"__main__","__file__":p})' /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_runtime_metadata_20261107_01/qualify_llm_intervention_runtime_metadata.py 9f728a3df959a0e69cc110d31cf11e5c07b9dde84ffe74eb914084c713b91b49 --packet /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_runtime_metadata_20261107_01/diagnostic.json --packet-sha256 a6f1fda50fc238b8c4b19e2113fc5bbc6fafdcf315cf21cf45677a3a41b3e555 --self-sha256 9f728a3df959a0e69cc110d31cf11e5c07b9dde84ffe74eb914084c713b91b49 --helper-sha256 7664772074cc84bdc9d668f5a99aae475a9e54aafc0c85486e23bf9b88ba747e --probe-sha256 8d6f9e2761922802712a8bf2e77df6bdec3708ce1e08272864bebbf76f0d1951
