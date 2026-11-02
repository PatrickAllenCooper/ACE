#!/bin/bash
#SBATCH --job-name=ace_llm_gpu03
#SBATCH --account=ucb736_asc1
#SBATCH --partition=aa100
#SBATCH --qos=gpu-testing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --time=00:15:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_gpu_20261102_03/logs/%j.out
#SBATCH --error=/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_gpu_20261102_03/logs/%j.err
set -euo pipefail
cd /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_gpu_20261102_03/input
sha256sum --strict --check source-pins.sha256
exec /projects/paco0228/miniconda3/envs/ace/bin/python -I -B supervise_llm_intervention_gpu.py --freeze freeze.json --freeze-sha256 2ef3056c6a26d9bb7ce89dd2577eaa890a8e5a5610aab9e796aec778b20281db --output /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_gpu_20261102_03/gpu03
