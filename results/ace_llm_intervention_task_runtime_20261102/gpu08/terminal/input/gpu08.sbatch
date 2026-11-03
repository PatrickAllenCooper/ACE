#!/bin/bash
#SBATCH --job-name=ace_llm_canon08
#SBATCH --account=ucb736_asc1
#SBATCH --partition=aa100
#SBATCH --qos=gpu-testing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_canonical_gpu_20261102_08/logs/%j.out
#SBATCH --error=/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_canonical_gpu_20261102_08/logs/%j.err
set -euo pipefail
cd /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_canonical_gpu_20261102_08/input
sha256sum --strict --check source-pins.sha256
exec /projects/paco0228/miniconda3/envs/ace/bin/python -I -B supervise_llm_intervention_canonical_gpu.py --freeze freeze.json --freeze-sha256 64ed97bb55fe7f97bd6ac36fc868c000673b6448cc78933f15dcb8abb26247d2 --output /scratch/alpine/paco0228/ACE/results/ace_llm_intervention_canonical_gpu_20261102_08/gpu08
